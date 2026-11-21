"""Identifiability of (g0..g3) independent of the learned estimator (reviewers DVmA, GBfw, VWKG).

1. Prior-width argument: |delta log Gamma(x)| <= max|g_k| x^k  -> the g3 prior (+-0.02) changes the
   adiabatic index by at most 2%, so "g3 is unconstrained" is largely a narrow-prior statement.
2. Local Fisher analysis of the TOV map  theta -> O = [R(m_i), log Lambda(m_i), M_max]:
   J by central differences in prior-std units, F = J^T Sigma^-1 J for stated per-observable noise,
   Gaussian-approximation posterior std with a unit prior: sqrt(diag((F + I)^-1)).
   Eigen-spectrum of F = number of data-constrained directions.
3. 1-D sensitivity scans across the full prior range of each g_k.
4. Model-free check: a gradient-boosted regressor from *noise-free* full curves to theta.
"""
from functools import partial

import numpy as np

from .simulator import interp_lambda


def prior_width_table(prior_bounds):
    rows = {}
    for k in range(4):
        lo, hi = prior_bounds[f"gamma{k}"]
        a = max(abs(lo), abs(hi))
        rows[f"gamma_{k}"] = {"max_abs": a, "max_frac_change_Gamma": float(np.expm1(a))}
    return rows


def observable_vector(gc, masses, n_pc=60):
    from .physics import compute_mr_curve
    from .simulator import stable_branch
    curve = compute_mr_curve(list(gc), n_pc=n_pc)
    if curve is None:
        return None
    Ms, Rs, Ls = stable_branch(curve)
    if len(Ms) < 6 or Ms[0] > masses[0] or Ms[-1] < masses[-1]:
        return None
    R = np.interp(masses, Ms, Rs)
    lL = np.log(interp_lambda(masses, Ms, Ls))
    return np.concatenate([R, lL, [Ms[-1]]])


def _jacobian_job(theta, prior_std, masses, n_pc, step_frac):
    theta = np.asarray(theta, float)
    o0 = observable_vector(theta, masses, n_pc)
    if o0 is None:
        return None
    out = {}
    for frac_name, frac in (("h", step_frac), ("h2", step_frac / 2)):
        cols = []
        for k in range(4):
            h = frac * prior_std[k]
            tp, tm = theta.copy(), theta.copy()
            tp[k] += h
            tm[k] -= h
            op, om = observable_vector(tp, masses, n_pc), observable_vector(tm, masses, n_pc)
            if op is None or om is None:
                return None
            cols.append((op - om) / (2 * frac))    # derivative per prior-std unit
        out[frac_name] = np.stack(cols, 1)
    return {"theta": theta, "O": o0, "J": out["h"], "J_half_step": out["h2"]}


def compute_jacobians(thetas, prior_std, masses, n_pc=60, step_frac=0.1, n_cores=None):
    import multiprocessing
    worker = partial(_jacobian_job, prior_std=np.asarray(prior_std), masses=np.asarray(masses),
                     n_pc=n_pc, step_frac=step_frac)
    with multiprocessing.Pool(n_cores or multiprocessing.cpu_count()) as pool:
        res = pool.map(worker, list(thetas), chunksize=1)
    return [r for r in res if r is not None]


def fisher_summary(jacs, masses, scenarios):
    """scenarios: name -> (sigma_R [km], sigma_logLambda, sigma_Mmax) per observable."""
    K = len(masses)
    J = np.stack([j["J"] for j in jacs])                       # (n, 2K+1, 4)
    Jh = np.stack([j["J_half_step"] for j in jacs])
    fd_rel = np.linalg.norm(J - Jh, axis=1) / (np.linalg.norm(J, axis=1) + 1e-30)
    out = {"n_theta": len(jacs), "finite_difference_rel_change_median": np.median(fd_rel, 0),
           "sensitivity_R_km_per_prior_sd_max": np.median(np.abs(J[:, :K]).max(1), 0),
           "sensitivity_logLambda_per_prior_sd_max": np.median(np.abs(J[:, K:2 * K]).max(1), 0),
           "sensitivity_Mmax_per_prior_sd": np.median(np.abs(J[:, -1]), 0), "scenarios": {}}
    for name, (sR, sL, sM) in scenarios.items():
        w = np.r_[np.full(K, sR ** -2), np.full(K, sL ** -2), [sM ** -2]]
        Fs = np.einsum("noi,o,noj->nij", J, w, J)
        post_sd = np.sqrt(np.stack([np.diag(np.linalg.inv(F + np.eye(4))) for F in Fs]))   # prior-sd units
        crb = np.stack([np.sqrt(np.diag(np.linalg.pinv(F))) for F in Fs])
        ev = np.stack([np.linalg.eigvalsh(F)[::-1] for F in Fs])
        vecs = np.stack([np.linalg.eigh(F)[1][:, ::-1] for F in Fs])
        # noise scale factor s needed so that the CRB of g_k reaches 0.5 prior sd (CRB ∝ s)
        s_needed = 0.5 / crb
        out["scenarios"][name] = {
            "sigma": [sR, sL, sM],
            "posterior_sd_over_prior_sd_median": np.median(post_sd, 0),
            "posterior_sd_over_prior_sd_p16_p84": np.percentile(post_sd, [16, 84], 0),
            "crb_over_prior_sd_median": np.median(crb, 0),
            "noise_scale_for_half_prior_sd_median": np.median(s_needed, 0),
            "fisher_eigenvalues_median": np.median(ev, 0),
            "n_directions_eig_gt_1_median": float(np.median((ev > 1).sum(1))),
            "top_eigvec_abs_loadings_median": np.median(np.abs(vecs[:, :, :2]), 0)}
    return out


def _scan_job(args, masses, n_pc):
    theta, k, values = args
    rows = []
    for v in values:
        t = np.array(theta, float)
        t[k] = v
        o = observable_vector(t, masses, n_pc)
        rows.append(np.full(2 * len(masses) + 1, np.nan) if o is None else o)
    return k, np.array(rows)


def sensitivity_scans(thetas, prior_bounds, masses, n_values=11, n_pc=60, n_cores=None):
    import multiprocessing
    jobs = []
    for i, th in enumerate(thetas):
        for k in range(4):
            lo, hi = prior_bounds[f"gamma{k}"]
            jobs.append((th, k, np.linspace(lo, hi, n_values)))
    worker = partial(_scan_job, masses=np.asarray(masses), n_pc=n_pc)
    with multiprocessing.Pool(n_cores or multiprocessing.cpu_count()) as pool:
        res = pool.map(worker, jobs, chunksize=1)
    return {"jobs": [(j[0], j[1], j[2]) for j in jobs], "observables": [r[1] for r in res]}


def curve_features(sample, masses):
    Ms, Rs, Ls = sample["curve"]
    return np.concatenate([np.interp(masses, Ms, Rs), np.log(interp_lambda(masses, Ms, Ls)), [Ms[-1]]])


def model_free_regression(train, test, masses, prior, seed=0, n_train=50000):
    """Noise-free full curves -> theta with gradient boosting; R^2 and MAE ratio to prior per parameter."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    tr = train[:n_train]
    Xtr = np.stack([curve_features(s, masses) for s in tr])
    Xte = np.stack([curve_features(s, masses) for s in test])
    ytr = np.array([s["theta_eos"] for s in tr])
    yte = np.array([s["theta_eos"] for s in test])
    out = {}
    for k in range(4):
        reg = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=seed).fit(Xtr, ytr[:, k])
        p = reg.predict(Xte)
        ss_res, ss_tot = np.sum((yte[:, k] - p) ** 2), np.sum((yte[:, k] - yte[:, k].mean()) ** 2)
        out[f"gamma_{k}"] = {"r2": float(1 - ss_res / ss_tot),
                             "mae_ratio_to_prior": float(np.mean(np.abs(p - yte[:, k])) /
                                                         np.mean(np.abs(prior["median"][k] - yte[:, k])))}
    return out
