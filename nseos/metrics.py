"""Posterior-quality metrics requested by the reviewers.

* per-parameter accuracy *relative to the truncated-prior baseline* and posterior contraction
* SBC ranks, central-interval coverage curves, TARP joint coverage
* Jensen-Shannon divergence **in bits** (log base 2, bounded in [0, 1]) between 1-D marginals,
  estimated with a boundary-reflected Gaussian KDE (Scott bandwidth) on a common grid,
  at **matched sample sizes**, with a finite-sample / sampler-noise floor
* classifier two-sample test (C2ST) on the joint 4-D posterior + correlation matrices
"""
import numpy as np

QUANTILES = (0.05, 0.16, 0.5, 0.84, 0.95)


def summarize_posterior(post, theta):
    """Reducer for inference.posterior_for_samples: keeps only small per-catalog summaries."""
    q = np.quantile(post, QUANTILES, axis=1)                    # (5, B, 4)
    return {"mean": post.mean(1), "std": post.std(1), "q": np.moveaxis(q, 0, 1),
            "rank": (post < theta[:, None, :]).sum(1), "theta": theta,
            "thin": post[:, :: max(1, post.shape[1] // 200)][:, :200]}


def accuracy_table(summ, prior, names):
    """MAE of posterior median vs truncated-prior median, contraction, z-scores."""
    theta, med = summ["theta"], summ["q"][:, 2]
    prior_var = np.asarray(prior["std"]) ** 2
    rows = {}
    for k, n in enumerate(names):
        mae = np.mean(np.abs(med[:, k] - theta[:, k]))
        mae_prior = np.mean(np.abs(prior["median"][k] - theta[:, k]))
        rows[n] = {"mae_post_median": mae, "mae_post_mean": np.mean(np.abs(summ["mean"][:, k] - theta[:, k])),
                   "rmse_post_mean": np.sqrt(np.mean((summ["mean"][:, k] - theta[:, k]) ** 2)),
                   "mae_prior_median": mae_prior, "mae_ratio_to_prior": mae / mae_prior,
                   "contraction": 1.0 - np.mean(summ["std"][:, k] ** 2) / prior_var[k],
                   "learned": bool(mae / mae_prior < 0.9)}
    return rows


def coverage_curve(ranks, n_samples, levels=np.linspace(0.02, 0.98, 49)):
    """Empirical coverage of central credible intervals, per parameter. ranks: (n_sims, d)."""
    u = (ranks + 0.5) / (n_samples + 1)
    cov = np.array([[np.mean(np.abs(u[:, k] - 0.5) < lv / 2) for k in range(u.shape[1])] for lv in levels])
    se = np.sqrt(levels * (1 - levels) / len(u))[:, None]
    return levels, cov, se


def coverage_error(ranks, n_samples):
    lv, cov, _ = coverage_curve(ranks, n_samples)
    return np.mean(np.abs(cov - lv[:, None]), axis=0)


def sbc_histogram(ranks, n_samples, n_bins=20):
    hist = np.stack([np.histogram(ranks[:, k], bins=n_bins, range=(0, n_samples + 1))[0]
                     for k in range(ranks.shape[1])])
    n = len(ranks)
    p = 1.0 / n_bins
    band = (n * p - 2.58 * np.sqrt(n * p * (1 - p)), n * p + 2.58 * np.sqrt(n * p * (1 - p)))   # 99 %
    return hist, band


def tarp(samples, theta, lo, hi, n_alpha=101, n_boot=100, seed=0):
    """TARP expected coverage (Lemos et al. 2023) in prior-normalised coordinates."""
    rng = np.random.default_rng(seed)
    s = (samples - lo) / (hi - lo)
    t = (theta - lo) / (hi - lo)
    ref = rng.uniform(0, 1, size=t.shape)
    d_s = np.linalg.norm(s - ref[:, None], axis=-1)
    d_t = np.linalg.norm(t - ref, axis=-1)
    f = np.mean(d_s < d_t[:, None], axis=1)
    alpha = np.linspace(0, 1, n_alpha)
    ecp = np.array([np.mean(f < a) for a in alpha])
    boot = np.array([[np.mean(fb < a) for a in alpha] for fb in
                     (f[rng.integers(0, len(f), len(f))] for _ in range(n_boot))])
    return alpha, ecp, np.percentile(boot, [2.5, 97.5], axis=0)


# ── 1-D JS divergence (bits) with a reflected KDE ───────────────────────────────
def kde_reflect(x, grid, lo, hi, weights=None):
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float)
    w = None if weights is None else np.asarray(weights, float)
    kde = gaussian_kde(x, weights=w)        # bandwidth from the unreflected sample (Scott)
    bw = kde.factor * np.sqrt(np.cov(x, aweights=w))
    xr = np.concatenate([x, 2 * lo - x, 2 * hi - x])
    wr = None if w is None else np.tile(w, 3)
    k = gaussian_kde(xr, bw_method=bw / np.sqrt(np.cov(xr, aweights=wr)), weights=wr)
    p = k(grid)
    p /= np.trapezoid(p, grid) if hasattr(np, "trapezoid") else np.trapz(p, grid)
    return p


def js_bits(p, q, grid):
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log2(p / m), 0.0)
        kl_qm = np.where(q > 0, q * np.log2(q / m), 0.0)
    return float(0.5 * trap(kl_pm, grid) + 0.5 * trap(kl_qm, grid))


def marginal_js(a, b, lo, hi, wa=None, wb=None, n_grid=400):
    """Per-parameter JS (bits) between sample sets a and b (each (n, d), optional weights)."""
    out = []
    for k in range(a.shape[1]):
        g = np.linspace(lo[k], hi[k], n_grid)
        out.append(js_bits(kde_reflect(a[:, k], g, lo[k], hi[k], wa),
                           kde_reflect(b[:, k], g, lo[k], hi[k], wb), g))
    return np.array(out)


def matched_js(sbi, ref, ref_w, lo, hi, n, repeats=200, seed=0, n_grid=400):
    """JS between the reference (weighted) and SBI subsamples of size n = ESS(reference),
    plus the SBI-vs-SBI finite-sample floor at the same n. Returns dict of (mean, std) arrays."""
    rng = np.random.default_rng(seed)
    js, floor = [], []
    for _ in range(repeats):
        i = rng.choice(len(sbi), size=2 * n, replace=False)
        js.append(marginal_js(sbi[i[:n]], ref, lo, hi, wb=ref_w, n_grid=n_grid))
        floor.append(marginal_js(sbi[i[:n]], sbi[i[n:]], lo, hi, n_grid=n_grid))
    js, floor = np.array(js), np.array(floor)
    return {"js_mean": js.mean(0), "js_std": js.std(0), "floor_sbi_mean": floor.mean(0),
            "floor_sbi_std": floor.std(0), "n": n, "repeats": repeats}


def weighted_resample(samples, weights, n, seed=0):
    rng = np.random.default_rng(seed)
    return samples[rng.choice(len(samples), size=n, replace=True, p=weights / weights.sum())]


def c2st(a, b, seed=0, folds=5):
    """Classifier two-sample test accuracy (0.5 = indistinguishable) on standardised samples."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    n = min(len(a), len(b))
    rng = np.random.default_rng(seed)
    a, b = a[rng.choice(len(a), n, replace=False)], b[rng.choice(len(b), n, replace=False)]
    X = np.concatenate([a, b])
    _, groups = np.unique(X, axis=0, return_inverse=True)
    folds = min(folds, len(np.unique(groups[:n])), len(np.unique(groups[n:])))
    if folds < 2:
        return float("nan")
    y = np.r_[np.zeros(n), np.ones(n)]
    accs = []
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            return float("nan")
        clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=1000, early_stopping=False,
                            random_state=seed)).fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def weighted_corr(x, w=None):
    c = np.cov(x.T, aweights=w)
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def kish_ess(w):
    w = np.asarray(w, float) / np.sum(w)
    return float(1.0 / np.sum(w ** 2))
