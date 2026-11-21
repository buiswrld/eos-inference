"""Simulator v2 — the explicit statistical model (numpy only, fork-safe).

Fixes relative to v1 (see docs/CHANGES_v2.md):
  * B1/B2  every sigma is drawn from a broad log-uniform range *independent of theta*.
           v1 used sigma_Lt = 0.3*Lt_true (which leaks Lt_true to the network) and a
           constant sigma_Mc = 0.01 (normalisation sd clipped to 1e-6, so real GW170817
           was pushed ~9000 sd out of distribution).
  * B6     N ~ U{1..50}, Dirichlet-multinomial composition (messengers may be absent),
           shuffled order, so catalog size is no longer confounded with messenger type.
  * The generative model is written down exactly so that ``likelihood.py`` can
    implement the same p(catalog | theta).

Generative model (per accepted EOS branch M -> (R, Lambda), M_max = branch[-1]):
  GW     : m1, m2 ~ iid U(1.1, min(M_max, 2.0)); sLt ~ LogU(40, 800); sMc ~ LogU(5e-4, 3e-2)
           obs = (Lt(m1,m2) + sLt*e1, Mc(m1,m2) + sMc*e2)
  NICER  : m ~ U(1.2, min(M_max, 2.2)); sM ~ LogU(0.02, 0.2); sR ~ LogU(0.3, 1.6)
           obs = (m + sM*e1, R(m) + sR*e2)            [uncorrelated Gaussian]
  Pulsar : m ~ U(1.4, min(M_max, 2.5)); s ~ LogU(0.01, 0.25); obs = m + s*e
"""
import numpy as np

from .config import DEFAULT_CONFIG

SIM_CFG = DEFAULT_CONFIG["sim"]
OBS_TYPES = ("gw", "nicer", "pulsar")
OBS_DIMS = {"gw": 2, "nicer": 2, "pulsar": 1}


def interp_lambda(m, Ms, Ls):
    """Lambda(M) interpolated in log space. Lambda ~ M^-6, so linear interpolation on a
    30-point branch erred by up to ~7 %; log-linear interpolation is used everywhere
    (simulator, likelihood, analyses) so all components share one forward model."""
    return np.exp(np.interp(m, Ms, np.log(np.maximum(Ls, 1e-12))))


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def lambda_tilde(m1, m2, L1, L2):
    return 16.0 / 13.0 * ((m1 + 12 * m2) * m1 ** 4 * L1 + (m2 + 12 * m1) * m2 ** 4 * L2) / (m1 + m2) ** 5


def _loguniform(rng, lo_hi):
    lo, hi = lo_hi
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def stable_branch(curve):
    """Stable branch (up to M_max) with strictly increasing masses, so that
    ``np.interp(m, Ms, .)`` is well defined."""
    Ms, Rs, Ls = (np.asarray(a, dtype=float) for a in curve)
    i = int(np.argmax(Ms))
    Ms, Rs, Ls = Ms[: i + 1], Rs[: i + 1], Ls[: i + 1]
    keep = np.r_[True, Ms[1:] > np.maximum.accumulate(Ms[:-1])]
    return Ms[keep], Rs[keep], Ls[keep]


def accept_eos(curve, cfg=SIM_CFG):
    """Acceptance rule that defines the truncated prior. Returns (branch, reason)."""
    if curve is None:
        return None, "tov_solver_none"
    Ms, Rs, Ls = stable_branch(curve)
    if len(Ms) < 3:
        return None, "insufficient_mr_points"
    if Ms[-1] < cfg["MMAX_MIN"]:
        return None, "mmax_below_min"
    if Ms[0] > cfg["BRANCH_MMIN_MAX"]:
        return None, "branch_starts_above_1.0"
    return (Ms, Rs, Ls), None


def draw_counts(rng, cfg=SIM_CFG, allowed_types=None):
    alpha = np.array(cfg["DIRICHLET_ALPHA"], dtype=float)
    if allowed_types is not None:
        alpha = np.array([a if t in allowed_types else 0.0 for a, t in zip(alpha, OBS_TYPES)])
    n = int(rng.integers(cfg["N_MIN"], cfg["N_MAX"] + 1))
    p = np.zeros(3)
    nz = alpha > 0
    p[nz] = rng.dirichlet(alpha[nz])
    return rng.multinomial(n, p)


def simulate_catalog(branch, rng, cfg=SIM_CFG, counts=None, sigma_scale=1.0, noise_scale=1.0,
                     sigma_fixed=None, allowed_types=None, shuffle=True):
    """Draw one catalog for an accepted EOS branch.

    counts        : (n_gw, n_nicer, n_psr); default: N ~ U{N_MIN..N_MAX}, Dirichlet-multinomial.
    sigma_scale   : multiply the drawn sigmas (covariate shift).
    noise_scale   : actual noise = noise_scale * sigma while sigma is reported (misspecification).
    sigma_fixed   : dict type -> tuple of fixed sigmas (consistency tests).
    allowed_types : restrict messengers (messenger ablation).
    """
    Ms, Rs, Ls = (np.asarray(a, dtype=float) for a in branch)
    Mmax = float(Ms[-1])
    if counts is None:
        counts = draw_counts(rng, cfg, allowed_types)
    n_gw, n_nicer, n_psr = (int(c) for c in counts)
    sf = sigma_fixed or {}
    obs = []

    hi = min(Mmax, cfg["GW_M_HI"])
    for _ in range(n_gw):
        m1, m2 = rng.uniform(cfg["GW_M_LO"], hi, size=2)
        L1, L2 = interp_lambda([m1, m2], Ms, Ls)
        Lt, Mc = float(lambda_tilde(m1, m2, L1, L2)), float(chirp_mass(m1, m2))
        sL, sMc = sf.get("gw", (_loguniform(rng, cfg["GW_SIG_LT"]), _loguniform(rng, cfg["GW_SIG_MC"])))
        sL, sMc = sL * sigma_scale, sMc * sigma_scale
        obs.append({"type": "gw",
                    "x": [Lt + noise_scale * sL * rng.standard_normal(),
                          Mc + noise_scale * sMc * rng.standard_normal()],
                    "sigma": [sL, sMc], "truth": [Lt, Mc, float(m1), float(m2)]})

    hi = min(Mmax, cfg["NICER_M_HI"])
    for _ in range(n_nicer):
        m = rng.uniform(cfg["NICER_M_LO"], hi)
        R = float(np.interp(m, Ms, Rs))
        sm, sR = sf.get("nicer", (_loguniform(rng, cfg["NICER_SIG_M"]), _loguniform(rng, cfg["NICER_SIG_R"])))
        sm, sR = sm * sigma_scale, sR * sigma_scale
        obs.append({"type": "nicer",
                    "x": [m + noise_scale * sm * rng.standard_normal(), R + noise_scale * sR * rng.standard_normal()],
                    "sigma": [sm, sR], "truth": [float(m), R]})

    hi = min(Mmax, cfg["PSR_M_HI"])
    for _ in range(n_psr):
        m = rng.uniform(cfg["PSR_M_LO"], hi)
        sm = sf.get("pulsar", (_loguniform(rng, cfg["PSR_SIG_M"]),))[0] * sigma_scale
        obs.append({"type": "pulsar", "x": [m + noise_scale * sm * rng.standard_normal()],
                    "sigma": [sm], "truth": [float(m)]})

    if shuffle:
        obs = [obs[i] for i in rng.permutation(len(obs))]
    return obs


def branch_summary(branch):
    Ms, Rs, Ls = branch
    return {"Mmax": float(Ms[-1]), "R14": float(np.interp(1.4, Ms, Rs)),
            "L14": float(interp_lambda(1.4, Ms, Ls))}


def restrict_catalog(catalog, allowed_types):
    return [o for o in catalog if o["type"] in allowed_types]
