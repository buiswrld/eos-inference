"""Likelihood implied *exactly* by the simulator (fix B3) + the TOV interpolation grid.

v1's Dynesty likelihood targeted a different posterior than the network:
  * GW: assumed m1 = m2 from Mc (simulator: m1, m2 ~ iid U(1.1, 2.0)) and ignored Mc noise;
  * NICER: used M_obs as the true mass (ignored sigma_M);
  * pulsar: one-sided soft penalty (simulator: m ~ U(1.4, min(Mmax, 2.5)) + noise);
  * prior: plain box (simulator: box truncated by accept_eos);
  * coarse 10^4 grid with NaN holes filled by neighbour averages, never validated.

Here every term is the marginal of the simulator's generative model:
  GW     : 2/(hi-lo)^2 ∫dMc N(Mc_obs; Mc, sMc) ∫_{q_min(Mc)}^1 dq Mc g(q)^2 N(Lt_obs; Lt(m1, m2), sLt),
           m1 = Mc g(q), m2 = q m1, g(q) = (1+q)^(1/5) q^(-3/5)     [(m1,m2) -> (Mc,q) Jacobian = Mc g^2]
  NICER  : 1/(hi-lo) ∫_lo^hi dm N(M_obs; m, sM) N(R_obs; R(m), sR)
  Pulsar : [Phi((u - m)/s) - Phi((1.4 - m)/s)] / (u - 1.4),  u = min(Mmax, 2.5)
  Prior  : uniform box x 1[accept_eos]
All integrals use Gauss-Legendre in log space. The core is numpy-only (unit-testable).
"""
import math

from .grid import build_grid, mass_grid

import numpy as np

from .simulator import SIM_CFG, chirp_mass, interp_lambda, lambda_tilde, stable_branch  # noqa: F401

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
_GL = {n: np.polynomial.legendre.leggauss(n) for n in (16, 24, 32, 48, 64, 96, 128, 192)}


def _gl_nodes(a, b, n):
    """Gauss-Legendre nodes/weights on [a, b] (a, b may be arrays -> broadcast on last axis)."""
    x, w = _GL[n]
    a, b = np.asarray(a, float)[..., None], np.asarray(b, float)[..., None]
    half = 0.5 * (b - a)
    return a + half * (x + 1.0), half * w


def _logsumexp(a):
    a = np.asarray(a, float)
    m = np.max(a)
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(a - m))))


def _lognorm(x, mu, s):
    return -0.5 * ((x - mu) / s) ** 2 - np.log(s) - LOG_SQRT_2PI


def _log_ndtr(z):
    if z > -30.0:
        return math.log(0.5 * math.erfc(-z / math.sqrt(2.0)))
    return -0.5 * z * z - math.log(-z) - LOG_SQRT_2PI + math.log1p(-1.0 / (z * z) + 3.0 / z ** 4)


def _log_ndtr_diff(b, a):
    """log(Phi(b) - Phi(a)) for b > a, stable in both tails."""
    if b <= a:
        return -np.inf
    if a > 0:  # upper tail: Phi(b)-Phi(a) = Phi(-a)-Phi(-b)
        b, a = -a, -b
    lb, la = _log_ndtr(b), _log_ndtr(a)
    d = la - lb
    if d >= 0:
        return -np.inf
    return lb + math.log1p(-math.exp(d))


def _g(q):
    return (1.0 + q) ** 0.2 * q ** -0.6


def _q_min(Mc, lo, hi, n_iter=50):
    """Smallest q in (0, 1] with m2 = Mc q g(q) >= lo and m1 = Mc g(q) <= hi (vectorised bisection).
    m1 decreases and m2 increases with q, so the support in q is the single interval [q_min, 1]."""
    Mc = np.asarray(Mc, float)
    a, b = np.full_like(Mc, 1e-4), np.ones_like(Mc)
    for _ in range(n_iter):
        mid = 0.5 * (a + b)
        ok = (Mc * mid * _g(mid) >= lo) & (Mc * _g(mid) <= hi)
        b = np.where(ok, mid, b)
        a = np.where(ok, a, mid)
    at1 = Mc * 2.0 ** 0.2
    feasible = (at1 >= lo) & (at1 <= hi)
    return np.where(feasible, b, np.nan)


class CurveEOS:
    """Exact EOS from a stored stable branch, interpolated exactly as the simulator does
    (linear in R, log-linear in Lambda), with the same interface as a grid point."""

    def __init__(self, branch):
        Ms, Rs, Ls = (np.asarray(a, float) for a in branch)
        self.Ms, self.Rs, self.Ls = Ms, Rs, Ls
        self.Mmax, self.Mmin, self.valid = float(Ms[-1]), float(Ms[0]), 1.0

    def R(self, m):
        return np.interp(m, self.Ms, self.Rs)

    def L(self, m):
        return interp_lambda(m, self.Ms, self.Ls)


class _GridPoint:
    def __init__(self, vals, m_grid):
        self.Mmax, self.Mmin, self.valid = float(vals[0]), float(vals[1]), float(vals[2])
        K = len(m_grid)
        self._R, self._logL, self.m_grid = vals[3:3 + K], vals[3 + K:3 + 2 * K], m_grid

    def R(self, m):
        return np.interp(m, self.m_grid, self._R)

    def L(self, m):
        return np.exp(np.interp(m, self.m_grid, self._logL))


def loglike_gw(obs, eos, cfg=SIM_CFG, n_mc=96, n_q=32):
    Lt_obs, Mc_obs = obs["x"]
    sLt, sMc = obs["sigma"]
    lo, hi = cfg["GW_M_LO"], min(eos.Mmax, cfg["GW_M_HI"])
    if hi <= lo:
        return -np.inf
    mc_lo, mc_hi = chirp_mass(lo, lo), chirp_mass(hi, hi)
    nearest = np.clip(Mc_obs, mc_lo, mc_hi)
    width = np.hypot(nearest - Mc_obs, 10 * sMc)
    a, b = max(mc_lo, Mc_obs - width), min(mc_hi, Mc_obs + width)
    # The limiting component for q_min switches at (hi, lo). Split there and
    # at the measured chirp mass to resolve narrow tidal ridges on either side.
    cuts = sorted({a, b, *(v for v in (chirp_mass(hi, lo), Mc_obs) if a < v < b)})
    edges = np.asarray(cuts)
    Mc, wMc = (v.ravel() for v in _gl_nodes(edges[:-1], edges[1:], n_mc))
    qmin = _q_min(Mc, lo, hi)
    ok = np.isfinite(qmin) & (qmin < 1.0)
    if not ok.any():
        return -np.inf
    Mc, wMc, qmin = Mc[ok], wMc[ok], qmin[ok]
    q, wq = _gl_nodes(qmin, np.ones_like(qmin), n_q)          # (n_mc, n_q)
    m1 = Mc[:, None] * _g(q)
    m2 = q * m1
    Lt = lambda_tilde(m1, m2, eos.L(m1), eos.L(m2))
    log_int = (np.log(wMc)[:, None] + np.log(wq) + _lognorm(Mc_obs, Mc, sMc)[:, None]
               + np.log(Mc[:, None] * _g(q) ** 2) + _lognorm(Lt_obs, Lt, sLt))
    return _logsumexp(log_int) + math.log(2.0) - 2.0 * math.log(hi - lo)


def loglike_nicer(obs, eos, cfg=SIM_CFG, n_m=48):
    M_obs, R_obs = obs["x"]
    sM, sR = obs["sigma"]
    lo, hi = cfg["NICER_M_LO"], min(eos.Mmax, cfg["NICER_M_HI"])
    if hi <= lo:
        return -np.inf
    nearest = np.clip(M_obs, lo, hi)
    width = np.hypot(nearest - M_obs, 10 * sM)
    a, b = max(lo, M_obs - width), min(hi, M_obs + width)
    m, w = _gl_nodes(a, b, n_m)              # shape (n_m,)
    log_int = np.log(w) + _lognorm(M_obs, m, sM) + _lognorm(R_obs, eos.R(m), sR)
    return _logsumexp(log_int) - math.log(hi - lo)


def loglike_pulsar(obs, eos, cfg=SIM_CFG):
    m_obs, s = obs["x"][0], obs["sigma"][0]
    lo, u = cfg["PSR_M_LO"], min(eos.Mmax, cfg["PSR_M_HI"])
    if u <= lo:
        return -np.inf
    return _log_ndtr_diff((u - m_obs) / s, (lo - m_obs) / s) - math.log(u - lo)


_LL = {"gw": loglike_gw, "nicer": loglike_nicer, "pulsar": loglike_pulsar}


def eos_in_support(eos, cfg=SIM_CFG):
    return eos.valid >= 0.5 and eos.Mmax >= cfg["MMAX_MIN"] and eos.Mmin <= cfg["BRANCH_MMIN_MAX"]


def loglike_catalog_eos(catalog, eos, cfg=SIM_CFG):
    if not eos_in_support(eos, cfg):
        return -np.inf
    total = 0.0
    for o in catalog:
        total += _LL[o["type"]](o, eos, cfg)
        if not np.isfinite(total):
            return -np.inf
    return total


# ── TOV interpolation grid ─────────────────────────────────────────────────────
class GridEOS:
    def __init__(self, grid):
        from scipy.interpolate import RegularGridInterpolator
        self.m_grid = np.asarray(grid["m_grid"])
        self.interp = RegularGridInterpolator(tuple(grid["axes"]), grid["values"], method="linear",
                                              bounds_error=False, fill_value=None)

    def __call__(self, theta):
        return _GridPoint(self.interp(np.atleast_2d(theta))[0], self.m_grid)

    def batch(self, thetas):
        return self.interp(np.atleast_2d(thetas))


def make_loglike(catalog, grid_eos, cfg=SIM_CFG):
    def ll(theta):
        return loglike_catalog_eos(catalog, grid_eos(theta), cfg)
    return ll


def validate_grid(grid_eos, samples, probe, cfg, m_max_check=2.2):
    """Interpolation error vs exact curves on held-out thetas + truncation-boundary accuracy."""
    m_grid = grid_eos.m_grid
    dR, dlogL, dMmax = [], [], []
    for s in samples:
        Ms, Rs, Ls = s["curve"]
        g = grid_eos(s["theta_eos"])
        m = m_grid[(m_grid >= 1.0) & (m_grid <= min(Ms[-1], m_max_check))]
        dR.append(g.R(m) - np.interp(m, Ms, Rs))
        dlogL.append(np.log(g.L(m)) - np.log(interp_lambda(m, Ms, Ls)))  # simulator convention
        dMmax.append(g.Mmax - Ms[-1])
    dR, dlogL, dMmax = np.concatenate(dR), np.concatenate(dlogL), np.array(dMmax)
    pred = np.array([eos_in_support(grid_eos(t), cfg["sim"]) for t in probe["theta"]])
    true = np.asarray(probe["accepted"])

    def q(a):
        a = np.abs(a)
        return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)), "max": float(a.max())}
    return {"abs_err_R_km": q(dR), "abs_err_logLambda": q(dlogL), "abs_err_Mmax": q(dMmax),
            "frac_R_err_gt_0.1km": float(np.mean(np.abs(dR) > 0.1)),
            "support_accuracy": float(np.mean(pred == true)),
            "support_false_accept": float(np.mean(pred & ~true)),
            "support_false_reject": float(np.mean(~pred & true)),
            "n_theta": len(samples), "n_probe": int(len(true))}
