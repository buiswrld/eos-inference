"""SLy4 low-density matter joined continuously to a four-parameter spectral core.
Radii and tidal deformabilities refer to the surface at the lowest tabulated pressure.
"""
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
from numpy.polynomial import Polynomial


from .simulator import interp_lambda, stable_branch  # re-export  # noqa: F401

# CGS constants
G_CGS = 6.67430e-8
C_LIGHT = 2.99792458e10
M_SUN_CGS = 1.98892e33
KM_CM = 1.0e5
MEV_FM3_TO_CGS = 1.6022e33
CGS_TO_MEV_FM3 = 1.0 / MEV_FM3_TO_CGS

RHO_NUC = 2.8e14
EPS_NUC_MEV = RHO_NUC * C_LIGHT ** 2 * CGS_TO_MEV_FM3
P_NUC_MEV = 3.0
GAMMA_CRUST = 1.35
P_MAX_MEV = 1500.0

LOG_P_NUC = np.log(P_NUC_MEV)
LOG_P_MAX = np.log(P_MAX_MEV)
LOG_P_RANGE = LOG_P_MAX - LOG_P_NUC


def spectral_gamma(P_mev, gc):
    """Gamma(x) = exp(g0 + g1 x + g2 x^2 + g3 x^3), x = log(P/P_nuc)/log(P_max/P_nuc) in [0, 1]."""
    x = np.clip((np.log(P_mev) - LOG_P_NUC) / LOG_P_RANGE, 0, 1)
    return np.exp(gc[0] + gc[1] * x + gc[2] * x ** 2 + gc[3] * x ** 3)


_SLY_P, _SLY_E = np.loadtxt(Path(__file__).with_name("sly4.dat")).T
P_MIN_CGS = _SLY_P[0] * MEV_FM3_TO_CGS
_MATCH_E = float(np.exp(np.interp(LOG_P_NUC, np.log(_SLY_P), np.log(_SLY_E))))


class _EnergyDensity:
    """Piecewise power law in pressure. The derivative and causal limit use the same segments."""
    def __init__(self, pressure, energy):
        self.x = np.log(pressure)
        self.y = np.log(energy)
        self.slope = np.diff(self.y) / np.diff(self.x)
        if np.any(self.slope <= 0) or not np.all(np.isfinite(self.slope)):
            raise ValueError("EOS energy must increase strictly with pressure")
        # Within a segment c_s^2 = P / (slope * epsilon) is monotone.
        left = pressure[:-1] / (self.slope * energy[:-1])
        right = pressure[1:] / (self.slope * energy[1:])
        bad = np.flatnonzero(np.maximum(left, right) >= 1)
        self.max_log_pressure = float(self.x[-1])
        if len(bad):
            i = bad[0]
            stop = self.x[i]
            if left[i] < 1:
                stop -= np.log(left[i]) / (1 - self.slope[i])
            self.max_log_pressure = float(stop - 1e-10)

    def __call__(self, log_pressure):
        return np.exp(np.interp(log_pressure, self.x, self.y))

    def gamma(self, log_pressure):
        i = np.clip(np.searchsorted(self.x, log_pressure, side="right") - 1, 0, len(self.slope) - 1)
        return (1 + np.exp(log_pressure) / self(log_pressure)) / self.slope[i]


class _SpectralEnergy:
    """Continuous crust/core match with independent one-sided derivatives."""
    def __init__(self, crust, log_p, eps, gc):
        self.crust = crust
        self.x = np.r_[crust.x[:-1], log_p]
        self.log_gamma = Polynomial(gc)(Polynomial([-LOG_P_NUC / LOG_P_RANGE, 1 / LOG_P_RANGE]))
        pressure = np.exp(log_p)
        gamma = np.exp(self.log_gamma(log_p))
        self.core = CubicHermiteSpline(log_p, np.log(eps), (1 + pressure / eps) / gamma)
        self.max_log_pressure = self._causal_end() if crust.max_log_pressure >= crust.x[-1] else crust.max_log_pressure

    def __call__(self, x):
        if np.isscalar(x):
            return self.crust(x) if x < LOG_P_NUC else np.exp(self.core(x))
        return np.where(x < LOG_P_NUC, self.crust(x), np.exp(self.core(np.maximum(x, LOG_P_NUC))))

    def gamma(self, x):
        if np.isscalar(x):
            return self.crust.gamma(x) if x < LOG_P_NUC else np.exp(self.log_gamma(x))
        return np.where(x < LOG_P_NUC, self.crust.gamma(x), np.exp(self.log_gamma(np.maximum(x, LOG_P_NUC))))

    def _causal_end(self):
        # H = epsilon + P - Gamma*P must stay positive. Differentiating H with
        # respect to log P gives H/Gamma + P*[2 - Gamma*(1 + d log Gamma/d log P)].
        # H times its positive integrating factor is monotone between roots of
        # the bracket. Partition there to catch even a short acausal excursion.
        rate = self.log_gamma.deriv()
        critical = (rate * (1 + rate) + rate.deriv()).roots()
        knots = sorted([LOG_P_NUC, LOG_P_MAX] + [z.real for z in critical if abs(z.imag) < 1e-10
                                                 and LOG_P_NUC < z.real < LOG_P_MAX])
        def bracket(x):
            return 2 - np.exp(self.log_gamma(x)) * (1 + rate(x))
        def margin(x):
            return np.exp(self.core(x) - x) + 1 - np.exp(self.log_gamma(x))
        if margin(LOG_P_NUC) <= 0:
            return LOG_P_NUC
        bounds = [LOG_P_NUC]
        for a, b in zip(knots, knots[1:]):
            if bracket(a) * bracket(b) < 0:
                bounds.append(brentq(bracket, a, b))
            bounds.append(b)
        for a, b in zip(bounds, bounds[1:]):
            if margin(b) <= 0:
                return brentq(margin, a, b, xtol=1e-14)
        return LOG_P_MAX


# Numerical search ceiling only. Reaching it never supplies a maximum mass.
PC_HARD_CAP = 2.4e5


class _ContinuedEnergy:
    """Constant dP/dε above the first causal crossing or the last spectral knot."""
    def __init__(self, base):
        self.base = base
        self.join = base.max_log_pressure
        self.pressure = np.exp(self.join)
        self.energy = base(self.join)
        self.kind = "causal" if self.join < LOG_P_MAX else "table_end"
        self.cs2 = (1.0 if self.kind == "causal" else
                    min(1.0, base.gamma(self.join) * self.pressure / (self.energy + self.pressure)))
        self.max_log_pressure = np.log(PC_HARD_CAP)
        self.x = np.r_[base.x[base.x < self.join], self.join, self.max_log_pressure]

    def __call__(self, x):
        if np.isscalar(x):
            if x <= self.join:
                return self.base(x)
            return self.energy + (np.exp(x) - self.pressure) / self.cs2
        return np.where(x <= self.join, self.base(np.minimum(x, self.join)),
                        self.energy + (np.exp(x) - self.pressure) / self.cs2)

    def gamma(self, x):
        if np.isscalar(x) and x <= self.join:
            return self.base.gamma(x)
        pressure = np.exp(x)
        continued = self.cs2 * (1 + self(x) / pressure)
        if np.isscalar(x):
            return continued
        return np.where(x <= self.join, self.base.gamma(np.minimum(x, self.join)), continued)


def build_eos(gc, n_pts=300, gamma_crust=None, *, continued=True):
    log_P = np.linspace(LOG_P_NUC, LOG_P_MAX, n_pts)
    P = np.exp(log_P)
    eps = np.zeros(n_pts)
    eps[0] = _MATCH_E
    h = log_P[1] - log_P[0]
    for i in range(n_pts - 1):
        Pi, ei = P[i], eps[i]
        k1 = (ei + Pi) / spectral_gamma(Pi, gc)
        Ph = np.exp(log_P[i] + h / 2)
        Gh = spectral_gamma(Ph, gc)
        k2 = (ei + h / 2 * k1 + Ph) / Gh
        k3 = (ei + h / 2 * k2 + Ph) / Gh
        Pn = P[i + 1]
        k4 = (ei + h * k3 + Pn) / spectral_gamma(Pn, gc)
        eps[i + 1] = ei + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    low = _SLY_P < P_NUC_MEV
    p_low = _SLY_P[low]
    # The optional polytrope is used only for the crust sensitivity experiment.
    if gamma_crust is None:
        e_low = _SLY_E[low]
    else:
        if gamma_crust <= 1:
            raise ValueError("crust index must exceed one")
        e_low = _MATCH_E * (p_low / P_NUC_MEV) ** (1 / gamma_crust)
    crust = _EnergyDensity(np.r_[p_low, P_NUC_MEV], np.r_[e_low, _MATCH_E])
    energy = _SpectralEnergy(crust, log_P, eps, gc)
    if continued:
        energy = _ContinuedEnergy(energy)
    return energy, energy.gamma


TIDAL_NU_FACTOR = 2.0     # correct coefficient (see solve_tov_single)
MAX_RHS_EVALS = 30000   # deterministic work limit (v1 used a 0.5 s wall-clock timeout -> load-dependent prior)


def solve_tov_single(Pc_mev, eps_i, gam_i, rtol=1e-6, max_evals=MAX_RHS_EVALS):
    """Integrate TOV + tidal perturbation. Returns (M_sun, R_km, Lambda) or Nones.
    Fix: the integration is abandoned after a fixed number of RHS evaluations instead of a
    wall-clock time, so acceptance no longer depends on machine load."""
    if Pc_mev <= P_MIN_CGS * CGS_TO_MEV_FM3 or np.log(Pc_mev) > getattr(eps_i, "max_log_pressure", np.inf) + 1e-13:
        return None, None, None
    n_evals = [0]
    Pc_cgs = Pc_mev * MEV_FM3_TO_CGS
    ec_cgs = eps_i(np.log(Pc_mev)) * MEV_FM3_TO_CGS
    r0 = 100.0
    m0 = (4.0 / 3.0) * np.pi * r0 ** 3 * ec_cgs / C_LIGHT ** 2
    gc2 = G_CGS / C_LIGHT ** 2
    gc4 = G_CGS / C_LIGHT ** 4

    def rhs(r, s):
        n_evals[0] += 1
        if n_evals[0] > max_evals:
            raise RuntimeError("TOV work limit exceeded")
        P, m, y = s
        if P <= P_MIN_CGS or r < 1.0:
            return [0.0, 0.0, 0.0]
        Pm = max(P * CGS_TO_MEV_FM3, 1e-20)
        eps = eps_i(np.log(Pm)) * MEV_FM3_TO_CGS
        gam = gam_i(np.log(Pm))
        cs2 = gam * P / (eps + P)
        if cs2 > 1.0 + 1e-12 or cs2 <= 0.0:
            return [0.0, 0.0, 0.0]  # causality / stability violation -> integration stalls -> rejected
        beta = 2 * G_CGS * m / (r * C_LIGHT ** 2)
        if beta >= 0.99:
            return [0.0, 0.0, 0.0]
        fac = 1.0 - beta
        A = m + 4 * np.pi * r ** 3 * P / C_LIGHT ** 2
        dPdr = -(eps + P) * gc2 * A / (r ** 2 * fac)
        dmdr = 4 * np.pi * r ** 2 * eps / C_LIGHT ** 2
        # nu' for the metric g_tt = -e^nu. TOV above is dP/dr = -(eps+P) nu'/2, so nu' carries a
        # factor 2. v1 (and v2 before this fix) used nu'/2 here, i.e. 1/4 of the nu'^2 term in Q
        # (Hinderer 2008; Postnikov et al. 2010, Q = ... - nu'^2). TIDAL_NU_FACTOR exists only so
        # the bias of the old coefficient can be measured; it must stay 2.0.
        dnu = TIDAL_NU_FACTOR * gc2 * A / (r ** 2 * fac)
        e2l = 1.0 / fac
        At = e2l * (1 + 4 * np.pi * r ** 2 * (P - eps) * gc4)
        Q = e2l * (4 * np.pi * gc4 * (5 * eps + 9 * P + (eps + P) / cs2) - 6 / r ** 2) - dnu ** 2
        dydr = (-y ** 2 - y * At - r ** 2 * Q) / r
        return [dPdr, dmdr, dydr]

    def surface(r, s):
        return s[0] - P_MIN_CGS
    surface.terminal = True
    surface.direction = -1

    try:
        sol = solve_ivp(rhs, [r0, 50 * KM_CM], [Pc_cgs, m0, 2.0], events=surface, method="RK45",
                        rtol=rtol, atol=1e-9 * rtol / 1e-6, max_step=5000.0)
        if sol.t_events[0].size == 0:
            return None, None, None
        Rcm = sol.t_events[0][0]
        _, Mg, ys = sol.y_events[0][0]
        R, M = Rcm / KM_CM, Mg / M_SUN_CGS
        if M <= 0 or R <= 0 or R > 30:
            return None, None, None
        C = G_CGS * Mg / (Rcm * C_LIGHT ** 2)
        num = 8 / 5 * C ** 5 * (1 - 2 * C) ** 2 * (2 + 2 * C * (ys - 1) - ys)
        den = (2 * C * (6 - 3 * ys + 3 * C * (5 * ys - 8))
               + 4 * C ** 3 * (13 - 11 * ys + C * (3 * ys - 2) + 2 * C ** 2 * (1 + ys))
               + 3 * (1 - 2 * C) ** 2 * (2 - ys + 2 * C * (ys - 1)) * np.log1p(-2 * C))
        # At small compactness the O(C) terms cancel to O(C^5). Expand D/C^5
        # before rounding so the Newtonian Love number remains measurable.
        if C < 0.025:
            coeff = [(48 + 16*ys)/5, -16*ys/5, -(96*ys + 32)/35, -(96*ys + 64)/35,
                     -(64*ys + 64)/21, -(384*ys + 512)/105, -(768*ys + 1280)/165]
            den = C**5 * np.polynomial.polynomial.polyval(C, coeff)
        if den <= 0:
            return M, R, None
        Lam = 2 / 3 * (num / den) / C ** 5
        if Lam < 0 or not np.isfinite(Lam):
            return M, R, None
        return M, R, Lam
    except Exception:
        return None, None, None


def solve_sequence(gc, n_pc=30, gamma_crust=None, rtol=1e-6, early_stop=True, refine_peak=4,
                   refine_end=8, policy="extended", pc_cap=PC_HARD_CAP):
    """Return the curve and endpoint diagnostics. Only the legacy policy permits an unbracketed endpoint."""
    if policy not in {"extended", "strict", "lenient"} or not 6 < pc_cap <= PC_HARD_CAP:
        raise ValueError("invalid causality policy or central pressure cap")
    ei, gi = build_eos(gc, gamma_crust=gamma_crust, continued=(policy == "extended"))
    boundary = ei.join if policy == "extended" else ei.max_log_pressure
    info = {"boundary_pressure": float(np.exp(boundary)), "boundary_kind": "causal" if boundary < LOG_P_MAX else "table_end",
            "strict": False, "used_extension": False, "reached_boundary": False, "reason": None}

    def star(lpc):
        M, R, L = solve_tov_single(np.exp(lpc), ei, gi, rtol=rtol)
        return None if (M is None or L is None) else (lpc, M, R, L)

    start, cap = np.log(2 * P_NUC_MEV), np.log(pc_cap)
    end = min(boundary, cap)
    if end <= start:
        info["reason"] = "spectral_domain_below_scan"
        return None, info
    grid = np.linspace(start, end, n_pc)
    step = np.clip(grid[1] - grid[0], 0.08, 0.25)
    if policy != "lenient" and end == boundary:
        # A peak just below the match can lie entirely within the final coarse interval.
        gap = grid[-1] - grid[-2]
        grid = np.r_[grid[:-1], end - gap/2, end - min(0.001, gap/8), end]
    if policy == "extended" and cap > end:
        # Preserve exactly the same sub-transition pressure grid for all three policies.
        grid = np.r_[grid, np.arange(end + step, cap, step), cap]
    pts = []   # (log Pc, M, R, Lambda)
    n_down, first_fail_after = 0, None
    for lpc in grid:
        if lpc >= boundary - 1e-12:
            info["reached_boundary"] = True
        s = star(lpc)
        if s is not None:
            n_down = n_down + 1 if (pts and s[1] < pts[-1][1]) else 0
            pts.append(s)
            first_fail_after = None
            if early_stop and n_down >= 2 and len(pts) >= 5:
                break
        elif pts and first_fail_after is None:
            first_fail_after = lpc
            break
    if len(pts) < 5:
        info["reason"] = "insufficient_stellar_models"
        return None, info
    i = int(np.argmax([p[1] for p in pts]))
    if policy == "lenient" and refine_end and i == len(pts) - 1 and first_fail_after is not None:
        lo, hi = pts[-1][0], first_fail_after
        for _ in range(refine_end):
            mid = 0.5 * (lo + hi)
            s = star(mid)
            if s is None:
                hi = mid
            else:
                pts.append(s)
                lo = mid
        pts.sort(key=lambda p: p[0])
        i = int(np.argmax([p[1] for p in pts]))
    bracketed = 0 < i < len(pts) - 1 and pts[i-1][1] < pts[i][1] > pts[i+1][1]
    if not bracketed and policy != "lenient":
        info["reason"] = ("solver_failure_before_turnover" if first_fail_after is not None else
                          "no_turnover_by_cap" if cap <= end or policy == "extended" else "no_turnover_before_boundary")
        return None, info
    if refine_peak and bracketed:
        a, b = pts[i - 1][0], pts[i + 1][0]
        g = (np.sqrt(5) - 1) / 2
        cache = {}

        def mass_at(lpc):
            if lpc not in cache:
                M, R, L = solve_tov_single(np.exp(lpc), ei, gi, rtol=rtol)
                cache[lpc] = (M, R, L)
            return cache[lpc][0] if cache[lpc][0] is not None and cache[lpc][2] is not None else -np.inf
        c, d = b - g * (b - a), a + g * (b - a)
        for _ in range(refine_peak):
            if mass_at(c) > mass_at(d):
                b, d = d, c
                c = b - g * (b - a)
            else:
                a, c = c, d
                d = a + g * (b - a)
        for lpc, (M, R, L) in cache.items():
            if M is not None and L is not None:
                pts.append((lpc, M, R, L))
        pts.sort(key=lambda p: p[0])
    _, Ms, Rs, Ls = (np.array(v) for v in zip(*pts))
    peak = pts[int(np.argmax(Ms))][0]
    info.update(peak_pressure=float(np.exp(peak)), bracketed=bool(bracketed),
                strict=bool(bracketed and peak < boundary),
                used_extension=bool(peak > boundary))
    return (Ms, Rs, Ls), info


def compute_mr_curve(gc, n_pc=30, gamma_crust=None, rtol=1e-6, early_stop=True, refine_peak=4, refine_end=8):
    return solve_sequence(gc, n_pc, gamma_crust, rtol, early_stop, refine_peak, refine_end)[0]


def branch_for_theta(gc, n_pc=30, gamma_crust=None):
    """Stable branch (Ms, Rs, Ls) as float32 array (3, n), or None (for plotting bands)."""
    try:
        curve = compute_mr_curve(list(gc), n_pc=n_pc, gamma_crust=gamma_crust)
    except Exception:
        return None
    if curve is None:
        return None
    br = stable_branch(curve)
    return np.stack(br).astype(np.float32) if len(br[0]) >= 3 else None


def accepted_branch(gc, n_pc=30, gamma_crust=None, cfg=None):
    """Branch if theta passes accept_eos (with an optional non-default crust), else None."""
    from .simulator import accept_eos, SIM_CFG
    try:
        curve = compute_mr_curve(list(gc), n_pc=n_pc, gamma_crust=gamma_crust)
    except Exception:
        return None
    br, _ = accept_eos(curve, cfg or SIM_CFG)
    return None if br is None else np.stack(br).astype(np.float32)


def derived_observables(gc, n_pc=30, gamma_crust=None, cfg=None):
    """(M_max, R_1.4, Lambda_1.4, inside_training_support) for one theta, or None.
    v1 took the nearest grid point to 1.4 Msun; here we interpolate on the stable branch."""
    from .simulator import accept_eos, SIM_CFG
    try:
        curve = compute_mr_curve(list(gc), n_pc=n_pc, gamma_crust=gamma_crust)
    except Exception:
        return None
    if curve is None:
        return None
    Ms, Rs, Ls = stable_branch(curve)
    if len(Ms) < 3 or Ms[-1] < 1.4 or Ms[0] > 1.4:
        return (float(Ms[-1]) if len(Ms) else np.nan, np.nan, np.nan, False)
    branch, reason = accept_eos(curve, cfg or SIM_CFG)
    return (float(Ms[-1]), float(np.interp(1.4, Ms, Rs)), float(interp_lambda(1.4, Ms, Ls)), reason is None)
