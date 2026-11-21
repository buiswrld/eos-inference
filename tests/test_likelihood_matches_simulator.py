"""The Dynesty likelihood must be the exact marginal of the simulator (fix B3).

Uses an analytic EOS branch, so it runs with numpy only (no TOV / scipy / torch):
    python -m pytest tests -q        (or)        python tests/test_likelihood_matches_simulator.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nseos.likelihood import (CurveEOS, loglike_catalog_eos, loglike_gw, loglike_nicer,  # noqa: E402
                              loglike_pulsar, _q_min, _g)
from nseos.simulator import SIM_CFG, chirp_mass, interp_lambda, lambda_tilde, simulate_catalog  # noqa: E402

MS = np.linspace(0.8, 2.3, 200)
BRANCH = (MS, 12.0 - 0.6 * (MS - 1.4) - 0.4 * (MS - 1.4) ** 2, 300.0 * (1.4 / MS) ** 6)
EOS = CurveEOS(BRANCH)


def _mc_density(obs_type, x_obs, sigma, n=400_000, seed=0):
    """Brute-force marginal density p(x_obs | theta) by Monte Carlo over the nuisance masses."""
    rng = np.random.default_rng(seed)
    Ms, Rs, Ls = BRANCH
    if obs_type == "pulsar":
        m = rng.uniform(1.4, min(Ms[-1], 2.5), n)
        return np.mean(np.exp(-0.5 * ((x_obs[0] - m) / sigma[0]) ** 2) / (sigma[0] * math.sqrt(2 * math.pi)))
    if obs_type == "nicer":
        m = rng.uniform(1.2, min(Ms[-1], 2.2), n)
        R = np.interp(m, Ms, Rs)
        return np.mean(np.exp(-0.5 * ((x_obs[0] - m) / sigma[0]) ** 2 - 0.5 * ((x_obs[1] - R) / sigma[1]) ** 2)
                       / (2 * math.pi * sigma[0] * sigma[1]))
    m1, m2 = rng.uniform(1.1, 2.0, (2, n))
    Lt = lambda_tilde(m1, m2, interp_lambda(m1, Ms, Ls), interp_lambda(m2, Ms, Ls))
    Mc = chirp_mass(m1, m2)
    return np.mean(np.exp(-0.5 * ((x_obs[0] - Lt) / sigma[0]) ** 2 - 0.5 * ((x_obs[1] - Mc) / sigma[1]) ** 2)
                   / (2 * math.pi * sigma[0] * sigma[1]))


def test_qmin_support():
    lo, hi = 1.1, 2.0
    Mc = np.array([1.0, 1.2, 1.5, 1.7])
    q = _q_min(Mc, lo, hi)
    m1, m2 = Mc * _g(q), Mc * q * _g(q)
    assert np.all((m2 >= lo - 1e-6) & (m1 <= hi + 1e-6))
    assert np.all((np.abs(m2 - lo) < 1e-6) | (np.abs(m1 - hi) < 1e-6))  # boundary is active


def test_pulsar_closed_form_vs_mc():
    for x, s in [(1.9, 0.02), (2.35, 0.17), (2.6, 0.05), (1.3, 0.1)]:
        ll = loglike_pulsar({"x": [x], "sigma": [s]}, EOS)
        # 6-sigma tail case has ~1.5% MC noise at 4e5 draws
        assert abs(math.exp(ll) / _mc_density("pulsar", [x], [s], n=2_000_000) - 1) < 0.02, (x, s)


def test_nicer_vs_mc():
    for x, s in [([1.4, 12.0], [0.05, 0.5]), ([2.07, 11.5], [0.067, 1.1]), ([1.42, 11.4], [0.02, 0.3]),
                 ([1.3, 13.5], [0.15, 1.2])]:
        ll = loglike_nicer({"x": x, "sigma": s}, EOS)
        mc = _mc_density("nicer", x, s)
        assert abs(math.exp(ll) / mc - 1) < 0.02, (x, s, math.exp(ll), mc)


def test_gw_vs_mc():
    # sMc must not be tiny here, otherwise brute-force MC has too few effective draws.
    for x, s in [([300.0, 1.20], [150.0, 0.02]), ([600.0, 1.05], [300.0, 0.03]),
                 ([100.0, 1.50], [60.0, 0.025]), ([250.0, 1.30], [800.0, 0.03])]:
        ll = loglike_gw({"x": x, "sigma": s}, EOS)
        mc = _mc_density("gw", x, s, n=2_000_000)
        assert abs(math.exp(ll) / mc - 1) < 0.03, (x, s, math.exp(ll), mc)


def test_gw_normalisation_tight_mc():
    """With sMc = 1e-3 (GW170817-like), ∫∫ p dLt dMc must still be 1."""
    s = [100.0, 1e-3]
    Lt = np.linspace(-600, 3000, 241)
    Mc = np.linspace(0.93, 1.77, 841)
    tot = 0.0
    for mc in Mc:
        row = [math.exp(loglike_gw({"x": [lt, mc], "sigma": s}, EOS)) for lt in Lt[::8]]
        tot += np.trapezoid(row, Lt[::8]) if hasattr(np, "trapezoid") else np.trapz(row, Lt[::8])
    tot *= Mc[1] - Mc[0]
    assert abs(tot - 1) < 0.03, tot


def test_simulator_draws_have_high_likelihood_under_truth():
    """log p(catalog | true EOS) > log p(catalog | shifted EOS) for most simulated catalogs."""
    rng = np.random.default_rng(1)
    Ms, Rs, Ls = BRANCH
    wrong = CurveEOS((Ms, Rs + 0.8, Ls * 1.6))
    wins = 0
    for _ in range(40):
        cat = simulate_catalog(BRANCH, rng, SIM_CFG, counts=(3, 3, 1))
        wins += loglike_catalog_eos(cat, EOS) > loglike_catalog_eos(cat, wrong)
    assert wins >= 36, wins


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
