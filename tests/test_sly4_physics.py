"""Cross-check the radial Riccati solver with pressure-coordinate metric perturbations."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from nseos import physics as p


def reference_star(pc, energy):
    # G=c=1, lengths in km. Integrate H and H' instead of the production y=rH'/H.
    unit = p.G_CGS * p.MEV_FM3_TO_CGS * 1e10 / p.C_LIGHT**4
    solar_km = p.G_CGS * p.M_SUN_CGS / (p.C_LIGHT**2 * 1e5)
    radius = 0.001
    ec, pressure = energy(np.log(pc)) * unit, pc * unit
    start = pressure - (2 * np.pi / 3) * (ec + pressure) * (ec + 3 * pressure) * radius**2
    state = [radius, 4 * np.pi * ec * radius**3 / 3, radius**2, 2 * radius]
    top = np.log(start / unit)
    # Restart at each slope change to evaluate one-sided derivatives unambiguously.
    for i in reversed(range(np.searchsorted(energy.x, top))):
        lower = energy.x[i]
        upper = energy.x[i + 1]

        def rhs(x, state):
            r, m, h, dh = state
            pr = np.exp(x) * unit
            e = energy(x) * unit
            a, b = max(x - 1e-6, lower), min(x + 1e-6, upper)
            d_e_dp = (energy(b) - energy(a)) / (np.exp(b) - np.exp(a))
            redshift = 1 - 2 * m / r
            gravity = (m + 4 * np.pi * r**3 * pr) / (r**2 * redshift)
            dr_dx = -pr / ((e + pr) * gravity)
            f = (1 + 4 * np.pi * r**2 * (pr - e)) / redshift
            q = (4 * np.pi * (5 * e + 9 * pr + (e + pr) * d_e_dp) - 6 / r**2) / redshift - 4 * gravity**2
            return dr_dx * np.array([1, 4 * np.pi * r**2 * e, dh, -(1 + f) * dh / r - q * h])

        solution = solve_ivp(rhs, (top, lower), state, method="DOP853", rtol=1e-9, atol=1e-16)
        assert solution.success
        state, top = solution.y[:, -1], lower
    r, m, h, dh = state
    c, y = m / r, r * dh / h
    a = 2 * c * (y - 1) - y + 2
    b = (2 * c * (6 - 3 * y + 3 * c * (5 * y - 8))
         + 4 * c**3 * (13 - 11 * y + c * (3 * y - 2) + 2 * c**2 * (1 + y))
         + 3 * (1 - 2 * c)**2 * a * np.log1p(-2 * c))
    return np.array([m / solar_km, r, (16 / 15) * (1 - 2 * c)**2 * a / b])


@pytest.mark.parametrize("theta,pc", [([1, 0, 0, 0], 100), ([1.5, -.25, 0, 0], 40)])
def test_pressure_coordinate_reference(theta, pc):
    energy, gamma = p.build_eos(theta)
    result = p.solve_tov_single(pc, energy, gamma)
    np.testing.assert_allclose(result, reference_star(pc, energy), rtol=1e-3)


def test_crust_derivative_join_and_causal_domain():
    energy, gamma = p.build_eos([1.5, -.25, 0, 0], continued=False)
    x = (energy.x[:-1] + energy.x[1:]) / 2
    eps = energy(x)
    cs2 = gamma(x) * np.exp(x) / (eps + np.exp(x))
    delta = 1e-7
    derivative = (np.exp(x + delta) - np.exp(x - delta)) / (energy(x + delta) - energy(x - delta))
    np.testing.assert_allclose(cs2, derivative, rtol=1e-6)
    for side in (-1, 1):
        join = np.log(p.P_NUC_MEV)
        side_x = join + side*delta
        diff = (np.exp(side_x) - np.exp(join)) / (energy(side_x) - energy(join))
        side_cs2 = gamma(side_x) * np.exp(side_x) / (energy(side_x) + np.exp(side_x))
        np.testing.assert_allclose(side_cs2, diff, rtol=1e-6)
    np.testing.assert_allclose(gamma(x[x > join]), p.spectral_gamma(np.exp(x[x > join]), [1.5, -.25, 0, 0]))
    assert np.all(cs2[x < energy.max_log_pressure] < 1)
    assert p.solve_tov_single(np.exp(energy.max_log_pressure + .01), energy, gamma) == (None, None, None)
    np.testing.assert_allclose(energy(np.log(3) + np.array([-delta, delta])), [p._MATCH_E] * 2, rtol=2e-7)
    changed, _ = p.build_eos([1.5, -.25, 0, 0], gamma_crust=1.4)
    assert abs(np.log(changed(np.log(.1)) / energy(np.log(.1)))) > .01



def test_full_sly4_canonical_star():
    from scipy.optimize import brentq
    eos = p._EnergyDensity(p._SLY_P, p._SLY_E)
    def star(pc):
        return p.solve_tov_single(pc, eos, eos.gamma)
    pc = brentq(lambda pc: star(pc)[0] - 1.4, 30, 200)
    mass, radius, tidal = star(pc)
    assert abs(radius - 11.7) < 0.15
    assert 280 < tidal < 320
    np.testing.assert_allclose([mass, radius, tidal], reference_star(pc, eos), rtol=1e-3)


def test_tidal_solution_in_luminal_core():
    eos, gamma = p.build_eos([1.5, -.25, 0, 0])
    pc = 2 * np.exp(eos.join)
    np.testing.assert_allclose(p.solve_tov_single(pc, eos, gamma), reference_star(pc, eos), rtol=1e-3)


def test_newtonian_n1_polytropic_love_number():
    # Analytic Newtonian n=1 fluid apsidal constant k2=(15-pi^2)/(2*pi^2).
    k, pc = 0.00012, 0.001
    def energy(logp):
        pressure = np.exp(logp)
        return np.sqrt(pressure / k) + pressure
    m, r, tidal = p.solve_tov_single(pc, energy, lambda x: 2.0)
    compactness = p.G_CGS * m * p.M_SUN_CGS / (p.C_LIGHT**2 * r * 1e5)
    assert compactness < 0.001
    k2 = 1.5 * tidal * compactness**5
    np.testing.assert_allclose(k2, (15 - np.pi**2) / (2*np.pi**2), rtol=0.01)
