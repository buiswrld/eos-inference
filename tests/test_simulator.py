"""Simulator properties that the v1 bugs violated (numpy only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nseos.data import compute_norm_stats  # noqa: E402
from nseos.simulator import SIM_CFG, accept_eos, simulate_catalog  # noqa: E402

MS = np.linspace(0.8, 2.3, 60)
BRANCH = (MS, 12.0 - 0.6 * (MS - 1.4), 300.0 * (1.4 / MS) ** 6)


def _many(n=3000, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [simulate_catalog(BRANCH, rng, SIM_CFG, **kw) for _ in range(n)]


def test_sigma_independent_of_truth():
    """B2: sigma_Lt must not encode Lt_true (v1: sigma = max(50, 0.3 Lt_true))."""
    obs = [o for c in _many(1500) for o in c if o["type"] == "gw"]
    lt = np.array([o["truth"][0] for o in obs])
    s = np.log([o["sigma"][0] for o in obs])
    assert abs(np.corrcoef(lt, s)[0, 1]) < 0.05


def test_sigma_mc_varies_and_covers_gw170817():
    """B1: sigma_Mc is not constant and the real value 0.001 is inside the training range."""
    s = np.array([o["sigma"][1] for c in _many(1500) for o in c if o["type"] == "gw"])
    assert s.std() > 0 and s.min() < 0.001 < s.max()
    stats = compute_norm_stats([{"theta_eos": [0, 0, 0, 0], "catalog": c} for c in _many(500)])
    z = (np.log(0.001) - stats["gw_s_mu"][1]) / stats["gw_s_sd"][1]
    assert abs(z) < 3, z     # v1: (0.001 - 0.01) / 1e-6 = -9000


def test_catalog_size_and_composition():
    """B6: N spans 1..50, messengers can be absent, order is shuffled."""
    cats = _many(3000)
    n = np.array([len(c) for c in cats])
    assert n.min() == 1 and n.max() == 50
    no_gw = np.mean([all(o["type"] != "gw" for o in c) for c in cats])
    assert 0.05 < no_gw < 0.5
    first_types = {c[0]["type"] for c in cats if len(c) > 5}
    assert first_types == {"gw", "nicer", "pulsar"}


def test_masses_within_branch_and_bounds():
    for c in _many(500):
        for o in c:
            if o["type"] == "gw":
                assert 1.1 <= o["truth"][2] <= 2.0 and 1.1 <= o["truth"][3] <= 2.0
            elif o["type"] == "nicer":
                assert 1.2 <= o["truth"][0] <= 2.2
            else:
                assert 1.4 <= o["truth"][0] <= 2.3


def test_accept_rule():
    assert accept_eos(BRANCH)[1] is None
    assert accept_eos((MS[MS < 1.45], MS[MS < 1.45], MS[MS < 1.45]))[1] == "mmax_below_min"
    hi = MS > 1.05
    assert accept_eos((MS[hi], MS[hi], MS[hi]))[1] == "branch_starts_above_1.0"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
