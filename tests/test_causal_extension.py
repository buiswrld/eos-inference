from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import shutil

import numpy as np
import pytest
import torch

from nseos.config import Paths, load_config
from nseos.physics import build_eos, solve_sequence, solve_tov_single
from nseos.importance import strict_posterior
from nseos.provenance import Fingerprints
from nseos.stages import _derived_test
from nseos.train import make_loader


@pytest.mark.parametrize("gc,kind", [([1.5, -.25, 0, 0], "causal"), ([.6, 0, 0, 0], "table_end")])
def test_constant_sound_speed_continuation(gc, kind):
    e, gamma = build_eos(gc)
    assert e.kind == kind
    x = e.join + np.log([1., 1.02, 2.])
    pressure = np.exp(x)
    np.testing.assert_allclose(e(e.join), e.base(e.join), rtol=1e-12)
    np.testing.assert_allclose(np.diff(pressure)/np.diff(e(x)), e.cs2, rtol=1e-10)
    np.testing.assert_allclose(gamma(x)*pressure/(e(x)+pressure), e.cs2, atol=1e-11)
    assert 0 < e.cs2 <= 1
    assert solve_tov_single(pressure[-1], e, gamma)[0] is not None


def test_peak_requires_two_sided_bracket_and_resolves_transition():
    stiff = [1.5, -.25, 0, 0]
    curve, info = solve_sequence(stiff, pc_cap=80)
    assert curve is None and info["reason"] == "no_turnover_by_cap"
    curve, info = solve_sequence(stiff)
    assert curve is not None and info["bracketed"] and info["used_extension"] and not info["strict"]
    assert solve_sequence(stiff, policy="strict")[0] is None
    # A turnover in the last spectral grid cell used to be missed by the strict scan.
    theta = [1.239408906995831, -.08025755953638081, -.21140145270560604, .016608308819422216]
    curve, info = solve_sequence(theta)
    restricted, other = solve_sequence(theta, policy="strict")
    assert info["strict"] and other["strict"]
    np.testing.assert_allclose(curve[0].max(), restricted[0].max(), rtol=1e-8)


def test_scoped_inputs_and_catalog_digest(tmp_path):
    cfg = load_config(overrides=[f"root={tmp_path}"])
    a = Paths(cfg)
    b = Paths({**cfg, "sampling": {**cfg["sampling"], "n_steps": 17}})
    obs = [{"type":"pulsar", "x":[1.9], "sigma":[.1]}]
    assert a.grid() == b.grid() and a.data("tov_dataset") == b.data("tov_dataset")
    assert a.run("main") == b.run("main") and a.tag_for("posterior") != b.tag_for("posterior")
    assert a.reference("real",obs) == b.reference("real",obs)
    assert a.reference("real",obs) != a.reference("real",[{**obs[0],"x":[1.8]}])
    package = tmp_path / "pkg"
    shutil.copytree(Path(__file__).resolve().parents[1]/"nseos", package, ignore=shutil.ignore_patterns("__pycache__"))
    def keys():
        fp = Fingerprints(cfg, package)
        return [fp.key(k) for k in ("simulate","grid-build","train","nested","summary")]
    before = keys()
    with (package/"plotting.py").open("a") as f:
        f.write("\n# rendering edit\n")
    assert keys() == before
    with (package/"likelihood.py").open("a") as f:
        f.write("\n# likelihood edit\n")
    likelihood = keys()
    assert likelihood[:3] == before[:3] and likelihood[3:] != before[3:]
    with (package/"sly4.dat").open("a") as f:
        f.write("\n# table edit\n")
    assert all(a != b for a,b in zip(likelihood,keys()))


class SerialPool:
    def map(self, f, values, chunksize=1):
        return list(map(f, values))


def test_conditioning_preserves_parent_weights(monkeypatch):
    def solve(theta,n_pc,sim):
        return (None,[2,12,300]) if theta[0] == 1 else ("no_turnover_before_boundary",None)
    monkeypatch.setattr("nseos.importance._strict_observables",solve)
    result = dict(samples=np.array([[1]*4,[0]*4,[1]*4,[0]*4]), weights=np.array([.1,.6,.3,0]),ess=2.2)
    s,w = strict_posterior(result,30,{},SerialPool())
    assert s["strict_posterior_mass"] == pytest.approx(.4) and s["n_unique_solved"] == 2
    assert s["ess"] == pytest.approx(1/.625) and s["status"] == "insufficient ESS"
    np.testing.assert_allclose(w,[.25,0,.75,0])


@pytest.mark.parametrize("all_failed",[False,True])
def test_derived_shortfall_alignment(monkeypatch, all_failed):
    samples = [{"idx":i, "Mmax":2, "R14":12, "L14":300} for i in [80,90]]
    p = np.ones((2,5,4))
    p[0] = np.nan
    if all_failed:
        p[1] = np.nan
    monkeypatch.setattr("nseos.inference.posterior_for_samples",lambda *a,**k:(p,None))
    monkeypatch.setattr("nseos.physics.derived_observables",lambda *a,**k:(2,12,300,True))
    cfg = load_config()
    cfg["derived"]["n_samples_per_catalog"] = 5
    ctx = SimpleNamespace(splits={"train":samples,"test":samples},cfg=cfg,lo=0,hi=1,pool=lambda:nullcontext(SerialPool()))
    d = _derived_test(ctx,None)
    assert d["excluded_catalog_ids"] == ([80,90] if all_failed else [80])
    assert d["n_requested_catalogs"] == 2
    assert d["R14"]["n_interval_catalogs"] == d["n_retained_catalogs"] == int(not all_failed)


def test_persistent_epoch_matches_fresh_worker():
    cfg = load_config()
    m = np.linspace(.9,2.2,16)
    samples = [{"idx":i,"theta_eos":[i]*4,"curve":(m,13-m,400*(1.4/m)**6)} for i in range(8)]
    def batches(loader, epoch):
        loader.dataset.epoch = epoch
        loader.sampler.generator.manual_seed(loader.dataset.base_seed + epoch)
        return list(loader)
    reused=make_loader(samples,cfg,True,augment=True,workers=1,batch_size=4)
    first=batches(reused,0)
    later=batches(reused,1)
    fresh=batches(make_loader(samples,cfg,True,augment=True,workers=1,batch_size=4),1)
    assert not torch.equal(first[0]["theta"],later[0]["theta"])
    for a,b in zip(later,fresh):
        assert all(torch.equal(a[k],b[k]) for k in a if torch.is_tensor(a[k]))
