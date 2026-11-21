import pickle

import numpy as np
import pytest
import torch

from nseos import importance, metrics
from nseos.config import Paths, load_config
from nseos.inference import sample_posterior_batch
from nseos.io import load_pickle, save_pickle
from nseos.likelihood import CurveEOS, loglike_gw
from nseos.reference import evidence_fields
from nseos.simulator import SIM_CFG, chirp_mass, lambda_tilde
from nseos.stages import _derived_table


def test_quadrature_over_noise_range():
    masses = np.linspace(.95, 2.4, 29)
    eos = CurveEOS((masses, 12.5 - 0.4*masses, 900*(1.4/masses)**7))
    for m1, m2 in ((1.12,1.11), (1.8,1.12), (1.4,1.39), (1.98,1.96)):
        lt = lambda_tilde(m1, m2, eos.L(m1), eos.L(m2))
        mc = chirp_mass(m1,m2)
        for sl in (40, 180, 800):
            for sm in (.0005, .005, .03):
                obs = {"x":[lt + 1.3*sl, mc - .7*sm], "sigma":[sl, sm]}
                log_ratio = loglike_gw(obs, eos) - loglike_gw(obs, eos, n_mc=192, n_q=192)
                assert abs(np.expm1(log_ratio)) < .001, (m1, m2, sl, sm, log_ratio)


def test_cache_rejects_relabelled_or_unversioned_products(tmp_path):
    cfg = load_config(overrides=[f"root={tmp_path}"])
    a = Paths(cfg)
    cfg["physics"]["n_pc"] += 1
    b = Paths(cfg)
    assert a.tag != b.tag
    src, dst = a.grid(), b.grid()
    save_pickle(src, {"marker": 3})
    assert load_pickle(src)["marker"] == 3
    from pathlib import Path
    for payload in (Path(src).read_bytes(), pickle.dumps({"marker": 3})):
        Path(dst).write_bytes(payload)
        with pytest.raises(ValueError):
            load_pickle(dst)
    from os import makedirs
    shards = a.data("tov_shards")
    makedirs(shards)
    shard = Path(shards) / "shard_00000.pkl"
    shard.write_bytes(pickle.dumps(["old"]))
    with pytest.raises(ValueError):
        load_pickle(shard)


class RejectAll:
    theta_mu = torch.zeros(4)

    def eval(self):
        return self

    def sample(self, batch, n_samples, n_steps, generator=None):
        return torch.full((batch["num_catalogs"], n_samples, 4), 20.)


def test_rejection_shortfalls_and_derived_support():
    args = (RejectAll(), {"num_catalogs": 1}, [-1]*4, [1]*4)
    with pytest.raises(RuntimeError, match="kept 0/8"):
        sample_posterior_batch(*args, n_samples=8, max_rounds=2)
    rows, _ = sample_posterior_batch(*args, n_samples=8, max_rounds=2, allow_shortfall=True)
    assert np.isnan(rows).all()
    derived = _derived_table(np.array([[2, 12, 300, 1], [3, 20, 3000, 0]]))
    assert derived["R14"]["median"] == 12 and derived["frac_outside_training_support"] == .5


def test_empty_weights_and_conditional_evidence(monkeypatch):
    monkeypatch.setattr(importance, "loglike_many", lambda cat, th, *args: np.full(len(th), -np.inf))
    monkeypatch.setattr(importance, "model_log_prob", lambda model, batch, th, steps: torch.zeros(th.shape[:-1]))
    obs = [{"type":"pulsar","x":[1.9],"sigma":[.1]}]
    r = importance.importance_sample(RejectAll(), obs, [-1]*4, [1]*4, "unused", SIM_CFG, n=12)
    assert r["status"] == "insufficient ESS" and r["ess"] == 0 and "weights" not in r
    import nseos.reference as ref
    monkeypatch.setattr(ref, "_support_fraction", lambda *args: (.25, 65536))
    meta = evidence_fields(np.log(.5), .01, "unused", [0]*4, [1]*4, SIM_CFG)
    assert meta["log_z_accepted_prior"] == pytest.approx(np.log(2))
    assert meta["log_z_accepted_prior_se"] > .01


def test_c2st_keeps_duplicate_rows_out_of_training_folds(monkeypatch):
    import sklearn.pipeline
    checked = []
    class Audit:
        def fit(self, x, y):
            self.rows = set(map(tuple, x))
            return self
        def score(self, x, y):
            assert self.rows.isdisjoint(map(tuple, x))
            checked.append(True)
            return .5
    monkeypatch.setattr(sklearn.pipeline, "make_pipeline", lambda *args: Audit())
    a = np.repeat(np.arange(30)[:, None], 2, axis=0)
    assert metrics.c2st(a, a + 100) == .5
    assert len(checked) == 5


def test_resume_after_last_shard_preserves_order_and_finalizes(tmp_path):
    from pathlib import Path
    from nseos.data import generate_dataset
    from nseos.io import save_json, Timings
    cfg = load_config(overrides=[f"root={tmp_path}", "data.n_samples=2"])
    paths = Paths(cfg)
    shard_dir = Path(paths.data("tov_shards"))
    shard_dir.mkdir()
    save_pickle(str(shard_dir / "shard_00000.pkl"), [{"idx": 0}, {"idx": 2}])
    save_json(str(shard_dir / "log.json"), dict(next_idx=3, n_attempted=3, rejections={"mass": 1},
                                                wall_s=3., n_cores=2, n_shards=1))
    timings = Timings(paths.result("timings", stage="simulate"))
    generate_dataset(cfg, paths, timings)
    final = load_pickle(paths.data("tov_dataset") + ".pkl")
    assert [s["idx"] for s in final["samples"]] == [0, 2]
    assert final["log"]["next_idx"] == 3 and timings.get("data_generation_s") == 3


def test_dynesty_resume_reuses_stopping_rule(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import dynesty
    import nseos.reference as ref
    checkpoint = tmp_path / "sampler.ckpt"
    checkpoint.touch()
    observed = {}
    sampler = SimpleNamespace(ncall=40, run_nested=lambda **kw: observed.update(kw),
                              results=SimpleNamespace(samples=np.zeros((2, 4)), logwt=np.log([.5, .5]),
                                                      logz=np.zeros(2), logzerr=[.01], ncall=[20, 20], niter=2))
    monkeypatch.setattr(dynesty.NestedSampler, "restore", lambda path: sampler)
    monkeypatch.setattr(ref, "evidence_fields", lambda *args: {})
    ref.run_dynesty([], "unused", SIM_CFG, [0]*4, [1]*4, maxcall=100, dlogz=.32,
                    checkpoint_file=str(checkpoint))
    assert observed["resume"] and observed["maxcall"] == 60 and observed["dlogz"] == .32
