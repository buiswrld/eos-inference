"""Synthetic data generation (seeded per index, resumable), split, normalisation.

Each stored sample keeps theta, one fixed noisy catalog (used for val/test) and
the stable M-R-Lambda curve, so that training can re-simulate fresh catalogs for
the same theta every epoch (fix B7: removes the v1 overfitting).
"""
import multiprocessing
import os
import time
from collections import Counter
from functools import partial

import numpy as np

from .io import load_json, load_pickle, save_json, save_pickle
from .simulator import OBS_DIMS, OBS_TYPES, accept_eos, branch_summary, simulate_catalog


def prior_lo_hi(cfg):
    pb = cfg["prior_bounds"]
    lo = np.array([pb[f"gamma{k}"][0] for k in range(4)], float)
    hi = np.array([pb[f"gamma{k}"][1] for k in range(4)], float)
    return lo, hi


def _gen_one(idx, seed, n_pc, sim_cfg, prior_bounds, stream=1):
    from .physics import solve_sequence
    rng = np.random.default_rng([seed, stream, int(idx)])
    gc = [float(rng.uniform(*prior_bounds[f"gamma{k}"])) for k in range(4)]
    try:
        curve, endpoint = solve_sequence(gc, n_pc=n_pc)
        branch, reason = accept_eos(curve, sim_cfg)
        reason = endpoint["reason"] or reason
        if branch is None:
            return {"__rejected__": reason, "idx": int(idx), "theta_eos": gc}
        out = {"idx": int(idx), "theta_eos": gc, "catalog": simulate_catalog(branch, rng, sim_cfg),
               "endpoint": endpoint, "curve": np.stack(branch).astype(np.float32)}
        out.update(branch_summary(branch))
        return out
    except Exception as e:  # pragma: no cover
        return {"__rejected__": f"exception:{type(e).__name__}", "idx": int(idx), "theta_eos": gc}


def generate_dataset(cfg, paths, timings, n_cores=None):
    """Generate (or top up) the cached dataset. Returns the list of samples."""
    path = paths.data("tov_dataset") + ".pkl"
    n_target = cfg["data"]["n_samples"]
    shard_dir = paths.data("tov_shards")
    log_path = os.path.join(shard_dir, "log.json")
    if os.path.exists(path):
        cache = load_pickle(path)
        samples, log = cache["samples"], cache["log"]
    else:
        # Resumable generation: every batch of ~1000 accepted samples is written once to its own
        # shard (append-only), so an interruption loses at most one batch.
        os.makedirs(shard_dir, exist_ok=True)
        log = load_json(log_path, {"next_idx": 0, "n_attempted": 0, "rejections": {}, "wall_s": 0.0,
                                   "n_shards": 0})
        samples = []
        for k in range(log["n_shards"]):
            samples += load_pickle(os.path.join(shard_dir, f"shard_{k:05d}.pkl"))
        if samples:
            print(f"Resuming generation: {len(samples)} samples in {log['n_shards']} shards")
    if len(samples) < n_target:
        n_cores = n_cores or multiprocessing.cpu_count()
        remaining = n_target - len(samples)
        start, n_try = log["next_idx"], int(remaining * cfg["data"]["attempt_factor"])
        rej = Counter(log["rejections"])
        worker = partial(_gen_one, seed=cfg["seed"], n_pc=cfg["physics"]["n_pc"],
                         sim_cfg=cfg["sim"], prior_bounds=cfg["prior_bounds"])
        print(f"Generating {remaining} samples (indices {start}..{start + n_try - 1}) on {n_cores} cores")
        t0, buf = time.time(), []
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            tqdm = lambda x, **k: x  # noqa: E731

        log["n_cores"] = n_cores

        def flush():
            if buf:
                save_pickle(os.path.join(shard_dir, f"shard_{log['n_shards']:05d}.pkl"), list(buf))
                samples.extend(buf)
                buf.clear()
                log["n_shards"] += 1
            log["rejections"] = dict(rej)
            log["wall_s"] += time.time() - flush.t
            flush.t = time.time()
            save_json(log_path, log)
        flush.t = t0
        with multiprocessing.Pool(n_cores) as pool:
            for r in tqdm(pool.imap(worker, range(start, start + n_try), chunksize=8), total=n_try):
                log["n_attempted"] += 1
                log["next_idx"] = max(log["next_idx"], r["idx"] + 1)
                if "__rejected__" in r:
                    rej[r["__rejected__"]] += 1
                else:
                    buf.append(r)
                if len(buf) >= cfg["data"].get("shard_size", 1000):
                    flush()
                if len(samples) + len(buf) >= n_target:
                    pool.terminate()
                    break
        flush()
        timings.set("data_generation_s", log["wall_s"])
        timings.set("data_generation_cores", n_cores)
        if len(samples) < n_target:
            raise RuntimeError(f"Saved {len(samples)}/{n_target} accepted EOSs. Resume simulation to top up.")
    if not os.path.exists(path):
        # Recover when the final shard was committed just before an interruption.
        save_pickle(path, {"samples": samples, "log": log})
        timings.set("data_generation_s", log["wall_s"])
        if log.get("n_cores"):
            timings.set("data_generation_cores", log["n_cores"])
    samples.sort(key=lambda s: s["idx"])
    samples = samples[:n_target]
    n_att = max(1, log["n_attempted"])
    save_json(paths.result("data_generation"),
              {"n_attempted": n_att, "n_accepted": len(samples), "acceptance": len(samples) / n_att,
               "rejections": log["rejections"], "wall_s": log.get("wall_s"), "n_cores": log.get("n_cores"),
               "sim_cfg": cfg["sim"], "n_pc": cfg["physics"]["n_pc"]})
    return samples


def run_prior_probe(cfg, paths, n_cores=None):
    """Box-uniform thetas through the acceptance rule (labels reused for grid validation)."""
    path = paths.data("prior_probe") + ".pkl"
    if os.path.exists(path):
        return load_pickle(path)
    worker = partial(_gen_one, seed=cfg["seed"], n_pc=cfg["physics"]["n_pc"], sim_cfg=cfg["sim"],
                     prior_bounds=cfg["prior_bounds"], stream=2)
    with multiprocessing.Pool(n_cores or multiprocessing.cpu_count()) as pool:
        res = pool.map(worker, range(cfg["data"]["n_probe"]), chunksize=10)
    probe = {"theta": np.array([r["theta_eos"] for r in res]),
             "accepted": np.array(["__rejected__" not in r for r in res]),
             "reason": [r.get("__rejected__") for r in res]}
    save_pickle(path, probe)
    return probe


def split_indices(n, cfg):
    rng = np.random.default_rng([cfg["seed"], 3])
    perm = rng.permutation(n)
    f_tr, f_va, _ = cfg["data"]["split"]
    n_tr, n_va = int(f_tr * n), int(f_va * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def compute_norm_stats(samples, sd_floor=1e-3):
    """x standardised; sigma enters as log(sigma) (fix B1)."""
    buf = {t: ([], []) for t in OBS_TYPES}
    for s in samples:
        for o in s["catalog"]:
            buf[o["type"]][0].append(o["x"])
            buf[o["type"]][1].append(np.log(np.maximum(o["sigma"], 1e-12)))
    stats = {}
    for t, (xs, ls) in buf.items():
        d = OBS_DIMS[t]
        if not xs:
            stats.update({f"{t}_x_mu": np.zeros(d), f"{t}_x_sd": np.ones(d),
                          f"{t}_s_mu": np.zeros(d), f"{t}_s_sd": np.ones(d)})
            continue
        xs, ls = np.array(xs), np.array(ls)
        stats[f"{t}_x_mu"], stats[f"{t}_x_sd"] = xs.mean(0), np.clip(xs.std(0), sd_floor, None)
        stats[f"{t}_s_mu"], stats[f"{t}_s_sd"] = ls.mean(0), np.clip(ls.std(0), sd_floor, None)
    thetas = np.array([s["theta_eos"] for s in samples])
    stats["theta_mu"], stats["theta_sd"] = thetas.mean(0), np.clip(thetas.std(0), 1e-6, None)
    return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def load_splits(cfg, paths, timings=None, generate=True):
    """Return dict(train, val, test, stats, prior) — generating data if needed."""
    from .io import ScopedTimings
    timings = timings or ScopedTimings(paths)
    samples = generate_dataset(cfg, paths, timings, n_cores=cfg.get("n_cores")) if generate else \
        sorted(load_pickle(paths.data("tov_dataset") + ".pkl")["samples"], key=lambda s: s["idx"])
    tr, va, te = split_indices(len(samples), cfg)
    out = {"train": [samples[i] for i in tr], "val": [samples[i] for i in va], "test": [samples[i] for i in te]}
    stats_path = paths.result("norm_stats")
    stats = load_json(stats_path)
    if stats is None or stats.get("_n_train") != len(tr):
        stats = compute_norm_stats(out["train"])
        prior = np.array([s["theta_eos"] for s in out["train"]])
        save_json(stats_path, {**stats, "_n_train": len(tr),
                               "prior_trunc_mean": prior.mean(0), "prior_trunc_median": np.median(prior, 0),
                               "prior_trunc_std": prior.std(0)})
        stats = load_json(stats_path)
    out["stats"] = {k: np.asarray(v, np.float32) for k, v in stats.items() if not k.startswith(("_", "prior"))}
    out["prior"] = {k[len("prior_trunc_"):]: np.asarray(v) for k, v in stats.items() if k.startswith("prior_trunc")}
    print(f"Split: {len(tr)} train / {len(va)} val / {len(te)} test")
    return out
