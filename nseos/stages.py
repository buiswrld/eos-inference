"""Pipeline stages. Each stage reads cached inputs, writes its outputs under cfg['root'], and is
skipped (or partially skipped) when its outputs exist, unless --force.

    python -m nseos.cli simulate train evaluate          # or: python -m nseos.cli all
"""
import itertools
import multiprocessing
import os
import time
from functools import partial

import numpy as np

from . import metrics as M
from . import plotting as P
from .config import Paths
from .data import load_splits, prior_lo_hi, run_prior_probe
from .io import ScopedTimings, cached, load_json, load_pickle, save_json, save_pickle
from .simulator import OBS_TYPES, interp_lambda, simulate_catalog

PARAM_KEYS = ["gamma_0", "gamma_1", "gamma_2", "gamma_3"]
N_BINS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 35), (36, 50)]


class Ctx:
    def __init__(self, cfg, force=False):
        self.cfg, self.force = cfg, force
        self.paths = Paths(cfg)
        self.timings = ScopedTimings(self.paths)
        self.lo, self.hi = prior_lo_hi(cfg)
        self.n_cores = cfg.get("n_cores") or multiprocessing.cpu_count()
        self._splits = self._main = None

    @property
    def splits(self):
        if self._splits is None:
            self._splits = load_splits(self.cfg, self.paths, self.timings)
        return self._splits

    @property
    def prior(self):
        return self.splits["prior"]

    def main_members(self, force=False):
        """Independently seeded copies of the main model (cached runs main, main_s1, ...)."""
        from .train import train_run
        out = []
        for s in self.cfg["train"].get("ensemble_seeds", [0]):
            out.append(train_run("main" if s == 0 else f"main_s{s}", self.cfg, self.paths, self.splits,
                                 seed=s, force=force))
        return out

    def main_model(self):
        """The headline model: an equal-weight ensemble of the seeded members (one member if a
        single seed is configured)."""
        if self._main is None:
            from .model import EnsembleModel
            members = [m for m, _ in self.main_members()]
            self._main = members[0] if len(members) == 1 else EnsembleModel(members).eval()
        return self._main

    def grid_path(self):
        return self.paths.grid()

    def root(self, sub):
        return os.path.join(self.cfg["root"], sub)

    def pool(self, n=None):
        return multiprocessing.Pool(n or self.n_cores)


def _summaries(ctx, model, samples, name, n_samples=None, allowed_types=None, seed=0, scope="posterior"):
    """Cached posterior summaries (median/quantiles/ranks/thin draws) for a list of samples."""
    from .inference import posterior_for_samples
    inputs = [seed, n_samples, allowed_types, len(samples)]
    path = ctx.paths.data(f"summ_{name}", scope, inputs) + ".pkl"
    if os.path.exists(path) and not ctx.force:
        return load_pickle(path)
    n = n_samples or ctx.cfg["sampling"]["n_eval_samples"]
    # Resumable: computed in chunks, each cached as soon as it finishes.
    chunk = 2000
    parts, fouts = [], []
    for k, i0 in enumerate(range(0, len(samples), chunk)):
        ppath = ctx.paths.data(f"summ_{name}_part{k}", scope, inputs) + ".pkl"
        if os.path.exists(ppath) and not ctx.force:
            part = load_pickle(ppath)
        else:
            s_, f_ = posterior_for_samples(model, samples[i0:i0 + chunk], ctx.lo, ctx.hi, n_samples=n,
                                           n_steps=ctx.cfg["sampling"]["n_steps"], seed=seed + 100000 * k,
                                           allowed_types=allowed_types, reduce=M.summarize_posterior,
                                           progress=True)
            part = {"summ": s_, "fout": f_}
            save_pickle(ppath, part)
        parts.append(part["summ"])
        fouts.append(part["fout"])
    parts = [p for p in parts if p]
    all_fouts = np.concatenate(fouts)
    keep = np.isfinite(all_fouts)
    failed = [samples[i].get("idx", int(i)) for i in np.flatnonzero(~keep)]
    if failed:
        save_json(ctx.paths.result(f"posterior_shortfall_{name}", stage=scope), {"excluded_catalog_ids": failed})
    if not parts:
        raise RuntimeError(f"No usable flow posteriors for {name}, excluded {len(failed)} catalogs")
    summ = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    summ["excluded_catalog_ids"] = failed
    summ["kept_indices"] = np.flatnonzero(keep)
    samples = [s for s, valid in zip(samples, keep) if valid]
    summ["frac_out"], summ["n_samples"] = all_fouts[keep], n
    cats = [s["catalog"] if allowed_types is None else [o for o in s["catalog"] if o["type"] in allowed_types]
            for s in samples]
    summ["n_obs"] = np.array([len(c) for c in cats])
    summ["counts"] = np.array([[sum(o["type"] == t for o in c) for t in OBS_TYPES] for c in cats])
    save_pickle(path, summ)
    for k in range(len(parts)):
        ppath = ctx.paths.data(f"summ_{name}_part{k}", scope, inputs) + ".pkl"
        if os.path.exists(ppath):
            os.remove(ppath)
    return summ


def _by_n(summ, prior):
    out = {}
    for a, b in N_BINS:
        sel = (summ["n_obs"] >= a) & (summ["n_obs"] <= b)
        if sel.sum() < 20:
            continue
        sub = {k: v[sel] for k, v in summ.items() if isinstance(v, np.ndarray) and len(v) == len(sel)}
        out[f"{a}-{b}"] = {"n": int(sel.sum()), **{k: {kk: vv for kk, vv in v.items() if kk in
                           ("mae_ratio_to_prior", "contraction")} for k, v in M.accuracy_table(sub, prior, PARAM_KEYS).items()}}
    return out


# ── stage: simulate ──────────────────────────────────────────────────────────
def stage_simulate(ctx):
    s = ctx.splits
    probe = run_prior_probe(ctx.cfg, ctx.paths, ctx.n_cores)
    print(f"prior probe: {probe['accepted'].mean() * 100:.1f}% of box draws accepted")
    print(f"truncated prior median {np.round(s['prior']['median'], 4)} std {np.round(s['prior']['std'], 4)}")


# ── stage: train ─────────────────────────────────────────────────────────────
def stage_train(ctx):
    runs = ctx.main_members(force=ctx.force)
    infos = [i for _, i in runs]
    walls = [i.get("wall_s") or 0.0 for i in infos]
    ctx.timings.set("train_main_s", walls[0])
    ctx.timings.set("train_ensemble_s", float(sum(walls)))
    save_json(ctx.paths.result("train_ensemble"),
              {"members": [{k: i.get(k) for k in ("seed", "best_epoch", "epochs_run", "best_val_loss", "wall_s",
                                                   "n_params")} for i in infos]})
    for i in infos:
        print({k: i.get(k) for k in ("seed", "best_epoch", "epochs_run", "best_val_loss", "wall_s", "n_params")})
    info = infos[0]
    h = info.get("history")
    if h:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(h["train"], label="train (re-simulated each epoch)")
        ax.plot(h["val"], label="validation (fixed catalogs, fixed RNG)")
        ax.axvline(info["best_epoch"], color="k", ls=":", label=f"best epoch {info['best_epoch']}")
        ax.set(xlabel="epoch", ylabel="flow-matching loss")
        ax.legend()
        P.save(fig, ctx.paths.figure("loss_main"))


# ── stage: evaluate ──────────────────────────────────────────────────────────
def _ridge_features(cat):
    f = []
    for t, d in (("gw", 2), ("nicer", 2), ("pulsar", 1)):
        rows = [o["x"] + list(np.log(o["sigma"])) for o in cat if o["type"] == t]
        f += list(np.mean(rows, 0)) if rows else [0.0] * (2 * d)
        f.append(len(rows))
    return f


def stage_evaluate(ctx):
    import torch
    from .model import move_to_device
    from .train import make_loader
    cfg, prior, test = ctx.cfg, ctx.prior, ctx.splits["test"]
    model = ctx.main_model()
    res = {}
    summ = _summaries(ctx, model, test, "main_test")
    res["excluded_catalog_ids"] = summ["excluded_catalog_ids"]
    res["accuracy"] = M.accuracy_table(summ, prior, PARAM_KEYS)
    res["by_n"] = _by_n(summ, prior)
    res["frac_outside_box_mean"] = float(summ["frac_out"].mean())
    # calibration. Headline: the identifiable parameters (g0, g1) and observables (derived test);
    # g2/g3 are reported but flagged, since a prior-dominated posterior is trivially calibrated.
    n_s = summ["n_samples"]
    res["coverage_error"] = M.coverage_error(summ["rank"], n_s)
    res["coverage_error_identifiable_g0_g1"] = float(np.mean(res["coverage_error"][:2]))
    res["prior_dominated"] = [p for p in PARAM_KEYS if res["accuracy"][p]["contraction"] < 0.2]
    # seed variability: each ensemble member on a common subset
    n_sub = min(cfg["ablation"]["n_eval"], len(test))
    members = getattr(model, "members", [model])
    per_seed = []
    for k, mem in enumerate(members):
        sm = _summaries(ctx, mem, test[:n_sub], f"main_member{k}")
        per_seed.append(M.accuracy_table(sm, prior, PARAM_KEYS))
    res["seed_variability"] = {p: {key: [float(np.mean([a[p][key] for a in per_seed])),
                                         float(np.std([a[p][key] for a in per_seed]))]
                                   for key in ("mae_ratio_to_prior", "contraction")} for p in PARAM_KEYS}
    res["n_ensemble_members"] = len(members)
    hist, band = M.sbc_histogram(summ["rank"], n_s)
    lv, cov, se = M.coverage_curve(summ["rank"], n_s, levels=np.unique(np.r_[np.linspace(0.02, 0.98, 49), 0.95]))
    res["coverage_at"] = {f"{round(l * 100)}%": cov[np.argmin(np.abs(lv - l))] for l in (0.5, 0.68, 0.9, 0.95)}
    k_t = min(cfg["evaluate"]["tarp_n"], len(summ["thin"]))
    tarp_res = M.tarp(summ["thin"][:k_t], summ["theta"][:k_t], ctx.lo, ctx.hi)
    res["tarp_max_abs_dev"] = float(np.max(np.abs(tarp_res[1] - tarp_res[0])))
    P.calibration_figure(summ["rank"], n_s, ctx.paths.figure("calibration_main"), hist, band, lv, cov, se, tarp_res)
    P.true_vs_pred(summ, prior, ctx.paths.figure("true_vs_pred_main"), n_obs=summ["n_obs"])
    P.bar_by_n(list(N_BINS), [[res["by_n"][f"{a}-{b}"][p]["mae_ratio_to_prior"] if f"{a}-{b}" in res["by_n"]
                               else np.nan for a, b in N_BINS] for p in PARAM_KEYS],
               "MAE / MAE(prior median)", ctx.paths.figure("mae_ratio_by_n"), ref_line=1.0)

    # nested catalogs for the same theta (controls for theta when varying N)
    rng = np.random.default_rng([cfg["seed"], 11])
    sel = test[: cfg["evaluate"]["nested_n_theta"]]
    Ns = cfg["evaluate"]["nested_N"]
    nested = {}
    full = [simulate_catalog(tuple(s["curve"]), rng, cfg["sim"], counts=rng.multinomial(50, rng.dirichlet([1, 1, 1])))
            for s in sel]
    for N in Ns:
        ss = [{"theta_eos": s["theta_eos"], "catalog": c[:N]} for s, c in zip(sel, full)]
        sm = _summaries(ctx, model, ss, f"main_nested_N{N}", scope="evaluate")
        acc = M.accuracy_table(sm, prior, PARAM_KEYS)
        nested[N] = {p: {"contraction": acc[p]["contraction"], "mae_ratio_to_prior": acc[p]["mae_ratio_to_prior"]}
                     for p in PARAM_KEYS}
    res["nested_N"] = nested

    # Ridge baseline on mean-pooled raw features (+ counts)
    from sklearn.linear_model import Ridge
    tr = ctx.splits["train"]
    Xtr = np.array([_ridge_features(s["catalog"]) for s in tr])
    ytr = np.array([s["theta_eos"] for s in tr])
    Xte = np.array([_ridge_features(test[i]["catalog"]) for i in summ["kept_indices"]])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    pred = Ridge(alpha=1.0).fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)
    res["ridge"] = {p: {"mae": float(np.mean(np.abs(pred[:, k] - summ["theta"][:, k]))),
                        "mae_ratio_to_prior": float(np.mean(np.abs(pred[:, k] - summ["theta"][:, k])) /
                                                    res["accuracy"][p]["mae_prior_median"])}
                    for k, p in enumerate(PARAM_KEYS)}

    # permutation invariance + padding/mask tests
    dev = model.theta_mu.device
    loader = make_loader(test[:64], cfg, shuffle=False, workers=0)
    batch = move_to_device(next(iter(loader)), dev)
    with torch.no_grad():
        perm_dev = pad_dev = 0.0
        for member in getattr(model, "members", [model]):
            s0 = member.encode_and_aggregate(batch)
            for t in OBS_TYPES:
                n = batch[f"{t}_x"].shape[1]
                if n > 1:
                    b2 = dict(batch)
                    idx = torch.randperm(n, device=dev)
                    for key in (f"{t}_x", f"{t}_sigma", f"{t}_mask"):
                        b2[key] = batch[key][:, idx]
                    perm_dev = max(perm_dev, (member.encode_and_aggregate(b2) - s0).abs().max().item())
            b3 = dict(batch)
            for t, d in (("gw", 2), ("nicer", 2), ("pulsar", 1)):
                B = batch["num_catalogs"]
                b3[f"{t}_x"] = torch.cat([batch[f"{t}_x"], torch.randn(B, 5, d, device=dev) * 1e3], 1)
                b3[f"{t}_sigma"] = torch.cat([batch[f"{t}_sigma"], torch.randn(B, 5, d, device=dev) * 1e3], 1)
                b3[f"{t}_mask"] = torch.cat([batch[f"{t}_mask"], torch.ones(B, 5, dtype=torch.bool, device=dev)], 1)
            pad_dev = max(pad_dev, (member.encode_and_aggregate(b3) - s0).abs().max().item())
    res["permutation_max_dev"], res["padding_max_dev"] = perm_dev, pad_dev

    # speed / NFE and ODE-step convergence
    res["speed"] = _speed(ctx, model, test)
    res["derived_test"] = _derived_test(ctx, model) if cfg.get("run_derived_test", True) else None
    save_json(ctx.paths.result("eval_main"), res)
    _print_accuracy("main model (test split)", res["accuracy"])
    print(f"coverage error per param: {np.round(res['coverage_error'], 4)}; TARP max dev {res['tarp_max_abs_dev']:.3f}")
    print(f"permutation dev {perm_dev:.2e}, padding dev {pad_dev:.2e}")


def _print_accuracy(title, acc):
    print(f"\n{title}\n{'param':>8} {'MAE':>9} {'MAE prior':>10} {'ratio':>7} {'contraction':>12}")
    for p, r in acc.items():
        print(f"{p:>8} {r['mae_post_median']:9.4f} {r['mae_prior_median']:10.4f} {r['mae_ratio_to_prior']:7.3f} "
              f"{r['contraction']:12.3f}")


def _speed(ctx, model, test):
    import torch
    from .inference import posterior_for_catalogs
    from .model import EOSCatalogDataset, collate_fn, move_to_device
    dev = model.theta_mu.device
    sc = ctx.cfg["speed"]

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
    cat = min(test, key=lambda s: abs(len(s["catalog"]) - 8))["catalog"]
    batch = move_to_device(collate_fn([EOSCatalogDataset([{"theta_eos": [0] * 4, "catalog": cat}])[0]]), dev)
    out = {"single_catalog": {}, "by_catalog_size": {}}
    rng = np.random.default_rng(52)
    full = simulate_catalog(test[0]["curve"], rng, ctx.cfg["sim"], counts=(17, 17, 16))
    for size in (1, 5, 10, 15, 50):
        item = EOSCatalogDataset([{"theta_eos": [0]*4, "catalog": full[:size]}])[0]
        sized = move_to_device(collate_fn([item]), dev)
        draws, steps = sc["n_list"][0], sc["steps"][-1]
        model.sample(sized, draws, steps)
        times = []
        for _ in range(5):
            sync()
            start = time.perf_counter()
            model.sample(sized, draws, steps)
            sync()
            times.append(time.perf_counter() - start)
        out["by_catalog_size"][size] = {"median_ms": 1000 * float(np.median(times)), "draws": draws,
                                        "steps": steps, "method": "raw flow"}

    model.sample(batch, 100, 10)
    for n in sc["n_list"]:
        for steps in sc["steps"]:
            ts = []
            for _ in range(5):
                sync(); t0 = time.perf_counter(); model.sample(batch, n, steps); sync()
                ts.append(time.perf_counter() - t0)
            out["single_catalog"][f"n{n}_steps{steps}"] = {"ms": 1e3 * float(np.median(ts)), "velocity_evals_per_trajectory": 4 * steps, "encoder_calls": min(n, len(getattr(model, "members", [model])))}
    cats = [s["catalog"] for s in test[: sc["throughput_catalogs"]]]
    sync(); t0 = time.perf_counter()
    posterior_for_catalogs(model, cats, ctx.lo, ctx.hi, n_samples=1000, batch_size=64)
    sync()
    out["throughput_catalogs_per_s_1000_samples"] = len(cats) / (time.perf_counter() - t0)
    # step convergence: identical z (same generator seed), medians in units of posterior std
    b = move_to_device(collate_fn([EOSCatalogDataset([s])[0] for s in test[: sc["conv_catalogs"]]]), dev)
    ref = model.sample(b, 1000, sc["conv_ref_steps"], generator=torch.Generator(device=dev).manual_seed(0)).cpu().numpy()
    conv = {}
    for steps in sc["steps"]:
        x = model.sample(b, 1000, steps, generator=torch.Generator(device=dev).manual_seed(0)).cpu().numpy()
        conv[steps] = float(np.max(np.abs(np.median(x, 1) - np.median(ref, 1)) / ref.std(1)))
    out["rk4_max_median_shift_vs_100_steps_in_post_sd"] = conv
    return out


def _derived_test(ctx, model):
    from .inference import posterior_for_samples
    from .physics import derived_observables
    dc = ctx.cfg["derived"]
    sel = ctx.splits["test"][: dc["n_test_catalogs"]]
    requested = len(sel)
    if requested:
        post, _ = posterior_for_samples(model, sel, ctx.lo, ctx.hi, n_samples=dc["n_samples_per_catalog"],
                                        n_steps=ctx.cfg["sampling"]["n_steps"], seed=5, allow_shortfall=True)
    else:
        post = np.empty((0, dc["n_samples_per_catalog"], 4))
    valid_catalog = np.isfinite(post).all(axis=(1, 2))
    excluded = [s.get("idx", i) for i, s in enumerate(sel) if not valid_catalog[i]]
    sel = [s for s, good in zip(sel, valid_catalog) if good]
    post = post[valid_catalog]
    thetas = post.reshape(-1, 4)
    r = []
    if len(thetas):
        with ctx.pool() as pool:
            r = pool.map(partial(derived_observables, n_pc=ctx.cfg["physics"]["n_pc"], cfg=ctx.cfg["sim"]), list(thetas), chunksize=8)
    arr = np.array([x if x is not None else (np.nan, np.nan, np.nan, False) for x in r], float)
    arr = arr.reshape(len(sel), dc["n_samples_per_catalog"], 4)
    truth = np.array([[s["Mmax"], s["R14"], s["L14"]] for s in sel]).reshape(-1, 3)
    train_q = np.array([[s["Mmax"], s["R14"], s["L14"]] for s in ctx.splits["train"][:20000]])
    out = {"n_requested_catalogs": requested, "n_retained_catalogs": len(sel), "excluded_catalog_ids": excluded,
           "frac_samples_outside_support": float(1 - np.mean(arr[..., 3])) if len(sel) else None}
    arr[..., :3] = np.where(arr[..., 3:4].astype(bool), arr[..., :3], np.nan)
    for j, name in enumerate(("Mmax", "R14", "L14")):
        values = arr[:, :, j]
        valid = np.isfinite(values).any(axis=1)
        med, lo_, hi_ = (np.nanpercentile(values[valid], q, axis=1) if valid.any() else np.array([])
                         for q in (50, 16, 84))
        eligible = np.sum(np.isfinite(values), axis=1) >= 5
        u = np.array([np.mean(v[np.isfinite(v)] < t) for v, t in zip(values[eligible], truth[eligible, j])])
        levels = np.linspace(0.1, 0.9, 9)
        cov = np.array([np.mean(np.abs(u - 0.5) < l / 2) for l in levels]) if len(u) else None
        out[name] = {"mae": float(np.mean(np.abs(med - truth[valid, j]))) if valid.any() else None,
                     "mae_prior": float(np.mean(np.abs(np.median(train_q[:, j]) - truth[valid, j]))) if valid.any() else None,
                     "coverage68": float(np.mean((truth[valid, j] >= lo_) & (truth[valid, j] <= hi_))) if valid.any() else None,
                     "coverage_curve": {"levels": levels, "empirical": cov},
                     "coverage_error": float(np.mean(np.abs(cov - levels))) if cov is not None else None,
                     "n_interval_catalogs": int(valid.sum()), "n_catalogs": int(len(u))}
    return out


# ── stage: architecture ablation ─────────────────────────────────────────────
def stage_ablate_arch(ctx):
    from .train import train_run
    ab = ctx.cfg["ablation"]
    test = ctx.splits["test"][: ab["n_eval"]]
    runs = {}
    for arch, seed in itertools.product(ab["archs"], ab["seeds"]):
        name = f"arch_{arch}_s{seed}"
        model, info = train_run(name, ctx.cfg, ctx.paths, ctx.splits, aggregator=arch, seed=seed,
                                max_epochs=ab["max_epochs"], patience=ab["patience"])
        summ = _summaries(ctx, model, test, name, seed=0, scope="ablate-arch")
        acc = M.accuracy_table(summ, ctx.prior, PARAM_KEYS)
        runs[name] = {"arch": arch, "seed": seed, "accuracy": acc, "by_n": _by_n(summ, ctx.prior),
                      "coverage_error": M.coverage_error(summ["rank"], summ["n_samples"]),
                      **{k: info.get(k) for k in ("best_val_loss", "best_epoch", "epochs_run", "wall_s", "n_params")}}
        del model
    table = {}
    for arch in ab["archs"]:
        rs = [r for r in runs.values() if r["arch"] == arch]
        row = {"n_seeds": len(rs), "n_params": rs[0]["n_params"]}
        for key, get in (("best_val_loss", lambda r: r["best_val_loss"]),
                         ("coverage_error_mean", lambda r: float(np.mean(r["coverage_error"])))):
            v = np.array([get(r) for r in rs], float)
            row[key] = [float(v.mean()), float(v.std())]
        for p in PARAM_KEYS:
            for key in ("mae_ratio_to_prior", "contraction"):
                v = np.array([r["accuracy"][p][key] for r in rs])
                row[f"{p}_{key}"] = [float(v.mean()), float(v.std())]
        table[arch] = row
    save_json(ctx.paths.result("ablation_arch"), {"table": table, "runs": runs})
    print("\narchitecture ablation (mean ± sd over seeds; identical decoder, data and budget)")
    for arch, row in table.items():
        print(f"{arch:>16}: val {row['best_val_loss'][0]:.4f}±{row['best_val_loss'][1]:.4f} | "
              + " | ".join(f"{p} {row[f'{p}_mae_ratio_to_prior'][0]:.3f}" for p in PARAM_KEYS[:2]))


# ── stage: messenger ablation ────────────────────────────────────────────────
MESSENGER_SETS = {"GW only": ["gw"], "NICER only": ["nicer"], "Pulsar only": ["pulsar"],
                  "GW+NICER": ["gw", "nicer"], "All": ["gw", "nicer", "pulsar"]}


def _nonempty(samples, types, n=None):
    out = [s for s in samples if any(o["type"] in types for o in s["catalog"])]
    return out[:n] if n else out


def stage_ablate_messenger(ctx):
    from .train import train_run
    mc, n_eval = ctx.cfg["messenger"], ctx.cfg["ablation"]["n_eval"]
    res = {"zero_shot_main": {}, "retrained": {}}
    main = ctx.main_model()
    fixed_n = min(6, ctx.cfg["sim"]["N_MAX"])
    paired = {}
    for key, types in MESSENGER_SETS.items():
        counts = np.zeros(3, dtype=int)
        for i in range(fixed_n):
            counts[OBS_TYPES.index(types[i % len(types)])] += 1
        paired[key] = [
            {**s, "catalog": simulate_catalog(s["curve"], np.random.default_rng([ctx.cfg["seed"], 81, s["idx"]]),
                                              ctx.cfg["sim"], counts=counts)}
            for s in ctx.splits["test"][:n_eval]]
    res["catalog_size"] = fixed_n
    # Nested subsets quantify the effect of adding observations to the same base catalog.
    res["nested_observations"] = {}
    for key, types in MESSENGER_SETS.items():
        nested = _nonempty(paired["All"], types)
        if nested:
            sm = _summaries(ctx, main, nested, f"msg_nested_{key.replace(' ', '_')}", allowed_types=types, scope="ablate-messenger")
            res["nested_observations"][key] = {"n_catalogs": len(nested), "n_obs": int(sm["n_obs"][0]),
                                               "accuracy": M.accuracy_table(sm, ctx.prior, PARAM_KEYS)}
    for key, types in MESSENGER_SETS.items():
        test = paired[key]
        summ = _summaries(ctx, main, test, f"msg0_{key.replace(' ', '_').replace('+', '_')}", allowed_types=types, scope="ablate-messenger")
        res["zero_shot_main"][key] = M.accuracy_table(summ, ctx.prior, PARAM_KEYS)
    if ctx.cfg.get("run_messenger_retrain", True):
        for key, types in MESSENGER_SETS.items():
            if key == "All":
                continue
            tag = key.replace(" ", "_").replace("+", "_")
            model, info = train_run(f"msg_{tag}", ctx.cfg, ctx.paths, ctx.splits, allowed_types=types,
                                    val_samples=_nonempty(ctx.splits["val"], types), max_epochs=mc["max_epochs"],
                                    patience=mc["patience"])
            test = paired[key]
            summ = _summaries(ctx, model, test, f"msg_{tag}", allowed_types=types, scope="ablate-messenger")
            res["retrained"][key] = {"accuracy": M.accuracy_table(summ, ctx.prior, PARAM_KEYS),
                                     **{k: info.get(k) for k in ("best_epoch", "epochs_run", "best_val_loss")}}
    save_json(ctx.paths.result("ablation_messenger"), res)
    for key, acc in res["zero_shot_main"].items():
        _print_accuracy(f"main model restricted to {key}", acc)


# ── stage: near-zero-noise ───────────────────────────────────────────────────
def pristine_catalog(branch, rng, zc, sim):
    Ms, Rs, Ls = (np.asarray(a, float) for a in branch)
    obs = []
    for m in np.sort(rng.uniform(1.0, Ms[-1], zc["n_nicer"])):
        obs.append({"type": "nicer", "x": [m + zc["sig_m"] * rng.standard_normal(),
                                           float(np.interp(m, Ms, Rs)) + zc["sig_r"] * rng.standard_normal()],
                    "sigma": [zc["sig_m"], zc["sig_r"]]})
    hi = min(Ms[-1], sim["GW_M_HI"])
    from .simulator import chirp_mass, lambda_tilde
    for _ in range(zc["n_gw"]):
        m1, m2 = rng.uniform(sim["GW_M_LO"], hi, 2)
        Lt = float(lambda_tilde(m1, m2, *interp_lambda([m1, m2], Ms, Ls)))
        obs.append({"type": "gw", "x": [Lt + zc["sig_lt"] * rng.standard_normal(),
                                        float(chirp_mass(m1, m2)) + zc["sig_mc"] * rng.standard_normal()],
                    "sigma": [zc["sig_lt"], zc["sig_mc"]]})
    return obs


def stage_zero_noise(ctx):
    from .data import compute_norm_stats
    from .identifiability import model_free_regression
    from .train import train_run
    zc, sim = ctx.cfg["zero_noise"], ctx.cfg["sim"]
    rng = np.random.default_rng([ctx.cfg["seed"], 21])
    fix = lambda ss: [{**s, "catalog": pristine_catalog(tuple(s["curve"]), rng, zc, sim)} for s in ss]  # noqa: E731
    tr = fix(ctx.splits["train"][: zc["n_train"]])
    va = fix(ctx.splits["val"][:2000])
    te = fix(ctx.splits["test"][: zc["n_test"]])
    stats = compute_norm_stats(tr)   # its own statistics (v1 reused the main model's -> ~1e4-sd inputs)
    model, info = train_run("zero_noise", ctx.cfg, ctx.paths, ctx.splits, train_samples=tr, val_samples=va,
                            stats=stats, catalog_fn=partial(pristine_catalog, zc=zc, sim=sim),
                            max_epochs=zc["max_epochs"], patience=zc["patience"])
    summ = _summaries(ctx, model, te, "zero_noise", scope="zero-noise")
    masses = np.linspace(1.0, 2.0, 17)
    res = {"flow_model": M.accuracy_table(summ, ctx.prior, PARAM_KEYS),
           "model_free_gbr_noise_free_curves": model_free_regression(ctx.splits["train"], ctx.splits["test"][:5000],
                                                                     masses, ctx.prior),
           "train_info": {k: info.get(k) for k in ("best_epoch", "epochs_run", "best_val_loss", "wall_s")},
           "v1_for_reference": {"n_train": 400, "gamma_2_mae_ratio": 1.002, "gamma_3_mae_ratio": 1.003,
                                "note": "v1 reused main-model normalisation; constant sigma=1e-4 inputs were ~1e4 sd"}}
    save_json(ctx.paths.result("zero_noise"), res)
    _print_accuracy("near-zero-noise flow model", res["flow_model"])
    print("model-free GBR on noise-free curves:", res["model_free_gbr_noise_free_curves"])


# ── stage: identifiability (Fisher / sensitivity) ────────────────────────────
def stage_identifiability(ctx):
    from .identifiability import compute_jacobians, fisher_summary, prior_width_table, sensitivity_scans
    fc = ctx.cfg["fisher"]
    jpath = ctx.paths.data("jacobians", "identifiability") + ".pkl"
    thetas = [s["theta_eos"] for s in ctx.splits["test"][: fc["n_theta"]]]
    if cached(jpath, ctx.force):
        jacs = load_pickle(jpath)
    else:
        jacs, block = [], 32
        for k, i0 in enumerate(range(0, len(thetas), block)):
            bpath = ctx.paths.data(f"jacobians_block{k}", "identifiability") + ".pkl"
            if os.path.exists(bpath) and not ctx.force:
                jb = load_pickle(bpath)
            else:
                jb = compute_jacobians(thetas[i0:i0 + block], ctx.prior["std"], fc["masses"], fc["n_pc"],
                                       fc["step_frac"], ctx.n_cores)
                save_pickle(bpath, jb)
            jacs += jb
        save_pickle(jpath, jacs)
    if not jacs:
        save_json(ctx.paths.result("identifiability"), {"n_requested": len(thetas), "n_eligible": 0, "n_excluded": len(thetas),
                  "reason": "No finite-difference neighborhood covers the configured mass range"})
        return
    res = {"n_requested": len(thetas), "n_eligible": len(jacs), "n_excluded": len(thetas) - len(jacs), "prior_width": prior_width_table(ctx.cfg["prior_bounds"]),
           "fisher": fisher_summary(jacs, fc["masses"], fc["scenarios"])}
    spath = ctx.paths.data("scans", "identifiability") + ".pkl"
    if cached(spath, ctx.force):
        scans = load_pickle(spath)
    else:
        scans = sensitivity_scans(thetas[:4], ctx.cfg["prior_bounds"], fc["masses"], n_pc=fc["n_pc"], n_cores=ctx.n_cores)
        save_pickle(spath, scans)
    import matplotlib.pyplot as plt
    K = len(fc["masses"])
    i14 = int(np.argmin(np.abs(np.array(fc["masses"]) - 1.4)))
    fig, ax = plt.subplots(2, 4, figsize=(16, 6.5), sharey="row")
    rng_max = {}
    for (th, k, vals), obs in zip(scans["jobs"], scans["observables"]):
        dR = obs[:, i14] - np.nanmedian(obs[:, i14])
        dM = obs[:, -1] - np.nanmedian(obs[:, -1])
        ax[0, k].plot(vals, dR, "o-", ms=3)
        ax[1, k].plot(vals, dM, "o-", ms=3)
        rng_max.setdefault(PARAM_KEYS[k], []).append([float(np.nanmax(obs[:, i14]) - np.nanmin(obs[:, i14])),
                                                      float(np.nanmax(obs[:, -1]) - np.nanmin(obs[:, -1]))])
    for k in range(4):
        ax[0, k].axhspan(-0.5, 0.5, color="0.9", zorder=0)
        ax[1, k].axhspan(-0.05, 0.05, color="0.9", zorder=0)
        ax[1, k].set_xlabel(P.TEX[k] + " (full prior range, others fixed)")
    ax[0, 0].set_ylabel(r"$\Delta R_{1.4}$ [km]")
    ax[1, 0].set_ylabel(r"$\Delta M_{\max}$ [$M_\odot$]")
    P.save(fig, ctx.paths.figure("sensitivity_scans"))
    res["scan_full_prior_range_change"] = {p: {"R14_km_median": float(np.median([r[0] for r in v])),
                                               "Mmax_median": float(np.median([r[1] for r in v]))}
                                           for p, v in rng_max.items()}
    save_json(ctx.paths.result("identifiability"), res)
    for name, sc in res["fisher"]["scenarios"].items():
        print(f"{name:>10}: posterior sd / prior sd (median) = {np.round(sc['posterior_sd_over_prior_sd_median'], 3)}"
              f" | constrained directions = {sc['n_directions_eig_gt_1_median']}")
    print("finite-difference stability (rel. change h vs h/2):",
          np.round(res["fisher"]["finite_difference_rel_change_median"], 4))


# ── stage: likelihood grid ───────────────────────────────────────────────────
def stage_grid(ctx):
    from .likelihood import GridEOS, build_grid, validate_grid
    from .reference import likelihood_eval_time
    gp = ctx.grid_path()
    if not cached(gp, ctx.force):
        grid = build_grid(ctx.cfg, ctx.n_cores, partial_path=gp + ".partial", block=ctx.cfg["grid"].get("block", 1000))
        ctx.timings.set("grid_s", grid["build_s"])
        save_pickle(gp, grid)
        if os.path.exists(gp + ".partial"):
            os.remove(gp + ".partial")
    grid = load_pickle(gp)
    geos = GridEOS(grid)
    probe = run_prior_probe(ctx.cfg, ctx.paths, ctx.n_cores)
    val = validate_grid(geos, ctx.splits["test"][: ctx.cfg["grid"]["n_validate"]], probe, ctx.cfg)
    cat = min(ctx.splits["test"], key=lambda s: abs(len(s["catalog"]) - 8))["catalog"]
    val["loglike_ms_per_eval_N8"] = 1e3 * likelihood_eval_time(cat, gp, ctx.cfg["sim"], ctx.lo, ctx.hi, n=500)
    val["frac_tov_ok"], val["shape"], val["grid_s"] = grid["frac_tov_ok"], grid["shape"], ctx.timings.get("grid_s")
    save_json(ctx.paths.result("grid_validation"), val)
    print(val)


# ── stage: real data ─────────────────────────────────────────────────────────
def _derive(ctx, thetas):
    from .physics import derived_observables
    with ctx.pool() as pool:
        r = pool.map(partial(derived_observables, n_pc=ctx.cfg["physics"]["n_pc"], cfg=ctx.cfg["sim"]), list(thetas), chunksize=8)
    return np.array([x if x is not None else (np.nan, np.nan, np.nan, False) for x in r], float)


def _derived_table(arr):
    out = {"frac_outside_training_support": float(1 - np.nanmean(arr[:, 3])),
           "frac_Mmax_below_2": float(np.nanmean(arr[:, 0] < 2.0))}
    for j, n in enumerate(("Mmax", "R14", "L14")):
        v = arr[:, j][arr[:, 3].astype(bool) & np.isfinite(arr[:, j])]
        if not len(v):
            out[n] = {"median": None, "ci90": [None, None], "ci68": [None, None], "n_valid": 0}
            continue
        out[n] = {"n_valid": len(v), "median": float(np.median(v)), "ci90": np.percentile(v, [5, 95]).tolist(),
                  "ci68": np.percentile(v, [16, 84]).tolist()}
    return out


def stage_realdata(ctx):
    import torch
    from .inference import sample_posterior_batch
    from .model import EOSCatalogDataset, collate_fn, move_to_device
    from .realdata import build_real_catalog, ood_report
    model = ctx.main_model()
    dev = model.theta_mu.device
    res = {}
    for policy in ("dedup", "duplicate"):
        cat, notes = build_real_catalog(ctx.root("data/gw_events"), ctx.cfg["sim"], policy,
                                        ctx.cfg["real"]["download_gw"])
        batch = move_to_device(collate_fn([EOSCatalogDataset([{"theta_eos": [0] * 4, "catalog": cat}])[0]]), dev)
        n = ctx.cfg["sampling"]["n_real_samples"]
        sample_posterior_batch(model, batch, ctx.lo, ctx.hi, 100, 10)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        post, fout = sample_posterior_batch(model, batch, ctx.lo, ctx.hi, n, ctx.cfg["sampling"]["n_steps"], seed=7)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_sbi = time.perf_counter() - t0
        post = post[0]
        np.save(ctx.paths.data(f"real_sbi_{policy}", "realdata") + ".npy", post)
        der = _derive(ctx, post[: ctx.cfg["derived"]["n_real"]])
        np.save(ctx.paths.data(f"real_sbi_derived_{policy}", "realdata") + ".npy", der)
        res[policy] = {"catalog": cat, "notes": notes, "ood": ood_report(cat, ctx.splits["train"]),
                       "sbi_time_s": t_sbi, "n_samples": n, "frac_outside_box": float(fout[0]),
                       "theta": {p: {"median": float(np.median(post[:, k])), "std": float(post[:, k].std()),
                                     "ci90": np.percentile(post[:, k], [5, 95]).tolist()}
                                 for k, p in enumerate(PARAM_KEYS)},
                       "derived": _derived_table(der)}
        if policy == "dedup":
            ctx.timings.set("sbi_real_10k_s", t_sbi)
        print(f"[{policy}] SBI {n} samples in {t_sbi * 1e3:.1f} ms | derived {res[policy]['derived']}")
        for r in res[policy]["ood"]:
            if r["ood"]:
                print(f"  OOD warning: {r['name']} percentiles {np.round(r['percentiles'], 2)}")
        for nline in notes:
            print("  note:", nline)
    save_json(ctx.paths.result("realdata"), res)


# ── stage: reference posteriors (real) ───────────────────────────────────────
def _ref_jobs_real(ctx):
    rd = load_json(ctx.paths.result("realdata"))
    if rd is None:
        raise RuntimeError("run the `realdata` stage first")
    jobs = []
    for policy, seeds in (("dedup", (1, 2)), ("duplicate", (1,))):
        for seed in seeds:
            jobs.append((ctx.paths.reference(f"dynesty_real_{policy}_s{seed}", rd[policy]["catalog"]), rd[policy]["catalog"], ctx.grid_path(),
                         ctx.cfg["sim"], ctx.lo, ctx.hi, seed, ctx.cfg["dynesty"]))
    return jobs


def stage_reference_real(ctx):
    from .reference import run_many
    if not os.path.exists(ctx.grid_path()):
        stage_grid(ctx)
    res = run_many(_ref_jobs_real(ctx), n_procs=3)
    for r in res:
        print(f"dynesty seed {r['seed']}: {r['wall_s'] / 60:.1f} min, ncall {r['ncall']}, ESS {r['ess']:.0f}, "
              f"logZ {r['logz']:.2f}±{r['logzerr']:.2f}")


def _compare(ctx, sbi, ref, ref2=None, seed=0, repeats=None, with_c2st=True):
    cc = ctx.cfg["compare"]
    repeats = repeats or cc["n_subsample_repeats"]
    n = int(min(ref["ess"], len(sbi) // 2))
    out = {"n_matched": n, "ess_ref": ref["ess"]}
    if n < 5 or ref["ess"] < 10:
        return {**out, "status": "insufficient ESS for posterior comparison"}
    if ref2 is not None and ref2["ess"] < 10:
        ref2 = None
    mj = M.matched_js(sbi, ref["samples"], ref["weights"], ctx.lo, ctx.hi, n, repeats=repeats, seed=seed,
                      n_grid=cc["kde_grid"])
    out.update({k: v for k, v in mj.items() if k not in ("n", "repeats")})
    w = ref["weights"]
    mu_r = np.average(ref["samples"], axis=0, weights=w)
    sd_r = np.sqrt(np.average((ref["samples"] - mu_r) ** 2, axis=0, weights=w))
    out["mean_sbi"], out["mean_ref"] = sbi.mean(0), mu_r
    out["std_sbi"], out["std_ref"] = sbi.std(0), sd_r
    out["std_ratio"] = sbi.std(0) / sd_r
    out["dmean_over_ref_sd"] = (sbi.mean(0) - mu_r) / sd_r
    out["corr_sbi"], out["corr_ref"] = M.weighted_corr(sbi), M.weighted_corr(ref["samples"], w)
    out["corr_max_abs_diff"] = float(np.max(np.abs(out["corr_sbi"] - out["corr_ref"])))
    ref_eq = M.weighted_resample(ref["samples"], w, n, seed=seed)
    if with_c2st:
        out["c2st"] = M.c2st(sbi[np.random.default_rng(seed).choice(len(sbi), n, replace=False)], ref_eq, seed)
    if ref2 is not None:
        out["floor_ref_js"] = M.marginal_js(ref["samples"], ref2["samples"], ctx.lo, ctx.hi, wa=ref["weights"],
                                            wb=ref2["weights"], n_grid=cc["kde_grid"])
        if with_c2st:
            out["floor_ref_c2st"] = M.c2st(ref_eq, M.weighted_resample(ref2["samples"], ref2["weights"], n, seed + 1), seed)
    return out


def stage_compare_real(ctx):
    from .physics import accepted_branch
    from .realdata import NICER, PULSARS
    rd = load_json(ctx.paths.result("realdata"))
    refs = {j[0]: load_pickle(j[0]) for j in _ref_jobs_real(ctx) if os.path.exists(j[0])}
    if not refs:
        raise RuntimeError("run `reference-real` first")
    res = {}
    for policy in ("dedup", "duplicate"):
        r1 = refs.get(ctx.paths.reference(f"dynesty_real_{policy}_s1", rd[policy]["catalog"]))
        if r1 is None:
            continue
        r2 = refs.get(ctx.paths.reference(f"dynesty_real_{policy}_s2", rd[policy]["catalog"]))
        sbi = np.load(ctx.paths.data(f"real_sbi_{policy}", "realdata") + ".npy")
        cmp_ = _compare(ctx, sbi, r1, r2)
        if "status" in cmp_:
            res[policy] = cmp_
            continue
        ref_eq = M.weighted_resample(r1["samples"], r1["weights"], ctx.cfg["derived"]["n_real"], seed=3)
        der_ref = _derive(ctx, ref_eq)
        der_sbi = np.load(ctx.paths.data(f"real_sbi_derived_{policy}", "realdata") + ".npy")
        cmp_["derived_sbi"], cmp_["derived_ref"] = _derived_table(der_sbi), _derived_table(der_ref)
        cmp_["dynesty"] = {k: r1[k] for k in ("wall_s", "ncall", "ess", "logz", "logzerr", "log_z_accepted_prior", "log_z_accepted_prior_se", "grid_prior_acceptance", "nlive", "sample")}
        cmp_["sbi_time_s"] = rd[policy]["sbi_time_s"]
        cmp_["speedup_wall"] = r1["wall_s"] / rd[policy]["sbi_time_s"]
        cmp_["speedup_per_effective_sample"] = (r1["wall_s"] / r1["ess"]) / (rd[policy]["sbi_time_s"] / len(sbi))
        res[policy] = cmp_
        if policy == "dedup":
            ctx.timings.set("dynesty_real_s", r1["wall_s"])
            P.corner_overlay([sbi, r1["samples"]], ["amortised (this work)", "dynesty (same likelihood)"],
                             [P.C_SBI, P.C_REF], ctx.lo, ctx.hi, ctx.paths.figure("real_corner_overlay"),
                             weights=[None, r1["weights"]])
            mg = np.linspace(1.0, 2.5, 151)
            solve = partial(accepted_branch, n_pc=ctx.cfg["physics"]["n_pc"], cfg=ctx.cfg["sim"])
            with ctx.pool() as pool:
                br_s = [b for b in pool.map(solve, list(sbi[: ctx.cfg["compare"]["n_band_curves"]])) if b is not None]
                br_r = [b for b in pool.map(solve, list(ref_eq[: ctx.cfg["compare"]["n_band_curves"]])) if b is not None]
            bands = []
            for brs in (br_s, br_r):
                qR = P.credible_band([(b[0], b[1]) for b in brs], mg)
                qL = P.credible_band([(b[0], b[2]) for b in brs], mg, interp_lambda)
                bands.append((qR, qL))
            cat = rd[policy]["catalog"]
            P.mr_lambda_figure(bands, ["amortised", "dynesty"], [P.C_SBI, P.C_REF],
                               [o for o in cat if o["type"] == "nicer"], [o for o in cat if o["type"] == "pulsar"],
                               [o for o in cat if o["type"] == "gw"], ctx.paths.figure("fig1_mr_mlambda"), mg)
    save_json(ctx.paths.result("compare_real"), res)
    for policy, c in res.items():
        if "status" in c:
            print(policy, c)
            continue
        print(f"\n[{policy}] JS (bits, n={c['n_matched']}): {np.round(c['js_mean'], 4)} ± {np.round(c['js_std'], 4)}")
        print(f"   SBI-vs-SBI floor: {np.round(c['floor_sbi_mean'], 4)}"
              + (f" | dynesty-vs-dynesty floor: {np.round(c['floor_ref_js'], 4)}" if "floor_ref_js" in c else ""))
        print(f"   C2ST {c.get('c2st'):.3f} (floor {c.get('floor_ref_c2st', float('nan')):.3f}) | std ratio "
              f"{np.round(c['std_ratio'], 3)} | Δmean/σ {np.round(c['dmean_over_ref_sd'], 3)}")
        print(f"   derived SBI {c['derived_sbi']['R14']} vs dynesty {c['derived_ref']['R14']}")


# ── stage: synthetic reference benchmark ─────────────────────────────────────
def _synth_catalogs(ctx):
    sc = ctx.cfg["synth_ref"]
    chosen = []
    for a, b in sc["bins"]:
        chosen += [s for s in ctx.splits["test"] if a <= len(s["catalog"]) <= b][: sc["n_per_bin"]]
    return chosen


def stage_reference_synth(ctx):
    from .inference import posterior_for_samples
    from .reference import run_many
    sc = ctx.cfg["synth_ref"]
    if not os.path.exists(ctx.grid_path()):
        stage_grid(ctx)
    chosen = _synth_catalogs(ctx)
    jobs, keys = [], []
    for i, s in enumerate(chosen):
        for seed in ((1, 2) if i < sc["n_floor"] else (1,)):
            jobs.append((ctx.paths.reference(f"dynesty_synth_{s['idx']}_seed{seed}", s["catalog"]), s["catalog"], ctx.grid_path(),
                         ctx.cfg["sim"], ctx.lo, ctx.hi, seed, ctx.cfg["dynesty"]))
            keys.append((s["idx"], seed))
    refs = run_many(jobs, ctx.n_cores)
    by = {}
    for (idx, seed), r in zip(keys, refs):
        by.setdefault(idx, {})[seed] = r
    post, _ = posterior_for_samples(ctx.main_model(), chosen, ctx.lo, ctx.hi, n_samples=sc["n_sbi_samples"], batch_size=4, seed=9)
    rows = []
    excluded = []
    for s, p in zip(chosen, post):
        rr = by[s["idx"]]
        c = _compare(ctx, p, rr[1], rr.get(2), repeats=50)
        if "status" in c:
            excluded.append({"idx": s["idx"], **c})
            continue
        th = np.array(s["theta_eos"])
        cover = lambda x, w=None: [bool(np.abs(np.mean(x[:, k] < th[k]) - 0.5) < 0.34) for k in range(4)]  # noqa: E731
        c.update({"idx": s["idx"], "N": len(s["catalog"]), "dynesty_wall_s": rr[1]["wall_s"],
                  "truth_in_68_sbi": cover(p), "truth_in_68_ref": cover(M.weighted_resample(rr[1]["samples"],
                                                                                            rr[1]["weights"], 4000))})
        rows.append(c)
    if not rows:
        save_json(ctx.paths.result("reference_synth"), {"summary": {"n_catalogs": 0}, "excluded": excluded, "rows": []})
        return
    js = np.array([r["js_mean"] for r in rows])
    fl = np.array([r["floor_sbi_mean"] for r in rows])
    scores = [r["c2st"] for r in rows if np.isfinite(r["c2st"])]
    floors = [r["floor_ref_c2st"] for r in rows if np.isfinite(r.get("floor_ref_c2st", np.nan))]
    summ = {"n_catalogs": len(rows), "js_median": np.median(js, 0), "js_p90": np.percentile(js, 90, 0),
            "floor_sbi_median": np.median(fl, 0),
            "floor_ref_js_median": np.median([r["floor_ref_js"] for r in rows if "floor_ref_js" in r], 0).tolist()
            if any("floor_ref_js" in r for r in rows) else None,
            "c2st_median": float(np.median(scores)) if scores else None,
            "c2st_floor_median": float(np.median(floors)) if floors else None,
            "n_valid_c2st": len(scores), "n_valid_floor_c2st": len(floors),
            "std_ratio_median": np.median([r["std_ratio"] for r in rows], 0),
            "abs_dmean_over_sd_median": np.median(np.abs([r["dmean_over_ref_sd"] for r in rows]), 0),
            "truth_in_68_rate_sbi": np.mean([r["truth_in_68_sbi"] for r in rows], 0),
            "truth_in_68_rate_ref": np.mean([r["truth_in_68_ref"] for r in rows], 0),
            "dynesty_wall_s_median": float(np.median([r["dynesty_wall_s"] for r in rows]))}
    save_json(ctx.paths.result("reference_synth"), {"summary": summ, "rows": rows, "excluded": excluded})
    ctx.timings.set("dynesty_synth_median_s", summ["dynesty_wall_s_median"])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].boxplot([js[:, k] for k in range(4)], positions=np.arange(4) - 0.15, widths=0.25)
    ax[0].boxplot([fl[:, k] for k in range(4)], positions=np.arange(4) + 0.15, widths=0.25)
    ax[0].set(xticks=range(4), xticklabels=P.TEX, ylabel="JS divergence [bits]",
              title="left: amortised vs dynesty, right: finite-sample floor")
    ax[1].hist(scores, bins=15, alpha=0.7, label="amortised vs dynesty")
    if summ["c2st_floor_median"] is not None:
        ax[1].hist(floors, bins=15, alpha=0.7,
                   label="dynesty vs dynesty")
    ax[1].axvline(0.5, color="k", ls="--")
    ax[1].set(xlabel="C2ST accuracy (joint 4-D)")
    ax[1].legend()
    P.save(fig, ctx.paths.figure("synthetic_reference_benchmark"))
    print(summ)


# ── stage: economics + amortised workflow ────────────────────────────────────
def stage_economics(ctx):
    T = ctx.timings.data
    rd = load_json(ctx.paths.result("realdata"), {})
    cr = load_json(ctx.paths.result("compare_real"), {})
    res = {"timings": T}
    t_sim, t_train, t_grid = T.get("data_generation_s"), T.get("train_ensemble_s"), T.get("grid_s")
    t_dyn = T.get("dynesty_real_s")
    t_sbi = T.get("sbi_real_10k_s")
    if None not in (t_sim, t_train, t_dyn, t_sbi):
        offline_sbi = t_sim + t_train
        offline_ref = t_grid or 0.0
        res["offline_s"] = {"sbi_simulation": t_sim, "sbi_training": t_train, "reference_grid": offline_ref}
        res["break_even_n_analyses"] = max(0, (offline_sbi - offline_ref) / (t_dyn - t_sbi)) if t_dyn > t_sbi else None
        res["note"] = ("Offline times are wall-clock on the recorded hardware (simulation on "
                       f"{T.get('data_generation_cores')} CPU cores, training on the recorded model device); the reference sampler also "
                       "needs the TOV grid once. Changing the prior, noise model or simulator requires re-simulation "
                       "and retraining for the amortised model.")
    else:
        res["missing"] = [k for k, v in {"data_generation_s": t_sim, "train_ensemble_s": t_train,
                                         "dynesty_real_s": t_dyn, "sbi_real_10k_s": t_sbi}.items() if v is None]
    if ctx.cfg.get("run_subset_workflow", True) and rd:
        from .inference import posterior_for_catalogs
        cat = rd["dedup"]["catalog"]
        subsets = [list(c) for r in range(1, len(cat) + 1) for c in itertools.combinations(range(len(cat)), r)]
        t0 = time.perf_counter()
        post, _ = posterior_for_catalogs(ctx.main_model(), [[cat[i] for i in s] for s in subsets], ctx.lo, ctx.hi,
                                         n_samples=ctx.cfg["economics"]["subset_n_samples"], batch_size=32)
        t_all = time.perf_counter() - t0
        loo = {}
        for i, o in enumerate(cat):
            idx = subsets.index([j for j in range(len(cat)) if j != i])
            loo[o["name"]] = _derived_table(_derive(ctx, post[idx][: ctx.cfg["economics"]["loo_n_samples"]]))
        full = subsets.index(list(range(len(cat))))
        loo["(full catalog)"] = _derived_table(_derive(ctx, post[full][: ctx.cfg["economics"]["loo_n_samples"]]))
        res["subset_workflow"] = {
            "n_subsets": len(subsets), "sbi_total_s": t_all,
            "dynesty_equivalent_s": len(subsets) * t_dyn if t_dyn else None,
            "gamma01_by_subset": [{"obs": [cat[i]["name"] for i in s],
                                   "g0_median": float(np.median(p[:, 0])), "g1_median": float(np.median(p[:, 1])),
                                   "g0_sd": float(p[:, 0].std()), "g1_sd": float(p[:, 1].std())}
                                  for s, p in zip(subsets, post)],
            "leave_one_out_R14": {k: v["R14"] for k, v in loo.items()}, "leave_one_out_derived": loo}
        print(f"all {len(subsets)} subsets in {t_all:.1f} s (dynesty-equivalent ≈ "
              f"{(len(subsets) * t_dyn / 3600) if t_dyn else float('nan'):.1f} h)")
    if "speedup_wall" in cr.get("dedup", {}):
        res["speedup_wall"] = cr["dedup"]["speedup_wall"]
        res["speedup_per_effective_sample"] = cr["dedup"]["speedup_per_effective_sample"]
    imp = load_json(ctx.paths.result("importance"), {}) or {}
    if "time_s" in imp.get("real", {}).get("dedup", {}) and t_dyn:
        r = imp["real"]["dedup"]
        t_is = r["time_s"]["total"]
        res["importance_corrected"] = {
            "time_s": t_is, "ess": r["ess"], "efficiency": r["efficiency"],
            "speedup_wall_exact": t_dyn / t_is,
            "break_even_n_analyses": ((t_sim + t_train) / (t_dyn - t_is)
                                      if t_sim is not None and t_train is not None and t_dyn > t_is else None),
            "speedup_per_effective_sample_exact": (t_dyn / cr["dedup"]["dynesty"]["ess"]) / (t_is / r["ess"])
            if "speedup_wall" in cr.get("dedup", {}) else None,
            "note": "IS-corrected posterior is asymptotically exact (same target as dynesty); its cost includes "
                    "flow sampling, flow log-density and exact-likelihood evaluation."}
    save_json(ctx.paths.result("economics"), res)
    print({k: v for k, v in res.items() if k not in ("subset_workflow", "timings")})


# ── stage: systematics ───────────────────────────────────────────────────────
def _crust_job(args):
    from .physics import accepted_branch
    gc, g_crust, n_pc, sim = args
    return accepted_branch(gc, n_pc=n_pc, gamma_crust=g_crust, cfg=sim)


def stage_systematics(ctx):
    model, prior, cfg = ctx.main_model(), ctx.prior, ctx.cfg
    rng = np.random.default_rng([cfg["seed"], 31])
    base = ctx.splits["test"][:2000]
    res = {}
    # (a) crust: same thetas, crust index ~ U(1.28, 1.42) actually used (fix B4)
    sel = base[:500]
    crusts = rng.uniform(1.28, 1.42, len(sel))
    with ctx.pool() as pool:
        brs = pool.map(_crust_job, [(s["theta_eos"], c, cfg["physics"]["n_pc"], cfg["sim"]) for s, c in zip(sel, crusts)])
    crust_s, ref_s = [], []
    res["crust_acceptance"] = {"attempted": len(sel), "accepted": sum(b is not None for b in brs)}
    for s, b in zip(sel, brs):
        if b is None:
            continue
        for target, branch in ((ref_s, s["curve"]), (crust_s, b)):
            paired_rng = np.random.default_rng([cfg["seed"], 32, s["idx"]])
            target.append({"theta_eos": s["theta_eos"], "catalog": simulate_catalog(branch, paired_rng, cfg["sim"])})
    for name, ss in (("crust_fixed", ref_s), ("crust_varied", crust_s)):
        if not ss:
            res[name] = {"n": 0}
            continue
        sm = _summaries(ctx, model, ss, f"sys_{name}", scope="systematics")
        res[name] = {"n": len(ss), "accuracy": M.accuracy_table(sm, prior, PARAM_KEYS),
                     "coverage_error": M.coverage_error(sm["rank"], sm["n_samples"])}
    # (b) noise: covariate shift (sigma x1.5, reported correctly) and misspecification (noise 1.3x reported)
    for name, kw in (("sigma_x1.5_shift", {"sigma_scale": 1.5}), ("noise_1.3x_misreported", {"noise_scale": 1.3}),
                     ("nominal", {})):
        ss = [{"theta_eos": s["theta_eos"], "catalog": simulate_catalog(tuple(s["curve"]), np.random.default_rng([cfg["seed"], 33, s["idx"]]), cfg["sim"], **kw)}
              for s in base]
        sm = _summaries(ctx, model, ss, f"sys_{name}", scope="systematics")
        lv, cov, se = M.coverage_curve(sm["rank"], sm["n_samples"])
        res[name] = {"accuracy": M.accuracy_table(sm, prior, PARAM_KEYS),
                     "coverage_error": M.coverage_error(sm["rank"], sm["n_samples"]),
                     "coverage_68": cov[np.argmin(np.abs(lv - 0.68))], "coverage_90": cov[np.argmin(np.abs(lv - 0.9))]}
    save_json(ctx.paths.result("systematics"), res)
    for k, v in res.items():
        if "coverage_error" in v:
            print(f"{k:>24}: coverage error {np.round(v['coverage_error'], 3)}")


# ── stage: summary ───────────────────────────────────────────────────────────
SUMMARY_INPUTS = ("data_generation", "grid_validation", "eval_main", "ablation_arch", "ablation_messenger", "zero_noise",
                  "identifiability", "realdata", "compare_real", "reference_synth", "importance", "convergence",
                  "economics", "systematics", "train_ensemble")


def stage_summary(ctx):
    allr = {n: load_json(ctx.paths.result(n)) for n in SUMMARY_INPUTS}
    save_json(ctx.paths.result("timings"), ctx.timings.data)
    info = load_json(os.path.join(ctx.paths.run("main"), "train_info.json"), {})
    import platform
    versions = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "torch", "sklearn", "dynesty", "h5py"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    out = {"config": ctx.cfg, "versions": versions, "train_main": {k: v for k, v in info.items() if k != "history"},
           "timings": ctx.timings.data, "available": {n: r is not None for n, r in allr.items()}}
    save_json(ctx.paths.result("summary"), out)
    lines = ["# Numbers for the paper (auto-generated)", ""]
    ev = allr["eval_main"]
    if ev:
        lines += ["## Test split (main model)", "| param | MAE | MAE prior | ratio | contraction |", "|---|---|---|---|---|"]
        lines += [f"| {p} | {r['mae_post_median']:.4f} | {r['mae_prior_median']:.4f} | {r['mae_ratio_to_prior']:.3f} | "
                  f"{r['contraction']:.3f} |" for p, r in ev["accuracy"].items()]
        lines += ["", f"Coverage error per param: {np.round(ev['coverage_error'], 4).tolist()}",
                  f"TARP max |ECP - alpha|: {ev['tarp_max_abs_dev']:.3f}", ""]
    if info:
        lines += [f"Training: best epoch {info.get('best_epoch')} of {info.get('epochs_run')} run "
                  f"(max {info.get('max_epochs')}, patience {info.get('patience')}), {info.get('wall_s', 0) / 3600:.2f} h "
                  f"on {info.get('device')}, {info.get('n_params')} parameters.", ""]
    cr = allr["compare_real"]
    if cr and "speedup_wall" in cr.get("dedup", {}):
        c = cr["dedup"]
        lines += ["## Real catalog vs dynesty (same likelihood)",
                  f"JS [bits], matched n={c['n_matched']}: {np.round(c['js_mean'], 4).tolist()} "
                  f"(SBI floor {np.round(c['floor_sbi_mean'], 4).tolist()}"
                  + (f", dynesty floor {np.round(c['floor_ref_js'], 4).tolist()})" if "floor_ref_js" in c else ")"),
                  f"C2ST: {c.get('c2st')} (floor {c.get('floor_ref_c2st')})",
                  f"std ratio: {np.round(c['std_ratio'], 3).tolist()}",
                  f"speed-up (wall): {c['speedup_wall']:.0f}x; per effective sample: {c['speedup_per_effective_sample']:.0f}x",
                  f"R1.4 SBI {c['derived_sbi']['R14']} | dynesty {c['derived_ref']['R14']}", ""]
    if ev:
        if ev.get("derived_test"):
            lines += ["## Observable-space calibration (posterior-predictive EOS -> TOV)",
                      "| quantity | MAE | MAE prior | 68% coverage | coverage error |", "|---|---|---|---|---|"]
            for q_ in ("Mmax", "R14", "L14"):
                d = ev["derived_test"][q_]
                fields = [format(d[k], ".3f") if d.get(k) is not None else "n/a"
                          for k in ("mae", "mae_prior", "coverage68", "coverage_error")]
                lines.append(f"| {q_} | " + " | ".join(fields) + " |")
            lines.append("")
        lines += [f"Coverage error, identifiable params (g0, g1): {ev.get('coverage_error_identifiable_g0_g1')}",
                  f"Prior-dominated parameters (contraction < 0.2): {ev.get('prior_dominated')}",
                  f"Ensemble members: {ev.get('n_ensemble_members')}; seed variability (mean, sd): "
                  f"{ev.get('seed_variability')}", ""]
    imp = allr["importance"]
    if imp and "log_z" in imp.get("real", {}).get("dedup", {}):
        r = imp["real"]["dedup"]
        lines += ["## Importance-sampling correction (exact likelihood, real catalog)",
                  f"efficiency {r['efficiency']:.3f} (ESS {r['ess']:.0f}); box-prior log Z {r['log_z']:.3f} ± {r['log_z_se']:.3f}"
                  + (f" vs dynesty {r['dynesty_log_z']:.3f} ± {r['dynesty_log_z_err']:.3f}" if "dynesty_log_z" in r else ""),
                  f"Accepted-prior log Z {r['log_z_accepted_prior']:.3f} ± {r['log_z_accepted_prior_se']:.3f} "
                  f"(grid acceptance {r['grid_prior_acceptance']:.4f})",
                  f"Strict-support sensitivity: {r.get('strict_sensitivity')}",
                  f"derived (IS-corrected): {r.get('derived_is_corrected')}"]
        if imp.get("test_efficiency_summary"):
            lines.append(f"test-catalog IS efficiency: {imp['test_efficiency_summary']}")
        if imp.get("synthetic_summary"):
            lines.append(f"synthetic catalogs: {imp['synthetic_summary']}")
        lines.append("")
    cv = allr["convergence"]
    if cv and cv.get("n_theta"):
        lines += ["## Forward-model convergence (production vs 4x denser / 100x tighter TOV)",
                  f"max|dR| p95 {cv['max_abs_dR_km']['p95']:.4f} km ({cv['p95_dR_over_min_sigmaR']:.3f} of the smallest "
                  f"sigma_R); max rel dLambda p95 {cv['max_rel_dLambda']['p95']:.4f}; |dMmax| p95 {cv['abs_dMmax']['p95']:.4f}", ""]
    with open(ctx.paths.result("paper_numbers", "md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


# ── stage: forward-model resolution convergence ──────────────────────────────
def _conv_job(args):
    from .physics import compute_mr_curve
    from .simulator import stable_branch
    gc, n_pc, rtol = args
    try:
        c = compute_mr_curve(list(gc), n_pc=n_pc, rtol=rtol)
    except Exception:
        return None
    return None if c is None else np.stack(stable_branch(c))


def stage_convergence(ctx):
    """Production forward model (n_pc, rtol 1e-6, linear interpolation in M) vs a 4x denser,
    100x tighter solve; errors are compared with the smallest training uncertainties."""
    cc, sim = ctx.cfg["convergence"], ctx.cfg["sim"]
    sel = ctx.splits["test"][: cc["n_theta"]]
    with ctx.pool() as pool:
        hi = pool.map(_conv_job, [(s["theta_eos"], cc["n_pc_hi"], cc["rtol_hi"]) for s in sel], chunksize=1)
    dR, dL, dM = [], [], []
    for s, h in zip(sel, hi):
        if h is None:
            continue
        Ms, Rs, Ls = s["curve"]
        m = np.linspace(1.0, min(Ms[-1], h[0][-1]) - 1e-3, 40)
        dR.append(np.abs(np.interp(m, Ms, Rs) - np.interp(m, h[0], h[1])).max())
        dL.append(np.abs(interp_lambda(m, Ms, Ls) / interp_lambda(m, h[0], h[2]) - 1).max())
        dM.append(abs(Ms[-1] - h[0][-1]))

    def q(a):
        return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(np.max(a))}
    if not dR:
        save_json(ctx.paths.result("convergence"), {"n_theta": 0, "n_requested": len(sel), "status": "no valid reference curves"})
        return
    res = {"n_theta": len(dR), "n_requested": len(sel), "max_abs_dR_km": q(dR), "max_rel_dLambda": q(dL), "abs_dMmax": q(dM),
           "reference": {"n_pc": cc["n_pc_hi"], "rtol": cc["rtol_hi"]},
           "smallest_training_sigma": {"R_km": sim["NICER_SIG_R"][0], "psr_M": sim["PSR_SIG_M"][0]}}
    res["p95_dR_over_min_sigmaR"] = res["max_abs_dR_km"]["p95"] / sim["NICER_SIG_R"][0]
    res["p95_dMmax_over_min_psr_sigma"] = res["abs_dMmax"]["p95"] / sim["PSR_SIG_M"][0]
    save_json(ctx.paths.result("convergence"), res)
    print(res)


# ── stage: importance sampling with the exact likelihood ─────────────────────
def _is(ctx, key, *args, **kw):
    """Per-catalog cached importance sampling (resumable stage)."""
    from .importance import importance_sample
    kw.setdefault("n_steps", ctx.cfg["sampling"]["n_steps"])
    path = ctx.paths.data(f"is_{key}", "is-sampler", [args[1], kw["n"], kw["defensive"], kw["seed"]]) + ".pkl"
    if os.path.exists(path) and not ctx.force:
        return load_pickle(path)
    r = importance_sample(*args, **kw)
    save_pickle(path, r)
    return r


def stage_importance(ctx):
    ic = ctx.cfg["importance"]
    if not os.path.exists(ctx.grid_path()):
        stage_grid(ctx)
    model, gp, sim = ctx.main_model(), ctx.grid_path(), ctx.cfg["sim"]
    res = {"real": {}, "synthetic": [], "test_efficiency": []}
    rd = load_json(ctx.paths.result("realdata"))
    with ctx.pool() as pool:
        for policy in ("dedup", "duplicate"):
            if not rd or policy not in rd:
                continue
            cat = rd[policy]["catalog"]
            r = _is(ctx, f"real_{policy}", model, cat, ctx.lo, ctx.hi, gp, sim, n=ic["n_real"], defensive=ic["defensive"],
                                  seed=11, pool=pool)
            if "status" in r:
                res["real"][policy] = r
                continue
            from .importance import strict_posterior
            strict, strict_w = strict_posterior(r, ctx.cfg["physics"]["n_pc"], sim, pool)
            np.savez(ctx.paths.data(f"real_is_{policy}", "importance") + ".npz", samples=r["samples"],
                     weights=r["weights"], strict_weights=strict_w)
            row = {k: r[k] for k in ("ess", "efficiency", "log_z", "log_z_se", "log_z_accepted_prior", "log_z_accepted_prior_se", "grid_prior_acceptance", "frac_flow_outside_support", "time_s")}
            row["strict_sensitivity"] = strict
            ref_path = ctx.paths.reference(f"dynesty_real_{policy}_s1", cat)
            if os.path.exists(ref_path) and r["ess"] >= 10:
                ref = load_pickle(ref_path)
                row["dynesty_log_z"], row["dynesty_log_z_err"] = ref["logz"], ref["logzerr"]
                row["delta_log_z"] = r["log_z"] - ref["logz"]
                eq = M.weighted_resample(r["samples"], r["weights"], int(r["ess"]), seed=1)
                row["vs_dynesty"] = _compare(ctx, eq, ref, repeats=max(3, ctx.cfg["compare"]["n_subsample_repeats"] // 4))
            row["comparison_status"] = "insufficient ESS" if r["ess"] < 10 else "eligible"
            if policy == "dedup" and r["ess"] >= 10:
                eq = M.weighted_resample(r["samples"], r["weights"], min(int(r["ess"]), ctx.cfg["derived"]["n_real"]), seed=2)
                row["derived_is_corrected"] = _derived_table(_derive(ctx, eq))
                ctx.timings.set("sbi_is_real_s", r["time_s"]["total"])
            res["real"][policy] = row
            msg = (f"[{policy}] IS efficiency {r['efficiency']:.3f} (ESS {r['ess']:.0f}/{r['n']}), "
                   f"logZ {r['log_z']:.2f}±{r['log_z_se']:.2f}")
            if "dynesty_log_z" in row:
                msg += f" vs dynesty {row['dynesty_log_z']:.2f}±{row['dynesty_log_z_err']:.2f}"
            print(msg)
        for s in _synth_catalogs(ctx):
            r = _is(ctx, f"synth_{s['idx']}", model, s["catalog"], ctx.lo, ctx.hi, gp, sim, n=ic["n_synth"],
                                  defensive=ic["defensive"], seed=s["idx"], pool=pool)
            if "status" in r:
                res["synthetic"].append({"idx": s["idx"], **r})
                continue
            row = {"idx": s["idx"], "N": len(s["catalog"]), "ess": r["ess"], "efficiency": r["efficiency"],
                   "log_z_accepted_prior": r["log_z_accepted_prior"], "log_z_accepted_prior_se": r["log_z_accepted_prior_se"],
                   "comparison_status": "insufficient ESS" if r["ess"] < 10 else "eligible", "log_z": r["log_z"],
                   "log_z_se": r["log_z_se"]}
            ref_path = ctx.paths.reference(f"dynesty_synth_{s['idx']}_seed1", s["catalog"])
            if os.path.exists(ref_path) and r["ess"] >= 10:
                ref = load_pickle(ref_path)
                row["delta_log_z"] = r["log_z"] - ref["logz"]
                row["dynesty_log_z_err"] = ref["logzerr"]
                eq = M.weighted_resample(r["samples"], r["weights"], int(r["ess"]), seed=1)
                c = _compare(ctx, eq, ref, repeats=10, with_c2st=True)
                if "status" in c:
                    row["comparison_status"] = c["status"]
                else:
                    row.update({"js_mean": c["js_mean"], "floor_sbi_mean": c["floor_sbi_mean"], "c2st": c["c2st"]})
            res["synthetic"].append(row)
        for s in ctx.splits["test"][: ic["n_test_catalogs"]]:
            r = _is(ctx, f"test_{s['idx']}", model, s["catalog"], ctx.lo, ctx.hi, gp, sim, n=ic["n_test"],
                                  defensive=ic["defensive"], seed=s["idx"], pool=pool)
            res["test_efficiency"].append({"N": len(s["catalog"]), "efficiency": r["efficiency"]})
    eff = np.array([r["efficiency"] for r in res["test_efficiency"]])
    if len(eff):
        res["test_efficiency_summary"] = {"median": float(np.median(eff)), "p10": float(np.percentile(eff, 10)),
                                          "frac_below_1pct": float(np.mean(eff < 0.01))}
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([r["N"] for r in res["test_efficiency"]], np.maximum(eff, 1e-5), s=12)
        ax.set(xlabel="catalog size N", ylabel="importance-sampling efficiency ESS/n", yscale="log")
        ax.grid(alpha=0.3)
        P.save(fig, ctx.paths.figure("is_efficiency_vs_N"))
    d = [r["delta_log_z"] for r in res["synthetic"] if "delta_log_z" in r]
    if d:
        res["synthetic_summary"] = {"median_abs_delta_log_z": float(np.median(np.abs(d))),
                                    "median_efficiency": float(np.median([r["efficiency"] for r in res["synthetic"]]))}
    save_json(ctx.paths.result("importance"), res)
    print({k: v for k, v in res.items() if k.endswith("summary")})


STAGES = {
    "simulate": stage_simulate, "train": stage_train, "evaluate": stage_evaluate, "grid": stage_grid,
    "realdata": stage_realdata, "reference-real": stage_reference_real, "compare-real": stage_compare_real,
    "identifiability": stage_identifiability, "zero-noise": stage_zero_noise, "ablate-arch": stage_ablate_arch,
    "ablate-messenger": stage_ablate_messenger, "reference-synth": stage_reference_synth,
    "importance": stage_importance, "convergence": stage_convergence,
    "economics": stage_economics, "systematics": stage_systematics, "summary": stage_summary,
}
ORDER = list(STAGES)
