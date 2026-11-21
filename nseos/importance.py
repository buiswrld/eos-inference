"""Importance-sampling correction of the amortised posterior with the exact likelihood
(neural importance sampling, cf. Dax et al. 2023).

Proposal: defensive mixture  q(θ) = (1-α) q_flow(θ | catalog) + α U(prior box).
Weights:  w = π(θ) L(θ) / q(θ), with π the uniform box and L the simulator-exact likelihood
(-inf outside the truncated prior support), i.e. *the same target as dynesty*.

Outputs per catalog:
  * sampling efficiency ε = ESS/n  — a per-catalog quality diagnostic of the network;
  * log-evidence  log Z = log mean(w)  with standard error — directly comparable to dynesty;
  * asymptotically exact, weighted posterior samples (the network only proposes).
"""
import math
import multiprocessing
import time

import numpy as np
import torch

from .model import EOSCatalogDataset, collate_fn, model_log_prob, move_to_device
from .reference import _grid_eos, evidence_fields
from .likelihood import loglike_catalog_eos


def _ll_chunk(args):
    catalog, grid_path, sim_cfg, thetas = args
    eos = _grid_eos(grid_path)
    return np.array([loglike_catalog_eos(catalog, eos(t), sim_cfg) for t in thetas])


def loglike_many(catalog, thetas, grid_path, sim_cfg, lo, hi, pool=None, n_chunks=64):
    inside = np.all((thetas >= lo) & (thetas <= hi), axis=1)
    out = np.full(len(thetas), -np.inf)
    idx = np.where(inside)[0]
    if len(idx):
        chunks = np.array_split(idx, min(n_chunks, len(idx)))
        jobs = [(catalog, grid_path, sim_cfg, thetas[c]) for c in chunks]
        res = pool.map(_ll_chunk, jobs) if pool is not None else [_ll_chunk(j) for j in jobs]
        for c, r in zip(chunks, res):
            out[c] = r
    return out


def importance_sample(model, catalog, lo, hi, grid_path, sim_cfg, n=20000, n_steps=50, defensive=0.1,
                      seed=0, pool=None):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    dev = model.theta_mu.device
    batch = move_to_device(collate_fn([EOSCatalogDataset([{"theta_eos": [0] * 4, "catalog": catalog}])[0]]), dev)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    rng = np.random.default_rng(seed)
    if n <= 0 or not 0 <= defensive <= 1:
        raise ValueError("n must be positive and defensive must be in [0, 1]")
    n_def = int(round(defensive * n))
    alpha = n_def / n
    t0 = time.perf_counter()
    th_flow = (model.sample(batch, n - n_def, n_steps, generator=g)[0].double().cpu().numpy()
               if n_def < n else np.empty((0, len(lo))))
    th = np.concatenate([th_flow, rng.uniform(lo, hi, size=(n_def, 4))])
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_sample = time.perf_counter() - t0
    t0 = time.perf_counter()
    lq_flow = model_log_prob(model, batch, torch.tensor(th, dtype=torch.float32, device=dev)[None], n_steps)
    lq_flow = lq_flow[0].double().cpu().numpy()
    t_logq = time.perf_counter() - t0
    log_vol = float(np.sum(np.log(hi - lo)))
    box_logq = np.where(np.all((th >= lo) & (th <= hi), axis=1), -log_vol, -np.inf)
    if alpha == 1:
        lq = box_logq
    elif alpha > 0:
        lq = np.logaddexp(math.log1p(-alpha) + lq_flow, math.log(alpha) + box_logq)
    else:
        lq = lq_flow
    t0 = time.perf_counter()
    ll = loglike_many(catalog, th, grid_path, sim_cfg, lo, hi, pool)
    t_ll = time.perf_counter() - t0
    logw = -log_vol + ll - lq
    logw[~np.isfinite(logw)] = -np.inf
    m = np.max(logw)
    if not np.isfinite(m):
        return {"status": "insufficient ESS", "reason": "all target weights are zero",
                "n": n, "ess": 0.0, "efficiency": 0.0}
    w = np.exp(logw - m)
    sw, sw2 = w.sum(), (w ** 2).sum()
    ess = float(sw ** 2 / sw2) if sw2 > 0 else 0.0
    log_z = float(m + math.log(sw / n)) if sw > 0 else -np.inf
    se_log_z = float(np.sqrt(max(np.mean(w ** 2) / np.mean(w) ** 2 - 1.0, 0.0) / n)) if sw > 0 else np.inf
    return {**evidence_fields(log_z, se_log_z, grid_path, lo, hi, sim_cfg), "samples": th, "weights": w / sw if sw > 0 else w, "ess": ess, "efficiency": ess / n,
            "log_z": log_z, "log_z_se": se_log_z, "n": n, "defensive": defensive,
            "frac_flow_outside_support": float(np.mean(~np.isfinite(ll[: n - n_def]))) if n_def < n else None,
            "time_s": {"sample": t_sample, "log_q": t_logq, "likelihood": t_ll,
                       "total": t_sample + t_logq + t_ll}}


def _strict_observables(theta, n_pc, sim):
    from .physics import solve_sequence
    from .simulator import accept_eos, branch_summary
    curve, info = solve_sequence(theta, n_pc=n_pc, policy="strict")
    branch, reason = accept_eos(curve, sim)
    if branch is None:
        return info["reason"] or reason, None
    values = branch_summary(branch)
    return None, [values[k] for k in ("Mmax", "R14", "L14")]


def strict_posterior(result, n_pc, sim, pool):
    """Condition the grid-likelihood importance posterior on exact strict TOV support."""
    from functools import partial
    th, weights = result["samples"], np.asarray(result["weights"], float)
    pos = np.flatnonzero(weights > 0)
    unique, inverse = np.unique(th[pos], axis=0, return_inverse=True)
    solved = pool.map(partial(_strict_observables, n_pc=n_pc, sim=sim), unique, chunksize=4)
    keep = np.zeros(len(th), dtype=bool)
    derived = []
    unresolved = 0.
    for i, j in zip(pos, inverse):
        reason, quantities = solved[j]
        if quantities is not None:
            keep[i] = True
            derived.append(quantities)
        elif reason in {"solver_failure_before_turnover", "insufficient_stellar_models"}:
            unresolved += weights[i]
    conditional = np.where(keep, weights, 0.)
    mass = float(conditional.sum())
    if mass:
        conditional /= mass
    ess = 1 / np.sum(conditional**2) if mass else 0.
    summary = {"strict_posterior_mass": mass, "unresolved_posterior_mass": float(unresolved),
               "ess": float(ess), "parent_ess": result["ess"], "n_positive": len(pos),
               "n_strict": int(keep.sum()), "n_unique_solved": len(unique),
               "status": "ok" if min(ess, result["ess"]) >= 10 else "insufficient ESS"}
    if summary["status"] == "ok":
        values = np.column_stack([th[keep], derived])
        summary["conditional_quantiles_05_50_95"] = {}
        for k, key in enumerate(("gamma0", "gamma1", "gamma2", "gamma3", "M_TOV", "R14", "L14")):
            order = np.argsort(values[:, k])
            cumulative = np.cumsum(conditional[keep][order])
            indices = np.searchsorted(cumulative, [.05, .5, .95])
            summary["conditional_quantiles_05_50_95"][key] = values[order[indices], k].tolist()
    return summary, conditional


def make_pool(n=None):
    return multiprocessing.Pool(n or multiprocessing.cpu_count())
