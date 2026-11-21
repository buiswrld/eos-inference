"""Run paired causality policies on deterministic production-prior draws."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nseos.config import load_config
from nseos.physics import solve_sequence
from nseos.simulator import accept_eos, branch_summary


def trial(job):
    idx, theta, policy, n_pc, sim = job
    start = time.perf_counter()
    curve, endpoint = solve_sequence(theta, n_pc=n_pc, policy=policy)
    branch, reason = accept_eos(curve, sim)
    return {"idx": idx, "theta": theta, "policy": policy, **endpoint,
            "reason": endpoint["reason"] or reason, "accepted": branch is not None,
            "seconds": time.perf_counter() - start,
            **(branch_summary(branch) if branch is not None else {})}


def compare(n, workers, output):
    cfg = load_config()
    bounds = list(cfg["prior_bounds"].values())
    thetas = [np.random.default_rng([cfg["seed"], 1, i]).uniform(*np.asarray(bounds).T).tolist() for i in range(n)]
    report = {"seed": cfg["seed"], "stream": 1, "n": n, "workers": workers, "n_pc": cfg["physics"]["n_pc"],
              "policies": {}, "rows": {}}
    with ProcessPoolExecutor(workers) as pool:
        for policy in ("lenient", "extended", "strict"):
            start = time.perf_counter()
            jobs = [(i, th, policy, report["n_pc"], cfg["sim"]) for i, th in enumerate(thetas)]
            rows = list(pool.map(trial, jobs, chunksize=1))
            wall = time.perf_counter() - start
            accepted = [r for r in rows if r["accepted"]]
            summary = {"accepted": len(accepted), "wall_seconds": wall, "wall_seconds_per_accepted": wall / max(1,len(accepted)),
                       "sum_worker_seconds_per_accepted": sum(r["seconds"] for r in rows) / max(1,len(accepted)),
                       "rejections": dict(Counter(r["reason"] for r in rows if not r["accepted"])),
                       "accepted_unbracketed": sum(not r["bracketed"] for r in accepted),
                       "accepted_peak_in_extension": sum(r["used_extension"] for r in accepted),
                       "accepted_peak_above_table_end_with_no_crossing": sum(r["used_extension"] and r["boundary_kind"] == "table_end" for r in accepted),
                       "all_peaks_above_table_end_with_no_crossing": sum(r["used_extension"] and r["boundary_kind"] == "table_end" for r in rows),
                       "draws_reaching_table_end_without_crossing": sum(r["reached_boundary"] and r["boundary_kind"] == "table_end" for r in rows)}
            report["policies"][policy], report["rows"][policy] = summary, rows
            print(policy, summary, flush=True)
    report["paired"] = {}
    for policy in ("lenient", "strict"):
        pairs = [(a,b) for a,b in zip(report["rows"]["extended"],report["rows"][policy]) if a["accepted"] and b["accepted"]]
        shifts = {"n_both": len(pairs)}
        for quantity in ("Mmax","R14","L14"):
            delta = np.array([a[quantity]-b[quantity] for a,b in pairs])
            shifts[quantity] = dict(zip(("p05","median","p95","max_abs"), [*np.percentile(delta,[5,50,95]), max(abs(delta))]))
        report["paired"]["extended_minus_"+policy] = shifts
    report["strict_membership_mismatches"] = sum(bool(c["accepted"] and c["strict"]) != s["accepted"]
                                          for c,s in zip(report["rows"]["extended"],report["rows"]["strict"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(report["paired"], "membership mismatches",report["strict_membership_mismatches"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=320)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.draws, args.workers, args.output)
