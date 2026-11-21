"""Construction of the cached TOV interpolation grid."""
import numpy as np

from .simulator import stable_branch


def mass_grid(gcfg):
    return np.round(np.arange(gcfg["m_lo"], gcfg["m_hi"] + 1e-9, gcfg["dm"]), 6)


def _grid_point_values(gc, n_pc, m_grid):
    from .physics import compute_mr_curve
    try:
        curve = compute_mr_curve(list(gc), n_pc=n_pc)
    except Exception:
        curve = None
    K = len(m_grid)
    if curve is None:
        return np.full(3 + 2 * K, np.nan)
    Ms, Rs, Ls = stable_branch(curve)
    if len(Ms) < 3:
        return np.full(3 + 2 * K, np.nan)
    R = np.interp(m_grid, Ms, Rs)                        # flat extension beyond the branch ends
    logL = np.interp(m_grid, Ms, np.log(np.maximum(Ls, 1e-12)))
    return np.concatenate([[Ms[-1], Ms[0], 1.0], R, logL])


def _fill_nans(arr, max_iter=30):
    """Fill NaN cells with the mean of valid neighbours in the 3^4 stencil (repeat)."""
    arr = arr.copy()
    for _ in range(max_iter):
        nan = np.isnan(arr)
        if not nan.any():
            break
        acc, cnt = np.zeros_like(arr), np.zeros_like(arr)
        for shift in np.ndindex(3, 3, 3, 3):
            s = tuple(int(v) - 1 for v in shift)
            if s == (0, 0, 0, 0):
                continue
            rolled = arr
            for ax, d in enumerate(s):
                if d:
                    rolled = np.roll(rolled, d, axis=ax)
                    idx = [slice(None)] * 4
                    idx[ax] = 0 if d == 1 else -1      # do not wrap around
                    rolled = rolled.copy()
                    rolled[tuple(idx)] = np.nan
            v = ~np.isnan(rolled)
            acc[v] += rolled[v]
            cnt[v] += 1
        fill = nan & (cnt > 0)
        arr[fill] = acc[fill] / cnt[fill]
    if np.isnan(arr).all():
        raise ValueError("No valid TOV solutions on the interpolation grid")
    if np.isnan(arr).any():
        arr[np.isnan(arr)] = np.nanmean(arr)
    return arr


def build_grid(cfg, n_cores=None, partial_path=None, block=1000):
    """TOV grid over the prior box. Resumable: with partial_path, finished blocks of grid points
    are saved as they complete and skipped on restart."""
    import multiprocessing
    import os
    from .io import load_pickle, save_pickle
    from functools import partial
    from itertools import product
    gcfg, pb = cfg["grid"], cfg["prior_bounds"]
    axes = [np.linspace(pb[f"gamma{k}"][0], pb[f"gamma{k}"][1], n) for k, n in enumerate(gcfg["shape"])]
    m_grid = mass_grid(gcfg)
    pts = list(product(*axes))
    worker = partial(_grid_point_values, n_pc=cfg["physics"]["n_pc"], m_grid=m_grid)
    import time
    done, elapsed = [], 0.0
    if partial_path and os.path.exists(partial_path):
        st = load_pickle(partial_path)
        done, elapsed = st["done"], st["elapsed"]
        print(f"grid: resuming at point {len(done)} / {len(pts)}")
    t0 = time.time()
    with multiprocessing.Pool(n_cores or multiprocessing.cpu_count()) as pool:
        for i0 in range(len(done), len(pts), block):
            done.extend(pool.map(worker, pts[i0:i0 + block], chunksize=10))
            if partial_path:
                save_pickle(partial_path, {"done": done, "elapsed": elapsed + time.time() - t0})
            print(f"grid: {len(done)} / {len(pts)} points", flush=True)
    vals = np.array(done).reshape(*gcfg["shape"], -1)
    tov_ok = ~np.isnan(vals[..., 0])
    for f in range(vals.shape[-1]):
        if f == 2:
            continue
        vals[..., f] = _fill_nans(vals[..., f])
    vals[..., 2] = tov_ok.astype(float)
    return {"axes": axes, "values": vals, "m_grid": m_grid, "shape": list(gcfg["shape"]),
            "n_pc": cfg["physics"]["n_pc"], "frac_tov_ok": float(tov_ok.mean()),
            "build_s": elapsed + time.time() - t0}
