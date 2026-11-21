"""Real multi-messenger catalog.

Fixes:
  B9  v1's HDF5 parser failed on both files (GWTC-1 stores component masses + lambda1/2,
      not chirp_mass/lambda_tilde; the GW190425 URL returned a non-HDF5 page), so the
      literature fallbacks were used without being reported. The parser below walks any
      HDF5 layout, derives source-frame Mc and Lambda-tilde, and reports what it used.
  B8  PSR J0740+6620's radio-timing mass is also the mass prior of its NICER M-R
      posterior; using both double-counts it. policy='dedup' (default) drops the radio entry.
  B1  ``ood_report`` checks every real feature against the training distribution.
"""
import os
import urllib.request

import numpy as np

from .simulator import chirp_mass, lambda_tilde

GW_SOURCES = {
    "GW170817": {"url": "https://dcc.ligo.org/public/0157/P1800370/005/GW170817_GWTC-1.hdf5",
                 "file": "GW170817_GWTC-1.hdf5", "prefer": ["lowSpin_posterior", "low_spin"]},
    # 661 MB PESummary file from LIGO-P2000026 (the v1 URL pointed at a non-existent file).
    "GW190425": {"url": "https://dcc.ligo.org/public/0165/P2000026/002/posterior_samples.h5",
                 "file": "GW190425_posterior_samples.h5", "prefer": ["LS", "low_spin", "lowSpin"]},
}

# Gaussianised literature summaries (used only if parsing fails; reported in the output).
GW_LITERATURE = {
    "GW170817": {"x": [300.0, 1.186], "sigma": [190.0, 0.001],
                 "ref": "Abbott et al. 2019, PRX 9, 011001 (low spin; Lt 90% HPD 70-720)"},
    "GW190425": {"x": [400.0, 1.44], "sigma": [300.0, 0.02],
                 "ref": "Abbott et al. 2020, ApJL 892, L3 (low spin; Lt < 600 at 90%)"},
}

NICER = [
    {"name": "PSR J0030+0451 (NICER)", "type": "nicer", "x": [1.34, 12.71], "sigma": [0.155, 1.165],
     "ref": "Riley et al. 2019, ApJL 887, L21"},
    {"name": "PSR J0740+6620 (NICER)", "type": "nicer", "x": [2.072, 12.39], "sigma": [0.067, 1.140],
     "ref": "Riley et al. 2021, ApJL 918, L27 (mass prior = radio timing)"},
    {"name": "PSR J0437-4715 (NICER)", "type": "nicer", "x": [1.418, 11.36], "sigma": [0.037, 0.790],
     "ref": "Choudhury et al. 2024, ApJL 971, L20"},
]
PULSARS = [
    {"name": "PSR J0740+6620 (radio)", "type": "pulsar", "x": [2.08], "sigma": [0.07],
     "ref": "Fonseca et al. 2021, ApJL 915, L12", "shared_with": "PSR J0740+6620 (NICER)"},
    {"name": "PSR J0952-0607 (optical companion)", "type": "pulsar", "x": [2.35], "sigma": [0.17],
     "ref": "Romani et al. 2022, ApJL 934, L17"},
    {"name": "PSR J1614-2230 (radio)", "type": "pulsar", "x": [1.908], "sigma": [0.016],
     "ref": "Arzoumanian et al. 2018, NANOGrav 11-year data set, Table 8 (ApJS 235, 37)"},
]

H0_KM_S_MPC = 67.9
C_KM_S = 299792.458


def _z_from_dl(dl_mpc):
    """Low-z inversion of d_L(z) ≈ (c/H0) z (1 + z/2)  (q0 ≈ -0.53 gives a <0.1% change at z~0.01)."""
    a = dl_mpc * H0_KM_S_MPC / C_KM_S
    return -1.0 + np.sqrt(1.0 + 2.0 * a)


def _walk(h5, prefix=""):
    import h5py
    for k in h5.keys():
        obj = h5[k]
        path = f"{prefix}/{k}"
        if isinstance(obj, h5py.Dataset):
            if obj.dtype.names:
                yield path, obj
        elif isinstance(obj, h5py.Group):
            yield from _walk(obj, path)


def _field(arr, *names):
    for n in names:
        if n in arr.dtype.names:
            return np.asarray(arr[n], float)
    return None


def parse_gw_file(path, prefer):
    """Return (Lt samples, source-frame Mc samples, dataset path) or None."""
    import h5py
    if not os.path.exists(path) or not h5py.is_hdf5(path):
        return None
    with h5py.File(path, "r") as f:
        cands = [(p, d) for p, d in _walk(f) if "prior" not in p.lower() and "posterior" in p.lower()]
        cands.sort(key=lambda pd: min([i for i, s in enumerate(prefer) if s.lower() in pd[0].lower()] or [99]))
        for p, d in cands:
            arr = d[()]
            lt = _field(arr, "lambda_tilde")
            mc = _field(arr, "chirp_mass_source")
            if lt is None or mc is None:
                m1 = _field(arr, "mass_1_source", "m1_source")
                m2 = _field(arr, "mass_2_source", "m2_source")
                if m1 is None:
                    m1d, m2d = _field(arr, "m1_detector_frame_Msun", "mass_1"), _field(arr, "m2_detector_frame_Msun", "mass_2")
                    dl = _field(arr, "luminosity_distance_Mpc", "luminosity_distance")
                    if m1d is None or dl is None:
                        continue
                    z = _z_from_dl(dl)
                    m1, m2 = m1d / (1 + z), m2d / (1 + z)
                l1, l2 = _field(arr, "lambda1", "lambda_1"), _field(arr, "lambda2", "lambda_2")
                if lt is None and l1 is not None and l2 is not None:
                    lt = lambda_tilde(m1, m2, l1, l2)
                mc = chirp_mass(m1, m2)
            if lt is None:
                continue
            ok = np.isfinite(lt) & np.isfinite(mc)
            if np.count_nonzero(ok) >= 2:
                return lt[ok], mc[ok], p
    return None


def gw_observation(name, data_dir, download=True):
    src = GW_SOURCES[name]
    path = os.path.join(data_dir, src["file"])
    status = {}
    if download and not os.path.exists(path):
        try:
            print(f"Downloading {name} posterior samples ({src['url']}) ...")
            urllib.request.urlretrieve(src["url"], path)
        except Exception as e:
            status["download_error"] = repr(e)
    parsed = None
    try:
        parsed = parse_gw_file(path, src["prefer"])
    except Exception as e:
        status["parse_error"] = repr(e)
    if parsed is not None:
        lt, mc, ds = parsed
        q16, q50, q84 = np.percentile(lt, [16, 50, 84])
        m16, m50, m84 = np.percentile(mc, [16, 50, 84])
        obs = {"name": name, "type": "gw", "x": [float(q50), float(m50)],
               "sigma": [float(0.5 * (q84 - q16)), float(0.5 * (m84 - m16))],
               "source": f"parsed {os.path.basename(path)}:{ds} ({len(lt)} samples)",
               "std": [float(lt.std()), float(mc.std())]}
    else:
        lit = GW_LITERATURE[name]
        obs = {"name": name, "type": "gw", "x": list(lit["x"]), "sigma": list(lit["sigma"]),
               "source": f"literature fallback: {lit['ref']}"}
    obs["status"] = status
    print(f"  {name}: Lt = {obs['x'][0]:.1f} ± {obs['sigma'][0]:.1f}, Mc = {obs['x'][1]:.4f} ± "
          f"{obs['sigma'][1]:.4f}  [{obs['source']}]")
    return obs


def clip_to_support(catalog, sim_cfg):
    """Sigmas below the training range are inflated to its lower edge (conservative); reported."""
    ranges = {"gw": [sim_cfg["GW_SIG_LT"], sim_cfg["GW_SIG_MC"]],
              "nicer": [sim_cfg["NICER_SIG_M"], sim_cfg["NICER_SIG_R"]], "pulsar": [sim_cfg["PSR_SIG_M"]]}
    notes = []
    for o in catalog:
        for j, (lo, hi) in enumerate(ranges[o["type"]]):
            if o["sigma"][j] < lo:
                notes.append(f"{o['name']}: sigma[{j}] {o['sigma'][j]:.4g} -> {lo:.4g} (inflated to training floor)")
                o["sigma"][j] = lo
            elif o["sigma"][j] > hi:
                notes.append(f"{o['name']}: sigma[{j}] {o['sigma'][j]:.4g} above training range ({hi:.4g}); "
                             "kept, flagged as out of distribution")
    return catalog, notes


def build_real_catalog(data_dir, sim_cfg, policy="dedup", download=True):
    os.makedirs(data_dir, exist_ok=True)
    gws = [gw_observation(n, data_dir, download) for n in GW_SOURCES]
    psr = [dict(p) for p in PULSARS if not (policy == "dedup" and p.get("shared_with"))]
    cat = gws + [dict(n) for n in NICER] + psr
    for o in cat:
        o["x"], o["sigma"] = list(o["x"]), list(o["sigma"])
    cat, notes = clip_to_support(cat, sim_cfg)
    return cat, notes


def ood_report(catalog, train_samples, max_obs=300_000, lo_pct=0.5, hi_pct=99.5):
    """Percentile of each real feature (x and log sigma) within the training distribution."""
    pools = {}
    n = 0
    for s in train_samples:
        for o in s["catalog"]:
            pools.setdefault(o["type"], []).append(o["x"] + list(np.log(o["sigma"])))
            n += 1
        if n > max_obs:
            break
    pools = {k: np.array(v) for k, v in pools.items()}
    rows = []
    for o in catalog:
        feats = o["x"] + list(np.log(o["sigma"]))
        ref = pools[o["type"]]
        pct = [float(np.mean(ref[:, j] < v) * 100) for j, v in enumerate(feats)]
        rows.append({"name": o["name"], "percentiles": pct,
                     "ood": bool(any(p < lo_pct or p > hi_pct for p in pct))})
    return rows
