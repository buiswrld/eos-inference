"""Paper figures for the ML4PS camera-ready version, built from a finished pipeline run.

    python paper/make_paper_figures.py --root ../run_v3 --out paper/ml4ps_camera_ready/figures
"""
import argparse
import glob
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde

LO = np.array([0.2, -1.6, -0.6, -0.02])
HI = np.array([2.0, 1.7, 0.6, 0.02])
C_SBI, C_REF = "#1f6fb4", "#d1495b"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7, "font.family": "serif"})


def _one(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[-1]


def _resample(x, w, n, seed=0):
    p = w / w.sum()
    return x[np.random.default_rng(seed).choice(len(x), size=n, p=p)]


def _levels(z, fracs=(0.9, 0.5)):
    """Density thresholds enclosing the given probability mass."""
    flat = np.sort(z.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return sorted(flat[np.searchsorted(cum, f)] for f in fracs)


def fig_real_posterior(root, out):
    sbi = np.load(_one(f"{root}/data/real_sbi_dedup_*.npy"))
    refs = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{root}/reference/dynesty_real_dedup_s*.pkl"))]
    ref = np.concatenate([_resample(r["samples"], r["weights"], 6000, seed=i) for i, r in enumerate(refs)])
    sbi = sbi[np.random.default_rng(1).choice(len(sbi), size=min(len(sbi), 12000), replace=False)]

    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.75), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    # joint gamma0-gamma1 with KDE contours (50 / 90 %)
    gx, gy = np.mgrid[0.4:1.9:160j, -1.6:1.7:160j]
    pts = np.vstack([gx.ravel(), gy.ravel()])
    for s, c, lab in ((ref, C_REF, "dynesty (same likelihood)"), (sbi, C_SBI, "amortized (this work)")):
        z = gaussian_kde(s[:, :2].T, bw_method=0.18)(pts).reshape(gx.shape)
        ax[0].contour(gx, gy, z, levels=_levels(z), colors=c, linewidths=[0.9, 1.4])
        ax[0].plot([], [], color=c, label=lab)
    ax[0].set(xlabel=r"$\gamma_0$", ylabel=r"$\gamma_1$", xlim=(0.5, 1.8), ylim=(-1.6, 1.7))
    ax[0].legend(loc="lower left", frameon=False, handlelength=1.2)
    for k, a in zip((0, 1, 2), ax[1:]):
        bins = np.linspace(LO[k], HI[k], 31)
        a.hist(ref[:, k], bins=bins, density=True, histtype="step", color=C_REF, lw=1.3)
        a.hist(sbi[:, k], bins=bins, density=True, histtype="step", color=C_SBI, lw=1.3)
        a.set(xlabel=fr"$\gamma_{k}$", yticks=[], xlim=(LO[k], HI[k]))
    ax[3].axhline(1 / (HI[2] - LO[2]), color="0.5", ls=":", lw=1)
    ax[3].text(0.03, 0.92, "box prior", color="0.4", transform=ax[3].transAxes, fontsize=7)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(f"{out}/real_posterior.pdf")
    plt.close(fig)


def crop_calibration(root, out):
    """Keep the coverage + TARP row of the calibration figure (the SBC histograms are summarised in text)."""
    im = Image.open(_one(f"{root}/figures/calibration_main_*.png"))
    w, h = im.size
    im.crop((0, int(0.505 * h), w, h)).save(f"{out}/coverage_tarp.png")


def fig_calibration_scaling(root, out):
    """(a) coverage error at four credible levels with the binomial band; (b) information vs catalog size."""
    import json
    ev = json.load(open(_one(f"{root}/results/eval_main_*.json")))
    levels = [0.50, 0.68, 0.90, 0.95]
    cov = np.array([ev["coverage_at"][f"{int(l * 100)}%"] for l in levels])      # (level, param)
    n_test = 15000
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 1.9), gridspec_kw={"width_ratios": [1, 1.15]})
    band = 2 * np.sqrt(np.array(levels) * (1 - np.array(levels)) / n_test) * 100
    ax[0].fill_between(np.arange(4), -band, band, color="0.85", label=r"$\pm2\sigma$ binomial ($n{=}15{,}000$)")
    ax[0].axhline(0, color="0.3", lw=0.8)
    cols = ["#1f6fb4", "#2a9d8f", "#e9a23b", "#8d6a9f"]
    for k in range(4):
        ax[0].plot(np.arange(4) + (k - 1.5) * 0.08, (cov[:, k] - levels) * 100, "o-", ms=3, lw=1,
                   color=cols[k], label=fr"$\gamma_{k}$")
    ax[0].set(xticks=range(4), xticklabels=["50%", "68%", "90%", "95%"], ylabel="coverage error [%]",
              xlabel="credible level", ylim=(-1.5, 1.1))
    ax[0].legend(ncol=3, frameon=False, loc="upper center", handlelength=1.2, columnspacing=0.8, fontsize=6.5)
    nn = ev["nested_N"]
    Ns = sorted(int(k) for k in nn)
    for k, c in zip((0, 1), cols):
        ax[1].plot(Ns, [nn[str(n)][f"gamma_{k}"]["mae_ratio_to_prior"] for n in Ns], "o-", ms=3, color=c,
                   label=fr"$\gamma_{k}$")
    for k, c in zip((2, 3), cols[2:]):
        ax[1].plot(Ns, [nn[str(n)][f"gamma_{k}"]["mae_ratio_to_prior"] for n in Ns], "o--", ms=2.5, lw=0.9,
                   color=c, label=fr"$\gamma_{k}$")
    ax[1].axhline(1, color="0.5", ls=":", lw=1)
    ax[1].set(xscale="log", xlabel="observations in catalog $N$ (nested subsets)", ylabel="MAE / prior MAE",
              ylim=(0.2, 1.08), xticks=Ns, xticklabels=[str(n) for n in Ns])
    ax[1].legend(ncol=4, frameon=False, loc="lower left", handlelength=1.4, columnspacing=0.8)
    ax[0].set_title("(a) calibration, 15,000 test catalogs", fontsize=8)
    ax[1].set_title("(b) information gain with catalog size", fontsize=8)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(f"{out}/calibration_scaling.pdf")
    plt.close(fig)


def copy_figs(root, out):
    for name in ("fig1_mr_mlambda",):
        Image.open(_one(f"{root}/figures/{name}_*.png")).save(f"{out}/{name}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fig_real_posterior(a.root, a.out)
    fig_calibration_scaling(a.root, a.out)
    copy_figs(a.root, a.out)
    print("figures written to", a.out)
