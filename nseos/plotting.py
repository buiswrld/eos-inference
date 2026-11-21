"""Figures (each saved as PNG + PDF)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

TEX = [r"$\gamma_0$", r"$\gamma_1$", r"$\gamma_2$", r"$\gamma_3$"]
C_SBI, C_REF = "#1f6fb4", "#d1495b"


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _hpd_levels(h, probs=(0.68, 0.95)):
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return sorted(flat[np.searchsorted(cum, p)] for p in probs)


def corner_overlay(sets, labels, colors, lo, hi, stem, weights=None, truth=None, title=None, bins=40):
    """Overlaid corner plot with 68/95% HPD contours (sets: list of (n, 4) arrays)."""
    weights = weights or [None] * len(sets)
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            a = ax[i, j]
            if j > i:
                a.axis("off")
                continue
            for s, w, c in zip(sets, weights, colors):
                if i == j:
                    a.hist(s[:, i], bins=bins, range=(lo[i], hi[i]), weights=w, density=True,
                           histtype="step", color=c, lw=1.6)
                else:
                    h, xe, ye = np.histogram2d(s[:, j], s[:, i], bins=bins, weights=w,
                                               range=[[lo[j], hi[j]], [lo[i], hi[i]]])
                    h = h.T
                    if h.sum() > 0:
                        lv = np.unique(_hpd_levels(h))       # 95% level first, then 68%
                        lv = lv[(lv > 0) & (lv < h.max())]
                        if len(lv):
                            a.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), h, levels=lv,
                                      colors=[c], linewidths=[0.9, 1.6][-len(lv):])
            if truth is not None:
                a.axvline(truth[j], color="k", lw=0.8, ls=":")
                if i != j:
                    a.axhline(truth[i], color="k", lw=0.8, ls=":")
            a.set_xlim(lo[j], hi[j])
            if i != j:
                a.set_ylim(lo[i], hi[i])
            else:
                a.set_yticks([])
            if i == 3:
                a.set_xlabel(TEX[j])
            else:
                a.set_xticklabels([])
            if j == 0 and i > 0:
                a.set_ylabel(TEX[i])
            elif j > 0:
                a.set_yticklabels([])
    for lab, c in zip(labels, colors):
        ax[0, 3].plot([], [], color=c, lw=2, label=lab)
    ax[0, 3].legend(loc="center", frameon=False)
    if title:
        fig.suptitle(title, y=0.995)
    fig.subplots_adjust(wspace=0.06, hspace=0.06)
    save(fig, stem)


def credible_band(curves, grid, interpolate=np.interp):
    """curves: list of (Ms, Y) stable branches -> 5/50/95% of Y at fixed M (NaN above M_max)."""
    Y = np.full((len(curves), len(grid)), np.nan)
    for i, (Ms, y) in enumerate(curves):
        ok = (grid >= Ms[0]) & (grid <= Ms[-1])
        Y[i, ok] = interpolate(grid[ok], Ms, y)
    q = np.full((3, len(grid)), np.nan)
    if len(curves):
        supported = np.mean(np.isfinite(Y), axis=0) >= 0.5
        if supported.any():
            q[:, supported] = np.nanpercentile(Y[:, supported], [5, 50, 95], axis=0)     # only where at least half of the EOS support that mass
    return q


def mr_lambda_figure(band_sets, labels, colors, nicer, pulsars, gws, stem, m_grid):
    """Fig. 1 replacement: (a) M-R 90% bands, NICER 1/2-sigma ellipses, pulsar bands;
    (b) M-Lambda 90% bands with GW Lambda-tilde constraints at the equal-mass component mass."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2))
    for (qR, qL), lab, c in zip(band_sets, labels, colors):
        a1.fill_betweenx(m_grid, qR[0], qR[2], color=c, alpha=0.25, lw=0)
        a1.plot(qR[1], m_grid, color=c, lw=1.5, label=f"{lab} (median, 90% band)")
        a2.fill_between(m_grid, qL[0], qL[2], color=c, alpha=0.25, lw=0)
        a2.plot(m_grid, qL[1], color=c, lw=1.5, label=lab)
    for o in nicer:
        (M, R), (sM, sR) = o["x"], o["sigma"]
        for k, ls in ((1, "-"), (2, "--")):
            a1.add_patch(Ellipse((R, M), 2 * k * sR, 2 * k * sM, fill=False, ec="0.2", lw=1, ls=ls))
        a1.annotate(o["name"].split(" (")[0], (R + sR * 1.05, M), fontsize=7, color="0.2")
    for o in pulsars:
        M, sM = o["x"][0], o["sigma"][0]
        a1.axhspan(M - sM, M + sM, color="orange", alpha=0.12, lw=0)
    for o in gws:
        m_eq = o["x"][1] * 2 ** 0.2      # Lambda-tilde = Lambda(m) exactly for m1 = m2 = Mc 2^(1/5)
        a2.errorbar([m_eq], [o["x"][0]], yerr=[[min(o["sigma"][0], o["x"][0] - 1)], [o["sigma"][0]]],
                    fmt="s", color="0.2", ms=5, capsize=3)
        a2.annotate(o["name"], (m_eq + 0.02, o["x"][0]), fontsize=7)
    a1.set(xlabel="Radius [km]", ylabel=r"Mass [$M_\odot$]", xlim=(8, 16), ylim=(1.0, 2.6))
    a2.set(xlabel=r"Mass [$M_\odot$]", ylabel=r"$\Lambda$", yscale="log", xlim=(1.0, 2.2), ylim=(5, 5000))
    a1.plot([], [], color="0.2", lw=1, label=r"NICER $1\sigma$ / $2\sigma$ (Gaussian summary)")
    a1.fill_between([], [], color="orange", alpha=0.3, label=r"pulsar mass $\pm1\sigma$")
    a1.legend(fontsize=8, loc="upper right", title="Bands conditional on reaching each mass", title_fontsize=7)
    a2.errorbar([], [], yerr=[], fmt="s", color="0.2", label=r"GW $\tilde\Lambda$ at $m=\mathcal{M}2^{1/5}$")
    a2.legend(fontsize=8)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    save(fig, stem)


def true_vs_pred(summ, prior, stem, n_obs=None, max_points=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(summ["theta"]), size=min(max_points, len(summ["theta"])), replace=False)
    th, q = summ["theta"][idx], summ["q"][idx]
    c = None if n_obs is None else np.asarray(n_obs)[idx]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
    for k in range(4):
        a = ax[k]
        lims = [min(th[:, k].min(), q[:, 0, k].min()), max(th[:, k].max(), q[:, 4, k].max())]
        a.axhspan(prior["median"][k] - prior["std"][k], prior["median"][k] + prior["std"][k], color="0.85",
                  zorder=0, label=r"prior median $\pm1\sigma$")
        a.errorbar(th[:, k], q[:, 2, k], yerr=[q[:, 2, k] - q[:, 1, k], q[:, 3, k] - q[:, 2, k]], fmt="none",
                   ecolor="0.6", elinewidth=0.4, alpha=0.5, zorder=1)
        sc = a.scatter(th[:, k], q[:, 2, k], c=c, s=6, cmap="viridis", zorder=2)
        a.plot(lims, lims, "r--", lw=1)
        a.set(xlabel=f"true {TEX[k]}", ylabel=f"posterior median {TEX[k]}", xlim=lims, ylim=lims)
        if k == 0:
            a.legend(fontsize=8, loc="upper left")
    if c is not None:
        fig.colorbar(sc, ax=ax, label="catalog size N", fraction=0.02)
    save(fig, stem)


def calibration_figure(ranks, n_samples, stem, sbc_hist, sbc_band, levels, cov, se, tarp_res=None):
    ncol = 5 if tarp_res is not None else 4
    fig, ax = plt.subplots(2, ncol, figsize=(4 * ncol, 7.5))
    for k in range(4):
        a = ax[0, k]
        nb = sbc_hist.shape[1]
        a.bar(np.arange(nb), sbc_hist[k], width=1.0, color="#7aa6c2", edgecolor="w")
        a.axhspan(*sbc_band, color="0.8", alpha=0.6, zorder=0)
        a.set(title=f"SBC ranks {TEX[k]}", xticks=[])
        b = ax[1, k]
        b.fill_between(levels, levels - 1.96 * se[:, 0], levels + 1.96 * se[:, 0], color="0.85", label="95% pointwise normal band")
        if k == 0:
            b.legend(fontsize=7)
        b.plot(levels, cov[:, k], color=C_SBI, lw=2)
        b.plot([0, 1], [0, 1], "k--", lw=1)
        b.set(xlabel="credible level", ylabel="empirical coverage", aspect="equal", xlim=(0, 1), ylim=(0, 1))
    if tarp_res is not None:
        alpha, ecp, band = tarp_res
        ax[0, 4].axis("off")
        b = ax[1, 4]
        b.fill_between(alpha, band[0], band[1], color="0.85")
        b.plot(alpha, ecp, color=C_SBI, lw=2)
        b.plot([0, 1], [0, 1], "k--", lw=1)
        b.set(title="TARP (joint)", xlabel="credibility", ylabel="expected coverage", aspect="equal")
    save(fig, stem)


def bar_by_n(bins, values_by_param, ylabel, stem, ref_line=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bins))
    for k, v in enumerate(values_by_param):
        ax.plot(x, v, "o-", label=TEX[k])
    if ref_line is not None:
        ax.axhline(ref_line, color="k", ls="--", lw=1)
    ax.set(xticks=x, xticklabels=[f"{a}-{b}" for a, b in bins], xlabel="catalog size N", ylabel=ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, stem)
