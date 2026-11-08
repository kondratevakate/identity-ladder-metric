"""Heatmap of metric agreement, grouped by metric family (results/donor_ladder_metrics_correlation.json). Pre-QC.

Left: Spearman rho over the ladder rungs (does a metric order the rungs the same way?). Right: rho over the 16
strangers of the same target (does it pick the same "worse wrong person"?). Mean over cells (target, cube, side),
Fisher z. Aggregates only.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import donor_ladder_metrics_blocks_figure as BF  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, MUTED, SURFACE = BF.INK, BF.MUTED, BF.SURFACE
FAMILIES = [("squared / absolute error", ["mse", "rmse", "rmse_rep", "psnr", "psnr_fixed", "mae"]),
            ("structure", ["ssim"]),
            ("correlation", ["pearson", "acc_dataset", "acc_cohort"]),
            ("protein: global", ["tm_like", "gdt_like"]),
            ("protein: local (lDDT)", ["lddt", "lddt_rep"])]
LABEL = {**BF.LABEL, "rmse_rep": "RMSE / repeat noise", "lddt_rep": "lDDT, repeat tol."}
CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#a9c8ee", "#2a78d6", "#0d3a75"])
VMIN = 0.4


def heat(ax, R, order, idx, title, n_cells, rungs_note):
    A = R[np.ix_([idx[m] for m in order], [idx[m] for m in order])]
    ax.imshow(np.clip(A, VMIN, 1), cmap=CMAP, vmin=VMIN, vmax=1, interpolation="nearest")
    for i in range(len(order)):
        for j in range(len(order)):
            v = A[i, j]
            dark = (v - VMIN) / (1 - VMIN) > 0.55
            ax.text(j, i, f"{v:.2f}".replace("1.00", "1"), ha="center", va="center", fontsize=5.6,
                    color="white" if dark else INK)
    k = 0
    for _, ms in FAMILIES[:-1]:
        k += len([m for m in ms if m in idx])
        ax.axhline(k - 0.5, color=SURFACE, lw=2.5)
        ax.axvline(k - 0.5, color=SURFACE, lw=2.5)
    labels = [LABEL.get(m, m) for m in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=6.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{title}\n{rungs_note}\n{n_cells:,} cells (target x cube x side)", fontsize=8, color=INK, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/donor_ladder_metrics_correlation.json")
    ap.add_argument("--out", default="results/figures/fig_metrics_correlation")
    a = ap.parse_args()
    d = json.load(open(os.path.join(ROOT, a.json)))
    idx = {m: i for i, m in enumerate(d["metrics"])}
    order = [m for _, ms in FAMILIES for m in ms if m in idx]
    missing = set(idx) - set(order)
    if missing:
        raise SystemExit(f"metrics without a family: {sorted(missing)}")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), facecolor=SURFACE)
    lad, strg = d["views"]["random_patches"], d["views"]["random_patches_strangers"]
    heat(axes[0], np.array(lad["rho"]), order, idx, "A. Do metrics order the ladder the same way?",
         lad["n_cells"], "Spearman over 8 rungs: mirror, look-alike, oracle look-alike,\n"
         "stranger, mean of 2, dataset mean, wrong detail, noise")
    heat(axes[1], np.array(strg["rho"]), order, idx, "B. Do metrics pick the same worse wrong person?",
         strg["n_cells"], "Spearman over the 16 strangers of each target\n"
         "(which unrelated person is the worse fill)")
    # family brackets on the left panel
    k = 0
    for name, ms in FAMILIES:
        n = len([m for m in ms if m in idx])
        axes[0].annotate(name, xy=(-0.5, k - 0.5 + n / 2), xytext=(-9.6, k - 0.5 + n / 2), fontsize=6.5,
                         color=MUTED, va="center", ha="left", annotation_clip=False)
        k += n
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(VMIN, 1))
    fig.subplots_adjust(left=0.235, right=0.9, bottom=0.2, top=0.82, wspace=0.42)
    cb = fig.colorbar(sm, cax=fig.add_axes([0.925, 0.3, 0.012, 0.45]))
    cb.set_label(f"mean Spearman rho (below {VMIN} drawn as {VMIN})", fontsize=7, color=MUTED)
    cb.ax.tick_params(labelsize=6.5, colors=MUTED)
    fig.text(0.01, 0.01, "QTIM, 32 random 16 mm cubes, both hemispheres, 221 targets. Each metric as a distance "
             "(similarities negated); mean of per-cell rho by Fisher z. " + d.get("status", ""),
             fontsize=6.5, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, f"{a.out}.{ext}"), dpi=200, facecolor=SURFACE)
    print(os.path.abspath(os.path.join(ROOT, f"{a.out}.png")))


if __name__ == "__main__":
    main()
