"""Ladder view of the metric panel: x = rung (Kate's order, from the repeat to a stranger), y = raw metric value.

One panel per metric (small multiples), plus the reference panel MSE / V (all cube voxels, the ISBI unit: 0 = perfect,
1 = dataset mean, 2 = stranger). Series = region views of one dataset (random cubes, deep box, parietal box), both
hemispheres pooled; points = mean over targets, bars = 95% interval by bootstrap over families. Controls (built from
the truth) sit right of a gap. Each panel title reports how ladder-like the metric is:
  "mean beats mirror / sibling" = whether the dataset mean scores closer to the target than the person's own mirror
  side / the sibling (YES = the metric rewards averaging). Kendall tau over repeat > mirror > sibling > stranger is
  printed to the console (it is 1 for every tier-1 metric, so it does not separate them).
Aggregates only (results/donor_ladder_brain_metrics.json, Amendment 10). Pre-QC.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# first three categorical slots of the dataviz reference palette (validated all-pairs), plus marker shape
VIEWS = [("random_patches/LR", "32 random cubes, 16 mm", "#2a78d6", "o"),
         ("deep/LR", "deep box", "#eb6834", "s"),
         ("parietal/LR", "parietal box", "#1baf7a", "^")]
RUNGS = [("repeat", "repeat"), ("mirror", "own mirror"), ("lookalike16", "16 look-alikes"),
         ("lookalike16_oracle", "oracle\nlook-alikes"), ("sibling", "sibling"), ("cohort_mean", "sex-age\nmean"),
         ("dataset_mean", "dataset\nmean"), ("mean_k8", "mean of 8"), ("mean_k4", "mean of 4"),
         ("mean_k2", "mean of 2"), ("stranger", "stranger")]
CONTROLS = [("shrink_repeat", "repeat\nshrunk"), ("shrink_mirror", "mirror\nshrunk"),
            ("wrong_detail", "wrong\ndetail"), ("noise", "white\nnoise")]
LABEL = {"mse_ratio": "MSE / V (reference unit)", "mse": "MSE", "rmse": "RMSE", "mae": "MAE", "psnr": "PSNR (dB)",
         "psnr_fixed": "PSNR, fixed range (dB)", "ssim": "SSIM", "pearson": "Pearson r",
         "acc_dataset": "ACC vs dataset mean", "acc_cohort": "ACC vs sex-age mean", "lddt": "lDDT-like"}
RELATED = ["repeat", "mirror", "sibling", "stranger"]


def value(view, fill, m):
    """(mean, lo, hi) of metric m for a fill; MSE / V divides by the dataset mean's mse_allvox."""
    f = view["fills"].get(fill)
    if f is None:
        return None
    if m == "mse_ratio":
        r, ref = f["mse_allvox"]["raw"], view["fills"]["dataset_mean"]["mse_allvox"]["raw"]
        if r is None or ref is None:
            return None
        return r["mean"] / ref["mean"], r["ci95"][0] / ref["mean"], r["ci95"][1] / ref["mean"]
    r = f[m]["raw"]
    return None if r is None else (r["mean"], r["ci95"][0], r["ci95"][1])


def ladder_scores(view, m, lower_better):
    """tau over the relatedness order, and whether the dataset mean scores closer than the mirror / the sibling."""
    vals = [value(view, f, m) for f in RELATED]
    if any(v is None for v in vals):
        return None, None, None
    o = np.array([(-v[0] if lower_better else v[0]) for v in vals])
    tau = stats.kendalltau(o, -np.arange(len(o))).statistic
    mean = value(view, "dataset_mean", m)

    def beats(other):
        ov = value(view, other, m)
        if mean is None or ov is None:
            return None
        return (mean[0] < ov[0]) if lower_better else (mean[0] > ov[0])
    return tau, beats("mirror"), beats("sibling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/donor_ladder_brain_metrics.json")
    ap.add_argument("--name", default="QTIM")
    ap.add_argument("--out", default="results/figures/fig_metrics_ladder")
    a = ap.parse_args()
    d = json.load(open(os.path.join(ROOT, a.json)))
    lower = set(d["lower_is_better"]) | {"mse_ratio"}
    metrics = ["mse_ratio"] + d["metrics"]
    xs = list(range(len(RUNGS))) + [len(RUNGS) + 0.7 + k for k in range(len(CONTROLS))]
    labels = [lab for _, lab in RUNGS] + [lab for _, lab in CONTROLS]
    fills = [f for f, _ in RUNGS] + [f for f, _ in CONTROLS]
    ncol = 4
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.3 * nrow), facecolor=SURFACE)
    axes = axes.ravel()
    for p, m in enumerate(metrics):
        ax = axes[p]
        ax.set_facecolor(SURFACE)
        for vi, (vk, vlab, col, mk) in enumerate(VIEWS):
            view = d["views"].get(vk)
            if view is None:
                continue
            off = (vi - 1) * 0.18
            pts = [(x + off, value(view, f, m)) for x, f in zip(xs, fills)]
            pts = [(x, v) for x, v in pts if v is not None]
            lad = [(x, v) for x, v in pts if x < len(RUNGS)]
            ax.plot([x for x, _ in lad], [v[0] for _, v in lad], color=col, lw=1.0, alpha=0.55, zorder=2)
            for x, v in pts:
                ax.plot([x, x], [v[1], v[2]], color=col, lw=1.0, zorder=3)
                ax.scatter(x, v[0], s=22, marker=mk, color=col, edgecolor=SURFACE, linewidth=0.8, zorder=4,
                           label=vlab if (p == 0 and x == pts[0][0]) else None)
        if m == "mse_ratio":
            for h in (1, 2):
                ax.axhline(h, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.axvline(len(RUNGS) - 0.15, color=GRID, lw=0.8, zorder=1)
        tau, over_mirror, over_sib = ladder_scores(d["views"]["random_patches/LR"], m, m in lower)
        arrow = "lower = closer" if m in lower else "higher = closer"
        yn = lambda b: "n/a" if b is None else ("YES" if b else "no")  # noqa: E731
        sub = f"mean beats mirror: {yn(over_mirror)}, sibling: {yn(over_sib)}"
        ax.set_title(f"{LABEL.get(m, m)}\n{sub}  ({arrow})", fontsize=7.6, color=INK, loc="left")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=90, fontsize=5.8, color=MUTED)
        ax.tick_params(axis="y", labelsize=6.5, colors=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
            ax.spines[s].set_linewidth(0.6)
    for ax in axes[len(metrics):]:
        ax.set_axis_off()
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower right", bbox_to_anchor=(0.97, 0.06), fontsize=8, frameon=False,
               title=f"{a.name}, both hemispheres", title_fontsize=8)
    fig.text(0.5, 0.005, "Points: mean over targets; bars: 95% interval (bootstrap over families). Right of the line: "
             "controls built from the truth. 'Mean beats' compares the dataset mean with the mirror and the sibling "
             "on the random cubes (YES = the metric rewards averaging).\n" + d.get("status", ""),
             ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, f"{a.out}.{ext}"), dpi=200, facecolor=SURFACE)
    print(os.path.abspath(os.path.join(ROOT, f"{a.out}.png")))
    for m in metrics:
        print(m, ladder_scores(d["views"]["random_patches/LR"], m, m in lower))


if __name__ == "__main__":
    main()
