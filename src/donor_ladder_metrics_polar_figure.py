"""Polar heatmap of the metric panel on the donor ladder: angle = metric, ring = rung, colour = calibrated c.

c = P(a stranger scores worse) + P(tie) / 2 against 16 strangers; 0.5 is the stranger level (grey), 1 = closer to the
target than every stranger (blue), 0 = worse than every stranger (red). One panel per dataset view. Reads aggregates
only (results/donor_ladder_brain_metrics.json, Amendment 10); writes results/figures/fig_metrics_polar.{png,pdf}.
Usage: python src/donor_ladder_metrics_polar_figure.py [--json results/x.json:Dataset ...]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
# diverging red <-> blue around a grey midpoint (dataviz reference palette poles)
CMAP = LinearSegmentedColormap.from_list("c", ["#9f2a2a", "#e34948", "#f0efec", "#3987e5", "#104281"])
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
METRIC_LABEL = {"mse": "MSE", "rmse": "RMSE", "mae": "MAE", "psnr": "PSNR", "psnr_fixed": "PSNR\nfixed",
                "ssim": "SSIM", "pearson": "Pearson r", "acc_dataset": "ACC\ndataset", "acc_cohort": "ACC\ncohort",
                "lddt": "lDDT-like"}
# inner -> outer; None leaves an empty ring between the ladder and the controls
RINGS = [("repeat", "repeat"), ("mirror", "mirror"), ("sibling", "sibling"), ("cohort_mean", "sex-age mean"),
         ("dataset_mean", "dataset mean"), ("lookalike16", "16 look-alikes"),
         ("lookalike16_oracle", "oracle look-alikes"), (None, ""), ("shrink_mirror", "mirror shrunk"),
         ("wrong_detail", "wrong detail"), ("noise", "white noise")]
HOLE = 3.2          # inner radius (rings start here)
GAP_DEG = 56        # empty wedge at the top that carries the ring labels


def _theta(j, n):
    """Centre angle of metric sector j, clockwise from the top, after the label wedge."""
    width = (2 * np.pi - np.deg2rad(GAP_DEG)) / n
    return np.deg2rad(GAP_DEG) / 2 + (j + 0.5) * width, width


def panel(ax, view, metrics, title):
    n = len(metrics)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for k, (fill, _) in enumerate(RINGS):
        if fill is None:
            continue
        r0 = HOLE + k
        for j, m in enumerate(metrics):
            th, width = _theta(j, n)
            entry = view["fills"].get(fill, {}).get(m, {})
            c = entry.get("c") if entry else None
            if c is None:
                ax.bar(th, 0.94, width=width * 0.97, bottom=r0, color=SURFACE, edgecolor="#c3c2b7",
                       hatch="////", linewidth=0.4)
                ax.text(th, r0 + 0.47, "n/a", ha="center", va="center", fontsize=5.2, color=MUTED)
                continue
            v = c["mean"]
            ax.bar(th, 0.94, width=width * 0.97, bottom=r0, color=CMAP(v), edgecolor=SURFACE, linewidth=0.6)
            ax.text(th, r0 + 0.47, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if abs(v - 0.5) > 0.33 else INK)
    rmax = HOLE + len(RINGS)
    for j, m in enumerate(metrics):
        th, _ = _theta(j, n)
        ax.text(th, rmax + 1.75, METRIC_LABEL.get(m, m), ha="center", va="center", fontsize=7, color=INK)
    for k, (fill, lab) in enumerate(RINGS):
        if fill is None:
            continue
        n_t = view["fills"].get(fill, {}).get("mse", {}).get("raw") or {}
        ax.text(0.0, HOLE + k + 0.47, f"{lab} ({n_t.get('n_targets', '-')})", ha="center", va="center",
                fontsize=5.6, color=MUTED)
    ax.set_ylim(0, rmax + 2.3)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, color=INK, pad=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", default=["results/donor_ladder_brain_metrics.json:QTIM"],
                    help="path:Dataset pairs; every view ending in /LR becomes a panel")
    ap.add_argument("--out", default="results/figures/fig_metrics_polar")
    a = ap.parse_args()
    panels = []
    for spec in a.json:
        path, name = spec.rsplit(":", 1)
        d = json.load(open(os.path.join(ROOT, path) if not os.path.isabs(path) else path))
        for view in ("random_patches/LR", "deep/LR", "parietal/LR"):
            if view in d["views"]:
                region = {"random_patches/LR": "32 random cubes, 16 mm", "deep/LR": "deep box",
                          "parietal/LR": "parietal box"}[view]
                panels.append((d["views"][view], d["metrics"], f"{name}: {region}", d.get("status", "")))
    fig = plt.figure(figsize=(6.4 * len(panels), 7.4), facecolor=SURFACE)
    for k, (view, metrics, title, _) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), k + 1, projection="polar", facecolor=SURFACE)
        panel(ax, view, metrics, title)
    cax = fig.add_axes([0.35, 0.09, 0.3, 0.02])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.ax.tick_params(labelsize=6.5, colors=MUTED)
    cb.outline.set_visible(False)
    cb.set_label("c = share of 16 strangers the fill beats (0.5 = stranger level; 1 = beats every stranger)",
                 fontsize=7, color=INK)
    status = panels[0][3] if panels else ""
    fig.text(0.5, 0.0, f"Ring labels: rung (targets). Both hemispheres pooled. {status}", ha="center",
             fontsize=6, color=MUTED)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, f"{a.out}.{ext}"), dpi=220, facecolor=SURFACE, bbox_inches="tight")
    print(os.path.abspath(os.path.join(ROOT, f"{a.out}.png")))


if __name__ == "__main__":
    main()
