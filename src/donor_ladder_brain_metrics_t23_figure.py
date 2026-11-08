"""Tier 2/3 metrics on the two frozen brain boxes (Amendment 20): where each donor sits between the repeat scan and a
stranger. Scale R: 0 = as close as the person's own repeat scan, 1 = as far as a random stranger (mean of 16).
One panel per metric (MS-SSIM, LPIPS, persistence distance); points = R, bars = 95% interval by bootstrap over
families; deep box and parietal box as two series. Values beyond the axis are drawn at the edge with their number.
A metric missing in a box (LPIPS where no central slice lies fully inside the brain) is written as missing, never
drawn at 0. Aggregates only (results/donor_ladder_brain_metrics_t23.json). Pre-QC, affine registration only.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "results", "donor_ladder_brain_metrics_t23.json")
OUT = os.path.join(ROOT, "results", "figures", "fig_brain_metrics_t23")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BOXES = [("deep/LR", "deep box", "#eb6834", "s"), ("parietal/LR", "parietal box", "#1baf7a", "^")]
RUNGS = [("repeat", "repeat scan"), ("mirror", "own other side"), ("mirror_block", "own other side, best fit"),
         ("lookalike_oracle", "best look-alike (knows answer)"), ("lookalike", "look-alike by context"),
         ("sibling", "sibling"), ("cotwin", "co-twin"), ("cohort_mean", "mean of same sex and age"),
         ("dataset_mean", "dataset mean"), ("mean_k2", "mean of 2 strangers"), ("stranger", "one stranger"),
         ("shrink_repeat", "repeat blended with mean"), ("wrong_detail", "mean + another's detail"),
         ("noise", "mean + white noise")]
METRICS = [("ms_ssim", "MS-SSIM"), ("lpips", "LPIPS (2.5D)"), ("persistence_w1", "persistence distance")]
XLIM = (-1.0, 3.5)


def points(view, metric):
    """[(row, R, lo, hi)] for the rungs present; None when the metric is missing in this view."""
    R = view["ladder_R"].get(metric)
    if not R:
        return None
    return [(k, R[f]["R"], *R[f]["ci95"]) for k, (f, _) in enumerate(RUNGS) if f in R]


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2), sharey=True, facecolor=SURFACE)
    for ax, (m, title) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        for x in (0, 1):
            ax.axvline(x, color=GRID, lw=1.2, zorder=0)
        ax.axhline(10.5, color=GRID, lw=0.8, ls=":")
        for j, (view, name, col, mk) in enumerate(BOXES):
            pts = points(d["views"][view], m)
            if pts is None:
                ax.text(XLIM[1] - 0.05, len(RUNGS) - 0.5 - j * 0.6, f"{name}: missing", ha="right", color=MUTED,
                        fontsize=8)
                continue
            dy = -0.15 + 0.3 * j
            for k, r, lo, hi in pts:
                y = k + dy
                ax.plot([max(lo, XLIM[0]), min(hi, XLIM[1])], [y, y], color=col, lw=2, solid_capstyle="round")
                rc = min(max(r, XLIM[0]), XLIM[1])
                ax.plot(rc, y, mk, color=col, ms=7, mec=SURFACE, mew=1.5, zorder=3)
                if r != rc:
                    ax.text(rc - 0.08, y, f"{r:.0f}", ha="right", va="center", fontsize=7, color=INK)
        miss = d["views"]["parietal/LR"]["missing"][m]
        n = d["views"]["parietal/LR"]["n_rows"]
        ax.set_title(title + (f"\nparietal: {miss / n:.0%} of rows missing" if miss else ""), fontsize=10, color=INK)
        ax.set_xlim(*XLIM)
        ax.set_xlabel("0 = repeat scan    1 = stranger", fontsize=9, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(range(len(RUNGS)), [lab for _, lab in RUNGS], fontsize=9)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], marker=mk, color=col, lw=2, ms=7, label=name) for _, name, col, mk in BOXES]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Brain, two fixed boxes, 221 targets: tier 2/3 metrics (pre-QC, affine registration only)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.01, "Below the dotted line: controls built from the truth. Bars: 95% interval, bootstrap over "
             "families. Numbers at the right edge: values beyond the axis; bars are cut at the axis edges.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=150, facecolor=SURFACE)
    print(f"saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
