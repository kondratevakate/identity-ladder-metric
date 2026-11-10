"""Fig. 2 B v3 (Kate, 10 Nov): the strips stacked in one column so near and far read straight down.

Panels top to bottom: error / similarity, LPIPS, topology (one shared linear x axis, 0 to 2 strangers, x aligned
across panels; a value beyond 2 sits at the edge with its number), then set level (FID / FWD, log axis, its own
scale). Overlap rule: no jitter. Each rung has a fixed lane inside every metric row (constant vertical offset, in
legend order), so the same rung always sits at the same height and no marker hides another; read left-right for
distance. Colour and shape as v2 (ladder cool, checks warm, each rung its own shape; stranger the grey reference);
sibling left out (2 of 9 sets). Values come from summary_figure.py's `<stem>_R.json`, unchanged.

Usage: python -I src/cross_modality/summary_figure_v3b.py results/cross_modality/fig_metric_summary_R.json
       results/cross_modality/fig_metric_summary_v3
Writes `<out>_B.{png,pdf}` and `<out>_B_plotted.json` (checked by a test).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
R = json.load(open(SRC, encoding="utf-8"))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8})

RUNGS = [  # (key, legend label, colour, marker, row 0 ladder / 1 checks); lane order = this order, top to bottom
    ("ref", "true person", "#2b5fb8", "o", 0),
    ("lookalike_oracle", "oracle look-alike*", "#38b2d8", "^", 0),
    ("lookalike16", "mean of 16 look-alikes*", "#6a4fd0", "h", 0),
    ("dataset_mean", "dataset mean", "#0f9a8a", "s", 0),
    ("stranger", "stranger", "#555555", "X", 0),
    ("shrink_ref", "shrunk true person", "#e8890c", "D", 1),
    ("noise", "white noise", "#c62828", "P", 1),
    ("wrong_detail", "wrong detail", "#e36fa8", "v", 1),
]
LANE = {k: 0.35 - i * 0.1 for i, (k, *_) in enumerate(RUNGS)}   # +0.35 (true person) ... -0.35 (wrong detail)
PANELS = [("Error / similarity", [("mse", "MSE"), ("ssim", "SSIM"), ("tm_like", "TM-like"), ("lddt", "lDDT")], False),
          ("LPIPS", [("lpips_alex", "AlexNet"), ("lpips_vgg", "VGG")], False),
          ("Topology", [("betti_matched", "Betti matching"), ("persistence", "Persistence")], False),
          ("Set level: FID / FWD as ratio to the stranger set (log scale, a different axis)",
           [("fid", "FID"), ("fwd", "FWD")], True)]
MS, EDGE = 4.6, 0.4
plotted = {}

fig = plt.figure(figsize=(7.0, 5.4))
gs = fig.add_gridspec(5, 1, height_ratios=[4, 2, 2, 0.45, 2], hspace=0.5, left=0.18, right=0.95, top=0.96, bottom=0.135)
axes = []
for pi, (title, mets, log) in enumerate(PANELS):
    ax = fig.add_subplot(gs[pi if pi < 3 else 4], sharex=axes[0] if (axes and not log) else None)
    axes.append(ax)
    x0, x1 = (np.log10(0.4), np.log10(300)) if log else (0.0, 2.0)
    tx = np.log10 if log else (lambda v: v)
    ys, labels = [], []
    for i, (m, lab) in enumerate(mets):
        y = -i
        p = R[m]
        ax.add_patch(plt.Rectangle((x0, y - 0.43), x1 - x0, 0.86, color="#f3f3f3", lw=0, zorder=0))
        for k in LANE:   # thin lane guides
            ax.plot([x0, x1], [y + LANE[k]] * 2, color="#e4e4e4", lw=0.4, zorder=1)
        plotted[m] = {}
        for k, _, col, mk, _ in RUNGS:
            v = p.get(k)
            if v is None:
                continue
            plotted[m][k] = v
            xv = tx(v)
            over = xv > x1
            xv = min(xv, x1)
            ax.plot(xv, y + LANE[k], marker=mk, ms=MS, color=col, mec="white", mew=EDGE, ls="", zorder=3)
            if over:
                ax.annotate(f"{v:.1f}→", (xv, y + LANE[k]), xytext=(4, -3), textcoords="offset points",
                            color="#444444")
        ys.append(y)
        labels.append(f"{lab} ({p['n_datasets']})")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(-len(mets) + 0.5, 0.5)
    ax.axvline(0 if log else 1, color="#555555", lw=0.8, ls=(0, (3, 3)), zorder=2)
    ax.set_xlim(x0 - 0.02 * (x1 - x0), x1 + 0.09 * (x1 - x0))
    if log:
        t = [0.5, 1, 2, 5, 10, 30, 100, 300]
        ax.set_xticks(np.log10(t))
        ax.set_xticklabels(["0.5", "1\nstranger", "2", "5", "10", "30", "100", "300"])
    else:
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
        ax.set_xticklabels(["0\ntarget", "", "0.5", "", "1\nstranger", "", "1.5", "", "2"])
        ax.tick_params(axis="x", length=2 if pi < 2 else 3.5, labelbottom=(pi == 2))
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

for row, (name, yy) in enumerate([("Ladder:", 0.05), ("Checks:", 0.012)]):
    sel = [r for r in RUNGS if r[4] == row]
    hh = [plt.Line2D([], [], marker=mk, ms=MS + 0.6, color=c, mec="white", mew=EDGE, ls="") for _, _, c, mk, _ in sel]
    fig.text(0.01, yy, name, fontweight="bold", va="center")
    fig.legend(hh, [l for _, l, *_ in sel], loc="center left", bbox_to_anchor=(0.085, yy), ncol=len(sel),
               frameon=False, handletextpad=0.15, columnspacing=0.9)
fig.savefig(OUT + "_B.png", dpi=300)
fig.savefig(OUT + "_B.pdf")
json.dump(plotted, open(OUT + "_B_plotted.json", "w", encoding="utf-8"), indent=1)
print("saved", OUT + "_B")
