"""Fig. 2 v4 (Kate, 10 Nov): one figure, A left and the stacked strips B right, compact labels, one legend.

A: per metric, stranger minus true person (x) and shrunk true person minus true person (y), rank gaps over the ladder
length, mean over image sets; shaded corner = both. B: groups stacked on one aligned linear x axis (target 0,
stranger 1; values beyond 2 at the edge with their number), set level (FID / FWD, ratio to the stranger set) as its
own log-axis panel at the bottom. Each rung has a fixed lane inside every metric row (no jitter), so no marker hides
another. Ladder rungs cool, checks warm, one shape per rung (palette validated with the dataviz validator); sibling
left out (2 of 9 sets). Formulas and caveats live in the caption, not on the axes.
Values come from summary_figure.py's `<stem>_R.json`, unchanged.

Usage: python -I src/cross_modality/summary_figure_v4.py results/cross_modality/fig_metric_summary_R.json
       results/cross_modality/fig_metric_summary_v4
Writes `<out>.{png,pdf}` and `<out>_plotted.json` (checked by a test).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
R = json.load(open(SRC, encoding="utf-8"))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.labelsize": 8})
GREY = "#8a8a8a"
RUNGS = [  # (key, legend label, colour, marker); lane order = this order, top to bottom
    ("ref", "true person", "#2b5fb8", "o"),
    ("lookalike_oracle", "oracle look-alike*", "#38b2d8", "^"),
    ("lookalike16", "mean of 16*", "#6a4fd0", "h"),
    ("dataset_mean", "dataset mean", "#0f9a8a", "s"),
    ("stranger", "stranger", "#555555", "X"),
    ("shrink_ref", "shrunk", "#e8890c", "D"),
    ("noise", "noise", "#c62828", "P"),
    ("wrong_detail", "wrong detail", "#e36fa8", "v"),
]
LANE = {k: 0.35 - i * 0.1 for i, (k, *_) in enumerate(RUNGS)}
PANELS = [("Error / similarity", [("mse", "MSE"), ("ssim", "SSIM"), ("tm_like", "TM-like"), ("lddt", "lDDT")], False),
          ("LPIPS", [("lpips_alex", "AlexNet"), ("lpips_vgg", "VGG")], False),
          ("Topology", [("betti_matched", "Betti"), ("persistence", "Persist.")], False),
          ("Set level (log)", [("fid", "FID"), ("fwd", "FWD")], True)]
GROUP_MARK = {"mse": "o", "ssim": "o", "tm_like": "o", "lddt": "o", "lpips_alex": "^", "lpips_vgg": "^",
              "betti_matched": "D", "persistence": "D", "fid": "P", "fwd": "P"}
A_LABEL = {"mse": "MSE", "ssim": "SSIM", "tm_like": "TM-like", "lddt": "lDDT", "lpips_alex": "AlexNet",
           "lpips_vgg": "VGG", "betti_matched": "Betti", "persistence": "Persistence", "fid": "FID", "fwd": "FWD"}
A_OFF = {"mse": (4, -4), "ssim": (4, -8), "tm_like": (4, 2), "lddt": (4, -4), "lpips_alex": (4, -8),
         "lpips_vgg": (-22, 3), "betti_matched": (-31, -3), "persistence": (-50, -3), "fid": (4, -4), "fwd": (4, -2)}
MS, EDGE = 4.4, 0.4
plotted = {"panel_A": {}, "panel_B": {}}

fig = plt.figure(figsize=(7.0, 4.85))
outer = fig.add_gridspec(1, 2, width_ratios=[1, 1.38], wspace=0.3, left=0.085, right=0.985, top=0.95, bottom=0.19)

# ---------------- A ----------------
ax = fig.add_subplot(outer[0])
ax.axhline(0, color="#cccccc", lw=0.7)
ax.axvline(0, color="#cccccc", lw=0.7)
XLO, XHI, YLO, YHI = -0.08, 1.0, -0.42, 0.72
ax.add_patch(plt.Rectangle((0.4, 0.25), XHI - 0.4, YHI - 0.25, color="#1baf7a", alpha=0.10, lw=0))
ax.text(XHI - 0.02, YHI - 0.02, "both", ha="right", va="top", color="#226b4f", fontweight="bold")
ax.text(XLO + 0.02, YHI - 0.02, "blind to\nthe person", ha="left", va="top", color=GREY, fontsize=7)
ax.text(XHI - 0.02, YLO + 0.02, "rewards\naveraging", ha="right", va="bottom", color=GREY, fontsize=7)
for m, mk in GROUP_MARK.items():
    a = R["panel_A"][m]
    x, y = a["x_stranger_gap"], a["y_shrunk_gap"]
    plotted["panel_A"][m] = [x, y]
    ax.plot(x, y, marker=mk, ms=MS, color="#333333", mec="white", mew=EDGE, ls="", zorder=3)
    ax.annotate(A_LABEL[m], (x, y), xytext=A_OFF[m], textcoords="offset points", color="#222222")
ax.set_xlim(XLO, XHI)
ax.set_ylim(YLO, YHI)
ax.set_xlabel("sees the person →")
ax.set_ylabel("rejects subtle averaging →")
ax.set_title("A", loc="left", fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color="#f0f0f0", lw=0.5, zorder=0)
hA = [plt.Line2D([], [], marker=mk, ms=MS, color="#333333", mec="white", mew=EDGE, ls="") for mk in "o^DP"]
ax.legend(hA, ["error / similarity", "LPIPS", "topology", "set level"], loc="lower left", frameon=False,
          handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2, fontsize=7)

# ---------------- B ----------------
inner = outer[1].subgridspec(5, 1, height_ratios=[4, 2, 2, 0.35, 2], hspace=0.42)
first = None
for pi, (title, mets, log) in enumerate(PANELS):
    bx = fig.add_subplot(inner[pi if pi < 3 else 4], sharex=first if (first is not None and not log) else None)
    if first is None:
        first = bx
    x0, x1 = (np.log10(0.4), np.log10(300)) if log else (0.0, 2.0)
    tx = np.log10 if log else (lambda v: v)
    ys, labels = [], []
    for i, (m, lab) in enumerate(mets):
        y = -i
        p = R[m]
        bx.add_patch(plt.Rectangle((x0, y - 0.43), x1 - x0, 0.86, color="#f3f3f3", lw=0, zorder=0))
        for k in LANE:
            bx.plot([x0, x1], [y + LANE[k]] * 2, color="#e6e6e6", lw=0.35, zorder=1)
        plotted["panel_B"][m] = {}
        for k, _, col, mk in RUNGS:
            v = p.get(k)
            if v is None:
                continue
            plotted["panel_B"][m][k] = v
            xv = tx(v)
            over = xv > x1
            xv = min(xv, x1)
            bx.plot(xv, y + LANE[k], marker=mk, ms=MS, color=col, mec="white", mew=EDGE, ls="", zorder=3)
            if over:
                bx.annotate(f"{v:.1f}→", (xv, y + LANE[k]), xytext=(3, -3), textcoords="offset points",
                            color="#444444", fontsize=7)
        ys.append(y)
        labels.append(f"{lab} ({p['n_datasets']})")
    bx.set_yticks(ys)
    bx.set_yticklabels(labels)
    bx.tick_params(axis="y", length=0, pad=2)
    bx.set_ylim(-len(mets) + 0.5, 0.5)
    bx.axvline(0 if log else 1, color="#555555", lw=0.7, ls=(0, (3, 3)), zorder=2)
    bx.set_xlim(x0 - 0.02 * (x1 - x0), x1 + 0.11 * (x1 - x0))
    if log:
        t = [0.5, 1, 2, 5, 10, 30, 100, 300]
        bx.set_xticks(np.log10(t))
        bx.set_xticklabels(["0.5", "1", "2", "5", "10", "30", "100", "300"])
        bx.set_xlabel("ratio to the stranger set", labelpad=1)
    else:
        bx.set_xticks([0, 0.5, 1, 1.5, 2])
        bx.set_xticklabels(["0", "0.5", "1", "1.5", "2"])
        bx.tick_params(axis="x", length=2.5, labelbottom=(pi == 2))
        if pi == 2:
            bx.set_xlabel("0 = target, 1 = stranger", labelpad=1)
    bx.set_title(("B   " if pi == 0 else "") + title, loc="left", fontweight="bold", pad=2)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)

hh = [plt.Line2D([], [], marker=mk, ms=MS + 0.4, color=c, mec="white", mew=EDGE, ls="") for _, _, c, mk in RUNGS]
for rows, yy in ((slice(0, 5), 0.045), (slice(5, 8), 0.0)):   # row 1 ladder rungs, row 2 checks
    fig.legend(hh[rows], [l for _, l, *_ in RUNGS[rows]], loc="lower center", bbox_to_anchor=(0.5, yy),
               ncol=len(RUNGS[rows]), frameon=False, handletextpad=0.1, columnspacing=1.0)
fig.savefig(OUT + ".png", dpi=300)
fig.savefig(OUT + ".pdf")
json.dump(plotted, open(OUT + "_plotted.json", "w", encoding="utf-8"), indent=1)
print("saved", OUT)
