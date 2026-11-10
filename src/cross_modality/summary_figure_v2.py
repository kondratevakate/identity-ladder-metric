"""Fig. 2 v2 for the paper (Kate, 10 Nov): the summary split into two figures, values unchanged.

A (single column, 3.3 in): per metric, how far the stranger is beyond the true person (x) and how far the true person
shrunk toward the mean is (y), rank gaps over the ladder length; the shaded corner marks metrics that would do both.
B (full width, 7 in, low): per metric group a small panel with each rung's position on the stranger scale (target 0,
stranger 1); FID / FWD as ratios to the stranger set on a log axis.
Encoding: ladder rungs in cool colours, checks in warm colours, every rung also its own marker shape (the palette was
validated with the dataviz validator: lightness band, chroma, CVD and normal-vision separation pass; the contrast
warning is met by shape plus legend). The stranger is the neutral grey reference. Sibling is left out of B (it exists
for 2 of 9 sets, so its mean is not comparable); it stays in the tables.

Reads the values written by summary_figure.py (`<stem>_R.json`), so A and B plot exactly the tables' numbers; writes
`<out>_A.{png,pdf}`, `<out>_B.{png,pdf}` and `<out>_plotted.json` (the plotted subset, checked by a test).
Usage: python -I src/cross_modality/summary_figure_v2.py results/cross_modality/fig_metric_summary_R.json
       results/cross_modality/fig_metric_summary_v2
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
R = json.load(open(SRC, encoding="utf-8"))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8})

# (key, legend label, colour, marker, row 0 = ladder / 1 = checks)
RUNGS = [
    ("ref", "true person", "#2b5fb8", "o", 0),
    ("lookalike_oracle", "oracle look-alike*", "#38b2d8", "^", 0),
    ("lookalike16", "mean of 16 look-alikes*", "#6a4fd0", "h", 0),
    ("dataset_mean", "dataset mean", "#0f9a8a", "s", 0),
    ("stranger", "stranger", "#555555", "X", 0),
    ("shrink_ref", "shrunk true person", "#e8890c", "D", 1),
    ("noise", "white noise", "#c62828", "P", 1),
    ("wrong_detail", "wrong detail", "#e36fa8", "v", 1),
]
GROUPS = [("Error / similarity", "o", [("mse", "MSE"), ("ssim", "SSIM"), ("tm_like", "TM-like"), ("lddt", "lDDT")]),
          ("Deep features (LPIPS)", "^", [("lpips_alex", "AlexNet"), ("lpips_vgg", "VGG")]),
          ("Topology", "D", [("betti_matched", "Betti matching"), ("persistence", "Persistence")]),
          ("Set level", "P", [("fid", "FID"), ("fwd", "FWD")])]
MS = 5.2          # marker size in points (legend and plot use the same)
EDGE = 0.5
plotted = {"panel_A": {}, "panel_B": {}}

# ---------------- A ----------------
figA, ax = plt.subplots(figsize=(3.3, 3.45))
ax.axhline(0, color="#cccccc", lw=0.8)
ax.axvline(0, color="#cccccc", lw=0.8)
XLO, XHI, YLO, YHI = -0.08, 1.05, -0.68, 0.78
ax.add_patch(plt.Rectangle((0.4, 0.25), XHI - 0.4, YHI - 0.25, color="#1baf7a", alpha=0.10, lw=0))
ax.text(XHI - 0.02, YHI - 0.02, "both", ha="right", va="top", color="#226b4f", fontweight="bold")
ax.text(XLO + 0.02, YHI - 0.02, "rejects averaging,\nblind to the person", ha="left", va="top", color="#999999")
ax.text(XHI - 0.02, YLO + 0.02, "sees the person,\nrewards averaging", ha="right", va="bottom", color="#999999")
LAB = {"mse": (4, -3), "ssim": (4, -8), "tm_like": (4, 2), "lddt": (4, -4), "lpips_alex": (3, -10),
       "lpips_vgg": (-24, 4), "betti_matched": (-62, -3), "persistence": (-50, -3), "fid": (4, -5), "fwd": (4, -2)}
for _, mk, mets in GROUPS:
    for m, lab in mets:
        a = R["panel_A"][m]
        x, y = a["x_stranger_gap"], a["y_shrunk_gap"]
        plotted["panel_A"][m] = [x, y]
        ax.plot(x, y, marker=mk, ms=MS, color="#333333", mec="white", mew=EDGE, ls="", zorder=3)
        ax.annotate(lab, (x, y), xytext=LAB[m], textcoords="offset points", color="#222222")
ax.set_xlim(XLO, XHI)
ax.set_ylim(YLO, YHI)
ax.set_xlabel("sees the person →\n(stranger − true person, rank gap)")
ax.set_ylabel("rejects subtle averaging →\n(shrunk − true person, rank gap)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color="#f0f0f0", lw=0.6, zorder=0)
hA = [plt.Line2D([], [], marker=mk, ms=MS, color="#333333", mec="white", mew=EDGE, ls="") for _, mk, _ in GROUPS]
ax.legend(hA, ["error / similarity", "LPIPS", "topology", "set level"], loc="lower left",
          bbox_to_anchor=(0.02, 0.0), frameon=False, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
figA.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.17)
figA.savefig(OUT + "_A.png", dpi=300)
figA.savefig(OUT + "_A.pdf")

# ---------------- B ----------------
figB = plt.figure(figsize=(7.0, 3.2))
gs = figB.add_gridspec(2, 2, height_ratios=[4, 2.25], hspace=0.62, wspace=0.42, left=0.1, right=0.985,
                       top=0.93, bottom=0.27)
cells = {0: gs[0, 0], 1: gs[1, 0], 2: gs[0, 1], 3: gs[1, 1]}
STACK = 0.26


def strip_panel(axx, title, mets, log):
    x0, x1 = (np.log10(0.4), np.log10(300)) if log else (0.0, 2.0)
    tx = np.log10 if log else (lambda v: v)
    step = 0.07 if log else 0.045
    ys, labels = [], []
    for i, (m, lab) in enumerate(mets):
        y = -i
        p = R[m]
        axx.plot([x0, x1], [y, y], color="#eeeeee", lw=3, solid_capstyle="round", zorder=1)
        placed = []
        plotted["panel_B"][m] = {}
        for k, _, col, mk, _ in RUNGS:
            v = p.get(k)
            if v is None:
                continue
            plotted["panel_B"][m][k] = v
            xv = tx(v)
            over = xv > x1
            xv = min(xv, x1)
            dy = 0.0
            while any(abs(xv - q) < step and abs(dy - e) < 0.2 for q, e in placed):
                dy = -dy + STACK if dy <= 0 else -dy
            placed.append((xv, dy))
            axx.plot(xv, y + dy, marker=mk, ms=MS, color=col, mec="white", mew=EDGE, ls="", zorder=3)
            if over:
                axx.annotate(f"{v:.1f}→", (xv, y + dy), xytext=(4, -3), textcoords="offset points", color="#444444")
        ys.append(y)
        labels.append(f"{lab} ({p['n_datasets']})")
    axx.set_yticks(ys)
    axx.set_yticklabels(labels)
    axx.tick_params(axis="y", length=0, pad=3)
    axx.set_ylim(-len(mets) + 0.45, 0.55)
    axx.set_xlim(x0 - 0.04 * (x1 - x0), x1 + 0.13 * (x1 - x0))
    axx.axvline(0 if log else 1, color="#555555", lw=0.8, ls=(0, (3, 3)), zorder=0)
    if log:
        t = [0.5, 1, 2, 5, 10, 30, 100, 300]
        axx.set_xticks(np.log10(t))
        axx.set_xticklabels(["0.5", "1", "2", "5", "10", "30", "100", "300"])
    else:
        axx.set_xticks([0, 0.5, 1, 1.5, 2])
        axx.set_xticklabels(["0", "0.5", "1", "1.5", "2"])
    axx.set_title(title + (" (ratio to stranger set, log)" if log else ""), loc="left", fontweight="bold")
    for s in ("top", "right", "left"):
        axx.spines[s].set_visible(False)


for gi, (title, _, mets) in enumerate(GROUPS):
    strip_panel(figB.add_subplot(cells[gi]), title, mets, log=(title == "Set level"))
figB.text(0.1, 0.165, "0 = target, 1 = stranger", color="#555555")
for row, (name, yy) in enumerate([("Ladder:", 0.085), ("Checks:", 0.025)]):
    sel = [r for r in RUNGS if r[4] == row]
    hh = [plt.Line2D([], [], marker=mk, ms=MS, color=c, mec="white", mew=EDGE, ls="") for _, _, c, mk, _ in sel]
    figB.text(0.01, yy + 0.012, name, fontweight="bold", va="center")
    figB.legend(hh, [l for _, l, *_ in sel], loc="center left", bbox_to_anchor=(0.09, yy + 0.012), ncol=len(sel),
                frameon=False, handletextpad=0.15, columnspacing=0.9)
figB.savefig(OUT + "_B.png", dpi=300)
figB.savefig(OUT + "_B.pdf")
json.dump(plotted, open(OUT + "_plotted.json", "w", encoding="utf-8"), indent=1)
print("saved", OUT + "_A", OUT + "_B")
