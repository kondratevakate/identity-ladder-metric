"""Metric summary figure v7 (15 Nov, teaser vocabulary): v4 values and layout in the shared paper style (paper/cvpr2027/fig_style.py).

Same data and encodings as summary_figure_v4.py (kept unchanged); changes are display only: full text width, Arial
at printed size, shared rung labels, colors and markers, bold capital panel letters, a more compact height.
A: per metric, stranger minus true person (x) and softened true person minus true person (y), rank gaps over the
ladder length, mean over image sets; shaded corner = both. B: each fill on the stranger scale, fixed lanes.

Usage: python -I src/cross_modality/summary_figure_v7.py results/cross_modality/fig_metric_summary_R.json
       paper/cvpr2027/fig/metric_summary_v7
Writes `<out>.{png,pdf}` and `<out>_plotted.json`.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "cvpr2027"))
import fig_style as S  # noqa: E402

SRC, OUT = sys.argv[1], sys.argv[2]
R = json.load(open(SRC, encoding="utf-8"))
S.apply()
LANE = {k: 0.35 - i * 0.1 for i, k in enumerate(S.ORDER)}
PANELS = [("Error and similarity", [("mse", "MSE"), ("ssim", "SSIM"), ("tm_like", "TM-like"), ("lddt", "lDDT")], False),
          ("LPIPS", [("lpips_alex", "AlexNet"), ("lpips_vgg", "VGG")], False),
          ("Topology", [("betti_matched", "Betti"), ("persistence", "Persistence")], False),
          ("Set level", [("fid", "FID"), ("fwd", "FWD")], True)]
GROUP_MARK = {"mse": "o", "ssim": "o", "tm_like": "o", "lddt": "o", "lpips_alex": "^", "lpips_vgg": "^",
              "betti_matched": "D", "persistence": "D", "fid": "P", "fwd": "P"}
A_LABEL = {"mse": "MSE", "ssim": "SSIM", "tm_like": "TM-like", "lddt": "lDDT", "lpips_alex": "LPIPS AlexNet",
           "lpips_vgg": "LPIPS VGG", "betti_matched": "Betti", "persistence": "Persistence", "fid": "FID", "fwd": "FWD"}
A_OFF = {"mse": (5, -3), "ssim": (-27, -3), "tm_like": (5, -3), "lddt": (5, -3), "lpips_alex": (5, 1),
         "lpips_vgg": (-50, -3), "betti_matched": (-25, -3), "persistence": (5, -2), "fid": (5, -3), "fwd": (5, -3)}
plotted = {"panel_A": {}, "panel_B": {}}

fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 3.75))
outer = fig.add_gridspec(1, 2, width_ratios=[1, 1.45], wspace=0.33, left=0.075, right=0.985, top=0.93, bottom=0.19)

# ---------------- A ----------------
ax = fig.add_subplot(outer[0])
ax.axhline(0, color="#cccccc", lw=0.6)
ax.axvline(0, color="#cccccc", lw=0.6)
XLO, XHI, YLO, YHI = -0.08, 1.0, -0.36, 0.66
ax.add_patch(plt.Rectangle((0.4, 0.25), XHI - 0.4, YHI - 0.25, color="#1baf7a", alpha=0.10, lw=0))
ax.text(XHI - 0.02, YHI - 0.02, "both", ha="right", va="top", color="#226b4f", fontweight="bold")
ax.text(XLO + 0.02, YHI - 0.02, "blind to\nthe person", ha="left", va="top", color=S.MUTED)
ax.text(XHI - 0.02, YLO + 0.02, "rewards\naveraging", ha="right", va="bottom", color=S.MUTED)
for m, mk in GROUP_MARK.items():
    a = R["panel_A"][m]
    x, y = a["x_stranger_gap"], a["y_shrunk_gap"]
    plotted["panel_A"][m] = [x, y]
    ax.plot(x, y, marker=mk, ms=S.MS, color="#333333", mec="white", mew=S.EDGE, ls="", zorder=3)
    ax.annotate(A_LABEL[m], (x, y), xytext=A_OFF[m], textcoords="offset points", color=S.INK)
ax.set_xlim(XLO, XHI)
ax.set_ylim(YLO, YHI)
ax.set_xlabel("sees the person →", labelpad=1)
ax.set_ylabel("rejects subtle averaging →", labelpad=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color="#f0f0f0", lw=0.5, zorder=0)
hA = [plt.Line2D([], [], marker=mk, ms=S.MS, color="#333333", mec="white", mew=S.EDGE, ls="") for mk in "o^DP"]
ax.legend(hA, ["error and similarity", "LPIPS", "topology", "set level"], loc="lower left", frameon=False,
          handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
S.letter(fig, 0.005, 0.955, "A")

# ---------------- B ----------------
inner = outer[1].subgridspec(5, 1, height_ratios=[4, 2, 2, 0.45, 2], hspace=0.5)
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
        plotted["panel_B"][m] = {}
        for k in S.ORDER:
            v = p.get(k)
            if v is None:
                continue
            plotted["panel_B"][m][k] = v
            xv = tx(v)
            over = xv > x1
            xv = min(xv, x1)
            bx.plot(xv, y + LANE[k], marker=S.marker(k), ms=S.MS - 0.6, color=S.color(k), mec="white", mew=S.EDGE,
                    ls="", zorder=3)
            if over:
                bx.annotate(f"{v:.1f}", (xv, y + LANE[k]), xytext=(4, -3), textcoords="offset points",
                            color=S.MUTED, fontsize=S.SIZE_SMALL)
        ys.append(y)
        labels.append(f"{lab} ({p['n_datasets']})")
    bx.set_yticks(ys)
    bx.set_yticklabels(labels)
    bx.tick_params(axis="y", length=0, pad=2)
    bx.set_ylim(-len(mets) + 0.5, 0.5)
    bx.axvline(0 if log else 1, color="#555555", lw=0.6, ls=(0, (3, 3)), zorder=2)
    bx.set_xlim(x0 - 0.02 * (x1 - x0), x1 + 0.1 * (x1 - x0))
    if log:
        t = [0.5, 1, 2, 5, 10, 30, 100, 300]
        bx.set_xticks(np.log10(t))
        bx.set_xticklabels([str(v) for v in t])
        bx.set_xlabel("set distance / random-person set distance (log)", labelpad=1)
    else:
        bx.set_xticks([0, 0.5, 1, 1.5, 2])
        bx.set_xticklabels(["0", "0.5", "1", "1.5", "≥2"])
        bx.tick_params(axis="x", length=2.5, labelbottom=(pi == 2))
        if pi == 2:
            bx.set_xlabel("0 = target, 1 = random person", labelpad=1)
    bx.set_title(title, loc="left", fontweight="bold", pad=2)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
S.letter(fig, 0.405, 0.955, "B")

for keys, yy in ((S.LADDER, 0.045), (S.CONTROLS, 0.0)):
    fig.legend(S.handles(keys), [S.label(k) for k in keys], loc="lower center", bbox_to_anchor=(0.53, yy),
               ncol=len(keys), frameon=False, handletextpad=0.1, columnspacing=1.2)
fig.savefig(OUT + ".png", dpi=300)
fig.savefig(OUT + ".pdf")
json.dump(plotted, open(OUT + "_plotted.json", "w", encoding="utf-8"), indent=1)
print("saved", OUT)
