"""One-column RQ4 figure: how close context-selected look-alikes come, by hidden-region size in object units.

Source: results/rq4_window_size.json (R_self under MSE, 0 = target, 1 = average of 16 random people).
Lines average the image sets of one modality that have all sizes (faces: CelebA, IMDb-WIKI; retina: GRAPE, PAPILA;
iris: UTIRIS NIR; brain: QTIM, which has 1/16, 1/8 and 1/4 only). UTIRIS VIS, UPOL and IXI have no size sweep.
Usage: python -I paper/cvpr2027/make_fig_rq4_size.py   (run from the repository root)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as S  # noqa: E402

SRC = "results/rq4_window_size.json"
OUT = "paper/cvpr2027/fig/rq4_size"
GROUPS = [("Brain (QTIM)", ["QTIM"], "#3b6e8f"), ("Faces", ["CelebA", "IMDb-WIKI"], "#7c6d9c"),
          ("Retina", ["GRAPE", "PAPILA"], "#33877a"), ("Iris (UTIRIS NIR)", ["UTIRIS NIR"], "#b9853a")]
PANELS = [("lookalike", "One look-alike"), ("lookalike16", "Mean of 16 look-alikes")]
TARGET_F = [1 / 16, 1 / 8, 1 / 4]

R = json.load(open(SRC, encoding="utf-8"))["datasets"]


def by_size(ds, rung):
    """(f, value) for every window of one dataset that has this rung; missing sizes are skipped, never zero."""
    out = []
    for entry in R[ds].values():
        if isinstance(entry, dict) and "R_self_mse" in entry and entry["R_self_mse"].get(rung):
            out.append((entry["f"], entry["R_self_mse"][rung][0]))
    return sorted(out)


def nearest(points, f):
    """Value at the window closest to fraction f, if within 25 % of it."""
    best = min(points, key=lambda p: abs(p[0] - f), default=None)
    return best[1] if best and abs(best[0] - f) <= 0.25 * f else None


S.apply()
fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75), sharey=True)
plotted = {}
for ax, (rung, title) in zip(axes, PANELS):
    dm = [v for g in GROUPS for ds in g[1] for _, v in by_size(ds, "dataset_mean")]
    ax.axhspan(min(dm), max(dm), color=S.RULE, alpha=0.6, lw=0)
    ax.axvline(1 / 8, color=S.MUTED, lw=0.6, ls="--")
    for name, sets, col in GROUPS:
        xs, ys = [], []
        for f in TARGET_F:
            vals = [nearest(by_size(ds, rung), f) for ds in sets]
            if all(v is not None for v in vals):
                xs.append(f)
                ys.append(sum(vals) / len(vals))
        ax.plot(xs, ys, marker="o", ms=2.6, lw=1.0, color=col, label=name)
        plotted[f"{rung}|{name}"] = [[x, round(y, 4)] for x, y in zip(xs, ys)]
    ax.set_xscale("log", base=2)
    ax.set_xticks(TARGET_F)
    ax.set_xticklabels(["1/16", "1/8", "1/4"])
    ax.set_xlim(0.05, 0.3)
    ax.set_ylim(0, 0.8)
    ax.set_title(title, loc="left", fontsize=S.SIZE_TITLE, fontweight="bold")
    ax.set_xlabel("hidden region / object size")
axes[0].set_ylabel("MSE, random-person scale")
axes[1].text(0.29, 0.52, "dataset\naverage", fontsize=S.SIZE_SMALL, color=S.MUTED, ha="right", va="bottom")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
           fontsize=S.SIZE_SMALL, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.14, right=0.99, top=0.88, bottom=0.42, wspace=0.08)
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
json.dump({"source": SRC, "plotted": plotted}, open(OUT + "_plotted.json", "w", encoding="utf-8"), indent=1)
print("saved", OUT)
