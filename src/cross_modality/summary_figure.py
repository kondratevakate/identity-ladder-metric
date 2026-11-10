"""Two views of the same rank tables, side by side.

Left: one point per metric. x = how far the metric puts a stranger beyond the true person; y = how far it puts the
dataset mean (crude averaging) beyond the true person. Both are gaps in the metric's own ordering of the ladder,
divided by the ladder length, mean over datasets.
Right: one strip per metric with the positions of the true person, the true person shrunk toward the mean (control
T2), the dataset mean and a stranger in that ordering (0 = closest to the target, 1 = furthest).
Source: committed rank tables results/cross_modality/ranks_all.json and ranks_generative.json (aggregates only);
panel B values from common_all.json and values_generative.json, written next to the figure as <stem>_R.json.
Written by the ISBI paper session (Kate saw this version on 9 Nov and asked for FID/FWD on a log axis, the other
metrics linear, and the two-row legend); owned by the cross-modality scripts from 9 Nov.
The IXI-only scanner-site stranger rungs are not drawn here: one dataset, so they would read as a cross-dataset mean.

Usage: python -I src/cross_modality/summary_figure.py results/cross_modality results/cross_modality/fig_metric_summary
       python -I src/cross_modality/summary_figure.py results/cross_modality results/cross_modality/fig_metric_summary_paper --paper
--paper: the paper variant (Fig. 2): 7 in wide, no suptitle and no footer (the LaTeX caption carries them), every text
at least 8 pt, shorter panel titles, legend across the bottom. Same values; the _R.json is written by both.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = "--paper" in sys.argv
CM, OUT = [a for a in sys.argv[1:] if a != "--paper"][:2]


def F(size):
    """Font size: as designed for the 17 in default; at least 8 pt (titles 8.5) in the 7 in paper variant."""
    return size if not PAPER else (8.5 if size >= 11 else 8)


def S(area):
    """Marker area (points^2) scaled to the figure width."""
    return area if not PAPER else area * 0.32
ranks = {**json.load(open(os.path.join(CM, "ranks_all.json"))),
         **json.load(open(os.path.join(CM, "ranks_generative.json")))}

GROUPS = [  # (heading, marker for the left panel, [(metric key, label)])
    ("Per-window error and similarity\n(BraTS; protein-style scores)", "o",
     [("mse", "MSE"), ("ssim", "SSIM"), ("tm_like", "TM-like"), ("lddt", "lDDT")]),
    ("Deep features (perceptual)", "^", [("lpips_alex", "LPIPS AlexNet"), ("lpips_vgg", "LPIPS VGG")]),
    ("Topology", "D", [("betti_matched", "Betti matching"), ("persistence", "Persistence")]),
    ("Whole set compared with whole set", "P", [("fid", "FID"), ("fwd", "FWD")]),
]
RUNGS = [  # (key, label, colour, marker, legend row); palette slots validated; stranger is the neutral reference
    ("ref", "true person (test-retest)", "#2a78d6", "o", 0),
    ("lookalike_oracle", "oracle look-alike*", "#008300", "^", 0),
    ("relative", "sibling (QTIM, IMDb-WIKI)", "#eda100", "p", 0),
    ("dataset_mean", "dataset mean", "#eb6834", "s", 0),
    ("stranger", "stranger", "#555555", "X", 0),
    ("shrink_ref", "true person shrunk to the mean", "#1baf7a", "D", 1),
    ("lookalike16", "mean of k look-alikes* (not faces)", "#6250d6", "h", 1),
    ("wrong_detail", "wrong detail", "#e87ba4", "v", 1),
    ("noise", "white noise", "#e34948", "P", 1),
]


def positions(metric):
    pos = {k: [] for k, *_ in RUNGS}
    nds = 0
    for key, v in ranks.items():
        m, _ = key.split("|")
        if m != metric:
            continue
        ref = "repeat" if "repeat" in v else ("other_side" if "other_side" in v else None)
        if ref is None:
            continue
        n = max(v.values())
        nds += 1
        for k, *_ in RUNGS:
            src = ref if k == "ref" else k
            if src in v:  # a rung missing on a dataset is left out, never filled
                pos[k].append((v[src] - 1) / (n - 1))
    return {k: (float(np.mean(p)) if p else np.nan) for k, p in pos.items()}, nds


data = {m: positions(m) for _, _, mets in GROUPS for m, _ in mets}

# ---- real scale for panel B: R_self (0 = the target itself, 1 = the 16 strangers); FID/FWD: ratio to stranger set
common = json.load(open(os.path.join(CM, "common_all.json")))["datasets"]
gen = json.load(open(os.path.join(CM, "values_generative.json")))["values"]
FLOOR = 0.2  # axis start; a value below would be drawn at the floor with an open marker (none at present)


def r_values(metric):
    """Mean over self-anchored datasets of each rung's R (or set ratio). Brain generative rows are repeat-anchored
    (not on the self scale) and are left out; common_all rows are self-anchored for every dataset."""
    vals = {k: [] for k, *_ in RUNGS}
    used = []
    srcs = []
    for key, blk in gen.items():
        m, d = key.split("|")
        if m == metric and blk.get("anchor") == "self":
            srcs.append((d, {r: (x.get("R", x.get("ratio_to_stranger"))) for r, x in blk["rungs"].items()}))
    alt = []  # common_all is self-anchored for every dataset; use it when it covers more datasets (persistence)
    for d, blk in common.items():
        R = blk.get("R_self", {})
        row = {r: R[r][metric][0] for r in R if isinstance(R[r], dict) and metric in R[r]}
        if row:
            alt.append((d, row))
    if len(alt) > len(srcs):
        srcs = alt
    for d, row in srcs:
        ref = "repeat" if "repeat" in row else ("other_side" if "other_side" in row else None)
        if ref is None:
            continue
        used.append(d)
        for k, *_ in RUNGS:
            src = ref if k == "ref" else k
            if k == "stranger":
                vals[k].append(1.0)  # by definition of the scale
            elif src in row and row[src] is not None:
                vals[k].append(float(row[src]))
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in vals.items()}, len(used)


rdata = {m: r_values(m) for _, _, mets in GROUPS for m, _ in mets}
json.dump({m: {"n_datasets": n, **{k: (None if np.isnan(x) else round(x, 3)) for k, x in p.items()}}
           for m, (p, n) in rdata.items()}, open(OUT + "_R.json", "w"), indent=1)

plt.rcParams.update({"font.size": F(10.5)})
fig = plt.figure(figsize=(17, 10.5) if not PAPER else (7.0, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3] if not PAPER else [1, 1.12], wspace=0.36 if not PAPER else 0.5)

# ---- left: crude averaging map ----
ax = fig.add_subplot(gs[0])
ax.axhline(0, color="#ccc", lw=1)
ax.axvline(0, color="#ccc", lw=1)
ax.add_patch(plt.Rectangle((0, 0), 1.05, 1.05, color="#1baf7a", alpha=0.07, lw=0))
ax.text(1.02, 1.02, "wanted corner:\nsees the person\nand rejects the mean" if not PAPER else "wanted corner", ha="right",
        va="top", fontsize=F(9), color="#226b4f")
if not PAPER:
    ax.text(-0.13, 0.72, "rejects the mean\nbut blind to the person", ha="left", va="top", fontsize=F(9), color="#777")
    ax.text(1.02, -0.27, "sees the person\nbut accepts the mean", ha="right", va="bottom", fontsize=F(9), color="#777")
LAB = {"mse": (8, -12), "ssim": (-38, 6), "tm_like": (8, 3), "lddt": (8, -4), "lpips_alex": (-88, 2),
       "lpips_vgg": (8, -4), "betti_matched": (8, -12), "persistence": (-80, 5), "fid": (8, 0), "fwd": (8, -10)}
if PAPER:   # label offsets in points for the 7 in figure
    LAB = {"mse": (4, -7), "ssim": (-27, 3), "tm_like": (4, -1), "lddt": (4, -3), "lpips_alex": (4, -8),
           "lpips_vgg": (-46, 3), "betti_matched": (4, -9), "persistence": (-48, 2), "fid": (4, -1), "fwd": (4, -8)}
for _, mk, mets in GROUPS:
    for m, lab in mets:
        p, _ = data[m]
        x, y = p["stranger"] - p["ref"], p["dataset_mean"] - p["ref"]
        ax.scatter(x, y, s=S(95), marker=mk, color="#333", edgecolor="white", linewidth=1.5 if not PAPER else 0.6, zorder=3)
        ax.annotate(lab, (x, y), xytext=LAB.get(m, (8, 3)), textcoords="offset points", fontsize=F(9.5), color="#222")
ax.set_xlim(-0.15, 1.05)
ax.set_ylim(-0.3, 1.05)
ax.set_xlabel("Sees the person  →\nhow much further the stranger is than the true person" if not PAPER
              else "sees the person →\n(stranger − true person, rank gap)")
ax.set_ylabel("Rejects the dataset mean  →\nhow much further the mean is than the true person" if not PAPER
              else "rejects the mean →\n(dataset mean − true person, rank gap)")
ax.set_title("A. Crude averaging: is the dataset mean pushed away?" if not PAPER else "A. Is the dataset mean pushed away?",
             loc="left", fontsize=F(11.5))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color="#f0f0f0", zorder=0)
handles = [plt.Line2D([], [], color="#333", marker=mk, ls="", ms=(9 if not PAPER else 5), markeredgecolor="white") for _, mk, _ in GROUPS]
if not PAPER:   # in the paper the marker shapes are explained in the caption
    ax.legend(handles, [h.split("\n")[0] for h, *_ in GROUPS], loc="lower left", ncol=1, frameon=True,
              framealpha=0.9, edgecolor="#ddd", fontsize=F(9))

# ---- right: strips; per-image metrics on a linear scale, set-level FID/FWD on a log scale ----
PAPER_HEAD = {GROUPS[0][0]: "Error / similarity", GROUPS[1][0]: "Deep features", GROUPS[2][0]: "Topology",
              GROUPS[3][0]: "Set level"}
PAPER_LEG = {"ref": "true person", "relative": "sibling", "shrink_ref": "shrunk true person",
             "lookalike16": "mean of k look-alikes*"}   # Kate's legend wording for the paper
XMAX = 2.0  # linear panel shows 0..2 strangers; a larger value sits at the edge with its number


def strips(axx, groups, tx, x0, x1, step):
    """Draw one strip per metric; tx maps a value to the axis coordinate. Returns the last y."""
    y, yt, yl = 0.0, [], []
    for head, _, mets in groups:
        y -= 0.3
        head = head if not PAPER else PAPER_HEAD.get(head, head)
        axx.text(x0 - 0.03 * (x1 - x0), y, head, ha="right", va="center", fontsize=F(9.5), fontweight="bold",
                 color="#222")
        y -= 0.75 if "\n" in head else 0.55
        for m, lab in mets:
            p, nds = rdata[m]
            axx.plot([x0, x1], [y, y], color="#eeeeee", lw=6 if not PAPER else 3.5, solid_capstyle="round", zorder=1)
            placed = []
            for k, _, col, mk, _row in RUNGS:
                if np.isnan(p[k]):
                    continue
                xv = tx(p[k])
                over = xv > x1
                xv = min(xv, x1)
                dy = 0.0
                while any(abs(xv - q) < step and abs(dy - e) < 0.2 for q, e in placed):
                    dy = -dy + 0.24 if dy <= 0 else -dy  # stack touching markers above and below the strip
                placed.append((xv, dy))
                axx.scatter(xv, y + dy, s=S(105), color=col, marker=mk, edgecolor="white",
                            linewidth=1.5 if not PAPER else 0.6, zorder=3)
                if over:
                    axx.annotate(f"{p[k]:.1f} →", (xv, y + dy), xytext=(8, -4), textcoords="offset points",
                                 fontsize=F(8.5), color="#444")
            yt.append(y)
            yl.append(f"{lab}  ({nds} datasets)" if not PAPER else f"{lab} ({nds})")
            y -= 1
    axx.set_yticks(yt)
    axx.set_yticklabels(yl, fontsize=F(10))
    axx.tick_params(axis="y", length=0, pad=8 if not PAPER else 3)
    axx.set_ylim(y + 0.3, 0.1)
    for s in ("top", "right", "left"):
        axx.spines[s].set_visible(False)
    return y


sub = gs[1].subgridspec(2, 1, height_ratios=[9.3, 2.6], hspace=0.32)
bx = fig.add_subplot(sub[0])
strips(bx, GROUPS[:3], lambda v: v, 0.0, XMAX, 0.045)
bx.axvline(1, color="#555", lw=1, ls=(0, (3, 3)), zorder=0)
bx.axvline(0.5, color="#eb6834", lw=0.8, ls=(0, (2, 4)), alpha=0.6, zorder=0)
bx.set_xlim(-0.04, XMAX + 0.2)
if not PAPER:
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.5, 2])
    bx.set_xticklabels(["0\ntarget", "0.25", "0.5", "0.75", "1\nstranger", "1.5", "2"], fontsize=F(9))
else:
    bx.set_xticks([0, 0.5, 1, 1.5, 2])
    bx.set_xticklabels(["0\ntarget", "0.5", "1\nstranger", "1.5", "2"], fontsize=F(9))
bx.set_title("B. How far each metric puts each fill, in units of a stranger (0 = the target itself, 1 = a stranger)"
             if not PAPER else "B. Distance in units of a stranger", loc="left", fontsize=F(11.5))

cx = fig.add_subplot(sub[1])
strips(cx, GROUPS[3:], np.log10, np.log10(0.3), np.log10(300), 0.07)
cx.axvline(0, color="#555", lw=1, ls=(0, (3, 3)), zorder=0)
cx.set_xlim(np.log10(0.3) - 0.05, np.log10(300) + 0.05)
lt = [0.3, 0.5, 1, 2, 5, 10, 30, 100, 300] if not PAPER else [0.5, 1, 2, 5, 10, 30, 100, 300]
cx.set_xticks(np.log10(lt))
cx.set_xticklabels((["0.3"] if not PAPER else []) + ["0.5", "1\nstranger", "2", "5", "10", "30", "100", "300"],
                   fontsize=F(9))
cx.set_xlabel("FID / FWD: set distance divided by the stranger set's distance (log scale)" if not PAPER
              else "set distance / stranger set's distance (log)", fontsize=F(9.5))
bx = cx  # legends go under the last axis
for row, (title, ypos) in enumerate([] if PAPER else [("Ladder:", -0.62), ("Checks:", -0.95)]):
    sel = [r for r in RUNGS if r[4] == row]
    hh = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=(9 if not PAPER else 5), markeredgecolor="white") for _, _, c, mk, _ in sel]
    lg = bx.legend(hh, [l for _, l, *_ in sel], loc="upper left", bbox_to_anchor=(-0.05, ypos), ncol=len(sel),
                   frameon=False, fontsize=F(9.5), handletextpad=0.3, columnspacing=1.1, title=title,
                   alignment="left")
    lg._legend_box.align = "left"
    lg.get_title().set_fontweight("bold")
    bx.add_artist(lg)

if PAPER:   # one legend across the bottom of the page-wide figure; caption text lives in LaTeX
    for row, (title, yy) in enumerate([("Ladder:", 0.075), ("Checks:", 0.02)]):
        sel = [r for r in RUNGS if r[4] == row]
        hh = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=5, markeredgecolor="white") for _, _, c, mk, _ in sel]
        fig.text(0.01, yy + 0.012, title, fontsize=F(9), fontweight="bold", va="center")
        fig.legend(hh, [PAPER_LEG.get(k, l) for k, l, *_ in sel], loc="center left", bbox_to_anchor=(0.085, yy + 0.012), ncol=len(sel),
                   frameon=False, fontsize=F(9), handletextpad=0.2, columnspacing=0.8)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.2)
    fig.savefig(OUT + ".png", dpi=300)
    fig.savefig(OUT + ".pdf")
    print("saved paper variant")
    sys.exit(0)
fig.suptitle("Metrics see the person or reject averaging; the shrunk true person (green) passes almost all of them",
             fontsize=F(13.5), x=0.5, y=0.99)
fig.text(0.01, 0.008,
         "A: per metric, from its ordering of the ladder (rank gaps / ladder length, mean over datasets): stranger minus "
         "true person (x), dataset mean minus true person (y). B: R = (d(fill) - d(target)) / (d(16 strangers) - "
         "d(target)), mean over datasets; per-image metrics on a linear axis, FID/FWD on a log axis.\n"
         "* Look-alikes: oracle = chosen with the hidden truth (upper bound); mean of k look-alikes (k = 16) on brain, fundus "
         "and iris (not faces). True person = repeat scan, or the other side where there is no repeat (PAPILA, IXI). Brain LPIPS and "
         "persistence are repeat-anchored and left out of B's means.\n"
         "Data: QTIM and IXI brain, CelebA, IMDb-WIKI, GRAPE, PAPILA, UTIRIS NIR/VIS, UPOL; brain, fundus, iris pre-QC. Source: "
         "results/cross_modality/common_all.json, values_generative.json, ranks_all.json, ranks_generative.json.",
         fontsize=F(7.8), color="#666")
fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.25)
fig.savefig(OUT + ".png", dpi=160)
fig.savefig(OUT + ".pdf")
print("saved")
