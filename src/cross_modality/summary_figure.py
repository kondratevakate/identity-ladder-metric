"""Two views of the same rank tables, side by side.

Left (A): one point per metric. x = how far the metric puts a stranger beyond the true person; y = how far it puts
the true person shrunk toward the dataset mean (subtle averaging, control T2) beyond the true person. Both are gaps
in the metric's own ordering of the ladder, divided by the ladder length, per dataset, then mean over the datasets
that have both rungs (Kate, 10 Nov: the earlier y, the crude dataset mean, put nearly every metric in the wanted
corner, which said nothing about subtle averaging). Shaded corner "both": x >= 0.4 and y >= 0.25.
Right (B): R on the distance scale per rung, mean over datasets (see r_values).
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
    ("lookalike16", "mean of k look-alikes*", "#6250d6", "h", 1),
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


SET_LEVEL = ("fid", "fwd")
MIN_SET_FRACTION = 0.5   # set-level rule (10 Nov, post hoc): FID / FWD of a rung count only if its set has at least
# half the windows of that dataset's stranger set; FID is biased upward for small sets (QTIM repeat: 138 of 1326
# windows, ratio 8.3). Size-matched subsampling would need the per-window features, held by the modality pipelines.


SET_LEVEL_EXCLUDED = {"QTIM": "brain excluded from FID/FWD: its repeat set (138 windows) is about 10x smaller than its "
                                "stranger set, and FID is biased upward for small sets"}   # ISBI paper session, 10 Nov


def size_ok(metric, d, rung):
    """False when a set-level rung's set is smaller than MIN_SET_FRACTION of the stranger set, or the dataset is excluded
    from set-level metrics (left out, never zero)."""
    if metric not in SET_LEVEL:
        return True
    if d in SET_LEVEL_EXCLUDED:
        return False
    blk = gen.get(f"{metric}|{d}", {}).get("rungs", {})
    r, st = blk.get(rung), blk.get("stranger")
    if not r or not st or not isinstance(r.get("n"), (int, float)) or not isinstance(st.get("n"), (int, float)):
        return True
    return r["n"] / st["n"] >= MIN_SET_FRACTION


def r_values(metric):
    """Mean over datasets of each rung's value on the stranger scale, merging the sources by dataset: common_all R_self
    where it has the metric, otherwise the generative block's value. Per-image generative values count only when
    self-anchored (brain R is on the repeat scale, so brain LPIPS and persistence are left out); set-level FID / FWD are
    ratios to the stranger set and do not depend on that anchor, so brain counts there. Returns (means, datasets)."""
    rows = {}
    for d, blk in common.items():
        R = blk.get("R_self", {})
        row = {r: R[r][metric][0] for r in R if isinstance(R[r], dict) and metric in R[r]}
        if row:
            rows[d] = row
    for key, blk in gen.items():
        m, d = key.split("|")
        if m != metric or d in rows:
            continue
        if metric in SET_LEVEL and d in SET_LEVEL_EXCLUDED:
            continue
        if blk.get("scale") == "ratio to stranger set" or blk.get("anchor") == "self":
            rows[d] = {r: (x.get("R", x.get("ratio_to_stranger")) if size_ok(metric, d, r) else None)
                       for r, x in blk["rungs"].items()}
    vals = {k: [] for k, *_ in RUNGS}
    used = []
    for d, row in rows.items():
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
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in vals.items()}, used


def panel_a(metric):
    """Panel A point: per dataset, (stranger - true person) and (shrunk true person - true person) rank gaps over the
    ladder length; mean over the datasets that have the rungs. Returns x, y and the number of datasets for each."""
    xs, ys = [], []
    for key, v in ranks.items():
        m, d = key.split("|")
        ref = "repeat" if "repeat" in v else ("other_side" if "other_side" in v else None)
        if m != metric or ref is None or not size_ok(m, d, ref):
            continue
        n = max(v.values())
        if "stranger" in v:
            xs.append((v["stranger"] - v[ref]) / (n - 1))
        if "shrink_ref" in v and size_ok(m, d, "shrink_ref"):
            ys.append((v["shrink_ref"] - v[ref]) / (n - 1))
    return (float(np.mean(xs)) if xs else np.nan, float(np.mean(ys)) if ys else np.nan, len(xs), len(ys))


rdata = {m: r_values(m) for _, _, mets in GROUPS for m, _ in mets}
adata = {m: panel_a(m) for _, _, mets in GROUPS for m, _ in mets}
out = {m: {"n_datasets": len(ds), "datasets": ds, **{k: (None if np.isnan(x) else round(x, 3)) for k, x in p.items()}}
       for m, (p, ds) in rdata.items()}
out["set_level_rule"] = {"min_set_fraction_of_stranger": MIN_SET_FRACTION, "datasets_excluded": SET_LEVEL_EXCLUDED,
                         "left_out": sorted(f"{k}:{r}" for k, b in gen.items() if k.split("|")[0] in SET_LEVEL
                                            for r in b["rungs"] if not size_ok(k.split("|")[0], k.split("|")[1], r))}
out["panel_A"] = {m: {"x_stranger_gap": round(a[0], 3), "y_shrunk_gap": round(a[1], 3), "n_x": a[2], "n_y": a[3]}
                  for m, a in adata.items()}
json.dump(out, open(OUT + "_R.json", "w"), indent=1)

plt.rcParams.update({"font.size": F(10.5)})
fig = plt.figure(figsize=(17, 12) if not PAPER else (7.0, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3] if not PAPER else [1, 1.12], wspace=0.36 if not PAPER else 0.5)

# ---- left: sees the person (x) and rejects subtle averaging (y) ----
ax = fig.add_subplot(gs[0])
ax.axhline(0, color="#ccc", lw=1)
ax.axvline(0, color="#ccc", lw=1)
YLO, YHI = -0.4, 0.75
ax.add_patch(plt.Rectangle((0.4, 0.25), 0.65, YHI - 0.25, color="#1baf7a", alpha=0.10, lw=0))
ax.text(1.02, YHI - 0.02, "both", ha="right", va="top", fontsize=F(9), color="#226b4f", fontweight="bold")
ax.text(-0.13, YHI - 0.02, "rejects averaging,\nblind to\nthe person", ha="left", va="top", fontsize=F(9), color="#999")
ax.text(1.02, YLO + 0.02, "sees the person,\nrewards averaging", ha="right", va="bottom", fontsize=F(9), color="#999")
LAB = {"mse": (9, -8), "ssim": (9, -16), "tm_like": (9, 5), "lddt": (9, -6), "lpips_alex": (9, -7),
       "lpips_vgg": (-80, 6), "betti_matched": (-118, -5), "persistence": (-98, -5), "fid": (9, -5), "fwd": (9, -5)}
if PAPER:   # label offsets in points for the 7 in figure
    LAB = {"mse": (5, -5), "ssim": (5, -10), "tm_like": (5, 3), "lddt": (-25, -4), "lpips_alex": (5, -5),
           "lpips_vgg": (-46, 4), "betti_matched": (-66, -4), "persistence": (-56, -4), "fid": (5, -4), "fwd": (5, -4)}
for _, mk, mets in GROUPS:
    for m, lab in mets:
        x, y = adata[m][0], adata[m][1]
        ax.scatter(x, y, s=S(95), marker=mk, color="#333", edgecolor="white", linewidth=1.5 if not PAPER else 0.6, zorder=3)
        ax.annotate(lab, (x, y), xytext=LAB.get(m, (8, 3)), textcoords="offset points", fontsize=F(9.5), color="#222")
ax.set_xlim(-0.15, 1.05)
ax.set_ylim(YLO, YHI)
ax.set_xlabel("Sees the person  →\nhow much further the stranger is than the true person (rank gap)" if not PAPER
              else "sees the person →\n(stranger − true person, rank gap)")
ax.set_ylabel("rejects subtle averaging →\n(shrunk true person − true person, rank gap)")
ax.set_title("A. Does the metric see the person and reject subtle averaging?" if not PAPER
             else "A. Does the metric see the person\nand reject subtle averaging?", loc="left", fontsize=F(11.5))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color="#f0f0f0", zorder=0)
handles = [plt.Line2D([], [], color="#333", marker=mk, ls="", ms=(9 if not PAPER else 5), markeredgecolor="white") for _, mk, _ in GROUPS]
if not PAPER:   # in the paper the marker shapes are explained in the caption
    ax.legend(handles, [h.split("\n")[0] for h, *_ in GROUPS], loc="center left", ncol=1, frameon=True,
              framealpha=0.9, edgecolor="#ddd", fontsize=F(9))

# ---- right: strips; per-image metrics on a linear scale, set-level FID/FWD on a log scale ----
PAPER_HEAD = {GROUPS[0][0]: "Error / similarity", GROUPS[1][0]: "Deep features", GROUPS[2][0]: "Topology",
              GROUPS[3][0]: "Set level"}
PAPER_LEG = {"ref": "true person", "relative": "sibling", "shrink_ref": "shrunk true person",
             "lookalike16": "mean of k look-alikes*"}   # Kate's legend wording for the paper
ROW = 1.2   # row step: 20% more than the first version, so stacked markers read as one row (Kate, 10 Nov)
LEG_MS = float(np.sqrt(S(105)))   # legend markers the same size as the plotted ones (scatter s is area in pt^2)
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
            p, ds_used = rdata[m]
            nds = len(ds_used)
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
            y -= ROW
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
    hh = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=LEG_MS, markeredgecolor="white",
              markeredgewidth=1.5 if not PAPER else 0.6) for _, _, c, mk, _ in sel]
    lg = bx.legend(hh, [l for _, l, *_ in sel], loc="upper left", bbox_to_anchor=(-0.05, ypos), ncol=len(sel),
                   frameon=False, fontsize=F(9.5), handletextpad=0.3, columnspacing=1.1, title=title,
                   alignment="left")
    lg._legend_box.align = "left"
    lg.get_title().set_fontweight("bold")
    bx.add_artist(lg)

if PAPER:   # one legend across the bottom of the page-wide figure; caption text lives in LaTeX
    for row, (title, yy) in enumerate([("Ladder:", 0.075), ("Checks:", 0.02)]):
        sel = [r for r in RUNGS if r[4] == row]
        hh = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=LEG_MS, markeredgecolor="white",
              markeredgewidth=1.5 if not PAPER else 0.6) for _, _, c, mk, _ in sel]
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
         "true person (x), true person shrunk to the mean minus true person (y); shaded: x >= 0.4 and y >= 0.25.\n"
         "B: R = (d(fill) - d(target)) / (d(16 strangers) - d(target)), mean over datasets; per-image metrics on a linear "
         "axis, FID/FWD on a log axis.\n"
         "* Look-alikes: oracle = chosen with the hidden truth (upper bound); mean of k look-alikes (k = 16), all nine sets. True person = repeat scan, or the other side where there is no repeat (PAPILA, IXI).\n"
         "Brain LPIPS, Betti and persistence (and IXI Betti) are on other anchors and left out of B's means. Brain is excluded "
         "from FID/FWD (its repeat set is about 10x smaller than its stranger set; FID is biased upward for small sets), and\n"
         "set-level rungs with under half the stranger set's windows are left out.\n"
         "Data: QTIM and IXI brain, CelebA, IMDb-WIKI, GRAPE, PAPILA, UTIRIS NIR/VIS, UPOL; brain, fundus, iris pre-QC. Source: "
         "results/cross_modality/common_all.json, values_generative.json, ranks_all.json, ranks_generative.json.",
         fontsize=F(7.8), color="#666")
fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.29)
fig.savefig(OUT + ".png", dpi=160)
fig.savefig(OUT + ".pdf")
print("saved")
