"""Explainer figure v2 (15 Nov, shared paper style from fig_style.py; v1 kept as make_fig_explainer.py): how one row (MSE) of the metric-summary figure is computed, traced on one QTIM brain cell.

Data: QTIM (OpenNeuro ds004169, CC0), the random-cube panel of src/donor_ladder_brain_metrics.py (16 mm cube,
4 mm context shell, both hemispheres, positions < 32) and the 16-look-alike fill of
src/donor_ladder_brain_lookalike16.py. Pre-QC: affine registration only.

The cell (target, cube, side) is chosen mechanically, before drawing: among cells that have every fill of the MSE row,
the one whose eight stranger-scale values are closest to the QTIM averages (sum of absolute log ratios). The fills of
that cell are rebuilt with the same code and seeds as the committed rows, and the script stops unless every rebuilt
MSE equals the saved row. Panel D uses the per-dataset values of results/cross_modality/common_all.json and the means
of results/cross_modality/fig_metric_summary_R.json (the MSE row of the summary figure).

Usage (from the repository root): python -I paper/cvpr2027/make_fig_explainer_v2.py
Prints counts and values only, never participant IDs or file names.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as BL  # noqa: E402
import donor_ladder_brain_metrics as M  # noqa: E402
from donor_ladder_brain_lookalike16 import nearest  # noqa: E402

ROWS = os.path.join(M.PRIVATE, "rows.csv.gz")
ROWS16 = os.path.join(B.PROJ, "donor-ladder-brain-lookalike16-20261109-v1", "parts", "p*.csv.gz")
COMMON = os.path.join(ROOT, "results", "cross_modality", "common_all.json")
SUMMARY = os.path.join(ROOT, "results", "cross_modality", "fig_metric_summary_R.json")
OUT = os.path.join(HERE, "fig", "fig_explainer_mse_v2")
TOL = 1e-4

# (row key in the rows, key in the summary JSON, origin line); label, color and marker come from fig_style.RUNG.
FILLS = [
    ("repeat", "ref", "same person,|second scan"),
    ("lookalike_oracle", "lookalike_oracle", "unrelated person most|like the hidden cube"),
    ("lookalike16", "lookalike16", "16 unrelated people with|the most similar ring"),
    ("dataset_mean", "dataset_mean", "average of all|unrelated people"),
    ("stranger", "stranger", "1 random|unrelated person"),
    ("shrink_repeat", "shrink_ref", "second scan, 85% of|its detail kept"),
    ("noise", "noise", "average + white noise,|same strength"),
    ("wrong_detail", "wrong_detail", "average + a stranger's|detail, same strength"),
]
STR = [f"str{j:02d}" for j in range(M.N_STRANGERS)]
DISPLAY = (-2.5, 2.5)
sys.path.insert(0, HERE)
import fig_style as S  # noqa: E402
S.apply()


def load_rows():
    keep = ["target", "patch", "side", "fill", "mse"]
    r = pd.read_csv(ROWS, usecols=keep)
    r16 = pd.concat([pd.read_csv(f, usecols=keep) for f in sorted(glob.glob(ROWS16))], ignore_index=True)
    r = pd.concat([r[r.patch < 32], r16[(r16.patch < 32) & (r16.fill == "lookalike16")]], ignore_index=True)
    return r.pivot_table(index=["target", "patch", "side"], columns="fill", values="mse", aggfunc="first")


def choose_cell(w, qtim):
    """Cell whose stranger-scale values are closest to the QTIM averages (sum of |log ratio| over the eight fills)."""
    need = [k for k, *_ in FILLS] + STR
    w = w.dropna(subset=need)
    den = w[STR].mean(1)
    R = pd.DataFrame({k: w[k] / den for k, *_ in FILLS if k != "stranger"})
    target = {k: qtim[s] for k, s, *_ in FILLS if k != "stranger"}
    score = sum(np.abs(np.log(R[k] / target[k])) for k in target)
    cell = score.idxmin()
    return cell, len(w), w.loc[cell], float(den.loc[cell])


def rebuild(cell, t, tg, pos, affine, lam):
    """Arrays of every fill for one cell, by the code path of run_position_at and the look-alike-16 script."""
    i, pi, side = int(cell[0]), int(cell[1]), cell[2]
    _, start, size = pos[pi]
    vol, mask = np.load(BL.region_paths(pi)[0]), np.load(BL.region_paths(pi)[1]).astype(bool)
    drawn = BL.side_of(start, size, affine)
    fo = BL.own_frame(side, drawn)
    shell, W = BL.SHELL, BL.W
    fam = tg.family.to_numpy()
    ix = tg["index"].to_numpy()
    row_of = {(p, s): k for k, (p, s) in enumerate(zip(t["person"], t["session"]))}
    win_sl = tuple(slice(W, W + s + 2 * shell) for s in size)
    cube_sl = tuple(slice(W + shell, W + shell + s) for s in size)
    sm = BL.shell_mask(size, shell)
    own = vol[:, fo]
    cube_all = own[(slice(None),) + cube_sl]
    mk_all = mask[:, fo][(slice(None),) + cube_sl]
    valid_all = mk_all.reshape(len(cube_all), -1).mean(1) >= BL.VALID
    X, MK, valid = cube_all[ix], mk_all[ix], valid_all[ix]
    win = own[(slice(None),) + win_sl][ix]
    wmask = mask[:, fo][(slice(None),) + win_sl][ix]
    C, CM = win[:, sm], wmask[:, sm]
    Xf, XM = X.reshape(len(X), -1), MK.reshape(len(X), -1)
    others = np.flatnonzero((fam != fam[i]) & valid)
    xi, ei = X[i], MK[i]
    mu_d, mu_ok = M.masked_mean(X[others], MK[others])
    pop = np.random.default_rng([B.SEED, i, pi, B.DRAWN.index("population")]).permutation(others)
    f = {}
    for j, s in enumerate(pop[:M.N_STRANGERS]):
        f[f"str{j:02d}"] = (X[s], MK[s])
    f["stranger"] = (X[pop[0]], MK[pop[0]])
    f["dataset_mean"] = (mu_d, mu_ok)
    d_s = M.masked_dist(C[others], CM[others], C[i], CM[i], int(sm.sum()))
    d_o = M.masked_dist(Xf[others], XM[others], Xf[i], XM[i], Xf.shape[1])
    j = others[int(np.argmin(d_o))]
    f["lookalike_oracle"] = (X[j], MK[j])
    near = others[nearest(d_s)]
    f["lookalike16"] = M.masked_mean(X[near], MK[near])
    j2 = row_of[(tg.person.iloc[i], "ses-02")]
    f["repeat"] = (cube_all[j2], mk_all[j2])
    f["shrink_repeat"] = (M.shrink(cube_all[j2], mu_d, lam["repeat"]), mk_all[j2] & mu_ok)
    e_wd = ei & mu_ok & MK[pop[0]]
    f["wrong_detail"] = (M.wrong_detail(xi, mu_d, X[pop[0]], e_wd), e_wd)
    e_nz = ei & mu_ok
    f["noise"] = (M.noise_fill(xi, mu_d, e_nz, np.random.default_rng([B.SEED, i, pi, BL.SIDES.index(side), 99])), e_nz)
    mse = {k: float(np.mean((xi[ei & ok].astype(np.float64) - a[ei & ok].astype(np.float64)) ** 2))
           for k, (a, ok) in f.items()}
    ctx = (win[i], wmask[i], sm)
    return xi, ei, f, mse, ctx, (start, size, fo, ix[i])


def brain_slice(t, k, start, size, fo, affine):
    """Axial slice of the whole target scan through the cube center, z-scored inside its brain mask as in extract()."""
    path = t.path.iloc[k]
    v = nib.load(path.replace("_brain.nii.gz", ".nii.gz")).get_fdata(dtype=np.float32)
    m = nib.load(path).get_fdata(dtype=np.float32) > 0.5
    z = np.where(m, (v - v[m].mean()) / (v[m].std() + 1e-6), np.nan)
    if fo == 1:
        z = z[B.mirror_index(affine, v.shape[0])]
    zc = start[2] + size[2] // 2
    return z[:, :, zc]


def main():
    common = json.load(open(COMMON, encoding="utf-8"))["datasets"]
    summ = json.load(open(SUMMARY, encoding="utf-8"))["mse"]
    qtim = {k: v["mse"][0] for k, v in common["QTIM"]["R_self"].items() if "mse" in v}
    qtim["ref"] = qtim["repeat"]
    w = load_rows()
    cell, n_cells, saved, den = choose_cell(w, qtim)
    print(json.dumps({"cells_with_all_fills": int(n_cells), "chosen_side": cell[2]}))

    t = B.scans()
    pos, affine, _ = B.positions(np.random.default_rng(B.SEED))
    tg = t[t.session == "ses-01"].reset_index()
    lam = M.rung_lambdas()
    xi, ei, f, mse, ctx, (start, size, fo, k) = rebuild(cell, t, tg, pos, affine, lam)
    bad = {n: (mse[n], float(saved[n])) for n in [x[0] for x in FILLS] + STR if abs(mse[n] - saved[n]) > TOL}
    if bad:
        raise SystemExit(f"rebuilt MSE differs from the saved rows: {bad}")
    den_new = float(np.mean([mse[s] for s in STR]))
    R = {key: mse[key] / den_new for key, *_ in FILLS}
    print(json.dumps({"rebuild_matches_rows": True, "mse_16_strangers_mean": round(den_new, 4),
                      "R_cell": {k: round(v, 3) for k, v in R.items()}}))

    # ------------------------------------------------------------------ figure
    R_plot = dict(R, stranger=1.0)        # the stranger marker is the average of the 16 strangers: 1 by definition
    LANE = {k: 0.35 - i * 0.1 for i, k in enumerate(S.ORDER)}
    skey_of = {k: s for k, s, _ in FILLS}
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 4.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.25, 0.95], hspace=0.62, left=0.015, right=0.985, top=0.905,
                          bottom=0.085)
    top = gs[0].subgridspec(2, 6, width_ratios=[1.35] + [1] * 5, wspace=0.16, hspace=1.05)
    mid = size[2] // 2

    # A: where the hidden cube sits
    ax = fig.add_subplot(top[:, 0])
    sl = brain_slice(t, k, start, size, fo, affine)
    rows_ok = np.flatnonzero(np.isfinite(sl).any(1))
    cols_ok = np.flatnonzero(np.isfinite(sl).any(0))
    ax.imshow(np.nan_to_num(sl, nan=DISPLAY[0]).T, cmap="gray", vmin=DISPLAY[0], vmax=DISPLAY[1], origin="lower")
    ax.add_patch(Rectangle((start[0] - 0.5, start[1] - 0.5), size[0], size[1], fill=False, ec="#f2b233", lw=1.0))
    s4 = BL.SHELL
    ax.add_patch(Rectangle((start[0] - s4 - 0.5, start[1] - s4 - 0.5), size[0] + 2 * s4, size[1] + 2 * s4,
                           fill=False, ec="#f2b233", lw=0.6, ls=(0, (2, 1.5))))
    ax.set_xlim(rows_ok.min() - 3, rows_ok.max() + 3)
    ax.set_ylim(cols_ok.min() - 3, cols_ok.max() + 3)
    ax.set_axis_off()
    ax.text(0.0, -0.03, "solid: hidden 16 mm cube\ndashed: 4 mm visible ring", transform=ax.transAxes,
            ha="left", va="top", fontsize=S.SIZE_SMALL, color=S.MUTED)

    def tile(axx, img, ok, title, sub, col=None, mk=None):
        shown = np.where(ok[:, :, mid], img[:, :, mid], np.nan)
        axx.imshow(np.nan_to_num(shown, nan=DISPLAY[0]).T, cmap="gray", vmin=DISPLAY[0], vmax=DISPLAY[1],
                   origin="lower", interpolation="nearest")
        axx.set_xticks([])
        axx.set_yticks([])
        for s in axx.spines.values():
            s.set_color(col or "#333333")
            s.set_linewidth(1.5 if col else 0.6)
        axx.set_title(title, fontsize=S.SIZE, pad=2)
        axx.text(0.5, -0.04, sub, transform=axx.transAxes, ha="center", va="top", fontsize=S.SIZE_SMALL,
                 color=S.MUTED, linespacing=1.05)
        if mk:
            axx.plot([0.13], [0.87], marker=mk, color=col, mec="white", mew=0.5, ms=5.5, transform=axx.transAxes)

    slots = [(r, c) for r in range(2) for c in range(1, 6)]
    ax0 = fig.add_subplot(top[slots[0]])
    tile(ax0, xi, ei, "Target", "hidden cube\n(true content)")
    ax0.text(0.5, -0.38, "MSE 0", transform=ax0.transAxes, ha="center", va="top")
    for (key, skey, sub), sl_ in zip(FILLS, slots[1:]):
        axx = fig.add_subplot(top[sl_])
        img, ok = f[key]
        lab = S.label(skey)
        tile(axx, img, ok, lab[0].upper() + lab[1:], sub.replace("|", chr(10)), S.color(skey), S.marker(skey))
        txt = (f"MSE {mse[key]:.2f}\n16 strangers: {den_new:.2f}" if key == "stranger"
               else f"MSE {mse[key]:.2f}\n= {R[key]:.2f} stranger")
        axx.text(0.5, -0.38, txt, transform=axx.transAxes, ha="center", va="top", linespacing=1.1)

    # C and D
    bot = gs[1].subgridspec(1, 2, width_ratios=[1, 1.3], wspace=0.28)

    def strip(axx, vals, y, big=True, alpha=1.0, lane=0.1):
        for key, skey, _ in FILLS:
            v = vals.get(skey if skey in vals else key)
            if v is None or not np.isfinite(v):
                continue
            axx.plot(min(v, 1.9), y + LANE[skey] * lane / 0.1, marker=S.marker(skey), color=S.color(skey),
                     mec="white", mew=S.EDGE, ms=S.MS + 0.6 if big else S.MS - 1.0, ls="", alpha=alpha, zorder=3)

    cx = fig.add_subplot(bot[0])
    strip(cx, R_plot, 0, lane=0.1)
    cx.axvline(0, color="#999999", lw=0.6)
    cx.axvline(1, color="#555555", lw=0.6, ls=(0, (3, 3)))
    cx.set_xlim(-0.05, 1.3)
    cx.set_ylim(-0.75, 0.75)
    cx.set_yticks([])
    cx.set_xticks([0, 0.5, 1])
    cx.set_xticklabels(["0\ntarget", "0.5", "1\nstranger"])
    for s in ("left", "right", "top"):
        cx.spines[s].set_visible(False)

    dx = fig.add_subplot(bot[1])
    groups = [("Brain MRI", ["QTIM", "IXI"]), ("Faces", ["CelebA", "IMDb-WIKI"]), ("Retina", ["GRAPE", "PAPILA"]),
              ("Iris", ["UTIRIS NIR", "UTIRIS VIS", "UPOL"])]
    if sorted(d for _, ds in groups for d in ds) != sorted(summ["datasets"]):
        raise SystemExit("modality groups do not cover the datasets of the summary row")
    group_vals = {}
    for r_, (g, ds) in enumerate(groups):
        per = []
        for d in ds:
            vals = {kk: v["mse"][0] for kk, v in common[d]["R_self"].items() if "mse" in v}
            vals["ref"] = vals.get("repeat", vals.get("other_side"))
            vals["stranger"] = 1.0
            per.append(vals)
        keys = {k for v in per for k in v}
        mean = {k: float(np.mean([v[k] for v in per if v.get(k) is not None])) for k in keys}
        group_vals[g] = mean
        y = len(groups) - r_ + 0.3
        strip(dx, mean, y, big=False, alpha=0.65, lane=0.08)
        dx.text(-0.07, y, f"{g} ({len(ds)})", ha="right", va="center", color=S.MUTED)
    print(json.dumps({"panel_D_groups": {g: {k: round(v, 3) for k, v in m.items() if k in
                                             ("ref", "shrink_ref", "dataset_mean", "lookalike16")}
                                         for g, m in group_vals.items()}}))
    strip(dx, {k: summ[k] for k in summ if isinstance(summ[k], float)}, 0, lane=0.1)
    dx.text(-0.07, 0, f"All {len(summ['datasets'])} datasets", ha="right", va="center", fontweight="bold")
    dx.axhline(0.75, color="#bbbbbb", lw=0.5)
    dx.axvline(1, color="#555555", lw=0.6, ls=(0, (3, 3)))
    dx.set_xlim(-0.05, 1.3)
    dx.set_ylim(-0.5, len(groups) + 0.75)
    dx.set_yticks([])
    dx.set_xticks([0, 0.5, 1])
    dx.set_xticklabels(["0\ntarget", "0.5", "1\nstranger"])
    for s in ("left", "right", "top"):
        dx.spines[s].set_visible(False)

    # panel letters and the B title, placed from the axes positions
    pa = ax.get_position()
    pb = ax0.get_position()
    S.letter(fig, 0.005, 0.955, "A")
    S.title(fig, 0.03, 0.955, "One target scan")
    S.letter(fig, pb.x0 - 0.02, 0.955, "B")
    S.title(fig, pb.x0 + 0.005, 0.955, "Every fill of the hidden cube, its MSE, and that MSE in strangers")
    pc, pd_ = cx.get_position(), dx.get_position()
    yt = pc.y1 + 0.03
    S.letter(fig, 0.005, yt, "C")
    S.title(fig, 0.03, yt, "This one cube on the stranger scale")
    xd = pd_.x0 - 0.16
    S.letter(fig, xd, yt, "D")
    S.title(fig, xd + 0.025, yt, "Mean over all cubes, by modality and over all datasets")
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=200)
    print(json.dumps({"written": [OUT + ".pdf", OUT + ".png"]}))


if __name__ == "__main__":
    main()
