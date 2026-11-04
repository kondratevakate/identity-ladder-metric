"""Case figure, version 2: the ISBI Fig. 1 layout with the bilateral and look-alike variants of Amendment 2.

Same target as paper/isbi2027/make_fig_real_case.py (first session-1 scan in participant order whose parietal box is
valid for the target, a sibling and a repeat scan), same box and display. The fills come from the region arrays of
donor_ladder_brain_bilateral (left side = the box as drawn). Numbers under each column are group means over both
sides (L+R view of results/donor_ladder_brain_bilateral.json). Pre-QC; affine only. Prints counts only, never IDs.
Output: results/figures/fig_real_case_parietal_v2.{png,pdf}.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "paper", "isbi2027"))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as L  # noqa: E402
import donor_ladder_brain_hemisphere as H  # noqa: E402
from make_figures import GRID, INK, MUTED, ORANGE  # noqa: E402  (also applies the paper rcParams)

CROP = (slice(30, 170), slice(20, 210))
DISPLAY = (-2.5, 2.5)
# key, title, (cohort, k) in the bilateral JSON; groups follow in GROUPS
COLUMNS = [("target", "Target", None),
           ("repeat", "Repeat\nscan", ("repeat", "1")),
           ("other_hemisphere", "Mirror,\nsame place", ("other_hemisphere", "1")),
           ("mirror_hemi", "Mirror, best\nin hemisphere", ("mirror_hemi", "1")),
           ("sibling", "Sibling", ("sibling", "1")),
           ("stranger", "Stranger,\nsame side", ("population", "1")),
           ("stranger_other_side", "Stranger,\nother side", ("stranger_other_side", "1")),
           ("knn_oracle", "16 look-alikes\nby true box*", ("knn_oracle", "16")),
           ("knn", "16 look-alikes\nby surround", ("knn", "16")),
           ("knn_mirror", "16 look-alikes\nby own mirror", ("knn_mirror", "16")),
           ("cohort", "Same sex\nand age", ("cohort", "all")),
           ("population", "Dataset\nmean", ("population", "all"))]
GROUPS = [(1, 3, "Same person"), (4, 6, "Another person"), (7, 11, "Averages of other people")]


def case():
    """Target slice and every fill of its parietal box on the drawn (left) side; arrays only."""
    t = B.scans()
    pos, affine, _ = B.positions(np.random.default_rng(B.SEED))
    pi = [p[0] for p in pos].index("parietal")
    _, start, size = pos[pi]
    if L.side_of(start, size, affine) != "L":
        raise RuntimeError("parietal box expected on the left")
    vol = np.load(L.region_paths(pi)[0], mmap_mode="r")
    mask = np.load(L.region_paths(pi)[1], mmap_mode="r")
    cube_sl = (slice(None),) + tuple(slice(L.W + L.SHELL, L.W + L.SHELL + s) for s in size)
    win_sl = tuple(slice(L.W, L.W + s + 2 * L.SHELL) for s in size)
    X_all = np.asarray(vol[:, 0][cube_sl]).reshape(len(t), -1)
    V_all = np.asarray(mask[:, 0][cube_sl]).reshape(len(t), -1).mean(1) >= L.VALID
    tg = t[t.session == "ses-01"].reset_index()
    ix = tg["index"].to_numpy()
    X, valid = X_all[ix], V_all[ix]
    M = np.asarray(vol[:, 1][cube_sl]).reshape(len(t), -1)[ix]
    mvalid = np.asarray(mask[:, 1][cube_sl]).reshape(len(t), -1).mean(1)[ix] >= L.VALID
    fam, age, sex = tg.family.to_numpy(), tg.age.to_numpy(), tg.sex.to_numpy()
    rep = {r.person: r.Index for r in t[t.session == "ses-02"].itertuples()}
    cands = []
    for i in np.flatnonzero(valid):
        sib = [j for j in np.flatnonzero((fam == fam[i]) & valid) if j != i and age[j] != age[i]]
        r = rep.get(tg.person.iloc[i])
        if sib and r is not None and V_all[r]:
            cands.append((i, sib[0], r))
    i, js, jr = cands[0]
    sm = L.shell_mask(size)
    win = np.asarray(vol[ix[i], 0][win_sl])
    C = np.stack([np.asarray(vol[ix[j], 0][win_sl])[sm] for j in range(len(ix))]).astype(float)
    others = np.flatnonzero((fam != fam[i]) & valid)
    coh = others[(sex[others] == sex[i]) & (np.abs(age[others] - age[i]) <= 2)]
    order = np.random.default_rng([B.SEED, i, pi, 0]).permutation(others)
    nn = lambda d: others[np.argsort(d[others], kind="stable")][:16]  # noqa: E731
    row = t.loc[ix[i]]
    v = nib.load(row.path.replace("_brain.nii.gz", ".nii.gz")).get_fdata(dtype=np.float32)
    m = nib.load(row.path).get_fdata(dtype=np.float32) > 0.5
    z = np.where(m, (v - v[m].mean()) / (v[m].std() + 1e-6), 0).astype(np.float32)
    a = L.mirror_constant(affine, v.shape[0])
    mi = B.mirror_index(affine, v.shape[0])
    _, hemi = H.search(win, H.prepare(z[mi], m[mi], "L", size, a))
    so = [j for j in order if mvalid[j]]
    fills = {"target": X[i], "repeat": X_all[jr], "other_hemisphere": M[i], "mirror_hemi": hemi.ravel(),
             "sibling": X[js], "stranger": X[order[0]], "stranger_other_side": M[so[0]],
             "knn_oracle": X[nn(((X - X[i]) ** 2).sum(1))].mean(0), "knn": X[nn(((C - C[i]) ** 2).sum(1))].mean(0),
             "knn_mirror": X[nn(((X - M[i]) ** 2).sum(1))].mean(0),
             "cohort": X[coh].mean(0), "population": X[others].mean(0)}
    zv = np.where(m, z, DISPLAY[0])  # outside brain: black
    return zv, start, size, fills, {"n_candidates": len(cands), "this_population": int(len(others)),
                                    "this_cohort": int(len(coh)), "mirror_valid": bool(mvalid[i])}


def draw(zv, start, size, fills, group, dest_pdf, dest_png):
    zc = size[2] // 2
    sl0 = zv[:, :, start[2] + zc]
    sx, sy = slice(start[0], start[0] + size[0]), slice(start[1], start[1] + size[1])
    fig, axes = plt.subplots(2, len(COLUMNS), figsize=(7.0, 2.3),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.05, "height_ratios": [1.36, 1]})
    fig.subplots_adjust(top=0.70, bottom=0.12, left=0.03, right=0.997)
    for j, (key, title, gkey) in enumerate(COLUMNS):
        cube = fills[key].reshape(size)[:, :, zc]
        sl = sl0.copy()
        sl[sx, sy] = cube
        axes[0, j].imshow(sl[CROP].T, origin="lower", cmap="gray", vmin=DISPLAY[0], vmax=DISPLAY[1],
                          interpolation="nearest")
        axes[0, j].add_patch(Rectangle((start[0] - CROP[0].start - .5, start[1] - CROP[1].start - .5), size[0],
                                       size[1], fill=False, ec=ORANGE, lw=0.7))
        axes[1, j].imshow(cube.T, origin="lower", cmap="gray", vmin=DISPLAY[0], vmax=DISPLAY[1],
                          interpolation="nearest")
        axes[0, j].set_title(title, fontsize=5.6, color=INK, pad=2, linespacing=1.0)
        if gkey is None:
            lab = "0 (itself)"
        else:
            v = group[gkey[0]][gkey[1]]
            lab = f"{v['err_over_V']:.2f}"
        axes[1, j].set_xlabel(lab, fontsize=6, color=INK, labelpad=2)
        for ax in axes[:, j]:
            ax.set(xticks=[], yticks=[])
            for sp in ax.spines.values():
                sp.set(color=INK, lw=0.5)
    axes[0, 0].set_ylabel("Slice", fontsize=6.5)
    axes[1, 0].set_ylabel("Box", fontsize=6.5)
    for a, b, text in GROUPS:
        x0, x1 = axes[0, a].get_position().x0 + 0.003, axes[0, b].get_position().x1 - 0.003
        y = 0.82
        fig.add_artist(plt.Line2D([x0, x0, x1, x1], [y - 0.015, y, y, y - 0.015], color=MUTED, lw=0.6))
        fig.text((x0 + x1) / 2, y + 0.008, text, ha="center", va="bottom", fontsize=6.5, color=MUTED)
    fig.text(0.5, 0.015, "MSE in units of the dataset-mean error, average of left and right boxes over all people. "
             "* upper bound: neighbours chosen with the answer. Pre-QC, affine registration only.",
             ha="center", fontsize=5.4, color=MUTED)
    fig.suptitle("Real brain: pixel error ranks the dataset mean above related anatomy", fontsize=9, y=0.995)
    fig.savefig(dest_png, dpi=250)
    fig.savefig(dest_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


BLUE, GREEN, AMBER, PINK, VIOLET, TEAL = "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300"
# rows of the theory panel (b): cohort, k, label, colour, prediction kind (None: no closed form)
THEORY_ROWS = [("repeat", "1", "Repeat scan", TEAL, None),
               ("other_hemisphere", "1", "Mirror, same place", PINK, None),
               ("mirror_block", "1", "Mirror, best of ±8 mm", PINK, None),
               ("mirror_hemi", "1", "Mirror, best in hemisphere", PINK, None),
               ("sibling", "1", "Sibling", VIOLET, None),
               ("population", "1", "Stranger, same side", BLUE, "stranger"),
               ("stranger_other_side", "1", "Stranger, other side", BLUE, "stranger"),
               ("stranger_block", "1", "Stranger, best of ±8 mm", BLUE, None),
               ("stranger_hemi", "1", "Stranger, best in hemisphere", BLUE, None),
               ("knn_oracle", "16", "16 look-alikes by true box*", AMBER, None),
               ("knn", "16", "16 look-alikes by surround", AMBER, None),
               ("knn_mirror", "16", "16 look-alikes by own mirror", AMBER, None),
               ("cohort", "all", "Same sex and age", GREEN, "cohort"),
               ("population", "all", "Dataset mean", BLUE, "mean")]
N_COHORT = 51   # median same-sex-and-age pool (paper Fig. 1 caption)


def theory(views, region_title, dest_pdf, dest_png):
    """Model prediction against the measured error, L+R mean with left and right marked.
    (a) averages of k donors; (b) every single fill and average, with predictions where the model has a closed
    form: 2 for one stranger (either side), (1 - R2)(1 + 1/n) for the same-sex-and-age mean, 1 for the mean."""
    lr, left, right = views["LR"], views["L"], views["R"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw={"width_ratios": [0.9, 1.1], "wspace": 0.75})
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.985)
    note = dict(fontsize=5.8, color=MUTED, linespacing=1.15)
    ks = np.array([1, 2, 4, 8, 16, 32, 64])
    kk = np.geomspace(1, 64, 200)
    r2 = lr["cohort"]["fit"]["R2"]
    a.plot(kk, 1 + 1 / kk, color=BLUE, lw=1.2, zorder=2, label="Model: k strangers, 1 + 1/k")
    a.plot(kk, (1 - r2) * (1 + 1 / kk), color=GREEN, lw=0.9, ls=(0, (3, 2)), zorder=2,
           label=f"Model: same sex and age, R$^2$ = {r2:.2f}")
    series = (("population", BLUE, "o", "Strangers"), ("cohort", GREEN, "o", "Same sex and age"),
              ("knn", AMBER, "o", "Look-alikes by surround"), ("knn_oracle", AMBER, "^", "Look-alikes by true box*"),
              ("knn_mirror", AMBER, "s", "Look-alikes by own mirror"))
    for n_s, (c, col, mk, lab) in enumerate(series):
        dx = 1 + 0.025 * (n_s - 2)
        y = np.array([lr[c][str(k)]["err_over_V"] for k in ks])
        lo = np.array([lr[c][str(k)]["ci95"][0] for k in ks])
        hi = np.array([lr[c][str(k)]["ci95"][1] for k in ks])
        a.errorbar(ks * dx, y, yerr=[y - lo, hi - y], fmt=mk, ms=3.0, color=col, mec="white", mew=0.4,
                   elinewidth=0.7, capsize=0, zorder=3, label=f"Measured: {lab}")
    a.axhline(1, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    a.text(1.1, 0.98, "dataset mean = 1", va="top", **note)
    a.set_xscale("log", base=2)
    a.set_xticks(ks, [str(k) for k in ks])
    a.set_xlim(0.8, 80)
    a.set_ylim(0.3, 2.1)
    a.set_xlabel("Number of people averaged, k")
    a.set_ylabel("Error, relative to the dataset mean")
    a.set_title("(a) Averages of k people", fontsize=7, loc="left", pad=4)
    a.legend(frameon=False, fontsize=5.2, loc="upper right", handlelength=1.6)

    pred = {"stranger": 2.0, "cohort": (1 - r2) * (1 + 1 / N_COHORT), "mean": 1.0}
    y = np.arange(len(THEORY_ROWS))[::-1]
    for yi, (c, k, lab, col, kind) in zip(y, THEORY_ROWS):
        v = lr.get(c, {}).get(k)
        if v is None:
            continue
        e = v["err_over_V"]
        if kind is not None:
            b.scatter(pred[kind], yi, s=24, marker="D", facecolor="white", edgecolor=col, lw=0.8, zorder=2)
        for side, mk in ((left, "<"), (right, ">")):
            sv = side.get(c, {}).get(k)
            if sv is not None:
                b.scatter(sv["err_over_V"], yi + 0.28, s=7, marker=mk, color=col, alpha=0.6, lw=0, zorder=2)
        b.errorbar(e, yi, xerr=[[e - v["ci95"][0]], [v["ci95"][1] - e]], fmt="o", ms=3.4, color=col, mec="white",
                   mew=0.4, elinewidth=0.8, capsize=0, zorder=3)
        shared = f"  r = {1 - e / 2:.2f}" if c in ("repeat", "other_hemisphere", "mirror_block", "mirror_hemi",
                                                    "sibling") else ""
        b.text(3.05, yi, f"{e:.2f}{shared}", va="center", ha="right", fontsize=5.8, color=INK)
    for x in (1, 2):
        b.axvline(x, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for h in (4.5, 9.5):     # same person | another person | averages
        b.axhline(h, color=GRID, lw=0.6)
    b.set_yticks(y, [r[2] for r in THEORY_ROWS], fontsize=6)
    b.set_ylim(-0.6, len(THEORY_ROWS) - 0.3)
    b.set_xlim(0, 3.1)
    b.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    b.scatter([], [], s=24, marker="D", facecolor="white", edgecolor=MUTED, lw=0.8, label="model prediction")
    b.scatter([], [], s=12, color=MUTED, label="measured, L+R (95% CI)")
    b.scatter([], [], s=7, marker="<", color=MUTED, label="left")
    b.scatter([], [], s=7, marker=">", color=MUTED, label="right")
    b.legend(frameon=False, fontsize=5.2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
             handletextpad=0.2, columnspacing=0.8)
    b.set_xlabel("Error, relative to the dataset mean (lower = 'better')")
    b.set_title("(b) Single donors and averages", fontsize=7, loc="left", pad=15)
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005, f"{region_title}. * upper bound: neighbours chosen with the answer. "
             "No closed form for search or look-alike fills. Pre-QC, affine registration only.",
             ha="center", fontsize=5.4, color=MUTED)
    fig.suptitle("Model and measurement on real brains, left and right", fontsize=9, y=0.995)
    fig.savefig(dest_png, dpi=250)
    fig.savefig(dest_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    res = json.load(open(L.OUT_JSON, encoding="utf-8"))
    regions = res["QTIM"]["regions"]
    figs = os.path.join(ROOT, "results", "figures")
    for reg, title in (("parietal", "Parietal box"), ("random_patches", "32 random 16 mm cubes")):
        views = {v: regions[f"{reg}/{v}"]["cohorts"] for v in ("LR", "L", "R")}
        out = os.path.join(figs, f"fig_real_theory_v2_{reg}")
        theory(views, title, out + ".pdf", out + ".png")
    if "--theory-only" in sys.argv:
        return
    group = regions["parietal/LR"]["cohorts"]
    zv, start, size, fills, info = case()
    out = os.path.join(ROOT, "results", "figures", "fig_real_case_parietal_v2")
    draw(zv, start, size, fills, group, out + ".pdf", out + ".png")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
