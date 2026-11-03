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
from make_figures import INK, MUTED, ORANGE  # noqa: E402  (also applies the paper rcParams)

CROP = (slice(30, 170), slice(20, 210))
DISPLAY = (-2.5, 2.5)
# key, title, (cohort, k) in the bilateral JSON; groups follow in GROUPS
COLUMNS = [("target", "Target", None),
           ("repeat", "Repeat\nscan", ("repeat", "1")),
           ("other_hemisphere", "Mirror,\nsame place", ("other_hemisphere", "1")),
           ("mirror_block", "Mirror,\nbest of ±8 mm", ("mirror_block", "1")),
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
    a = L.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    allowed = L.allowed_offsets(start[0], size[0], a)
    other = np.asarray(vol[ix[i], 1])
    _, block = L.block_search(win, other, size, *L.search_terms(other, np.asarray(mask[ix[i], 1]), size), allowed)
    so = [j for j in order if mvalid[j]]
    fills = {"target": X[i], "repeat": X_all[jr], "other_hemisphere": M[i], "mirror_block": block.ravel(),
             "sibling": X[js], "stranger": X[order[0]], "stranger_other_side": M[so[0]],
             "knn_oracle": X[nn(((X - X[i]) ** 2).sum(1))].mean(0), "knn": X[nn(((C - C[i]) ** 2).sum(1))].mean(0),
             "knn_mirror": X[nn(((X - M[i]) ** 2).sum(1))].mean(0),
             "cohort": X[coh].mean(0), "population": X[others].mean(0)}
    row = t.loc[ix[i]]
    v = nib.load(row.path.replace("_brain.nii.gz", ".nii.gz")).get_fdata(dtype=np.float32)
    m = nib.load(row.path).get_fdata(dtype=np.float32) > 0.5
    zv = np.where(m, (v - v[m].mean()) / (v[m].std() + 1e-6), DISPLAY[0])
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


def main():
    res = json.load(open(L.OUT_JSON, encoding="utf-8"))
    group = res["QTIM"]["regions"]["parietal/LR"]["cohorts"]
    zv, start, size, fills, info = case()
    out = os.path.join(ROOT, "results", "figures", "fig_real_case_parietal_v2")
    draw(zv, start, size, fills, group, out + ".pdf", out + ".png")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
