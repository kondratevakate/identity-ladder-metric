"""The columns of the ISBI Fig. 1 as brain maps, on a common scale and with a per-map range (Amendment 7). Pre-QC.

Columns as in the case figure: Target (anatomy: the MNI template), Repeat scan, Mirror side, Sibling, Stranger,
16 look-alikes, Same sex and age, Dataset mean. Fixed donors come from the 2 mm maps of Amendments 5-6; 16
look-alikes (4 mm shell) from the 8 mm grid of Amendment 7, shown with nearest-neighbour upsampling. The dataset
mean has ratio 1 everywhere by definition, so its column shows V itself (z units). `common`: one diverging scale
(1 = dataset mean, 2 = stranger). `contrast`: each map spans its own 2nd-98th percentile around its median, so
regional differences inside a map are visible. Under each map: MSE over the whole map in units of the mean, and
the split-half r over two family halves (how much of the pattern replicates). Also writes the agreement between
the 8 mm grid and the 2 mm maps for the fixed donors into the grid JSON.
Outputs: results/figures/fig_ladder_maps_{common,contrast}.{png,pdf}.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "paper", "isbi2027"))
import donor_ladder_brain_map_grid as G  # noqa: E402
import donor_ladder_brain_maps as P  # noqa: E402
from make_figures import INK, MUTED  # noqa: E402  (also applies the paper rcParams)
from donor_ladder_brain_maps_figure import SLICES, cut  # noqa: E402

# key, title, source file, split-half source ("fine" 2 mm maps or "grid" 8 mm JSON key)
COLUMNS = [("target", "Target", None, None),
           ("repeat", "Repeat scan", "repeat_ratio.nii.gz", ("fine", "repeat")),
           ("mirror", "Mirror side", "mirror_ratio.nii.gz", ("fine", "mirror")),
           ("sibling", "Sibling", "sibling_ratio.nii.gz", ("fine", "sibling")),
           ("stranger", "Stranger", "stranger_ratio.nii.gz", ("fine", "stranger")),
           ("knn16_s4", "16 look-alikes*", "grid_knn16_s4_ratio.nii.gz", ("grid", "knn16_s4")),
           ("cohort", "Same sex and age", "cohort_ratio.nii.gz", ("fine", "cohort")),
           ("V", "Dataset mean", "V.nii.gz", None)]
GROUPS = [(1, 4, "Real anatomy from another scan or person"), (5, 7, "Averages of other people")]


def load_maps():
    from nilearn.datasets import load_mni152_template
    from nilearn.image import resample_to_img
    ref = nib.load(os.path.join(P.PRIVATE, "V.nii.gz"))
    maps = {}
    for key, _, f, _ in COLUMNS:
        if f is None:
            continue
        img = nib.load(os.path.join(P.PRIVATE, f))
        if f.startswith("grid_"):
            img = resample_to_img(img, ref, interpolation="nearest", force_resample=True, copy_header=True)
        maps[key] = img
        hf = [os.path.join(P.PRIVATE, f.replace("_ratio.nii.gz", f"_ratio_h{h}.nii.gz")) for h in (0, 1)]
        if key != "V" and all(os.path.exists(x) for x in hf):          # independent half maps (Amendment 8)
            hs = [nib.load(x) for x in hf]
            if f.startswith("grid_"):
                hs = [resample_to_img(x, ref, interpolation="nearest", force_resample=True, copy_header=True)
                      for x in hs]
            maps[key + "_halves"] = hs
    under = resample_to_img(load_mni152_template(resolution=1), ref, interpolation="continuous",
                            force_resample=True, copy_header=True)
    return maps, under


def cross_check():
    """Voxel-wise correlation between the 8 mm grid and the 2 mm map at the grid centres, per fixed donor."""
    from nilearn.image import resample_to_img
    res = {}
    for f in ("repeat", "mirror", "sibling", "stranger", "cohort"):
        g = nib.load(os.path.join(P.PRIVATE, f"grid_{f}_ratio.nii.gz"))
        fine = resample_to_img(nib.load(os.path.join(P.PRIVATE, f"{f}_ratio.nii.gz")), g, interpolation="nearest",
                               force_resample=True, copy_header=True)
        a, b = np.asanyarray(g.dataobj).ravel(), np.asanyarray(fine.dataobj).ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        res[f] = {"r": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else None,
                  "median_grid": float(np.median(a[ok])), "median_fine": float(np.median(b[ok])),
                  "centres": int(ok.sum())}
    return res


def frame(ax):
    ax.set(xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)


def below(fig, ax, text):
    fig.text((ax.get_position().x0 + ax.get_position().x1) / 2, 0.095, text, ha="center", va="top", fontsize=5.4,
             color=INK)


def draw(maps, under, rel, mode, dest):
    fig, axes = plt.subplots(len(SLICES), len(COLUMNS), figsize=(7.2, 4.6),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    fig.subplots_adjust(top=0.82, bottom=0.2, left=0.05, right=0.995)
    common = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.4)
    for j, (key, title, _, src) in enumerate(COLUMNS):
        axes[0, j].set_title(title, fontsize=6.6, color=INK, pad=2)
        if key == "target":
            for i, (axis, mm, lab) in enumerate(SLICES):
                axes[i, j].imshow(cut(under, axis, mm), origin="lower", cmap="gray", interpolation="nearest")
                frame(axes[i, j])
                axes[i, j].set_ylabel(lab, fontsize=5.6)
            below(fig, axes[-1, j], "MNI template\n(anatomy)")
            continue
        data = np.asanyarray(maps[key].dataobj, dtype=np.float32)
        fin = data[np.isfinite(data) & ((data != 0) if key == "V" else True)]
        if key == "V":
            lo, hi = np.percentile(fin, [2, 98])
            norm, cmap = plt.Normalize(lo, hi), "viridis"
        elif mode == "common":
            norm, cmap = common, "RdBu_r"
        else:
            lo, med, hi = np.percentile(fin, [2, 50, 98])
            norm, cmap = TwoSlopeNorm(vmin=lo, vcenter=med, vmax=max(hi, med + 1e-6)), "PuOr_r"
        for i, (axis, mm, _) in enumerate(SLICES):
            ax = axes[i, j]
            ax.imshow(cut(under, axis, mm), origin="lower", cmap="gray", interpolation="nearest")
            m = cut(maps[key], axis, mm)
            m = np.ma.masked_invalid(np.where(m == 0, np.nan, m) if key == "V" else m)
            halves = maps.get(key + "_halves")
            if mode == "contrast" and halves is not None:
                # faded where the two independent halves deviate from the map's median in opposite directions
                med = float(np.median(fin))
                agree = np.sign(cut(halves[0], axis, mm) - med) == np.sign(cut(halves[1], axis, mm) - med)
                ax.imshow(m, origin="lower", cmap=cmap, norm=norm, alpha=0.25, interpolation="nearest")
                im = ax.imshow(np.ma.masked_where(~agree, m), origin="lower", cmap=cmap, norm=norm, alpha=0.95,
                               interpolation="nearest")
            else:
                im = ax.imshow(m, origin="lower", cmap=cmap, norm=norm, alpha=0.9, interpolation="nearest")
            frame(ax)
        pos = axes[-1, j].get_position()
        cb = fig.colorbar(im, cax=fig.add_axes([pos.x0 + 0.006, 0.12, pos.width - 0.012, 0.012]),
                          orientation="horizontal")
        cb.ax.tick_params(labelsize=4.6, length=1.5, pad=1)
        cb.ax.locator_params(nbins=3)
        if key == "V":
            below(fig, axes[-1, j], "MSE = 1 × mean\n(shown: V, z units)")
        else:
            r = rel[src[0]][src[1]]
            note = "≈ 2 by construction" if key == "stranger" else f"split-half r {r['split_half_r']:.2f}"
            below(fig, axes[-1, j], f"MSE = {r['pooled_ratio']:.2f} × mean\n{note}")
    for a, b, text in GROUPS:
        x0, x1 = axes[0, a].get_position().x0 + 0.003, axes[0, b].get_position().x1 - 0.003
        y = 0.895
        fig.add_artist(plt.Line2D([x0, x0, x1, x1], [y - 0.012, y, y, y - 0.012], color=MUTED, lw=0.6))
        fig.text((x0 + x1) / 2, y + 0.006, text, ha="center", va="bottom", fontsize=6.4, color=MUTED)
    scale = ("One scale: 1 = dataset mean (white), 2 = stranger; blue beats the mean." if mode == "common" else
             "Own range per map: 2nd-98th percentile around its median (white); faded where two independent family "
             "halves fall on opposite sides of the median (look-alikes: not yet, halves pending).")
    fig.text(0.5, 0.03, scale, ha="center", fontsize=5.0, color=MUTED)
    fig.text(0.5, 0.008, "* 8 mm grid, neighbours chosen by the 4 mm shell only. QTIM session 1, 16 mm window. "
             "Pre-QC, affine registration only.", ha="center", fontsize=5.0, color=MUTED)
    fig.suptitle("Real brain, every location: pixel error ranks the dataset mean above related anatomy"
                 + ("" if mode == "common" else " (contrast per map)"), fontsize=8.2, y=0.99)
    fig.savefig(dest + ".png", dpi=250)
    fig.savefig(dest + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def core(maps, under, rel, dest):
    """Repeat and mirror on the common scale next to the number of targets per voxel (coverage)."""
    cols = [("repeat", "Repeat scan", "ratio"), ("mirror", "Mirror side", "ratio"),
            ("repeat", "Repeat: targets per voxel", "n"), ("mirror", "Mirror: targets per voxel", "n")]
    fig, axes = plt.subplots(len(SLICES), len(cols), figsize=(4.6, 4.2), gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    fig.subplots_adjust(top=0.9, bottom=0.2, left=0.08, right=0.99)
    common = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.4)
    for j, (key, title, kind) in enumerate(cols):
        img = maps[key] if kind == "ratio" else nib.load(os.path.join(P.PRIVATE, f"{key}_n.nii.gz"))
        nmax = rel["fine"][key]["max_targets_per_voxel"]
        norm, cmap = (common, "RdBu_r") if kind == "ratio" else (plt.Normalize(0, nmax), "cividis")
        for i, (axis, mm, lab) in enumerate(SLICES):
            ax = axes[i, j]
            ax.imshow(cut(under, axis, mm), origin="lower", cmap="gray", interpolation="nearest")
            m = cut(img, axis, mm)
            m = np.ma.masked_invalid(np.where(m == 0, np.nan, m))
            im = ax.imshow(m, origin="lower", cmap=cmap, norm=norm, alpha=0.9, interpolation="nearest")
            frame(ax)
            if j == 0:
                ax.set_ylabel(lab, fontsize=5.6)
        axes[0, j].set_title(title, fontsize=6.4, color=INK, pad=2)
        pos = axes[-1, j].get_position()
        cb = fig.colorbar(im, cax=fig.add_axes([pos.x0 + 0.01, 0.12, pos.width - 0.02, 0.014]), orientation="horizontal")
        cb.ax.tick_params(labelsize=4.8, length=1.5, pad=1)
        cb.ax.locator_params(nbins=3)
        r = rel["fine"][key]
        below(fig, axes[-1, j], (f"MSE = {r['pooled_ratio']:.2f} × mean\nsplit-half r {r['split_half_r']:.2f}"
                                 if kind == "ratio" else f"max {nmax} targets"))
    fig.text(0.5, 0.012, "1 = dataset mean (white); 16 mm window, QTIM session 1. Pre-QC, affine registration only.",
             ha="center", fontsize=5.0, color=MUTED)
    fig.suptitle("Same person and own mirror, with coverage", fontsize=8.2, y=0.99)
    fig.savefig(dest + ".png", dpi=250)
    fig.savefig(dest + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    rel = {"fine": json.load(open(P.OUT_JSON, encoding="utf-8"))["maps"],
           "grid": json.load(open(G.OUT_JSON, encoding="utf-8"))["maps"]}
    maps, under = load_maps()
    for mode in ("common", "contrast"):
        draw(maps, under, rel, mode, os.path.join(ROOT, "results", "figures", f"fig_ladder_maps_{mode}"))
    core(maps, under, rel, os.path.join(ROOT, "results", "figures", "fig_ladder_maps_core"))
    g = json.load(open(G.OUT_JSON, encoding="utf-8"))
    g["cross_check_grid_vs_2mm"] = cross_check()
    with open(G.OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(g, fh, indent=1, allow_nan=False)
    print(json.dumps(g["cross_check_grid_vs_2mm"]))


if __name__ == "__main__":
    main()
