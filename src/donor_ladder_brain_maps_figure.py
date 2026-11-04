"""Figure of the brain-wide variability maps (Amendment 5). Pre-QC, exploratory.

Columns: V (individual variation, z units) and the error of each fixed donor in units of V (1 = the dataset mean,
2 = a stranger). Rows: axial slices at z = 55 mm (parietal box level) and z = 10 mm (deep box level), and a
sagittal slice at x = -30 mm. Underlay: the nilearn MNI152 template. Under each column: pooled ratio over the map
and the split-half correlation. Output: results/figures/fig_variability_maps.{png,pdf}.
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
import donor_ladder_brain_maps as P  # noqa: E402
from make_figures import INK, MUTED  # noqa: E402  (also applies the paper rcParams)

COLUMNS = [("V", "Individual variation V\n(z units)"), ("repeat", "Repeat scan"), ("mirror", "Own mirror"),
           ("sibling", "Sibling"), ("stranger", "Stranger")]
SLICES = [("z", 55, "axial z = 55 mm"), ("z", 10, "axial z = 10 mm"), ("x", -30, "sagittal x = -30 mm")]


def cut(img, axis, mm):
    """2D slice of a volume at a world coordinate, oriented for display (neurological: left on the left)."""
    inv = np.linalg.inv(img.affine)
    k = int(round((inv @ np.array([mm if axis == "x" else 0, 0, mm if axis == "z" else 0, 1]))["xyz".index(axis)]))
    a = np.asanyarray(img.dataobj, dtype=np.float32)
    return (a[:, :, k] if axis == "z" else a[k]).T


def main():
    from nilearn.datasets import load_mni152_template
    from nilearn.image import resample_to_img
    res = json.load(open(P.OUT_JSON, encoding="utf-8"))
    maps = {"V": nib.load(os.path.join(P.PRIVATE, "V.nii.gz"))}
    for d in P.DONORS:
        maps[d] = nib.load(os.path.join(P.PRIVATE, f"{d}_ratio.nii.gz"))
    under = resample_to_img(load_mni152_template(resolution=1), maps["V"], interpolation="continuous")
    fig, axes = plt.subplots(len(SLICES), len(COLUMNS), figsize=(7.0, 4.9),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    fig.subplots_adjust(top=0.88, bottom=0.17, left=0.07, right=0.99)
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.4)
    for j, (key, title) in enumerate(COLUMNS):
        for i, (axis, mm, lab) in enumerate(SLICES):
            ax = axes[i, j]
            ax.imshow(cut(under, axis, mm), origin="lower", cmap="gray", interpolation="nearest")
            m = cut(maps[key], axis, mm)
            m = np.ma.masked_invalid(np.where(m == 0, np.nan, m)) if key == "V" else np.ma.masked_invalid(m)
            if key == "V":
                im_v = ax.imshow(m, origin="lower", cmap="viridis", alpha=0.85, interpolation="nearest")
            else:
                im_r = ax.imshow(m, origin="lower", cmap="RdBu_r", norm=norm, alpha=0.85, interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j == 0:
                ax.set_ylabel(lab, fontsize=6)
        axes[0, j].set_title(title, fontsize=6.8, color=INK, pad=3, linespacing=1.0)
        if key != "V":
            d = res["maps"][key]
            rel = ("≈ 2 by construction" if key == "stranger"  # exact expectation: split-half r is not informative
                   else f"split-half r {d['split_half_r']:.2f}")
            axes[-1, j].set_xlabel(f"pooled {d['pooled_ratio']:.2f}\n{rel}", fontsize=5.8, color=INK, labelpad=2)
    cb1 = fig.add_axes([0.08, 0.075, 0.15, 0.015])
    fig.colorbar(im_v, cax=cb1, orientation="horizontal").ax.tick_params(labelsize=5)
    cb1.set_title("V", fontsize=5.6, pad=2)
    cb2 = fig.add_axes([0.42, 0.075, 0.45, 0.015])
    cb = fig.colorbar(im_r, cax=cb2, orientation="horizontal")
    cb.ax.tick_params(labelsize=5)
    cb.set_ticks([0, 0.5, 1, 1.5, 2])
    cb2.set_title("error / dataset-mean error  (blue: beats the mean, red: worse; 2 = stranger)", fontsize=5.6, pad=2)
    fig.text(0.5, 0.012, "QTIM session 1; 16 mm window on a 2 mm grid; fixed donors, no selection. Mirror map: "
             "|x| < 8 mm excluded. Pre-QC, affine registration only: maps mix anatomy with misregistration.",
             ha="center", fontsize=5.4, color=MUTED)
    fig.suptitle("How alike each part of the brain is: same person, own mirror, sibling, stranger", fontsize=8.5)
    out = os.path.join(ROOT, "results", "figures", "fig_variability_maps")
    fig.savefig(out + ".png", dpi=250)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
