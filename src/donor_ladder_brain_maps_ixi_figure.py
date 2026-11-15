"""IXI brain maps in the layout of the QTIM figure `fig_ladder_maps_common` (Amendment 25). Pre-QC.

Same columns, slices, common diverging scale (1 = dataset mean, white) and Gaussian maps as
`donor_ladder_brain_ladder_maps_figure.py`; maps from `donor_ladder_brain_maps_ixi.py`. Columns IXI cannot have
(repeat: no repeat scans; sibling: no families; same sex and age when no target has 10 such donors) show the
template with "n/a" and the reason, never a value.
Usage: python -I src/donor_ladder_brain_maps_ixi_figure.py [--smoke N]
Outputs: results/figures/fig_ladder_maps_common_ixi.{png,pdf} (smoke: the private smoke folder).
"""
import argparse
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
sys.path.insert(0, HERE)
import torchvision_guard  # noqa: E402
torchvision_guard.ensure()   # a figure import pulls torchvision; Application Control may block its _C.pyd
import donor_ladder_brain_ladder_maps_figure as LF  # noqa: E402  (applies the paper rcParams)
import donor_ladder_brain_maps_ixi as X  # noqa: E402
from donor_ladder_brain_maps_figure import SLICES, cut  # noqa: E402

COLUMNS = LF.COLUMNS


def load(d):
    from nilearn.datasets import load_mni152_template
    from nilearn.image import resample_to_img
    ref = nib.load(os.path.join(d, "V.nii.gz"))
    maps = {}
    for key, _, f, _ in COLUMNS:
        if f is None or not os.path.exists(os.path.join(d, f)):
            continue
        img = nib.load(os.path.join(d, f))
        maps[key] = LF.upsample(img, ref) if f.startswith("grid_") else img
    under = resample_to_img(load_mni152_template(resolution=1), ref, interpolation="continuous",
                            force_resample=True, copy_header=True)
    return maps, under


def draw(maps, under, rel, na, dest, label):
    fig, axes = plt.subplots(len(SLICES), len(COLUMNS), figsize=(7.2, 4.6), gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    fig.subplots_adjust(top=0.82, bottom=0.2, left=0.05, right=0.995)
    common = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.4)
    for j, (key, title, _, src) in enumerate(COLUMNS):
        axes[0, j].set_title(title, fontsize=6.6, color=LF.INK, pad=2)
        for i, (axis, mm, lab) in enumerate(SLICES):
            axes[i, j].imshow(cut(under, axis, mm), origin="lower", cmap="gray", interpolation="nearest")
            LF.frame(axes[i, j])
            if j == 0:
                axes[i, j].set_ylabel(lab, fontsize=5.6)
        if key == "target":
            LF.below(fig, axes[-1, j], "MNI template\n(anatomy)")
            continue
        if key not in maps:
            for i in range(len(SLICES)):
                axes[i, j].text(0.5, 0.5, "n/a", transform=axes[i, j].transAxes, ha="center", va="center",
                                fontsize=8, color="#7a2e0e", fontweight="bold")
            LF.below(fig, axes[-1, j], "n/a in IXI\n" + na.get(key, "not available"))
            continue
        data = np.asanyarray(maps[key].dataobj, dtype=np.float32)
        fin = data[np.isfinite(data) & ((data != 0) if key == "V" else True)]
        if key == "V":
            lo, hi = np.percentile(fin, [2, 98])
            norm, cmap = plt.Normalize(lo, hi), "viridis"
        else:
            norm, cmap = common, "RdBu_r"
        for i, (axis, mm, _) in enumerate(SLICES):
            m = cut(maps[key], axis, mm)
            m = np.ma.masked_invalid(np.where(m == 0, np.nan, m) if key == "V" else m)
            im = axes[i, j].imshow(m, origin="lower", cmap=cmap, norm=norm, alpha=0.9, interpolation="nearest")
        pos = axes[-1, j].get_position()
        cb = fig.colorbar(im, cax=fig.add_axes([pos.x0 + 0.006, 0.12, pos.width - 0.012, 0.012]),
                          orientation="horizontal")
        cb.ax.tick_params(labelsize=4.6, length=1.5, pad=1)
        cb.ax.locator_params(nbins=3)
        if key == "V":
            LF.below(fig, axes[-1, j], "MSE = 1 × mean\n(shown: V, z units)")
        else:
            r = rel[src[0]].get(src[1], {})
            sh = r.get("split_half_r")
            note = "≈ 2 by construction" if key == "stranger" else (f"split-half r {sh:.2f}" if sh is not None else
                                                                   "split-half r n/a")
            LF.below(fig, axes[-1, j], f"MSE = {r['pooled_ratio']:.2f} × mean\n{note}")
    for a, b, text in LF.GROUPS:
        x0, x1 = axes[0, a].get_position().x0 + 0.003, axes[0, b].get_position().x1 - 0.003
        y = 0.895
        fig.add_artist(plt.Line2D([x0, x0, x1, x1], [y - 0.012, y, y, y - 0.012], color=LF.MUTED, lw=0.6))
        fig.text((x0 + x1) / 2, y + 0.006, text, ha="center", va="bottom", fontsize=6.4, color=LF.MUTED)
    fig.text(0.5, 0.03, "One scale: 1 = dataset mean (white), 2 = stranger; blue beats the mean.", ha="center",
             fontsize=5.0, color=LF.MUTED)
    fig.text(0.5, 0.008, f"* 8 mm grid, Gaussian-weighted 16 mm cube, neighbours chosen by the 4 mm shell only. {label}. "
             "Pre-QC, affine registration only.", ha="center", fontsize=5.0, color=LF.MUTED)
    fig.suptitle("IXI brain, every location: the same Gaussian maps as QTIM", fontsize=8.2, y=0.99)
    fig.savefig(dest + ".png", dpi=250)
    fig.savefig(dest + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        d = os.path.join(X.PRIVATE, f"smoke{a.smoke}")
        rel = {"fine": json.load(open(os.path.join(d, "summary.json")))["maps"],
               "grid": json.load(open(os.path.join(d, "grid_summary.json")))["maps"]}
        dest, label = os.path.join(d, "qc_maps"), f"IXI smoke, first {a.smoke} scans"
    else:
        d = os.path.join(X.PRIVATE, "maps")
        rel = {"fine": json.load(open(X.OUT_FINE))["maps"], "grid": json.load(open(X.OUT_GRID))["maps"]}
        dest, label = os.path.join(LF.ROOT, "results", "figures", "fig_ladder_maps_common_ixi"), "IXI, 100 scans"
    na = {"repeat": "(no repeat scans)", "sibling": "(no families)"}
    coh = rel["fine"].get("cohort", {})
    if coh.get("status") == "n/a":
        na["cohort"] = "(< 10 same-sex donors\nwithin 2 years)"
    for key, _, _, src in COLUMNS:
        if src and key not in na and rel[src[0]].get(src[1], {}).get("pooled_ratio") is None:
            na[key] = "(no target has\nenough donors)"
    maps, under = load(d)
    for k in [k for k in maps if k in na]:
        del maps[k]
    draw(maps, under, rel, na, dest, label)
    print(dest + ".png")


if __name__ == "__main__":
    main()
