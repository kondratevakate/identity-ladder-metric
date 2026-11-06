"""Fundus ladder maps (general protocol, section 9): the case-figure columns over the whole fundus frame. Pre-QC.

Columns: Target (mean canonical fundus of the dataset, anatomy only), Repeat (other visit; GRAPE only), Mirror side
(fellow eye at the same place), Stranger (exact expectation), Same sex and age, Dataset mean (V itself, z units).
Look-alikes need a search per location and are not mapped here. Rows: one per dataset. `common` uses one scale
(1 = dataset mean, 2 = stranger); `contrast` gives each map its own 2nd-98th percentile range around its median and
fades pixels where the two independent person halves fall on opposite sides of the median. Under each map: median
ratio over the frame and split-half r. Output: results/figures/fig_ladder2d_maps_{common,contrast}.{png,pdf}.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "paper", "isbi2027"))
import donor_ladder_2d as D  # noqa: E402
import donor_ladder_fundus as F  # noqa: E402
from make_figures import INK, MUTED  # noqa: E402  (also applies the paper rcParams)

COLUMNS = [("target", "Target"), ("other_visit", "Repeat (other visit)"), ("fellow_eye", "Mirror side (fellow eye)"),
           ("stranger", "Stranger"), ("cohort", "Same sex and age"), ("V", "Dataset mean")]


def mean_image(name, tag=""):
    path = (os.path.join(F.PRIVATE, f"{name}_canonical.npz") if not tag else
            os.path.join(D.PRIVATE, f"{name}_canonical{tag}.npz"))
    img, fov, ok, _ = F.load_all(None, path)
    m = img[ok].mean(0)
    return np.where(fov[ok].mean(0) > 0.5, m, np.nan)


def draw(mode, res, dest, tag=""):
    names = [n for n in ("PAPILA", "GRAPE") if os.path.exists(os.path.join(D.PRIVATE, f"{n}_maps{tag}.npz"))]
    h = 1.75 * len(names) + 0.7
    fig, axes = plt.subplots(len(names), len(COLUMNS), figsize=(7.2, h), squeeze=False,
                             gridspec_kw={"wspace": 0.05, "hspace": 0.9})
    fig.subplots_adjust(top=1 - 0.42 / h, bottom=0.62 / h + 0.06, left=0.06, right=0.995)
    common = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.4)
    for r, name in enumerate(names):
        z = np.load(os.path.join(D.PRIVATE, f"{name}_maps{tag}.npz"))
        under = mean_image(name, tag)
        rel = res[name]["maps"]
        shown = np.isfinite(z["all_V"])                         # crop every panel to the mapped area
        ys, xs = np.flatnonzero(shown.any(1)), np.flatnonzero(shown.any(0))
        crop = (slice(max(ys[0] - 12, 0), ys[-1] + 13), slice(max(xs[0] - 12, 0), xs[-1] + 13))
        under = under[crop]
        for j, (key, title) in enumerate(COLUMNS):
            ax = axes[r, j]
            ax.imshow(under, cmap="gray", interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=6.2, color=INK, pad=2)
            if j == 0:
                ax.set_ylabel(name, fontsize=6.4)
            if key == "target":
                ax.set_xlabel("mean fundus\n(anatomy)", fontsize=5.4, color=INK, labelpad=2)
                continue
            m = z[f"all_{key}"][crop]
            fin = m[np.isfinite(m)]
            if fin.size == 0:
                ax.set_xlabel("not available", fontsize=5.4, color=MUTED, labelpad=2)
                continue
            if key == "V":
                lo, hi = np.percentile(fin, [2, 98])
                norm, cmap = plt.Normalize(lo, hi), "viridis"
            elif mode == "common":
                norm, cmap = common, "RdBu_r"
            else:
                lo, med, hi = np.percentile(fin, [2, 50, 98])
                norm, cmap = TwoSlopeNorm(vmin=lo, vcenter=med, vmax=max(hi, med + 1e-6)), "PuOr_r"
            mm = np.ma.masked_invalid(m)
            if mode == "contrast" and key != "V":
                med = float(np.median(fin))
                agree = np.sign(z[f"h0_{key}"][crop] - med) == np.sign(z[f"h1_{key}"][crop] - med)
                ax.imshow(mm, cmap=cmap, norm=norm, alpha=0.25, interpolation="nearest")
                im = ax.imshow(np.ma.masked_where(~agree, mm), cmap=cmap, norm=norm, alpha=0.95,
                               interpolation="nearest")
            else:
                im = ax.imshow(mm, cmap=cmap, norm=norm, alpha=0.9, interpolation="nearest")
            pos = ax.get_position()
            cb = fig.colorbar(im, cax=fig.add_axes([pos.x0 + 0.01, pos.y0 - 0.035, pos.width - 0.02, 0.012]),
                              orientation="horizontal")
            cb.ax.tick_params(labelsize=4.4, length=1.5, pad=1)
            cb.ax.locator_params(nbins=3)
            if key == "V":
                ax.text(0.5, -0.32, "MSE = 1 × mean\n(shown: V, z units)", transform=ax.transAxes, ha="center",
                        va="top", fontsize=5.2, color=INK)
            else:
                rr = rel[key]
                sh = rr["split_half_r"]
                note = ("≈ 2 by construction" if key == "stranger" else
                        "split-half r n/a" if sh is None else f"split-half r {sh:.2f}")
                ax.text(0.5, -0.32, f"median {rr['median_ratio']:.2f} × mean\n{note}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=5.2, color=INK)
    scale = ("One scale: 1 = dataset mean (white), 2 = stranger; blue beats the mean." if mode == "common" else
             "Own range per map around its median (white); faded where two independent person halves disagree.")
    prep = ("P7: PAPILA expert-disc translation frame, GRAPE disc-fovea frame; green / background, z-scored"
            if tag == "_p7" else "P0: disc-fovea frame (fovea unreliable); green z-scored")
    fig.text(0.5, 0.03, f"{scale} {prep} in the field of view, 48 px window on the "
             "256 grid. Pre-QC.", ha="center", fontsize=5.0, color=MUTED)
    fig.suptitle(f"Fundus{(' ' + tag[1:].upper()) if tag else ''}, every location: error of each donor in units of the dataset mean"
                 + ("" if mode == "common" else " (contrast per map)"), fontsize=8.2, y=0.99)
    fig.savefig(dest + ".png", dpi=250)
    fig.savefig(dest + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    tag = next((f"_{a.split('=', 1)[1]}" for a in sys.argv if a.startswith("--arm=")), "")   # preprocessing version
    res = json.load(open(D.OUT_JSON.replace(".json", f"{tag}.json"), encoding="utf-8"))["datasets"]
    for mode in ("common", "contrast"):
        draw(mode, res, os.path.join(ROOT, "results", "figures", f"fig_ladder2d_maps{tag}_{mode}"), tag)


if __name__ == "__main__":
    main()
