"""Fundus ladder under the general protocol: sensitivity to patch size and ring width (section 9). Pre-QC.

Reads results/donor_ladder_2d.json (L+R views). One row per dataset: (a) against the ring width at a 48 px
patch, (b) against the patch size at a 16 px ring, on the 32 random patches of the 256 grid. Bands: 95% person
bootstrap. Output: results/figures/fig_ladder2d_sensitivity.{png,pdf}.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "paper", "isbi2027"))
import donor_ladder_2d as D  # noqa: E402
from make_figures import INK, MUTED  # noqa: E402  (also applies the paper rcParams)

FILLS = [("population", "1", "Stranger", "#2a78d6", "-"),
         ("stranger_other_side", "1", "Stranger, other eye", "#2a78d6", ":"),
         ("stranger_block", "1", "Stranger, best of ±8 px", "#2a78d6", "--"),
         ("fellow_eye", "1", "Fellow eye, same place", "#e87ba4", "-"),
         ("fellow_block", "1", "Fellow eye, best of ±8 px", "#e87ba4", "--"),
         ("fellow_block_oracle", "1", "Fellow eye, ±8 px by true patch*", "#e87ba4", ":"),
         ("other_visit", "1", "Other visit (GRAPE)", "#008300", "-"),
         ("knn", "16", "16 look-alikes by ring", "#eda100", "--"),
         ("knn_oracle", "16", "16 look-alikes by true patch*", "#eda100", ":"),
         ("knn_fellow", "16", "16 look-alikes by fellow eye", "#eda100", "-."),
         ("cohort", "all", "Same sex and age", "#1baf7a", "-")]


def series(configs, keys, c, k):
    out = []
    for key in keys:
        v = configs.get(key, {}).get("random_patches/LR", {}).get("cohorts", {}).get(c, {}).get(k)
        out.append((np.nan,) * 3 if v is None or v["err_over_V"] is None else (v["err_over_V"], *v["ci95"]))
    return np.array(out)


def main():
    res = json.load(open(D.OUT_JSON, encoding="utf-8"))["datasets"]
    names = [n for n in ("PAPILA", "GRAPE") if n in res]
    fig, axes = plt.subplots(len(names), 2, figsize=(7.0, 2.6 * len(names) + 0.6), squeeze=False,
                             gridspec_kw={"wspace": 0.28, "hspace": 0.55})
    fig.subplots_adjust(top=0.84, bottom=0.2 if len(names) == 1 else 0.14, left=0.08, right=0.72)
    for r, name in enumerate(names):
        cfg = res[name]["configs"]
        for col, (xs, keyf, xlabel) in enumerate(((D.RINGS, lambda v: f"p48_r{v}", "Ring width, px (patch 48 px)"),
                                                  (D.PATCHES, lambda v: f"p{v}_r16", "Patch, px (ring 16 px)"))):
            ax = axes[r, col]
            for c, k, lab, colr, ls in FILLS:
                y = series(cfg, [keyf(x) for x in xs], c, k)
                if np.all(np.isnan(y[:, 0])):
                    continue
                ax.plot(xs, y[:, 0], ls=ls, color=colr, lw=1.0, marker="o", ms=2.5, label=lab)
                ax.fill_between(xs, y[:, 1], y[:, 2], color=colr, alpha=0.12, lw=0)
            ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (2, 2)))
            ax.set_xticks(xs)
            ax.set_xlabel(xlabel)
            ax.set_title(f"({'ab'[col]}) {name}", fontsize=7, loc="left")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[r, 0].set_ylabel("Error / dataset-mean error")
    h, lab = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, frameon=False, fontsize=5.6, loc="center left", bbox_to_anchor=(0.73, 0.55))
    fig.text(0.5, 0.02, "Fundus, disc-right frame, green channel z-scored in the field of view; 32 random patches, "
             "mean of right and left eyes; bands: 95% person bootstrap. * upper bound: chosen with the answer. "
             "Pre-QC.", ha="center", fontsize=5.4, color=MUTED, wrap=True)
    fig.suptitle("Fundus ladder: patch size, ring width and the selection controls", fontsize=9, color=INK, y=0.99)
    out = os.path.join(ROOT, "results", "figures", "fig_ladder2d_sensitivity")
    fig.savefig(out + ".png", dpi=250)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
