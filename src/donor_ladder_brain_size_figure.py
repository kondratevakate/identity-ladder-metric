"""Dependence of the donor-ladder errors on cube size, shell thickness and their ratio (Amendment 4). Pre-QC.

Reads results/donor_ladder_brain_size.json (L+R aggregates). Top row, 32 random cubes: (a) against cube edge at a
4 mm shell, (b) against shell at a 16 mm cube, (c) the shell-based fills against shell / cube for all 20
configurations. Bottom row: the frozen deep and parietal boxes against shell. Error in units of the dataset-mean
error; bands are 95% family-bootstrap intervals. Output: results/figures/fig_size_shell_dependence.{png,pdf}.
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
import donor_ladder_brain_size as S  # noqa: E402
from make_figures import INK, MUTED  # noqa: E402  (also applies the paper rcParams)

# cohort, k, label, colour, line style
FILLS = [("population", "1", "Stranger", "#2a78d6", "-"),
         ("sibling", "1", "Sibling", "#6250d6", "-"),
         ("other_hemisphere", "1", "Mirror, same place", "#e87ba4", "-"),
         ("mirror_block", "1", "Mirror, best of ±8 mm", "#e87ba4", "--"),
         ("mirror_block_oracle", "1", "Mirror, ±8 mm by true box*", "#e87ba4", ":"),
         ("stranger_block", "1", "Stranger, best of ±8 mm", "#2a78d6", "--"),
         ("repeat", "1", "Repeat scan", "#008300", "-"),
         ("knn", "16", "16 look-alikes by surround", "#eda100", "--"),
         ("knn_oracle", "16", "16 look-alikes by true box*", "#eda100", ":"),
         ("knn_mirror", "16", "16 look-alikes by own mirror", "#eda100", "-."),
         ("cohort", "all", "Same sex and age", "#1baf7a", "-")]
SHELL_BASED = ("knn", "mirror_block", "stranger_block")


def value(cfg, region, c, k):
    v = cfg["regions"].get(f"{region}/LR", {}).get(c, {}).get(k)
    return (np.nan, np.nan, np.nan) if v is None or v["err_over_V"] is None else (v["err_over_V"], *v["ci95"])


def curve(ax, configs, region, xs, keyf, fills, xlabel):
    for c, k, lab, col, ls in fills:
        y = np.array([value(configs[keyf(x)], region, c, k) for x in xs if keyf(x) in configs])
        x = np.array([x for x in xs if keyf(x) in configs])
        if not len(y) or np.all(np.isnan(y[:, 0])):
            continue
        ax.plot(x, y[:, 0], ls=ls, color=col, lw=1.0, marker="o", ms=2.5, label=lab)
        ax.fill_between(x, y[:, 1], y[:, 2], color=col, alpha=0.12, lw=0)
    ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks(xs)
    ax.set_xlabel(xlabel)


def main():
    res = json.load(open(S.OUT_JSON, encoding="utf-8"))
    configs = res["configs"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0), gridspec_kw={"hspace": 0.55, "wspace": 0.32})
    fig.subplots_adjust(top=0.9, bottom=0.17, left=0.07, right=0.99)
    a, b, c = axes[0]
    curve(a, configs, "random_patches", list(S.PATCHES), lambda p: f"p{p}_s4", FILLS, "Cube edge, mm (shell 4 mm)")
    a.set_ylabel("Error / dataset-mean error")
    a.set_title("(a) Random cubes: cube size", fontsize=7, loc="left")
    curve(b, configs, "random_patches", list(S.SHELLS), lambda s: f"p16_s{s}", FILLS, "Shell, mm (cube 16 mm)")
    b.set_title("(b) Random cubes: shell thickness", fontsize=7, loc="left")
    for (cc, k, lab, col, ls) in [f for f in FILLS if f[0] in SHELL_BASED]:
        for p, mk in zip(S.PATCHES, "v<o>^"):
            pts = [(s / p, *value(configs[f"p{p}_s{s}"], "random_patches", cc, k)) for s in S.SHELLS
                   if f"p{p}_s{s}" in configs]
            if pts:
                pts = np.array(pts)
                c.plot(pts[:, 0], pts[:, 1], ls=ls, color=col, lw=0.7, marker=mk, ms=3, alpha=0.85)
    for p, mk in zip(S.PATCHES, "v<o>^"):
        c.scatter([], [], marker=mk, color=MUTED, s=9, label=f"cube {p} mm")
    c.axhline(1, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    c.set_xscale("log", base=2)
    c.set_xticks([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1], ["1/16", "1/8", "1/4", "1/2", "1"])
    c.set_xlabel("Shell / cube edge")
    c.set_title("(c) Shell-based fills: ratio", fontsize=7, loc="left")
    c.legend(frameon=False, fontsize=5, loc="lower right")
    for ax, region, title in ((axes[1, 0], "deep", "(d) Deep box: shell"), (axes[1, 1], "parietal", "(e) Parietal box: shell")):
        curve(ax, configs, region, list(S.SHELLS), lambda s: f"p16_s{s}", FILLS, "Shell, mm")
        ax.set_title(title, fontsize=7, loc="left")
    axes[1, 0].set_ylabel("Error / dataset-mean error")
    axes[1, 2].axis("off")
    h, lab = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(h, lab, frameon=False, fontsize=5.6, loc="center left")
    for ax in axes.ravel():
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.5, 0.02, "QTIM, session 1, mean of left and right boxes; bands: 95% family bootstrap. "
             "* upper bound: chosen with the answer. Pre-QC, affine registration only.",
             ha="center", fontsize=5.6, color=MUTED)
    fig.suptitle("How cube size and context shell change each fill", fontsize=9, color=INK)
    out = os.path.join(ROOT, "results", "figures", "fig_size_shell_dependence")
    fig.savefig(out + ".png", dpi=250)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
