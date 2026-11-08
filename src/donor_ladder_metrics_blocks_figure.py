"""Ladder by blocks (PROTOCOL_donor_ladder_brain.md, Amendment 11): x = rungs grouped by source of information,
y = R = (d - d_repeat) / (d_stranger - d_repeat) (repeat = 0, stranger = 1), one panel per metric, plus a grid of the
pre-registered expectations E1-E5 (pass / fail per metric, random cubes).

Shaded bands show where an identity-faithful metric is expected to put a rung: below 1 for the own mirror and the
sibling, at or above 0.9 for averages and for wrong detail / noise. Aggregates only
(results/donor_ladder_brain_metrics.json). Pre-QC.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BAND = "#eaf2fc"
GOOD, BAD = "#0ca30c", "#d03b3b"
VIEWS = [("random_patches/LR", "32 random cubes, 16 mm", "#2a78d6", "o"),
         ("deep/LR", "deep box", "#eb6834", "s"),
         ("parietal/LR", "parietal box", "#1baf7a", "^")]
BLOCKS = [("same person", [("repeat", "repeat"), ("mirror", "own mirror")]),
          ("relatives", [("cotwin", "co-twin"), ("sibling", "sibling")]),
          ("selected by scaffold (achievable)", [("lookalike", "look-alike"), ("mirror_block", "mirror + search"),
                                                ("stranger_block", "stranger + search")]),
          ("no selection (averages)", [("stranger", "stranger"), ("mean_k2", "mean of 2"),
                                       ("cohort_mean", "sex-age mean"), ("dataset_mean", "dataset mean")]),
          ("oracle bound (unreachable)", [("lookalike_oracle", "oracle look-alike"),
                                          ("mirror_block_oracle", "oracle mirror"),
                                          ("stranger_block_oracle", "oracle stranger")]),
          ("controls", [("wrong_detail", "wrong detail"), ("noise", "white noise")])]
# expected region of R per rung under reality B, "do not lie" (Amendment 13); (lo, hi), None = open
EXPECTED = {"mirror": (None, 1.0), "sibling": (None, 1.0), "mean_k2": (None, 1.0), "dataset_mean": (None, 1.0),
            "wrong_detail": (0.9, None), "noise": (1.0, None)}
LABEL = {"mse": "MSE", "rmse": "RMSE", "mae": "MAE", "psnr": "PSNR", "psnr_fixed": "PSNR, fixed range",
         "ssim": "SSIM", "pearson": "Pearson r", "acc_dataset": "ACC vs dataset mean",
         "acc_cohort": "ACC vs sex-age mean", "lddt": "lDDT-like", "tm_like": "TM-like",
         "gdt_like": "GDT-like", "lddt_rep": "lDDT, repeat tolerance", "rmse_rep": "RMSE / repeat noise"}
SHORT = {"E1 mirror closer than stranger": "E1 mirror<str", "E2 sibling closer than stranger": "E2 sib<str",
         "E3B dataset mean better than stranger": "E3B mean<str", "E3B mean of 2 better than stranger": "E3B mean2<str",
         "E3B averages ordered by k": "E3B order by k", "E5B wrong detail not better than stranger": "E5B wrong~str",
         "E6B white noise worse than stranger": "E6B noise>str"}


# main-figure subset with plain labels (Kate, 6 Nov 2026); everything else goes to the _full version
CORE = {"repeat": "repeat scan", "mirror": "other hemisphere", "sibling": "sibling", "stranger": "unrelated person",
        "lookalike": "most similar person\n(by surroundings)", "lookalike_oracle": "best person in pool\n(oracle)",
        "mean_k2": "average of\n2 people", "dataset_mean": "average of\nall people",
        "noise": "white noise,\nsame amplitude"}


def layout(core=False):
    xs, fills, labels, spans, x = [], [], [], [], 0.0
    for name, rungs in BLOCKS:
        if core:
            rungs = [(f, CORE[f]) for f, _ in rungs if f in CORE]
            if not rungs:
                continue
        x0 = x
        for f, lab in rungs:
            xs.append(x)
            fills.append(f)
            labels.append(lab)
            x += 1
        spans.append((name, x0, x - 1))
        x += 0.8
    return xs, fills, labels, spans


def panel(ax, d, m, xs, fills, labels, spans, ylim):
    ax.set_facecolor(SURFACE)
    for (name, a, b) in spans:
        ax.axvspan(a - 0.45, b + 0.45, color="#f4f3f0", zorder=0)
    for x, f in zip(xs, fills):
        lo, hi = EXPECTED.get(f, (None, None))
        if lo is not None:
            ax.fill_between([x - 0.42, x + 0.42], lo, ylim[1], color=BAND, zorder=1)
        if hi is not None:
            ax.fill_between([x - 0.42, x + 0.42], ylim[0], hi, color=BAND, zorder=1)
    for h in (0, 1):
        ax.axhline(h, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2)
    for (name, a, b) in spans:
        ax.text((a + b) / 2, ylim[0] + 0.03, " ".join(name.split()), ha="center", va="bottom", fontsize=4.8,
                color=MUTED, zorder=5)
    for vi, (vk, vlab, col, mk) in enumerate(VIEWS):
        R = d["views"].get(vk, {}).get("ladder_R", {}).get(m)
        if not R:
            continue
        off = (vi - 1) * 0.2
        for x, f in zip(xs, fills):
            e = R.get(f)
            if not e:
                continue
            lo, hi = np.clip(e["ci95"], ylim[0], ylim[1] - 0.02)
            ax.plot([x + off] * 2, [lo, hi], color=col, lw=1.0, zorder=3)
            if e["R"] > ylim[1] - 0.02:             # off the scale: arrow at the top with the value
                ax.annotate(f"{e['R']:.1f}", (x + off, ylim[1] - 0.02), ha="center", va="top", fontsize=5,
                            color=col, xytext=(0, -2), textcoords="offset points")
                ax.scatter(x + off, ylim[1] - 0.12, s=18, marker="^", color=col, zorder=4)
                continue
            ax.scatter(x + off, e["R"], s=20, marker=mk, color=col, edgecolor=SURFACE, linewidth=0.7, zorder=4,
                       label=vlab if f == "repeat" else None)
    ex = d["views"]["random_patches/LR"].get("expectations", {}).get(m, {})
    passed = [v["pass"] for k, v in ex.items() if isinstance(v, dict)]
    ax.set_title(f"{LABEL.get(m, m)}: {sum(passed)}/{len(passed)} expectations met", fontsize=8, color=INK,
                 loc="left")
    ax.set_ylim(*ylim)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.6, color=MUTED)
    ax.tick_params(axis="y", labelsize=6.5, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)


def grid(ax, d, metrics):
    ex = d["views"]["random_patches/LR"].get("expectations", {})
    names = list(SHORT)
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylim(len(metrics) - 0.5, -0.5)
    for i, m in enumerate(metrics):
        for j, n in enumerate(names):
            v = ex.get(m, {}).get(n)
            if not isinstance(v, dict):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color=MUTED)
                continue
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, color=GOOD if v["pass"] else BAD, alpha=0.18,
                                       lw=0))
            val = v["value"]
            shown = "/".join(f"{x:.2f}" for x in val) if isinstance(val, list) else f"{val:.2f}"
            ax.text(j, i, ("pass" if v["pass"] else "fail") + f"\n{shown}", ha="center", va="center",
                    fontsize=5.6, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([SHORT[n] for n in names], rotation=90, fontsize=6, color=MUTED)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([LABEL.get(m, m) for m in metrics], fontsize=6.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Expectations, random cubes (value = R or contrast)", fontsize=8, color=INK, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/donor_ladder_brain_metrics.json")
    ap.add_argument("--name", default="QTIM")
    ap.add_argument("--out", default="results/figures/fig_metrics_blocks")
    ap.add_argument("--full", action="store_true", help="all rungs (supplement); default: the core subset")
    ap.add_argument("--metrics", default="", help="comma-separated metric keys (reduced panel); default: all")
    a = ap.parse_args()
    d = json.load(open(os.path.join(ROOT, a.json)))
    metrics = a.metrics.split(",") if a.metrics else d["metrics"]
    xs, fills, labels, spans = layout(core=not a.full)
    vals = [e["ci95"][k] for vk, *_ in VIEWS for m in metrics
            for f, e in d["views"].get(vk, {}).get("ladder_R", {}).get(m, {}).items()
            if not f.startswith("_") for k in (0, 1)]
    ylim = (-0.45, 1.6)     # fixed: repeat 0, stranger 1; values beyond are drawn as arrows with their value
    ncol = 3
    nrow = int(np.ceil((len(metrics) + 2) / ncol))
    fig = plt.figure(figsize=(6.0 * ncol, 3.6 * nrow), facecolor=SURFACE)
    gs = fig.add_gridspec(nrow, ncol)
    for p, m in enumerate(metrics):
        panel(fig.add_subplot(gs[p // ncol, p % ncol]), d, m, xs, fills, labels, spans, ylim)
    first = fig.axes[0]
    k = len(metrics)
    gax = fig.add_subplot(gs[k // ncol, k % ncol:])
    grid(gax, d, metrics)
    h, lab = first.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", fontsize=8, frameon=False, title=f"{a.name}, both hemispheres",
               title_fontsize=8)
    fig.text(0.5, 0.003, "R = (d - d_repeat) / (d_stranger - d_repeat): 0 = repeat scan, 1 = stranger (mean of 16). "
             "Bars: 95% interval, bootstrap over families. Blue bands: where an identity-faithful metric is expected "
             "(mirror and sibling below 1; averages, wrong detail and noise at or above 0.9).\n" + d.get("status", ""),
             ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 0.93, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, f"{a.out}.{ext}"), dpi=200, facecolor=SURFACE)
    print(os.path.abspath(os.path.join(ROOT, f"{a.out}.png")))


if __name__ == "__main__":
    main()
