"""Axiomatic target ladder of reality B ("do not lie", Amendments 13 and 14) next to measured metrics (QTIM, random
cubes).

Reality B: correct individual detail is good, wrong detail is worse than none, and something that is not anatomy is
worse than a wrong person. The target on R = (d - d_repeat) / (d_stranger - d_repeat) uses no measured metric value
(Amendment 14): repeat 0; no information (population mean) 0.5; mean of k unrelated people 0.5 + 0.5/k; wrong detail
1; white noise above 1 (arrow); partly correct rungs (mirror, sibling, look-alike, oracle) only "below 1", no value.
Each panel reports the value fit (RMS over valued rungs) and the side constraints. `rung_r` (r from squared error)
is kept only for the superseded Amendment 13 target. Blends with the mean are a separate table (Amendment 13).
Aggregates only. Pre-QC.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import donor_ladder_metrics_blocks_figure as BF  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
INK, MUTED, SURFACE = BF.INK, BF.MUTED, BF.SURFACE
THEORY, MEAS = "#0b0b0b", "#2a78d6"
SINGLE = ["repeat", "mirror", "cotwin", "sibling", "lookalike", "mirror_block", "stranger_block", "stranger",
          "lookalike_oracle", "mirror_block_oracle", "stranger_block_oracle"]
SHOWN = [("mse", "MSE"), ("ssim", "SSIM"), ("lddt", "lDDT-like"), ("acc_cohort", "ACC vs sex-age mean")]
NOISE_TARGET = 1.25
NOISE_TARGET = 1.25          # drawn position of the "above 1" target for white noise (an arrow, not a value)


def rung_r(view):
    """r per single-donor rung from its squared error over all cube voxels, in units of the dataset mean's error."""
    V = view["fills"]["dataset_mean"]["mse_allvox"]["raw"]["mean"]
    out = {}
    for f in SINGLE:
        e = view["fills"].get(f, {}).get("mse_allvox", {}).get("raw")
        if e:
            out[f] = 1 - (e["mean"] / V) / 2
    return out


# Amendment 14: axiomatic target of reality B, no measured metric value used
AXIOM = {"repeat": 0.0, "dataset_mean": 0.5, "cohort_mean": 0.5, "mean_k2": 0.75, "stranger": 1.0,
         "wrong_detail": 1.0}
PARTLY = ["lookalike_oracle", "lookalike", "mirror", "sibling", "cotwin", "mirror_block", "stranger_block",
          "mirror_block_oracle", "stranger_block_oracle"]
VALUED = ["dataset_mean", "cohort_mean", "mean_k2", "wrong_detail"]
ORDER = ["repeat", "lookalike_oracle", "mirror_block_oracle", "stranger_block_oracle", "lookalike", "mirror_block",
         "stranger_block", "mirror", "cotwin", "sibling", "dataset_mean", "cohort_mean", "mean_k2", "stranger",
         "wrong_detail", "noise"]


def score_B(view, m):
    """Amendment 14 score of one metric against the axiomatic reality-B target."""
    R = view["ladder_R"].get(m, {})
    dev = {f: R[f]["R"] - AXIOM[f] for f in VALUED if f in R}
    out = {"dev": dev, "rms": float(np.sqrt(np.mean(np.square(list(dev.values()))))) if dev else np.nan}
    out["noise_above_1"] = bool(R["noise"]["ci95"][0] > 1) if "noise" in R else None
    part = [f for f in PARTLY if f in R]
    out["partly_below_1"] = (sum(R[f]["ci95"][1] < 1 for f in part), len(part))
    out["oracle_le_lookalike"] = (bool(R["lookalike_oracle"]["R"] <= R["lookalike"]["R"])
                                  if "lookalike_oracle" in R and "lookalike" in R else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/donor_ladder_brain_metrics.json")
    ap.add_argument("--out", default="results/figures/fig_metrics_expected")
    ap.add_argument("--full", action="store_true", help="all rungs (supplement); default: the core subset")
    ap.add_argument("--metrics", default="", help="comma-separated metric keys (reduced panel); default: SHOWN")
    a = ap.parse_args()
    d = json.load(open(os.path.join(ROOT, a.json)))
    shown = [(m, BF.LABEL.get(m, m)) for m in a.metrics.split(",")] if a.metrics else SHOWN
    view = d["views"]["random_patches/LR"]
    _, fills0, labels0, spans = BF.layout(core=not a.full)
    block_of = {f: name for name, rungs in BF.BLOCKS for f, _lab in rungs}
    lab_of = dict(zip(fills0, labels0))
    fills = [f for f in ORDER if f in fills0]
    xs = list(range(len(fills)))
    block_names = [name for name, _ in BF.BLOCKS]
    block_col = dict(zip(block_names, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
    ylim = (-0.45, 2.3)
    fig, axes = plt.subplots(1, len(shown), figsize=(5.2 * len(shown), 5.9), facecolor=SURFACE, sharey=True)
    for k_ax, (ax, (m, mlab)) in enumerate(zip(axes, shown)):
        ax.set_facecolor(SURFACE)
        for h in (0, 1):
            ax.axhline(h, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        for x, f in zip(xs, fills):
            ax.add_patch(plt.Rectangle((x - 0.42, ylim[0]), 0.84, 0.05 * (ylim[1] - ylim[0]),
                                       color=block_col[block_of[f]], lw=0, zorder=2))
        vx = [(x, AXIOM[f]) for x, f in zip(xs, fills) if f in AXIOM]
        ax.scatter([x for x, _ in vx], [v for _, v in vx], s=46, marker="D", facecolor="white", edgecolor=THEORY,
                   linewidth=1.1, zorder=4, label="target value (axioms of reality B)")
        px = [x for x, f in zip(xs, fills) if f in PARTLY]
        if px:
            ax.scatter(px, [1.0] * len(px), s=60, marker="v", facecolor="white", edgecolor=THEORY, linewidth=1.0,
                       zorder=4, label="target: below 1, no value (partly correct)")
        if "noise" in fills:
            xn = xs[fills.index("noise")]
            ax.annotate("", (xn, 1.55), (xn, 1.02), arrowprops=dict(arrowstyle="->", color=THEORY, lw=1.4))
            ax.text(xn - 0.1, 1.58, "> 1", ha="center", fontsize=7, color=INK)
        R = view["ladder_R"].get(m, {})
        pts = [(x, R[f]) for x, f in zip(xs, fills) if R.get(f)]
        ax.plot([x + 0.18 for x, _ in pts], [min(e["R"], ylim[1] - 0.05) for _, e in pts], color=MEAS, lw=0.9,
                alpha=0.7, zorder=3)
        for x, e in pts:
            ax.plot([x + 0.18] * 2, np.clip(e["ci95"], ylim[0], ylim[1] - 0.05), color=MEAS, lw=1.0, zorder=3)
        ax.scatter([x + 0.18 for x, _ in pts], [min(e["R"], ylim[1] - 0.05) for _, e in pts], s=20, color=MEAS,
                   zorder=5, label=f"measured {mlab}")
        sc = score_B(view, m)
        nb, nt = sc["partly_below_1"]
        ax.set_title(f"{mlab}: value fit RMS {sc['rms']:.2f}; noise > 1: {'yes' if sc['noise_above_1'] else 'no'}; "
                     f"partly correct < 1: {nb}/{nt}", fontsize=8, color=INK, loc="left")
        ax.set_ylim(*ylim)
        ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab_of[f] for f in fills], rotation=90, fontsize=6, color=MUTED)
        ax.tick_params(axis="y", labelsize=7, colors=MUTED)
        if k_ax == 0:
            ax.set_ylabel("R  (0 = repeat scan, 1 = unrelated person)", fontsize=7.5, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=block_col[n]) for n in block_names if n in set(block_of[f] for f in fills)]
    names = [n for n in block_names if n in set(block_of[f] for f in fills)]
    fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.05), title="block (strip under each rung)", title_fontsize=7)
    fig.text(0.5, 0.005, "Target (reality B, 'do not lie', Amendment 14), from axioms only: no information = 0.5; "
             "mean of k unrelated people = 0.5 + 0.5/k; wrong detail = 1; white noise above 1; partly correct rungs "
             "below 1 with no value. No measured metric enters the target. " + d.get("status", ""), ha="center",
             fontsize=6, color=MUTED, wrap=True)
    print("metric | dev dataset mean | dev sex-age mean | dev mean of 2 | dev wrong detail | RMS | noise>1 | "
          "partly<1 | oracle<=look-alike")
    for m in d["metrics"]:
        sc = score_B(view, m)
        dv = sc["dev"]
        print(" | ".join([m] + [f"{dv[f]:+.2f}" if f in dv else "n/a" for f in VALUED] +
                         [f"{sc['rms']:.2f}", str(sc["noise_above_1"]), f"{sc['partly_below_1'][0]}/{sc['partly_below_1'][1]}",
                          str(sc["oracle_le_lookalike"])]))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, f"{a.out}.{ext}"), dpi=200, facecolor=SURFACE)
    print(os.path.abspath(os.path.join(ROOT, f"{a.out}.png")))


if __name__ == "__main__":
    main()
