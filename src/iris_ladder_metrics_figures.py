"""Iris metric panel figures (docs/PROTOCOL_iris_metrics.md), in the face / fundus style: R per rung by block for
every computed metric (UTIRIS NIR, UPOL, UTIRIS visible light) and the summed squared error relative to the
dataset mean per rung (spec block). Reads only the aggregate JSON (results/iris_ladder_metrics.json); styling
constants are imported from the brain figure script without editing it.
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
VIEWS = [("UTIRIS_NIR", "UTIRIS near-infrared (primary)", "#2a78d6", "o"), ("UPOL", "UPOL visible light", "#eb6834", "s"),
         ("UTIRIS_VIS", "UTIRIS visible light", "#3b9a5b", "D")]
BLOCKS = [("same eye", [("repeat", "another photo,\nsame eye")]),
          ("same person", [("fellow", "other eye,\nmirrored")]),
          ("selected by context (achievable)", [("lookalike", "most similar eye\n(by surroundings)"),
                                                ("lookalike16", "average of 16\nmost similar")]),
          ("no selection (averages)", [("stranger", "unrelated person"), ("mean_k2", "average of\n2 people"),
                                       ("dataset_mean", "average of\nall people")]),
          ("oracle bound (unreachable)", [("lookalike_oracle", "best eye in pool\n(oracle)")]),
          ("controls", [("wrong_detail", "wrong detail\n(stranger texture,\nsame amplitude)"),
                        ("noise", "white noise,\nsame amplitude"),
                        ("shrink_repeat", "another photo\npulled to the mean"),
                        ("shrink_fellow", "other eye\npulled to the mean")])]
CORE = {"repeat", "fellow", "lookalike", "stranger", "mean_k2", "dataset_mean", "lookalike_oracle", "wrong_detail", "noise",
        "shrink_repeat"}
EXPECTED = dict(BF.EXPECTED, fellow=(None, 1.0), lookalike=(None, 1.0), lookalike16=(None, 1.0),
                lookalike_oracle=(None, 1.0), repeat=(None, 1.0))
LABEL = dict(BF.LABEL, ms_ssim="MS-SSIM", ampdev_dataset="amplitude vs dataset mean", persistence="intensity persistence")
YLIM = (-0.6, 1.8)
SHORT = {"same eye": "same eye", "same person": "same\nperson", "selected by context (achievable)": "by context",
         "no selection (averages)": "averages", "oracle bound (unreachable)": "oracle", "controls": "controls"}


def layout(full):
    xs, fills, labels, spans, x = [], [], [], [], 0.0
    for name, rungs in BLOCKS:
        rungs = [(f, lab) for f, lab in rungs if full or f in CORE]
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


def style(ax, xs, labels, ylim):
    ax.set_ylim(*ylim)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.6, color=BF.MUTED)
    ax.tick_params(axis="y", labelsize=6.5, colors=BF.MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BF.MUTED)
        ax.spines[s].set_linewidth(0.6)


def blocks_figure(d, out, full):
    panel = d["panel"]
    metrics = [m for m in ("mse", "rmse", "mae", "ssim", "ms_ssim", "acc_dataset", "lddt", "persistence", "pearson",
                           "tm_like", "gdt_like", "lddt_rep", "rmse_rep")
               if any(m in panel.get(v, {}).get("R_self", {}) for v, *_ in VIEWS)]
    if not full:
        metrics = [m for m in metrics if m in ("mse", "ssim", "ms_ssim", "acc_dataset", "lddt", "persistence")]
    xs, fills, labels, spans = layout(full)
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig = plt.figure(figsize=(6.4 * ncol, 4.2 * nrow + 0.6), facecolor=BF.SURFACE)
    gs = fig.add_gridspec(nrow, ncol)
    for p, m in enumerate(metrics):
        ax = fig.add_subplot(gs[p // ncol, p % ncol])
        ax.set_facecolor(BF.SURFACE)
        for (name, a, b) in spans:
            ax.axvspan(a - 0.45, b + 0.45, color="#f4f3f0", zorder=0)
            ax.text((a + b) / 2, YLIM[0] + 0.03, SHORT.get(name, name), ha="center", va="bottom", fontsize=5.2,
                    color=BF.MUTED, zorder=5)
        for x, f in zip(xs, fills):
            lo, hi = EXPECTED.get(f, (None, None))
            if lo is not None:
                ax.fill_between([x - 0.42, x + 0.42], lo, YLIM[1], color=BF.BAND, zorder=1)
            if hi is not None:
                ax.fill_between([x - 0.42, x + 0.42], YLIM[0], hi, color=BF.BAND, zorder=1)
        for h in (0, 1):
            ax.axhline(h, color=BF.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2)
        tt = []
        for vi, (vk, vlab, col, mk) in enumerate(VIEWS):
            R = panel.get(vk, {}).get("R_self", {}).get(m)
            if not R:
                continue
            off = (vi - 1) * 0.22
            for x, f in zip(xs, fills):
                e = R.get(f)
                if not e or e.get("R") is None:
                    continue
                lo, hi = np.clip(e["ci95"], YLIM[0] + 0.02, YLIM[1] - 0.02)
                ax.plot([x + off] * 2, [lo, hi], color=col, lw=1.0, zorder=3)
                if not YLIM[0] + 0.02 < e["R"] < YLIM[1] - 0.02:
                    top = e["R"] > 0
                    yv = YLIM[1] - 0.12 if top else YLIM[0] + 0.12
                    ax.scatter(x + off, yv, s=18, marker="^" if top else "v", color=col, zorder=4)
                    ax.annotate(f"{e['R']:.1f}", (x + off, yv), ha="center", va="bottom" if top else "top", fontsize=5,
                                color=col, xytext=(0, 3 if top else -3), textcoords="offset points")
                    continue
                ax.scatter(x + off, e["R"], s=20, marker=mk, color=col, edgecolor=BF.SURFACE, linewidth=0.7, zorder=4,
                           label=vlab if (p == 0 and f == fills[0]) else None)
            ex = panel.get(vk, {}).get("expectations", {}).get(m, {})
            passed = [v["pass"] for v in ex.values() if isinstance(v, dict) and "pass" in v]
            if passed:
                tt.append(f"{vlab.split(' (')[0]} {sum(passed)}/{len(passed)}")
        ax.set_title(f"{LABEL.get(m, m)}\nexpectations met: " + ", ".join(tt), fontsize=7.5, color=BF.INK, loc="left")
        style(ax, xs, labels, YLIM)
    h, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.035),
               title="iris, 16 px windows on the unwrapped strip", title_fontsize=8)
    fig.text(0.5, 0.003, "R = (d - d_self) / (d_stranger - d_self): 0 = the target itself, 1 = unrelated person (mean of 16). "
             "Bars: 95% interval, bootstrap over people. Blue bands: expected side (partly correct rungs and averages "
             "below 1; wrong detail at or above 0.9; noise above 1).\n" + d.get("status", ""),
             ha="center", fontsize=6.5, color=BF.MUTED)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out)


def ratio_figure(d, out):
    """Summed squared error over the dataset mean's on the same pixels, per rung (1 = mean, 2 = stranger)."""
    xs, fills, labels, spans = layout(True)
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=BF.SURFACE)
    ax.set_facecolor(BF.SURFACE)
    for (name, a, b) in spans:
        ax.axvspan(a - 0.45, b + 0.45, color="#f4f3f0", zorder=0)
        ax.text((a + b) / 2, 0.03, SHORT.get(name, name), ha="center", va="bottom", fontsize=6, color=BF.MUTED, zorder=5)
    for h in (1, 2):
        ax.axhline(h, color=BF.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2)
    for vi, (vk, vlab, col, mk) in enumerate(VIEWS):
        res = d["results"].get(vk, {})
        off = (vi - 1) * 0.22
        first = True
        for x, f in zip(xs, fills):
            e = res.get(f)
            if not isinstance(e, dict) or "ratio" not in e:
                continue
            ax.plot([x + off] * 2, e["ci95"], color=col, lw=1.0, zorder=3)
            ax.scatter(x + off, e["ratio"], s=22, marker=mk, color=col, edgecolor=BF.SURFACE, linewidth=0.7, zorder=4,
                       label=vlab if first else None)
            first = False
    ax.set_ylabel("squared error / dataset mean's", fontsize=7, color=BF.MUTED)
    style(ax, xs, labels, (0, 2.4))
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.text(0.5, 0.005, "1 = the average of all people, 2 = an unrelated person (theory). Bars: 95% interval, bootstrap "
             "over people.\n" + d.get("status", ""), ha="center", fontsize=6, color=BF.MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out)


def save(fig, out):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, f"{out}.{ext}"), dpi=200, facecolor=BF.SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/iris_ladder_metrics.json")
    ap.add_argument("--prefix", default="results/figures/fig_iris_metrics")
    a = ap.parse_args()
    d = json.load(open(a.json if os.path.isabs(a.json) else os.path.join(ROOT, a.json), encoding="utf-8"))
    blocks_figure(d, a.prefix + "_blocks", full=False)
    blocks_figure(d, a.prefix + "_blocks_full", full=True)
    ratio_figure(d, a.prefix + "_ratio")
    print("written", a.prefix + "_{blocks,blocks_full,ratio}.{png,pdf}")


if __name__ == "__main__":
    main()
