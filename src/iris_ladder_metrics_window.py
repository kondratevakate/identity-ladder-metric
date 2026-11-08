"""Amendment P4: iris panel at the reference window (16 px) against f = 1/8 (10 px), UTIRIS_NIR.

Reads only the aggregate JSONs (results/iris_ladder_metrics.json and results/iris_ladder_metrics_w10.json), adds a
`window_comparison` block to the 10 px JSON (per metric: R_self per rung at both sizes, Spearman correlation of the
rung order, and which ordered rung pairs flip) and draws results/figures/fig_iris_window_f8.{png,pdf} (charts only).

    python -I src/iris_ladder_metrics_window.py --set UTIRIS_NIR
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import donor_ladder_metrics_blocks_figure as BF  # noqa: E402
import iris_ladder_metrics_figures as FG  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REF, NEW = "results/iris_ladder_metrics.json", "results/iris_ladder_metrics_w10.json"
RUNGS = ("repeat", "fellow", "lookalike", "lookalike16", "lookalike_oracle", "stranger", "mean_k2", "dataset_mean",
         "wrong_detail", "noise", "shrink_repeat", "shrink_fellow")
PAIRS = (("repeat", "lookalike"), ("lookalike", "stranger"), ("dataset_mean", "stranger"), ("fellow", "stranger"),
         ("repeat", "shrink_repeat"), ("wrong_detail", "stranger"))
TIE = 0.1                              # differences below 0.1 on R are not interpreted (protocol precision)


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


def compare(R16, R10):
    """Per metric: Spearman of the rung R values between sizes, and ordered pairs that change side by more than the
    0.1 tolerance."""
    out = {}
    for m in sorted(set(R16) & set(R10)):
        rr = [f for f in RUNGS if f in R16[m] and f in R10[m]]
        a = np.array([R16[m][f]["R"] for f in rr])
        b = np.array([R10[m][f]["R"] for f in rr])
        rho = float(stats.spearmanr(a, b).statistic) if len(rr) >= 3 else None
        flips = []
        for x, y in PAIRS:
            if x in rr and y in rr:
                d16 = R16[m][x]["R"] - R16[m][y]["R"]
                d10 = R10[m][x]["R"] - R10[m][y]["R"]
                s16 = 0 if abs(d16) < TIE else np.sign(d16)
                s10 = 0 if abs(d10) < TIE else np.sign(d10)
                if s16 != s10:
                    flips.append({"pair": f"{x} vs {y}", "diff_16px": d16, "diff_10px": d10})
        out[m] = {"rungs": rr, "R_16px": dict(zip(rr, a.tolist())), "R_10px": dict(zip(rr, b.tolist())),
                  "ci95_10px": {f: R10[m][f]["ci95"] for f in rr}, "spearman_rung_order": rho,
                  "order_changes_beyond_0.1": flips}
    return out


def figure(cmp_, name, out):
    metrics = list(cmp_)
    xs, fills, labels, spans = FG.layout(True)
    ncol = 4
    nrow = int(np.ceil(len(metrics) / ncol))
    fig = plt.figure(figsize=(5.2 * ncol, 4.0 * nrow + 0.6), facecolor=BF.SURFACE)
    gs = fig.add_gridspec(nrow, ncol)
    for p, m in enumerate(metrics):
        ax = fig.add_subplot(gs[p // ncol, p % ncol])
        ax.set_facecolor(BF.SURFACE)
        for (_, a, b) in spans:
            ax.axvspan(a - 0.45, b + 0.45, color="#f4f3f0", zorder=0)
        for h in (0, 1):
            ax.axhline(h, color=BF.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2)
        c = cmp_[m]
        for off, key, col, mk, lab in ((-0.14, "R_16px", "#2a78d6", "o", "16 px window (f = 0.2, reference)"),
                                       (0.14, "R_10px", "#e8a200", "D", "10 px window (f = 1/8)")):
            first = True
            for x, f in zip(xs, fills):
                if f not in c[key]:
                    continue
                v = float(np.clip(c[key][f], FG.YLIM[0] + 0.1, FG.YLIM[1] - 0.1))
                if key == "R_10px":
                    lo, hi = np.clip(c["ci95_10px"][f], FG.YLIM[0] + 0.02, FG.YLIM[1] - 0.02)
                    ax.plot([x + off] * 2, [lo, hi], color=col, lw=1.0, zorder=3)
                ax.scatter(x + off, v, s=18, marker=mk, color=col, edgecolor=BF.SURFACE, linewidth=0.6, zorder=4,
                           label=lab if (p == 0 and first) else None)
                first = False
        rho = c["spearman_rung_order"]
        fl = ", ".join(x["pair"].replace("_", " ") for x in c["order_changes_beyond_0.1"]) or "none"
        ax.set_title(f"{FG.LABEL.get(m, m)}\nSpearman of rung order {rho:.2f}; order differs: {fl}", fontsize=6.8,
                     color=BF.INK, loc="left")
        FG.style(ax, xs, labels, FG.YLIM)
    h, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.03),
               title=f"iris {name.replace('_', ' ')}, window in object units (D = 256 / pi strip columns)",
               title_fontsize=8)
    fig.text(0.5, 0.003, "R = (d - d_self) / (d_stranger - d_self): 0 = the target itself, 1 = unrelated person (mean "
             "of 16). Bars: 95% interval at 10 px, bootstrap over people. Order differs = a pair of steps is separated by "
             "more than 0.1 on R at one size but not at the other (tie), or reversed. Amendment P4.", ha="center", fontsize=6.3, color=BF.MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FG.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    a = ap.parse_args()
    ref, new = load(REF), load(NEW)
    if new.get("window_px") != 10 or ref.get("window_px") != 16:
        raise SystemExit("expected the 16 px reference and the 10 px run")
    cmp_ = compare(ref["panel"][a.set]["R_self"], new["panel"][a.set]["R_self"])
    new["window_comparison"] = {"set": a.set, "reference_window_px": 16, "window_px": 10,
                                "tolerance_R": TIE,
                                "order_change_definition": "sign of the R difference of a rung pair, with |diff| < 0.1 "
                                                           "counted as a tie, differs between the two sizes",
                                "per_metric": cmp_}
    with open(os.path.join(ROOT, NEW), "w", encoding="utf-8") as fh:
        json.dump(new, fh, indent=1, ensure_ascii=False, allow_nan=False)
    figure(cmp_, a.set, "results/figures/fig_iris_window_f8")
    for m, c in cmp_.items():
        print(f"{m}: rho {c['spearman_rung_order']:.2f}, changes {[x['pair'] for x in c['order_changes_beyond_0.1']]}")


if __name__ == "__main__":
    main()
