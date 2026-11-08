"""Shrinkage experiment table (Amendment 13): does a metric prefer a real donor faded towards the dataset mean?

Blend = mean + lambda (donor - mean), lambda = r of the donor's rung (the squared-error optimum). For each metric:
R of the donor, R of the blend, and the difference blend - donor with its 95% interval (shared bootstrap). Verdict:
"prefers the faded donor" if the whole interval is below 0, "indifferent" if it contains 0, "penalises fading" if it
is above 0. Random cubes, both hemispheres. Aggregates only (results JSON). Pre-QC.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import donor_ladder_metrics_blocks_figure as BF  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PAIRS = [("repeat", "shrink_repeat", "A-E4a repeat blend not rewarded", "repeat scan"),
         ("mirror", "shrink_mirror", "A-E4b mirror blend not rewarded", "other hemisphere")]
GOOD, BAD, NEUTRAL = "#d5ecd5", "#f6d6d6", "#f0efec"


def verdict(lo, hi):
    if hi < 0:
        return "prefers the faded donor", BAD
    if lo > 0:
        return "penalises fading", GOOD
    return "indifferent", NEUTRAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/donor_ladder_brain_metrics.json")
    ap.add_argument("--out", default="results/figures/fig_metrics_shrink_table")
    a = ap.parse_args()
    d = json.load(open(os.path.join(ROOT, a.json)))
    v = d["views"]["random_patches/LR"]
    lam = d.get("lambda_shrink", {})
    metrics = d["metrics"]
    header = ["metric"]
    for _, _, _, lab in PAIRS:
        header += [f"{lab}\nR donor", "R blend", "blend - donor\n[95% CI]", "verdict"]
    rows, colors = [], []
    for m in metrics:
        R = v["ladder_R"].get(m, {})
        ex = v.get("expectations_realityA", {}).get(m, {})
        row, col = [BF.LABEL.get(m, m)], ["white"]
        for donor, blend, key, _ in PAIRS:
            e = ex.get(key)
            if not e or donor not in R or blend not in R:
                row += ["n/a"] * 4
                col += ["white"] * 4
                continue
            txt, c = verdict(e["lower95"], e["upper95"])
            row += [f"{R[donor]['R']:.2f}", f"{R[blend]['R']:.2f}",
                    f"{e['value']:+.2f} [{e['lower95']:+.2f}, {e['upper95']:+.2f}]", txt]
            col += ["white", "white", c, c]
        rows.append(row)
        colors.append(col)
        print(" | ".join(x.replace("\n", " ") for x in row))
    fig, ax = plt.subplots(figsize=(15, 0.42 * len(rows) + 1.6), facecolor=BF.SURFACE)
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=header, cellColours=colors, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.5)
    tab.scale(1, 1.35)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor("#d9d8d4")
        if i == 0:
            cell.set_text_props(weight="bold", fontsize=7)
            cell.set_height(cell.get_height() * 1.6)
    lam_txt = ", ".join(f"{k} {x:.2f}" for k, x in lam.items())
    ax.set_title("Shrinkage test: a real donor faded towards the dataset mean, blend = mean + lambda (donor - mean), "
                 f"lambda = r of the rung ({lam_txt}). R: 0 = repeat, 1 = unrelated person; lower = judged closer.",
                 fontsize=8.5, loc="left")
    fig.text(0.01, 0.01, "Intervals: bootstrap over families and positions (random cubes, both hemispheres). "
             + d.get("status", ""), fontsize=6.5, color=BF.MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, f"{a.out}.{ext}"), dpi=200, facecolor=BF.SURFACE)
    print(os.path.abspath(os.path.join(ROOT, f"{a.out}.png")))


if __name__ == "__main__":
    main()
