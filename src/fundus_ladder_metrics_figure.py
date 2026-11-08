"""Figure of the Amendment 9.15 fundus metrics from the committed aggregate only (no images, no rows).

Input: results/fundus_ladder_metrics_a915.json. Output: results/figures/fig_fundus_metrics_a915.{png,pdf}.
Left: per-target distance relative to the stranger rung (R_self, person bootstrap 95% CI) for LPIPS AlexNet, LPIPS
VGG (subset S), Betti matching matched and unmatched. Middle, right: set-level FID and FWD ratio to the stranger set
(point values only; their bootstrap intervals are invalid, Amendment 9.15 outcome). A rung missing in a dataset is
left empty, never drawn as zero; a ratio clamped to 0 is labelled, not drawn on the log axis.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "fundus_ladder_metrics_a915.json")
OUT = os.path.join(ROOT, "results", "figures", "fig_fundus_metrics_a915")
ORDER = ("repeat", "shrink_repeat", "fellow", "shrink_fellow", "lookalike", "mean_k2", "dataset_mean", "wrong_detail",
         "stranger", "noise")
LABEL = {"repeat": "repeat visit", "shrink_repeat": "repeat, shrunk", "fellow": "fellow eye",
         "shrink_fellow": "fellow, shrunk", "lookalike": "look-alike", "mean_k2": "mean of 2",
         "dataset_mean": "dataset mean", "wrong_detail": "wrong detail", "stranger": "stranger", "noise": "noise"}
PER_TARGET = (("lpips_alex", "LPIPS AlexNet"), ("lpips_vgg", "LPIPS VGG (subset S)"),
              ("betti_matched", "Betti matching, matched"), ("betti_unmatched", "Betti matching, unmatched"))
DATASETS = ("GRAPE", "PAPILA")


def per_target_points(panel, metric):
    """(x, R, lo, hi) per rung present for this metric; absent rungs are skipped."""
    R = panel["R_self"].get(metric, {})
    return [(i, R[f]["R"], R[f]["ci95"][0], R[f]["ci95"][1]) for i, f in enumerate(ORDER)
            if isinstance(R.get(f), dict) and "R" in R[f]]


def set_points(panel, metric):
    """(x, ratio) per rung with a computed set-level value; 'not run' rungs are skipped."""
    sl = panel["amendment_9_15"]["set_level"][metric]
    return [(i, sl[f]["ratio_to_stranger"]) for i, f in enumerate(ORDER)
            if isinstance(sl.get(f), dict) and "ratio_to_stranger" in sl[f]]


def draw(src=SRC, out=OUT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(src, encoding="utf-8") as fh:
        d = json.load(fh)
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.4))
    for r, ds in enumerate(DATASETS):
        p = d["panel"][ds]
        a = ax[r, 0]
        for k, (m, ml) in enumerate(PER_TARGET):
            pts = per_target_points(p, m)
            if not pts:
                continue
            x, y, lo, hi = map(np.array, zip(*pts))
            a.errorbar(x + (k - 1.5) * 0.15, y, yerr=[y - lo, hi - y], fmt="o", ms=3, lw=0.8, label=ml)
        a.set_ylabel(f"{ds}\ndistance / stranger distance", fontsize=8)
        if r == 0:
            a.set_title("Per-target metrics (lower = closer to the target)", fontsize=9)
            a.legend(fontsize=6.5, loc="upper left")
        for c, m in ((1, "fid"), (2, "fwd")):
            a = ax[r, c]
            pts = set_points(p, m)
            for i, v in pts:
                if v > 0:
                    a.bar(i, v, color="#4a7bb7")
                    a.text(i, v * 1.06, f"{v:.2f}", ha="center", fontsize=6)
                else:
                    a.text(i, 1.0, "0 (clamped)", ha="center", va="bottom", rotation=90, fontsize=6, color="#a00")
            if r == 0:
                a.set_title(f"{m.upper()}, set level: ratio to the stranger set", fontsize=9)
            a.set_ylabel("set distance / stranger set distance", fontsize=8)
        for a in ax[r]:
            a.axhline(1, c="grey", lw=0.6, ls="--")
            a.set_yscale("log")
            a.set_xticks(range(len(ORDER)))
            a.set_xticklabels([LABEL[o] for o in ORDER], rotation=45, ha="right", fontsize=7)
    fig.suptitle("Fundus donor ladder, Amendment 9.15 metrics (GRAPE P8, PAPILA P9, 16 px window). "
                 "Aggregates only; empty = rung not available. Pre-QC.", fontsize=10)
    fig.text(0.5, 0.005, "Per-target: R_self with person-bootstrap 95% CI. Set level: point values; their bootstrap "
             "intervals are biased and not shown (Amendment 9.15 outcome).", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(draw(*sys.argv[1:3]))
