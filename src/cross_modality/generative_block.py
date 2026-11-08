"""Generative, perceptual and topology block for the cross-modality tiles (aggregates only).

Sources, read only: fundus `results/fundus_ladder_metrics_a915.json` (Amendment 9.15, identity-ladder-bilateral),
iris `results/iris_ladder_metrics.json` key `amendment_p3`. Brain (Amendment 20, LPIPS 2.5D) and faces (F2) are listed
as "not run" until their summary JSONs carry these metrics; that is a stated limitation, not a zero.
Per-target metrics (LPIPS AlexNet, LPIPS VGG, Betti matching matched / unmatched, intensity persistence, iris-code
Hamming distance) use R_self per rung (target 0, mean of 16 strangers 1, lower = closer). Set-level metrics (FID,
FWD) use the per-rung set value (lower = closer). The tile rank is the rung's place among the rungs present.
Writes `ranks_generative.json`, `generative_status.json` and `generative_tiles.png`.

Usage: python -I src/cross_modality/generative_block.py results/cross_modality <bilateral_worktree>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

OUT, BIL = sys.argv[1], sys.argv[2]
HERE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
COLS = ["repeat", "other_side", "relative", "lookalike_oracle", "lookalike", "cohort_mean", "mean_k2", "dataset_mean",
        "stranger", "shrink_ref", "wrong_detail", "noise"]
LAB = {"repeat": "repeat image", "other_side": "other side (mirror / eye)", "relative": "relative (sibling or twin)",
       "lookalike_oracle": "best look-alike", "lookalike": "look-alike", "cohort_mean": "sex-age mean",
       "mean_k2": "mean of 2 strangers", "dataset_mean": "dataset mean   [T1]", "stranger": "stranger",
       "shrink_ref": "repeat shrunk to mean   [T2]", "wrong_detail": "wrong detail   [T3a]", "noise": "white noise   [T3b]"}
MET = [("lpips_alex", "LPIPS (AlexNet)", "per target"), ("lpips_vgg", "LPIPS (VGG)", "per target"),
       ("betti_matched", "Betti matching, matched", "per target"), ("betti_unmatched", "Betti matching, unmatched", "per target"),
       ("persistence", "intensity persistence", "per target"), ("iriscode_hd", "iris code Hamming distance", "per target"),
       ("fid", "FID (set level)", "set"), ("fwd", "FWD (set level)", "set")]
NOT_RUN = {"QTIM": "brain: Amendment 20 run restarted after the 8 Nov reboot; Betti 3D needs a C++ build",
           "CelebA": "faces: F2 panel run restarted after the 8 Nov reboot; summary not yet in the JSON",
           "IMDb-WIKI": "faces: F2 panel run restarted after the 8 Nov reboot; summary not yet in the JSON"}


def ren(f, has_repeat):
    f = {"fellow": "other_side", "sexage": "cohort_mean"}.get(f, f)
    if f == ("shrink_repeat" if has_repeat else "shrink_fellow"):
        return "shrink_ref"
    return f


def per_rung(R, sets, m):
    """{rung: value} for one metric, lower = closer; None if the metric is absent."""
    if m in ("fid", "fwd"):
        blk = (sets or {}).get(m)
        if not blk:
            return None
        has_rep = "repeat" in blk
        return {ren(f, has_rep): e["value"] for f, e in blk.items() if isinstance(e, dict) and e.get("value") is not None}
    blk = (R or {}).get(m)
    if not blk:
        return None
    has_rep = "repeat" in blk
    return {ren(f, has_rep): e["R"] for f, e in blk.items() if isinstance(e, dict) and e.get("R") is not None}


src = {}
a915 = json.load(open(os.path.join(BIL, "results", "fundus_ladder_metrics_a915.json")))["panel"]
for d in ("GRAPE", "PAPILA"):
    g = a915[d]
    src[d] = (g["R_self"], g["amendment_9_15"]["set_level"])
iris = json.load(open(os.path.join(HERE, "results", "iris_ladder_metrics.json")))["amendment_p3"]["sets"]
for d in ("UTIRIS_NIR", "UTIRIS_VIS", "UPOL"):
    s = iris[d]
    src[d.replace("_", " ")] = (s["R_self"], s["set_level"])

ranks, status = {}, {"method": __doc__.strip(), "not_run": NOT_RUN, "present": {}}
for d, (R, S) in src.items():
    status["present"][d] = []
    for m, _, _ in MET:
        v = per_rung(R, S, m)
        if not v:
            continue
        it = [(r, v[r]) for r in COLS if r in v]
        if len(it) < 3:
            continue
        order = sorted(it, key=lambda x: x[1])
        ranks[f"{m}|{d}"] = {r: i + 1 for i, (r, _) in enumerate(order)}
        status["present"][d].append(m)
json.dump(ranks, open(os.path.join(OUT, "ranks_generative.json"), "w"), indent=1)
json.dump(status, open(os.path.join(OUT, "generative_status.json"), "w"), indent=1)

DS = ["QTIM", "CelebA", "IMDb-WIKI", "GRAPE", "PAPILA", "UTIRIS NIR", "UTIRIS VIS", "UPOL"]
cmap = LinearSegmentedColormap.from_list("rank", ["#0b5d63", "#3aa6a8", "#bfe3dc", "#f6e7c8", "#f0a868", "#b7471a"])
rows = []
for m, name, kind in MET:
    blk = [(d, ranks.get(f"{m}|{d}")) for d in DS]
    if not any(r for _, r in blk):
        continue
    rows.append(("m", f"{name}  ({'lower = closer' if kind == 'per target' else 'set level, lower = closer'})"))
    rows += [("r", (d, r)) for d, r in blk]
    rows.append(("g", None))
H = sum(1 if k != "g" else 0.4 for k, _ in rows) + 7
fig, ax = plt.subplots(figsize=(11.5, 0.24 * H + 0.6))
for j, r in enumerate(COLS):
    ax.text(j + 0.5, -0.25, LAB[r], fontsize=7.5, rotation=50, rotation_mode="anchor", ha="left", va="bottom",
            color="#7a2e0e" if "[T" in LAB[r] else "#222", fontweight="bold" if "[T" in LAB[r] else "normal")
y = 0.0
for k, b in rows:
    if k == "g":
        y += 0.4
        continue
    if k == "m":
        ax.text(-0.2, y + 0.5, b, fontsize=8, fontweight="bold", ha="right", va="center")
        y += 1
        continue
    d, rank = b
    ax.text(-0.2, y + 0.46, d, fontsize=7.5, ha="right", va="center", color="#1f6f3a" if d in ("CelebA", "IMDb-WIKI") else "#8a6400")
    if not rank:
        ax.add_patch(Rectangle((0, y), len(COLS) - 0.05, 0.92, facecolor="#f4f4f4", edgecolor="#dddddd", linewidth=0.5))
        why = NOT_RUN.get(d, "metric not defined or not computed for this dataset")
        ax.text(0.15, y + 0.46, "not run: " + why if d in NOT_RUN else why, fontsize=6.3, va="center", color="#888888")
        y += 1
        continue
    n = len(rank)
    for j, r in enumerate(COLS):
        if r not in rank:
            ax.add_patch(Rectangle((j, y), 0.95, 0.92, facecolor="none", edgecolor="#dddddd", linewidth=0.5))
            continue
        t = (rank[r] - 1) / max(1, n - 1)
        ax.add_patch(Rectangle((j, y), 0.95, 0.92, facecolor=cmap(t), edgecolor="none"))
        ax.text(j + 0.475, y + 0.46, str(rank[r]), fontsize=7, ha="center", va="center",
                color="white" if (t < 0.25 or t > 0.8) else "#222")
    y += 1
ax.text(0, y + 1.2, "Same columns and colours as the main rank tiles (1 = what the metric calls closest). Per-target metrics are "
        "ranked by R_self, set-level FID and FWD\nby their per-rung set value. Fundus Amendment 9.15 and iris Amendment P3; brain "
        "and faces not run tonight (grey rows, reason given). All pre-QC except faces.", fontsize=7, color="#333", va="center")
ax.set_xlim(-7.2, len(COLS) + 0.2)
ax.set_ylim(y + 2.2, -5.2)
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "generative_tiles.png"), dpi=170, facecolor="white")
print({d: v for d, v in status["present"].items()})
