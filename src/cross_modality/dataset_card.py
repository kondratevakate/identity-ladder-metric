"""Dataset card table: one row per dataset, from committed aggregate JSONs only (no rows, no IDs).

Columns: persons (families for brain and IMDb-WIKI), target images, images in the frame where the JSON records them,
window positions, rungs available, V and three ladder ratios. V = mean squared error of the dataset-mean fill (z units
squared, the dataset's own frame), from `common_all.json`; ratio = mean squared error of a rung / V, so 1 = dataset
mean and about 2 = stranger. The same computation for every dataset (cell means, `common_frame.py`); the per-modality
files compute their ratios per person and can differ slightly. Missing is never zero: a missing count or rung is
"n/a".

Usage: python -I src/cross_modality/dataset_card.py results/cross_modality <bilateral_worktree>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
BIL = sys.argv[2]
HERE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
C = json.load(open(os.path.join(OUT, "common_all.json")))["datasets"]
B = json.load(open(os.path.join(HERE, "results", "donor_ladder_brain_metrics.json")))["counts"]
Fa = json.load(open(os.path.join(HERE, "results", "face_ladder_metrics.json")))["counts"]
Fu = json.load(open(os.path.join(BIL, "results", "fundus_ladder_metrics.json")))["panel"]
Ir = json.load(open(os.path.join(HERE, "results", "iris_ladder_metrics.json")))["panel"]
QC = {"QTIM": "pre-QC", "CelebA": "QC", "IMDb-WIKI": "QC", "GRAPE": "pre-QC", "PAPILA": "pre-QC", "UTIRIS NIR": "pre-QC",
      "UTIRIS VIS": "pre-QC", "UPOL": "pre-QC"}
MOD = {"QTIM": "brain MRI", "CelebA": "face photo", "IMDb-WIKI": "face photo", "GRAPE": "fundus", "PAPILA": "fundus",
       "UTIRIS NIR": "iris NIR", "UTIRIS VIS": "iris visible", "UPOL": "iris visible"}
RUNG_SHORT = {"repeat": "repeat", "other_side": "other side", "relative": "relative", "lookalike_oracle": "best look-alike",
              "lookalike": "look-alike", "cohort_mean": "sex-age mean", "mean_k2": "mean of 2", "dataset_mean": "dataset mean",
              "stranger": "stranger", "shrink_ref": "shrunk ref.", "wrong_detail": "wrong detail", "noise": "noise"}
NA = "n/a"


def counts(d):
    """persons, target images, images in frame, positions; n/a where the JSON does not record it."""
    if d == "QTIM":
        return (f"{B['n_families']} families", B["n_targets"], B["n_targets"] + B["n_repeat_scans"], B["n_positions"])
    if d in ("CelebA", "IMDb-WIKI"):
        c = Fa["celeba" if d == "CelebA" else "imdb"]
        return (f"{c['n_families']} families" if d == "IMDb-WIKI" else c["n_families"], c["n_targets"], NA, c["n_positions"])
    if d in ("GRAPE", "PAPILA"):
        c = Fu[d]["counts"]
        return (c["persons"], c["units"], NA, c["positions"])
    c = Ir[d.replace(" ", "_")]["counts"]
    return (c["persons"], c["units"], c.get("images", NA), c["positions"])


rows = []
for d, v in C.items():
    raw = v["raw"]
    mse = {r: raw[r]["mse"][0] for r in raw if "mse" in raw[r]}
    V = mse.get("dataset_mean")
    ratio = lambda r: NA if (V is None or r not in mse) else round(mse[r] / V, 2)
    rungs = [RUNG_SHORT[r] for r in RUNG_SHORT if r in mse]
    p, t, im, pos = counts(d)
    rows.append({"dataset": d, "modality": MOD[d], "status": QC[d], "persons": p, "target_images": t, "images_in_frame": im,
                 "positions": pos, "rungs_available": rungs, "n_rungs": len(rungs),
                 "V_dataset_mean_mse": NA if V is None else round(V, 3),
                 "ratio_repeat": ratio("repeat"), "ratio_other_side": ratio("other_side"), "ratio_lookalike": ratio("lookalike"),
                 "ratio_stranger": ratio("stranger")})
json.dump({"method": __doc__.strip(), "rows": rows}, open(os.path.join(OUT, "dataset_card.json"), "w"), indent=1)

cols = [("dataset", "dataset"), ("modality", "modality"), ("status", "status"), ("persons", "persons"),
        ("target_images", "target\nimages"), ("images_in_frame", "images\nin frame"), ("positions", "windows"),
        ("n_rungs", "rungs"), ("V_dataset_mean_mse", "V (MSE of\ndataset mean)"), ("ratio_repeat", "repeat\n/ V"),
        ("ratio_other_side", "other side\n/ V"), ("ratio_lookalike", "look-alike\n/ V"), ("ratio_stranger", "stranger\n/ V")]
fig, ax = plt.subplots(figsize=(13.5, 0.36 * len(rows) + 1.3))
ax.axis("off")
cell = [[str(r[k]) for k, _ in cols] for r in rows]
t = ax.table(cellText=cell, colLabels=[h for _, h in cols], loc="upper center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(8)
t.scale(1, 1.6)
for (i, j), c in t.get_celld().items():
    c.set_edgecolor("#dddddd")
    if i == 0:
        c.set_facecolor("#f1efe9")
        c.set_height(c.get_height() * 1.5)
    elif cell[i - 1][j] == NA:
        c.get_text().set_color("#999999")
ax.set_title("Dataset card: who is in each dataset and where its ladder sits (V = error of the dataset mean)", fontsize=10,
             loc="left")
fig.text(0.01, 0.02, "Ratios = mean squared error of the rung / V, same computation for every dataset; 1 = dataset mean, "
         "about 2 = stranger. n/a = not recorded or rung absent (never 0).\nRungs listed in dataset_card.json. Brain, fundus "
         "and iris pre-QC; faces QC approved. V is in each dataset's own z-scored frame, so V values are not comparable "
         "across modalities.", fontsize=7.5, color="#333")
plt.savefig(os.path.join(OUT, "dataset_card.png"), dpi=160, facecolor="white", bbox_inches="tight")
for r in rows:
    print(r["dataset"], r["persons"], r["target_images"], r["n_rungs"], r["V_dataset_mean_mse"], r["ratio_repeat"],
          r["ratio_lookalike"], r["ratio_stranger"])
