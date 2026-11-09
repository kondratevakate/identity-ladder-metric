"""Rank tile figure, fixed column order. Rows = one metric on one dataset; columns = rungs in identity-ladder order
(the same for every row); tile colour and number = the place the metric gives that rung (1 = closest to the person),
from the metric's own raw values. Right margin: reference tests T1-T3b computed with one rule for every dataset
(tests_uniform.py). Missing rungs are left empty. Aggregates only."""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

SCR = sys.argv[1]
c = json.load(open(SCR + "/common_all.json"))
TST = json.load(open(SCR + "/tests_all.json"))["datasets"]
QC = {"QTIM": "pre-QC", "CelebA": "QC", "IMDb-WIKI": "QC", "GRAPE": "pre-QC", "PAPILA": "pre-QC", "UTIRIS NIR": "pre-QC", "UTIRIS VIS": "pre-QC", "UPOL": "pre-QC"}
COLS = ["repeat", "other_side", "relative", "lookalike_oracle", "lookalike", "lookalike16", "cohort_mean", "mean_k2",
        "dataset_mean", "stranger", "shrink_ref", "wrong_detail", "noise"]
LAB = {"repeat": "repeat image", "other_side": "other side (mirror / eye)", "relative": "relative (sibling or twin)", "cotwin": "co-twin",
       "lookalike_oracle": "best look-alike", "lookalike": "look-alike", "lookalike16": "mean of 16 look-alikes", "cohort_mean": "sex-age mean", "mean_k2": "mean of 2 strangers",
       "dataset_mean": "dataset mean   [T1]", "stranger": "16 strangers", "shrink_ref": "repeat shrunk to mean   [T2]",
       "wrong_detail": "wrong detail   [T3a]", "noise": "white noise   [T3b]"}
TCOL = [("T1", "T1 repeat beats mean"), ("T2", "T2 repeat beats repeat shrunk to mean"), ("T3a", "T3a repeat beats wrong detail"),
        ("T3b", "T3b repeat beats white noise")]
FAM = [("BraTS panel", ["mse", "rmse", "mae", "psnr", "psnr_fixed", "ssim"]),
       ("Correlation and anomaly", ["pearson", "acc_dataset", "acc_cohort", "ampdev_dataset", "ampdev_cohort"]),
       ("Structure", ["ms_ssim", "lddt", "lddt_rep", "tm_like", "gdt_like", "rmse_rep", "persistence"]),
       ("Identity", ["arcface_cos"])]
MN = {"mse": "MSE", "rmse": "RMSE", "mae": "MAE", "psnr": "PSNR", "psnr_fixed": "PSNR, fixed range", "ssim": "SSIM",
      "pearson": "Pearson r", "acc_dataset": "ACC vs dataset mean", "acc_cohort": "ACC vs matched mean",
      "ampdev_dataset": "|1 - amplitude|, dataset mean", "ampdev_cohort": "|1 - amplitude|, matched mean", "ms_ssim": "MS-SSIM",
      "lddt": "lDDT-like", "lddt_rep": "lDDT, tolerance = repeat noise", "tm_like": "TM-like", "gdt_like": "GDT-like",
      "rmse_rep": "RMSE / repeat noise", "persistence": "intensity persistence", "arcface_cos": "ArcFace identity cosine"}
LOW = {"mse", "rmse", "mae", "rmse_rep", "persistence", "ampdev_dataset", "ampdev_cohort"}
cmap = LinearSegmentedColormap.from_list("rank", ["#0b5d63", "#3aa6a8", "#bfe3dc", "#f6e7c8", "#f0a868", "#b7471a"])

MODE = sys.argv[2] if len(sys.argv) > 2 else "metric"     # "metric": blocks by metric; "dataset": blocks by dataset
RANK = {}
for fam, ms in FAM:
    for m in ms:
        for d, v in c["datasets"].items():
            raw = v["raw"]
            it = [(r, raw[r][m][0]) for r in COLS if r in raw and m in raw[r]]
            if len(it) < 3:
                continue
            order = sorted(it, key=lambda x: x[1] if m in LOW else -x[1])
            RANK[(m, d)] = ({r: i + 1 for i, (r, _) in enumerate(order)}, len(order))
json.dump({f"{m}|{d}": v[0] for (m, d), v in RANK.items()}, open(SCR + "/ranks_all.json", "w"), indent=1)
rows = []
if MODE == "metric":
    for fam, ms in FAM:
        blk = [(m, d) + RANK[(m, d)] for m in ms for d in c["datasets"] if (m, d) in RANK]
        if blk:
            rows.append(("fam", fam))
            last = None
            for b in blk:
                if last is not None and b[0] != last:
                    rows.append(("gap", None))
                last = b[0]
                rows.append(("row", b))
else:
    for d in c["datasets"]:
        rows.append(("fam", f"{d}  ({QC.get(d, '')})"))
        first = True
        for fam, ms in FAM:
            blk = [(m, d) + RANK[(m, d)] for m in ms if (m, d) in RANK]
            if blk and not first:
                rows.append(("gap", None))
            first = first and not blk
            rows += [("row", b) for b in blk]
        rows.append(("gap", None))

TW, TH = 1.0, 0.92
X0 = len(COLS) * TW + 0.6
TCOL = []   # tests removed from this figure (Kate, 7 Nov); they stay in tests_all.json
H = sum(1.0 if k in ("row", "fam") else 0.35 for k, _ in rows) + 10
fig, ax = plt.subplots(figsize=(11.5, 0.24 * H + 0.8))
TB = np.load(SCR + "/drawn_face_thumbs.npz")
TCTRL = {"dataset_mean": "T1", "shrink_ref": "T2", "wrong_detail": "T3a", "noise": "T3b"}
TY0, TY1 = -2.0, -0.2          # thumbnail band (y grows downward)
HY = TY0 - 0.15


def thumb(x, key):
    ax.imshow(TB[key], extent=(x, x + TW * 0.95, TY1, TY0), interpolation="nearest", zorder=2, aspect="auto")


for j, r in enumerate(COLS):
    ax.text(j * TW + TW / 2, HY, LAB[r], fontsize=7.5, ha="left", va="bottom", rotation=50, rotation_mode="anchor",
            color="#7a2e0e" if r in TCTRL else "#222", fontweight="bold" if r in TCTRL else "normal")
    if r in TB.files:
        thumb(j * TW, r)
    else:
        ax.add_patch(Rectangle((j * TW, TY0), TW * 0.95, TY1 - TY0, facecolor="#f4f4f4", edgecolor="#ccc", linewidth=0.5))
        ax.text(j * TW + TW * 0.475, (TY0 + TY1) / 2, "not|drawn".replace("|", chr(10)), fontsize=5.5, ha="center", va="center", color="#888")
thumb(-1.25, "target")
ax.text(-1.25 + TW * 0.475, TY1 + 0.08, "the person", fontsize=6.5, ha="center", va="top", color="#333")
ax.text(-1.45, (TY0 + TY1) / 2, "drawn face, no real person:|the red box (brow and eye)|is replaced by each rung;|the rest stays the person".replace("|", chr(10)),
        fontsize=6.5, ha="right", va="center", color="#333")
y = 0.0
seen = set()
for kind, b in rows:
    if kind == "fam":
        ax.text(-5.6, y + 0.55, b.upper() if MODE == "metric" else b, fontsize=8.5, fontweight="bold", color="#555", va="center")
        y += 1.0
        continue
    if kind == "gap":
        y += 0.35
        continue
    m, d, rank, n = b
    if MODE == "metric" and m not in seen:
        seen.add(m)
        ax.text(-2.1, y + TH / 2, MN[m], fontsize=8, ha="right", va="center", fontweight="bold")
        ax.text(-2.1, y + TH / 2 + 0.85, "lower = closer" if m in LOW else "higher = closer", fontsize=6.5, ha="right",
                va="center", color="#777")
    if MODE == "metric":
        ax.text(-0.2, y + TH / 2, d, fontsize=7.5, ha="right", va="center", color="#1f6f3a" if QC.get(d) == "QC" else "#8a6400")
    else:
        ax.text(-0.2, y + TH / 2, f"{MN[m]}  ({'lower' if m in LOW else 'higher'} = closer)", fontsize=7.5, ha="right", va="center")
    for j, r in enumerate(COLS):
        if r not in rank:
            ax.add_patch(Rectangle((j * TW, y), TW * 0.95, TH, facecolor="none", edgecolor="#ddd", linewidth=0.5))
            continue
        t = (rank[r] - 1) / max(1, n - 1)
        ax.add_patch(Rectangle((j * TW, y), TW * 0.95, TH, facecolor=cmap(t), edgecolor="none"))
        ax.text(j * TW + TW * 0.475, y + TH / 2, str(rank[r]), fontsize=7, ha="center", va="center",
                color="white" if (t < 0.25 or t > 0.8) else "#222")
    tt = TST.get(d, {}).get("tests", {}).get(m, {})
    for k, (tn, _) in enumerate(TCOL):
        e = tt.get(tn, {})
        x = X0 + k * TW
        if "pass" in e:
            ok = e["pass"]
            ax.add_patch(Rectangle((x, y), TW * 0.95, TH, facecolor="#2f7d4f" if ok else "#c0392b", edgecolor="none"))
            ax.text(x + TW * 0.475, y + TH / 2, f"{round(100 * e['share_targets_pass'])}%", fontsize=6.3, ha="center",
                    va="center", color="white")
        else:
            ax.add_patch(Rectangle((x, y), TW * 0.95, TH, facecolor="none", edgecolor="#ddd", linewidth=0.5))
            ax.text(x + TW * 0.475, y + TH / 2, "n/a", fontsize=5.6, ha="center", va="center", color="#999")
    y += 1.0
ly = y + 0.8
for i in range(len(COLS)):
    ax.add_patch(Rectangle((i * TW, ly), TW * 0.95, 0.6, facecolor=cmap(i / (len(COLS) - 1)), edgecolor="none"))
ax.text(0, ly + 1.15, "1 = what the metric calls closest", fontsize=7.5, ha="left", va="center")
ax.text(len(COLS) * TW, ly + 1.15, "last = farthest", fontsize=7.5, ha="right", va="center")
ax.text(0, ly + 3.2, "Columns are in identity-ladder order for every row; a metric that follows the ladder shows a smooth teal-to-orange|"
        "gradient. Empty = rung not available. ArcFace has no averaged or noise fills.|"
        "Top: what each column puts into the compared window, on a drawn face. [T1]-[T3b] = the four reference tests: the repeat must come out|"
        "closer than this control. Test results are in the table on the page.|"
        "Brain (QTIM), GRAPE, PAPILA and iris (UTIRIS, UPOL; frame P2, lid-shadow sheets not yet re-shown) pre-QC; faces QC approved.".replace("|", chr(10)),
        fontsize=7, color="#333", ha="left", va="center")
ax.set_xlim(-8.6, len(COLS) * TW + 0.3)
ax.set_ylim(ly + 5.6, -7.6)
ax.axis("off")
plt.tight_layout()
SUF = "" if MODE == "metric" else "_by_dataset"
plt.savefig(SCR + f"/rank_tiles{SUF}.png", dpi=170, facecolor="white")
plt.savefig(SCR + f"/rank_tiles{SUF}.svg", facecolor="white")
print("rows", sum(1 for k, _ in rows if k == "row"))
