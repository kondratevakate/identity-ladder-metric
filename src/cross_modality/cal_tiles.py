"""Calibrated score c per rung (share of the 16 matched strangers that the rung beats; 0.5 = no better than a stranger,
1 = closer than every stranger). Same column order as rank_tiles.py. Aggregates only."""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

SCR = sys.argv[1]
B = json.load(open("C:/Projects/02_academia/studies/_worktrees/identity-ladder-cvpr2027/results/donor_ladder_brain_metrics.json"))["views"]["random_patches/LR"]["calibrated"]
F = json.load(open("C:/Projects/02_academia/studies/_worktrees/identity-ladder-bilateral/results/fundus_ladder_metrics.json"))["panel"]
I = json.load(open("C:/Projects/02_academia/studies/_worktrees/identity-ladder-cvpr2027/results/iris_ladder_metrics.json"))["panel"]
Fc = json.load(open(SCR + "/cal_faces.json"))["datasets"]
REN = {"mirror": "other_side", "fellow": "other_side", "sexage": "cohort_mean"}
brain = {}
for f, blk in B.items():
    for k, e in blk.items():
        if k.startswith("c_"):
            brain.setdefault(k[2:], {})[REN.get(f, f)] = e["mean"]
def flat(src):
    out = {}
    for m, blk in src.items():
        o = {REN.get(f, f): e["c"]["mean"] for f, e in blk.items() if e and e.get("c")}
        rel = [(e["c"]["mean"], e["c"]["n_targets"]) for f, e in blk.items() if f in ("sibling", "cotwin") and e and e.get("c")]
        if rel:   # pooled relative rung, weighted by targets
            o["relative"] = sum(v * n for v, n in rel) / sum(n for _, n in rel)
        out[m] = o
    return out
DS = {"QTIM": flat(json.load(open(SCR + "/cal_brain.json"))["datasets"]["QTIM"]), "CelebA": flat(Fc["CelebA"]), "IMDb-WIKI": flat(Fc["IMDb-WIKI"]), "GRAPE": flat(F["GRAPE"]["calibrated"]),
      "PAPILA": flat(F["PAPILA"]["calibrated"]), "UTIRIS NIR": flat(I["UTIRIS_NIR"]["calibrated"]),
      "UTIRIS VIS": flat(I["UTIRIS_VIS"]["calibrated"]), "UPOL": flat(I["UPOL"]["calibrated"])}
json.dump(DS, open(SCR + "/cal_c_all.json", "w"), indent=1)
COLS = ["repeat", "other_side", "relative", "lookalike_oracle", "lookalike", "cohort_mean", "mean_k2", "dataset_mean", "wrong_detail", "noise"]
LAB = {"repeat": "repeat image", "other_side": "other side (mirror / eye)", "relative": "relative (sibling or twin)", "lookalike_oracle": "best look-alike",
       "lookalike": "look-alike", "cohort_mean": "sex-age mean", "mean_k2": "mean of 2 strangers", "dataset_mean": "dataset mean",
       "wrong_detail": "wrong detail", "noise": "white noise"}
MS = [("mse", "MSE"), ("ssim", "SSIM"), ("ms_ssim", "MS-SSIM"), ("pearson", "Pearson r"), ("acc_cohort", "ACC vs matched mean"),
      ("lddt", "lDDT-like"), ("persistence", "intensity persistence"), ("arcface_cos", "ArcFace identity cosine")]
cmap = LinearSegmentedColormap.from_list("c", ["#b7471a", "#f0a868", "#f7f4ee", "#7cc3bf", "#0b5d63"])
rows = []
for m, mn in MS:
    blk = [(d, v[m]) for d, v in DS.items() if m in v and len(v[m]) >= 2]
    if blk:
        rows.append(("m", mn)); rows += [("r", b) for b in blk]; rows.append(("g", None))
H = sum(1 if k != "g" else 0.4 for k, _ in rows) + 6
fig, ax = plt.subplots(figsize=(10.5, 0.24 * H + 0.6))
for j, r in enumerate(COLS):
    ax.text(j + 0.5, -0.25, LAB[r], fontsize=7.5, rotation=50, rotation_mode="anchor", ha="left", va="bottom")
y = 0
for k, b in rows:
    if k == "g":
        y += 0.4; continue
    if k == "m":
        ax.text(-2.6, y + 0.5, b, fontsize=8, fontweight="bold", va="center", ha="right"); y += 1; continue
    d, v = b
    ax.text(-0.2, y + 0.46, d, fontsize=7.5, ha="right", va="center", color="#1f6f3a" if d in ("CelebA", "IMDb-WIKI") else "#8a6400")
    for j, r in enumerate(COLS):
        if r not in v:
            ax.add_patch(Rectangle((j, y), 0.95, 0.92, facecolor="none", edgecolor="#ddd", lw=0.5)); continue
        c = v[r]
        ax.add_patch(Rectangle((j, y), 0.95, 0.92, facecolor=cmap(c), edgecolor="none"))
        ax.text(j + 0.475, y + 0.46, f"{round(100 * c)}", fontsize=7, ha="center", va="center", color="white" if abs(c - 0.5) > 0.32 else "#222")
    y += 1
for i in range(11):
    ax.add_patch(Rectangle((i, y + 0.8), 0.95, 0.6, facecolor=cmap(i / 10), edgecolor="none"))
ax.text(0, y + 1.9, "0: every stranger is closer", fontsize=7.5, va="center")
ax.text(5.5, y + 1.9, "50: like a stranger", fontsize=7.5, va="center", ha="center")
ax.text(11, y + 1.9, "100: closer than all 16 strangers", fontsize=7.5, va="center", ha="right")
ax.text(0, y + 3.4, "c x 100 = share of the 16 matched strangers (same window) that the rung beats, ties half; mean over targets.\n"
        "Same computation for every dataset (brain Amendment 12, group 2). Faces and brain c computed here from saved rows (validated on GRAPE and brain z).\n"
        "Brain, fundus and iris pre-QC; faces QC approved. PAPILA has no repeat images.", fontsize=7, color="#333", va="center")
ax.set_xlim(-6, 11.2); ax.set_ylim(y + 4.4, -4.6); ax.axis("off")
plt.tight_layout(); plt.savefig(SCR + "/cal_tiles.png", dpi=170, facecolor="white")
print("rows", sum(1 for k, _ in rows if k == "r"))
