"""Agreement of rung orderings (from ranks_all.json, the ranks shown in the tiles; aggregates only).
Datasets agree: for one metric, Spearman rho between two datasets' rung ranks over the rungs both have (>= 5);
mean over metrics gives the dataset x dataset matrix; mean over dataset pairs gives each metric's cross-dataset agreement.
Metrics agree: for one dataset, rho between two metrics' rung ranks over shared rungs (>= 5); mean over datasets gives
the metric x metric matrix; mean over metric pairs gives each dataset's agreement between metrics.
MSE, RMSE and PSNR are monotone in each other on means of the same rows only approximately; equal ranks give rho = 1."""
import itertools
import json
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = sys.argv[1]
R = {tuple(k.split("|")): v for k, v in json.load(open(SCR + "/ranks_all.json")).items()}
try:  # generative, perceptual and topology block (generative_block.py); absent file = block not built yet
    R.update({tuple(k.split("|")): v for k, v in json.load(open(SCR + "/ranks_generative.json")).items()
              if tuple(k.split("|")) not in R})
except FileNotFoundError:
    pass
MET = [m for m in dict.fromkeys(m for m, _ in R) if m != "arcface_cos"];  # ArcFace: 5 rungs on one dataset only, too thin
DS = list(dict.fromkeys(d for _, d in R))
MN = {"mse": "MSE", "rmse": "RMSE", "mae": "MAE", "psnr": "PSNR", "psnr_fixed": "PSNR fixed", "ssim": "SSIM", "pearson": "Pearson r",
      "acc_dataset": "ACC vs dataset mean", "acc_cohort": "ACC vs matched mean", "ampdev_dataset": "|1-amp|, dataset",
      "ampdev_cohort": "|1-amp|, matched", "ms_ssim": "MS-SSIM", "lddt": "lDDT", "lddt_rep": "lDDT rep", "tm_like": "TM-like",
      "gdt_like": "GDT-like", "rmse_rep": "RMSE / rep", "persistence": "persistence", "arcface_cos": "ArcFace",
      "lpips_alex": "LPIPS Alex", "lpips_vgg": "LPIPS VGG", "betti_matched": "Betti matched", "betti_unmatched": "Betti unmatched",
      "iriscode_hd": "iris code HD", "fid": "FID (set)", "fwd": "FWD (set)"}

def rho(a, b):
    sh = [r for r in a if r in b]
    if len(sh) < 5:
        return None
    return float(stats.spearmanr([a[r] for r in sh], [b[r] for r in sh])[0])

def matrix(items, other, key):
    n = len(items); S = np.zeros((n, n)); C = np.zeros((n, n))
    for o in other:
        for i, j in itertools.combinations(range(n), 2):
            a, b = R.get(key(items[i], o)), R.get(key(items[j], o))
            if a and b:
                r = rho(a, b)
                if r is not None:
                    S[i, j] += r; S[j, i] += r; C[i, j] += 1; C[j, i] += 1
    with np.errstate(invalid="ignore"):
        M = S / C
    np.fill_diagonal(M, 1.0)
    return M, C

Md, Cd = matrix(DS, MET, lambda d, m: (m, d))          # dataset x dataset, averaged over metrics
Mm, Cm = matrix(MET, DS, lambda m, d: (m, d))          # metric x metric, averaged over datasets
per_metric = {}
for m in MET:
    rs = [rho(R[(m, a)], R[(m, b)]) for a, b in itertools.combinations(DS, 2) if (m, a) in R and (m, b) in R]
    rs = [r for r in rs if r is not None]
    per_metric[m] = (float(np.mean(rs)), float(np.min(rs)), len(rs)) if rs else None
per_ds = {}
for d in DS:
    rs = [rho(R[(a, d)], R[(b, d)]) for a, b in itertools.combinations(MET, 2) if (a, d) in R and (b, d) in R]
    rs = [r for r in rs if r is not None]
    per_ds[d] = (float(np.mean(rs)), float(np.min(rs)), len(rs)) if rs else None
json.dump({"method": __doc__.strip(), "datasets": DS, "metrics": MET, "dataset_matrix": np.round(Md, 3).tolist(),
           "dataset_pairs_n": Cd.astype(int).tolist(), "metric_matrix": np.round(Mm, 3).tolist(), "metric_pairs_n": Cm.astype(int).tolist(),
           "per_metric_cross_dataset": per_metric, "per_dataset_cross_metric": per_ds}, open(SCR + "/agreement.json", "w"), indent=1)
print("cross-dataset agreement per metric (mean rho, min, pairs):")
for m, v in sorted(per_metric.items(), key=lambda x: -(x[1][0] if x[1] else -9)):
    print(f"  {MN[m]:22s}", None if v is None else f"{v[0]:.2f}  min {v[1]:.2f}  n {v[2]}")
print("cross-metric agreement per dataset:")
for d, v in per_ds.items():
    print(f"  {d:11s}", None if v is None else f"{v[0]:.2f}  min {v[1]:.2f}  n {v[2]}")

cm = matplotlib.colors.LinearSegmentedColormap.from_list("r", ["#b7471a", "#f0a868", "#f7f4ee", "#7cc3bf", "#0b5d63"])
fig, ax = plt.subplots(1, 2, figsize=(17, 8.6), gridspec_kw={"width_ratios": [len(DS) + 2.5, len(MET) + 1.5]})
for a, M, lab, title, per in ((ax[0], Md, DS, "Do datasets agree?  (rho of rung order, mean over metrics)", per_ds),
                              (ax[1], Mm, [MN[m] for m in MET], "Do metrics agree?  (rho of rung order, mean over datasets)", per_metric)):
    a.imshow(M, cmap=cm, vmin=-1, vmax=1)
    for i in range(len(lab)):
        for j in range(len(lab)):
            if np.isfinite(M[i, j]):
                a.text(j, i, f"{M[i, j]:.2f}".replace("0.", ".").replace("1.00", "1"), ha="center", va="center", fontsize=6.5 if len(lab) > 10 else 8,
                       color="white" if abs(M[i, j]) > 0.65 else "#222")
    a.set_xticks(range(len(lab))); a.set_yticks(range(len(lab)))
    a.set_xticklabels(lab, rotation=55, ha="right", fontsize=7.5); a.set_yticklabels(lab, fontsize=7.5)
    a.set_title(title, fontsize=9, loc="left")
    for s in a.spines.values():
        s.set_visible(False)
fig.text(0.01, 0.015, "rho = Spearman correlation of the rung order (the tile ranks) between two datasets or two metrics, over rungs both have (at least 5). "
         "1 = same order, 0 = unrelated, -1 = reversed. ArcFace left out (5 shared rungs on one dataset).\nLPIPS, Betti, persistence, FID, FWD "
         "from fundus 9.15, iris P3, faces F2 and brain Amendment 20 (brain: LPIPS and persistence only). MSE, RMSE, PSNR and PSNR fixed give the same order by construction. Brain, fundus and iris pre-QC; faces QC approved.", fontsize=7.5, color="#333")
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.savefig(SCR + "/agreement.png", dpi=160, facecolor="white")
