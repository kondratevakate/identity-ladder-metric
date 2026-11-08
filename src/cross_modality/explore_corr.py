"""Per-target metric correlation across identity-ladder rungs, all datasets (read-only, aggregates only).

Method ported from src/donor_ladder_metrics_correlation.py (brain session): every metric is turned into a
distance (lower = closer), rungs are ranked within each cell (target, position[, side]), Spearman per cell,
Fisher-z mean over cells, average linkage on 1 - rho. Output contains no IDs, only matrices and counts.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

P = "C:/Projects/data/projects/identity-ladder-metric/"
OUT = sys.argv[1]
CUT = 0.8
STR = [f"str{j:02d}" for j in range(16)]
LOWER = {"mae", "mse", "rmse", "rmse_rep", "ampdev_dataset", "ampdev_cohort", "persistence"}

BASE = ["mse", "rmse", "mae", "psnr", "psnr_fixed", "ssim", "pearson", "acc_dataset", "acc_cohort", "lddt",
        "tm_like", "gdt_like", "lddt_rep", "rmse_rep"]
SOURCES = {
    "QTIM brain (check)": dict(path=P + "donor-ladder-brain-metrics-20261106-v3/rows.csv.gz", keys=["target", "patch", "side"],
                               mirror="mirror", metrics=BASE, filt=("patch_lt", 32)),
    "CelebA": dict(path=P + "face-ladder-metrics-20261106-v1/rows_v6f.csv.gz", keys=["target", "patch"], mirror="mirror",
                   metrics=BASE + ["arcface_cos"], filt=("src", "celeba")),
    "IMDb-WIKI": dict(path=P + "face-ladder-metrics-20261106-v1/rows_v6f.csv.gz", keys=["target", "patch"], mirror="mirror",
                      metrics=BASE + ["arcface_cos"], filt=("src", "imdb")),
    "GRAPE": dict(path=P + "fundus-ladder-metrics-20261106-v1/rows_GRAPE_p8_full.csv.gz", keys=["unit", "patch"],
                  mirror="fellow", metrics=BASE + ["ms_ssim", "persistence", "ampdev_dataset", "ampdev_cohort"], filt=None),
    "PAPILA": dict(path=P + "fundus-ladder-metrics-20261106-v1/rows_PAPILA_p9_full.csv.gz", keys=["unit", "patch"],
                   mirror="fellow", metrics=BASE + ["ms_ssim", "persistence", "ampdev_dataset", "ampdev_cohort"], filt=None),
}


def rho_cells(a, b):
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ra * rb).sum(1) / np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1))


def matrix(d, metrics, rungs, keys):
    d = d[d.fill.isin(rungs)]
    cube = d.set_index(keys + ["fill"])[metrics].unstack("fill").reindex(
        columns=pd.MultiIndex.from_product([metrics, rungs]))
    a = np.stack([cube[m].to_numpy(dtype=float) for m in metrics], axis=2)
    defined = ~np.isnan(a).all(0)
    R = np.full((len(metrics), len(metrics)), np.nan)
    N = np.zeros((len(metrics), len(metrics)), dtype=int)
    for p in range(len(metrics)):
        for q in range(p, len(metrics)):
            use = defined[:, p] & defined[:, q]
            if use.sum() < 4:
                continue
            x, y = a[:, use, p], a[:, use, q]
            ok = ~(np.isnan(x).any(1) | np.isnan(y).any(1))
            v = rho_cells(x[ok], y[ok])
            v = v[~np.isnan(v)]
            if len(v):
                R[p, q] = R[q, p] = np.tanh(np.mean(np.arctanh(np.clip(v, -0.999999, 0.999999))))
                N[p, q] = N[q, p] = len(v)
    return R, N, len(cube)


def clusters(R, metrics):
    ok = ~np.isnan(R).all(0)
    m2 = [m for m, o in zip(metrics, ok) if o]
    D = 1 - np.nan_to_num(R[np.ix_(ok, ok)], nan=0.0)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(np.clip((D + D.T) / 2, 0, None), checks=False), "average")
    lab = fcluster(Z, 1 - CUT, criterion="distance")
    g = {}
    for m, l in zip(m2, lab):
        g.setdefault(int(l), []).append(m)
    order = [m2[i] for i in __import__("scipy.cluster.hierarchy", fromlist=["leaves_list"]).leaves_list(Z)]
    return sorted(g.values(), key=lambda x: m2.index(x[0])), order


out = {}
for name, s in SOURCES.items():
    cols = s["keys"] + ["fill"] + s["metrics"] + (["src"] if s["filt"] and s["filt"][0] == "src" else [])
    r = pd.read_csv(s["path"], usecols=lambda c: c in set(cols))
    if s["filt"] and s["filt"][0] == "src":
        r = r[r.src == s["filt"][1]]
    elif s["filt"]:
        r = r[r.patch < s["filt"][1]]
    if "drawn" in r.columns:
        pass
    fills = set(r.fill.unique())
    stranger = "stranger" if "stranger" in fills else "str00"   # one fixed stranger when no pooled stranger row
    for m in s["metrics"]:
        if m not in LOWER:
            r[m] = -r[m]
    views = {}
    main = [s["mirror"], "lookalike", stranger, "mean_k2", "dataset_mean", "lookalike_oracle", "wrong_detail", "noise"]
    main = [x for x in main if x in fills]
    views["main rungs"] = main
    if "repeat" in fills:
        views["with repeat (targets that have one)"] = ["repeat"] + main
    views["16 strangers only"] = [x for x in STR if x in fills]
    out[name] = {"metrics": s["metrics"], "stranger_rung_used": stranger, "views": {}}
    for vname, rungs in views.items():
        R, N, ncell = matrix(r, s["metrics"], rungs, s["keys"])
        g, order = clusters(R, s["metrics"])
        out[name]["views"][vname] = {"rungs": rungs, "n_cells": int(ncell), "n_cells_min_pair": int(N[N > 0].min()) if (N > 0).any() else 0,
                                     "rho": np.round(R, 3).tolist(), "clusters": g, "order": order}
        print(name, vname, ncell, g, flush=True)
json.dump(out, open(OUT, "w"), indent=1, allow_nan=True)
