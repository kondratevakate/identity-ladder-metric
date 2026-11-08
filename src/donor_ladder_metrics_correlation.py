"""Which metrics say the same thing? Rank correlation across ladder rungs, per target, then clustering
(PROTOCOL_donor_ladder_brain.md, Amendment 12, last item: a reduced panel of weakly correlated metrics). Pre-QC.

For every cell (target, position, side) the rungs are ranked by each metric's distance d (lower = closer;
similarities negated) and Spearman rho is taken between every pair of metrics; rho is averaged over cells (Fisher z).
Calibrated variants (Tnorm z, EVT, posterior) are monotone per cell, so they rank like their base metric and are left
out. Clusters: average linkage on 1 - rho, cut at rho = CUT. Segmentation metrics (Amendment 15) enter only on the
boxes of the segmented targets. Aggregates only.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain_metrics as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_JSON = os.path.join(ROOT, "results", "donor_ladder_metrics_correlation.json")
# rungs present for (almost) every target: repeat and relatives exist for few targets and are left out of the ranks
RUNGS = ("mirror", "lookalike", "stranger", "mean_k2", "dataset_mean", "lookalike_oracle", "wrong_detail", "noise")
STRANGERS = tuple(f"str{j:02d}" for j in range(M.N_STRANGERS))   # finer view: which wrong person is worse
CUT = 0.8
SEG = os.path.join(M.B.PROJ, "donor-ladder-brain-seg-20261106-v1")
SEG_RUNGS = ("repeat", "mirror", "sibling", "stranger", "lookalike", "lookalike_oracle", "mean_k2", "dataset_mean",
             "noise")


def distances(r, metrics):
    """Every metric as a distance (lower = closer): similarities negated, segmentation scores as 1 - score."""
    d = r.copy()
    for m in metrics:
        if m in ("seg_dice", "seg_acc"):
            d[m] = 1 - d[m]
        elif m not in M.LOWER_BETTER:
            d[m] = -d[m]
    return d


def _rho(a, b):
    """Spearman per cell between a and b (cells, rungs)."""
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ra * rb).sum(1) / np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1))


def cell_rho(d, metrics, rungs, keys):
    """Spearman per cell over the rungs, for every metric pair: (cells, metrics, metrics) and the cell count.
    A pair uses the rungs where both metrics are defined (acc_dataset has no value for the dataset mean); cells
    missing any of those rungs are NaN for that pair and drop out of the mean."""
    d = d[d.fill.isin(rungs)]
    cube = d.set_index(keys + ["fill"])[list(metrics)].unstack("fill").reindex(columns=pd.MultiIndex.from_product(
        [list(metrics), list(rungs)]))
    if len(cube) == 0:
        return None, 0
    a = np.stack([cube[m].to_numpy(dtype=float) for m in metrics], axis=2)       # cells, rungs, metrics
    defined = ~np.isnan(a).all(0)                                                 # rungs, metrics
    rho = np.full((len(cube), len(metrics), len(metrics)), np.nan)
    for p in range(len(metrics)):
        for q in range(p, len(metrics)):
            use = defined[:, p] & defined[:, q]
            x, y = a[:, use, p], a[:, use, q]
            ok = ~(np.isnan(x).any(1) | np.isnan(y).any(1))
            v = np.full(len(cube), np.nan)
            if use.sum() >= 4 and ok.any():
                v[ok] = _rho(x[ok], y[ok])
            rho[:, p, q] = rho[:, q, p] = v
    n = int((~np.isnan(rho).all((1, 2))).sum())
    return rho, n


def fisher_mean(rho):
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    return np.tanh(np.nanmean(z, axis=0))


def cluster(R, metrics, cut=CUT):
    D = 1 - R
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    Z = linkage(squareform(np.clip(D, 0, None), checks=False), "average")
    lab = fcluster(Z, 1 - cut, criterion="distance")
    groups = {}
    for m, g in zip(metrics, lab):
        groups.setdefault(int(g), []).append(m)
    return sorted(groups.values(), key=lambda g: metrics.index(g[0])), Z.tolist()


def seg_rows(r):
    path = os.path.join(SEG, "scores_fastsurfer.csv")
    if not os.path.exists(path):
        path = os.path.join(SEG, "scores_synthseg.csv")
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path)
    s = s[~s.failed & s.fill.isin(SEG_RUNGS)]
    pis = {"deep": 32, "parietal": 33}
    s = s.assign(patch=s.box.map(pis)).rename(columns={"dice_mean": "seg_dice", "accuracy": "seg_acc"})
    base = r[r.patch.isin(pis.values()) & r.drawn & r.fill.isin(SEG_RUNGS)]
    return base.merge(s[["target", "patch", "fill", "seg_dice", "seg_acc"]], on=["target", "patch", "fill"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=float, default=CUT)
    a = ap.parse_args()
    metrics = list(M.METRICS)
    r = pd.read_csv(os.path.join(M.PRIVATE, "rows.csv.gz"), usecols=["target", "patch", "side", "drawn", "fill"]
                    + metrics)
    pos_names = {pi: name for pi, (name, _, _) in enumerate(M.B.positions(np.random.default_rng(M.SEED))[0])}
    r["region"] = r.patch.map(pos_names)
    out = {"status": "pre-QC: affine registration only", "rungs": list(RUNGS), "cut_rho": a.cut,
           "method": "Spearman over rungs per (target, position, side), Fisher-z mean over cells; average linkage "
                     "on 1 - rho", "metrics": metrics, "views": {}}
    rp, bx = r[r.region == "random_patches"], r[r.region != "random_patches"]
    for view, sub, rungs in (("random_patches", rp, RUNGS), ("boxes", bx, RUNGS),
                             ("random_patches_strangers", rp, STRANGERS)):
        rho, n = cell_rho(distances(sub, metrics), metrics, rungs, ["target", "patch", "side"])
        R = fisher_mean(rho)
        groups, Z = cluster(R, metrics, a.cut)
        out["views"][view] = {"n_cells": n, "rungs": list(rungs), "rho": np.round(R, 3).tolist(), "clusters": groups, "linkage": Z}
        print(view, n, groups)
    s = seg_rows(r)
    if s is not None and len(s):
        mm = metrics + ["seg_dice", "seg_acc"]
        rho, n = cell_rho(distances(s, mm), mm, SEG_RUNGS, ["target", "patch"])
        if rho is not None:
            R = fisher_mean(rho)
            groups, Z = cluster(R, mm, a.cut)
            out["views"]["boxes_with_segmentation"] = {"n_cells": n, "metrics": mm, "rungs": list(SEG_RUNGS),
                                                       "rho": np.round(R, 3).tolist(), "clusters": groups,
                                                       "linkage": Z}
            print("boxes_with_segmentation", n, groups)
    else:
        out["views"]["boxes_with_segmentation"] = {"not_run": "no segmentation scores yet (Amendment 15)"}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(M.finite(out), fh, indent=1, allow_nan=False)


if __name__ == "__main__":
    main()
