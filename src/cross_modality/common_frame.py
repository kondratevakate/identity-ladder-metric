"""One comparison frame for all datasets, from existing private rows (aggregates only, no IDs).

R_self = (mean d(rung) - d(self)) / (mean d(16 strangers) - d(self)), means over (target, window) cells,
d = distance (similarities negated: d(self) = -1; errors: d(self) = 0). Same function for every dataset.
Metrics with no finite self value (PSNR) or no closeness orientation (amplitude deviation) are reported in
their own units: PSNR as dB minus the strangers' dB, amplitude deviation |1 - amplitude| as a raw mean.
Interval: bootstrap over persons/families (brain: targets, no family column in rows), 1000 reps.
"""
import json
import sys

import numpy as np
import pandas as pd

P = "C:/Projects/data/projects/identity-ladder-metric/"
OUT = sys.argv[1]
SIM = ["ssim", "ms_ssim", "pearson", "acc_dataset", "acc_cohort", "lddt", "lddt_rep", "tm_like", "gdt_like", "arcface_cos"]
ERR = ["mse", "rmse", "mae", "rmse_rep", "persistence"]
DB = ["psnr", "psnr_fixed"]
RAW = ["ampdev_dataset", "ampdev_cohort"]
ALL = ERR[:3] + DB + ["ssim", "ms_ssim", "pearson", "acc_dataset", "acc_cohort", "ampdev_dataset", "ampdev_cohort",
                      "lddt", "lddt_rep", "tm_like", "gdt_like", "rmse_rep", "persistence", "arcface_cos"]
STR = [f"str{j:02d}" for j in range(16)]
RUNGS = ["repeat", "other_side", "relative", "lookalike_oracle", "lookalike", "cohort_mean", "mean_k2", "dataset_mean", "wrong_detail", "noise", "shrink_ref"]
SRC = {
    "QTIM": dict(path=P + "donor-ladder-brain-metrics-20261106-v3/rows.csv.gz", cell=["target", "patch", "side"], unit="target",
                 ren={"mirror": "other_side"}, filt=lambda r: r[r.patch < 32]),
    "CelebA": dict(path=P + "face-ladder-metrics-20261106-v1/rows_v6f.csv.gz", cell=["target", "patch"], unit="family",
                   ren={"mirror": "other_side"}, filt=lambda r: r[r.src == "celeba"]),
    "IMDb-WIKI": dict(path=P + "face-ladder-metrics-20261106-v1/rows_v6f.csv.gz", cell=["target", "patch"], unit="family",
                      ren={"mirror": "other_side"}, filt=lambda r: r[r.src == "imdb"]),
    "GRAPE": dict(path=P + "fundus-ladder-metrics-20261106-v1/rows_GRAPE_p8_full.csv.gz", cell=["unit", "patch"], unit="person",
                  ren={"fellow": "other_side", "sexage": "cohort_mean"}, filt=None),
    "PAPILA": dict(path=P + "fundus-ladder-metrics-20261106-v1/rows_PAPILA_p9_full.csv.gz", cell=["unit", "patch"], unit="person",
                   ren={"fellow": "other_side", "sexage": "cohort_mean"}, filt=None),
}

IRIS = "C:/Projects/data/projects/identity-ladder-metric/iris-ladder-metrics-20261107-v1/rows_{}_P2_full.csv.gz"
for _ds, _f in (("UTIRIS NIR", "UTIRIS_NIR"), ("UTIRIS VIS", "UTIRIS_VIS"), ("UPOL", "UPOL")):
    SRC[_ds] = dict(path=IRIS.format(_f), cell=["unit", "patch"], unit="person", ren={"fellow": "other_side"}, filt=None)
rng = np.random.default_rng(20261106)
out = {"method": __doc__.strip(), "metrics": ALL, "rungs": RUNGS, "kind": {**{m: "R" for m in SIM + ERR}, **{m: "dB" for m in DB},
                                                                         **{m: "raw" for m in RAW}}, "datasets": {}}
for name, s in SRC.items():
    need = set(s["cell"] + [s["unit"], "fill", "src", "amplitude"] + ALL)
    r = pd.read_csv(s["path"], usecols=lambda c: c in need)
    if s["filt"]:
        r = s["filt"](r)
    if "ampdev_dataset" not in r.columns and "amplitude" in r.columns:
        r["ampdev_dataset"] = (1.0 - r["amplitude"]).abs()          # same definition as fundus_ladder_metrics.py
    r = r.replace([np.inf, -np.inf], np.nan)
    mets = [m for m in ALL if m in r.columns and r[m].notna().any()]
    r["fill"] = r.fill.replace(s["ren"])
    # T2 control as a rung: the reference (repeat, else other side) shrunk toward the dataset mean
    sh = "shrink_repeat" if (r.fill == "repeat").any() else next(f for f in ("shrink_other_side", "shrink_mirror", "shrink_fellow") if (r.fill == f).any())
    r.loc[r.fill == sh, "fill"] = "shrink_ref"
    # one relative rung (Kate, 7 Nov): sibling and co-twin pooled; in the brain this mixes same-age and different-age relatives
    r.loc[r.fill.isin(["sibling", "cotwin"]), "fill"] = "relative"
    for m in mets:
        if m in SIM:
            r[m] = -r[m]
    keys = list(dict.fromkeys(s["cell"] + [s["unit"]]))
    st = r[r.fill.isin(STR)].groupby(keys)[mets].mean().reset_index()
    st["fill"] = "_strangers"
    rr = pd.concat([r[r.fill.isin(RUNGS)], st], ignore_index=True)
    rr["u"] = rr[s["unit"]].astype("category").cat.codes
    nu = int(rr.u.max() + 1)
    W = rng.multinomial(nu, np.full(nu, 1 / nu), size=1000).astype(float)
    agg = {}
    for f, g in rr.groupby("fill"):
        S = np.zeros((nu, len(mets)))
        C = np.zeros((nu, len(mets)))
        for k, m in enumerate(mets):
            gg = g[g[m].notna()]
            np.add.at(S[:, k], gg.u.to_numpy(), gg[m].to_numpy())
            np.add.at(C[:, k], gg.u.to_numpy(), 1)
        agg[f] = (S, C)
    S0, C0 = agg["_strangers"]
    res = {}
    raw = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for f in RUNGS + ["_strangers"]:
            if f not in agg:
                continue
            S, C = agg[f]
            res[f] = {}
            for k, m in enumerate(mets):
                if C[:, k].sum() == 0 or C0[:, k].sum() == 0:
                    continue
                a, ab = S[:, k].sum() / C[:, k].sum(), (W @ S[:, k]) / (W @ C[:, k])
                z, zb = S0[:, k].sum() / C0[:, k].sum(), (W @ S0[:, k]) / (W @ C0[:, k])
                if m in DB:
                    v, vb = a - z, ab - zb
                elif m in RAW:
                    v, vb = a, ab
                else:
                    d0 = -1.0 if m in SIM else 0.0
                    v, vb = (a - d0) / (z - d0), (ab - d0) / (zb - d0)
                sg = -1.0 if m in SIM else 1.0
                rlo, rhi = sorted(np.nanpercentile(sg * ab, [2.5, 97.5]))
                raw.setdefault("stranger" if f == "_strangers" else f, {})[m] = [round(float(sg * a), 5), round(float(rlo), 5),
                                                                                   round(float(rhi), 5), int((C[:, k] > 0).sum())]
                lo, hi = np.nanpercentile(vb, [2.5, 97.5])
                res[f][m] = [round(float(v), 4), round(float(lo), 4), round(float(hi), 4), int((C[:, k] > 0).sum())]
    out["datasets"][name] = {"unit": s["unit"], "n_units": nu, "metrics": mets, "R_self": res, "raw": raw}
    print(name, nu, mets, flush=True)
json.dump(out, open(OUT, "w"), indent=1)
