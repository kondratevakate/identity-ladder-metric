"""T1-T3b for every dataset with one function (aggregates only, no IDs).

Same rule as src/donor_ladder_brain_metrics.py:tests and fundus_ladder_metrics.py:paired_tests:
d = distance(control) - distance(reference) per cell -> mean per target -> mean per family/person;
one-sided Wilcoxon on those means (>= 10 units), Holm over all metric x test hypotheses of the dataset.
Reference = repeat image; where a dataset has none (PAPILA), the other side, and T1 is not available.
T1 control = dataset mean, T2 = reference shrunk toward the mean, T3a = wrong detail, T3b = white noise.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

P = "C:/Projects/data/projects/identity-ladder-metric/"
OUT = sys.argv[1]
SIM = {"ssim", "ms_ssim", "pearson", "acc_dataset", "acc_cohort", "lddt", "lddt_rep", "tm_like", "gdt_like", "arcface_cos",
       "psnr", "psnr_fixed"}
ALL = ["mse", "rmse", "mae", "psnr", "psnr_fixed", "ssim", "ms_ssim", "pearson", "acc_dataset", "acc_cohort", "ampdev_dataset",
       "ampdev_cohort", "lddt", "lddt_rep", "tm_like", "gdt_like", "rmse_rep", "persistence", "arcface_cos"]
SRC = {
    "QTIM": dict(path=P + "donor-ladder-brain-metrics-20261106-v3/rows.csv.gz", target=["target"], unit="target",
                 ren={"mirror": "other_side", "shrink_mirror": "shrink_other_side"}, filt=lambda r: r[r.patch < 32]),
    "CelebA": dict(path=P + "face-ladder-metrics-20261106-v1/rows_v6f.csv.gz", target=["target"], unit="family",
                   ren={"mirror": "other_side", "shrink_mirror": "shrink_other_side"}, filt=lambda r: r[r.src == "celeba"]),
    "IMDb-WIKI": dict(path=P + "face-ladder-metrics-20261106-v1/rows_v6f.csv.gz", target=["target"], unit="family",
                      ren={"mirror": "other_side", "shrink_mirror": "shrink_other_side"}, filt=lambda r: r[r.src == "imdb"]),
    "GRAPE": dict(path=P + "fundus-ladder-metrics-20261106-v1/rows_GRAPE_p8_full.csv.gz", target=["unit"], unit="person",
                  ren={"fellow": "other_side", "shrink_fellow": "shrink_other_side"}, filt=None),
    "PAPILA": dict(path=P + "fundus-ladder-metrics-20261106-v1/rows_PAPILA_p9_full.csv.gz", target=["unit"], unit="person",
                   ren={"fellow": "other_side", "shrink_fellow": "shrink_other_side"}, filt=None),
}

IRIS = "C:/Projects/data/projects/identity-ladder-metric/iris-ladder-metrics-20261107-v1/rows_{}_P2_full.csv.gz"
for _ds, _f in (("UTIRIS NIR", "UTIRIS_NIR"), ("UTIRIS VIS", "UTIRIS_VIS"), ("UPOL", "UPOL")):
    SRC[_ds] = dict(path=IRIS.format(_f), target=["unit"], unit="person", ren={"fellow": "other_side", "shrink_fellow": "shrink_other_side"}, filt=None)


def holm(ps):
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


out = {"method": __doc__.strip(), "datasets": {}}
for name, s in SRC.items():
    need = set(s["target"] + [s["unit"], "fill", "src", "patch", "amplitude"] + ALL)
    r = pd.read_csv(s["path"], usecols=lambda c: c in need)
    if s["filt"]:
        r = s["filt"](r)
    if "ampdev_dataset" not in r.columns and "amplitude" in r.columns:
        r["ampdev_dataset"] = (1.0 - r["amplitude"]).abs()
    r["fill"] = r.fill.replace(s["ren"])
    mets = [m for m in ALL if m in r.columns and r[m].notna().any()]
    for m in mets:
        r[m] = r[m].replace([np.inf, -np.inf], np.nan)
        if m in SIM:
            r[m] = -r[m]
    ref = "repeat" if (r.fill == "repeat").any() else "other_side"
    T = {"T1": "dataset_mean", "T2": "shrink_" + ref, "T3a": "wrong_detail", "T3b": "noise"}
    tkeys = list(dict.fromkeys(s["target"] + [s["unit"]]))
    res, ps = {}, []
    for m in mets:
        for tn, ctrl in T.items():
            if tn == "T1" and ref != "repeat":
                res.setdefault(m, {})[tn] = {"status": "not available (no repeat images)"}
                continue
            w = r[r.fill.isin([ref, ctrl])].pivot_table(index=tkeys + ["patch"], columns="fill", values=m).dropna()
            if ref not in w or ctrl not in w or w.empty:
                res.setdefault(m, {})[tn] = {"status": "not available (no pairs)"}
                continue
            w = w.reset_index()
            w["d"] = w[ctrl] - w[ref]  # distance of control minus distance of reference: > 0 = reference closer
            pt = w.groupby(tkeys)["d"].mean().reset_index()
            pu = pt.groupby(s["unit"])["d"].mean()
            e = {"control": ctrl, "reference": ref, "n_targets": int(len(pt)), "n_units": int(len(pu)),
                 "mean_diff": float(pt.d.mean()), "share_targets_pass": float((pt.d > 0).mean())}
            if len(pu) >= 10 and np.any(pu.to_numpy() != 0):
                e["p_one_sided"] = float(stats.wilcoxon(pu.to_numpy(), alternative="greater").pvalue)
                ps.append((m, tn, e["p_one_sided"]))
            else:
                e["status"] = "descriptive only (fewer than 10 units)"
            res.setdefault(m, {})[tn] = e
    if ps:
        for (m, tn, _), pa in zip(ps, holm(np.array([p for *_, p in ps]))):
            res[m][tn]["p_holm"] = float(pa)
            res[m][tn]["pass"] = bool(pa < 0.05)
    out["datasets"][name] = {"reference": ref, "n_hypotheses_holm": len(ps), "tests": res}
    print(name, ref, len(ps), {m: "".join("+" if res[m][t].get("pass") else ("." if "p_holm" in res[m][t] else "x") for t in T)
                               for m in mets}, flush=True)
json.dump(out, open(OUT, "w"), indent=1)
