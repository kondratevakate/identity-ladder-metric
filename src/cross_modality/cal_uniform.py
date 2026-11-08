"""Group-2 calibrations (brain Amendment 12, as fundus_ladder_metrics.py:calibrate) for any dataset, aggregates only.
z = T-norm against the 16 strangers of the same cell; nlp = -log10 EVT P (GEV fitted on pooled stranger z per position
[and side]); post = P(same | z) from Gaussian stranger vs repeat z; c = P(stranger worse) + P(tie)/2.
Per-target means, then means over targets; 95% CI by bootstrap over persons/families (2000 reps)."""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

P = "C:/Projects/data/projects/identity-ladder-metric/"
OUT = sys.argv[1]
ONLY = sys.argv[2:] or None
LOWER = {"mse", "rmse", "mae", "rmse_rep", "persistence"}
MET = ["mse", "rmse", "mae", "psnr", "psnr_fixed", "ssim", "ms_ssim", "pearson", "acc_dataset", "acc_cohort", "lddt", "lddt_rep",
       "tm_like", "gdt_like", "rmse_rep", "persistence", "arcface_cos"]
SRC = {
    "QTIM": dict(path=P + "donor-ladder-brain-metrics-20261106-v3/rows.csv.gz", t="target", u="target", grp=["patch", "side"], filt=lambda r: r[r.patch < 32], cell=["target", "patch", "side"]),
    "GRAPE": dict(path=P + "fundus-ladder-metrics-20261106-v1/rows_GRAPE_p8_full.csv.gz", t="unit", u="person", grp=["patch", "side"], filt=None),
    "CelebA": dict(path=P + "face-ladder-metrics-20261106-v1/rows_v6f.csv.gz", t="target", u="family", grp=["patch"], filt=lambda r: r[r.src == "celeba"]),
    "IMDb-WIKI": dict(path=P + "face-ladder-metrics-20261106-v1/rows_v6f.csv.gz", t="target", u="family", grp=["patch"], filt=lambda r: r[r.src == "imdb"]),
}
isref = lambda f: len(f) == 5 and f.startswith("str") and f[3:].isdigit()
rng = np.random.default_rng(20261102)
out = {"method": __doc__.strip(), "datasets": {}}
for name, s in SRC.items():
    if ONLY and name not in ONLY:
        continue
    r = pd.read_csv(s["path"], usecols=lambda c: c in set([s["t"], s["u"], "fill", "src", "patch", "side"] + MET))
    if s["filt"]:
        r = s["filt"](r).reset_index(drop=True)
    keys = s.get("cell", [s["t"], "patch"])
    is_ref = r.fill.map(isref).to_numpy()
    is_rep = (r.fill == "repeat").to_numpy()
    persons = np.array(sorted(r[s["u"]].unique()))
    bidx = rng.integers(0, len(persons), (2000, len(persons)))
    res = {}
    for m in [m for m in MET if m in r.columns and r[m].notna().any()]:
        o = r[m].to_numpy(np.float64)
        o = -o if m in LOWER else o
        o = np.where(np.isfinite(o), o, np.nan)
        df = r[keys + [x for x in s["grp"] if x not in keys] + ["fill"]].assign(o=o)
        ref = df[is_ref]
        g = ref.groupby(keys)["o"].agg(["mean", "std"])
        mi = pd.MultiIndex.from_frame(df[keys])
        mu, sd = g["mean"].reindex(mi).to_numpy(), g["std"].reindex(mi).to_numpy()
        z = (o - mu) / np.where(sd > 0, sd, np.nan)
        piv = ref.pivot_table(index=keys, columns="fill", values="o", aggfunc="first")
        Sm = piv.reindex(mi).to_numpy(np.float64)
        n_s = np.isfinite(Sm).sum(1)
        with np.errstate(invalid="ignore"):
            c = ((Sm < o[:, None]).sum(1) + 0.5 * (Sm == o[:, None]).sum(1)) / np.where(n_s > 0, n_s, np.nan)
        c = np.where(np.isfinite(o), c, np.nan)
        nlp, post = np.full(len(df), np.nan), np.full(len(df), np.nan)
        for _, ix in df.groupby(s["grp"]).indices.items():
            zs = z[ix][is_ref[ix]]; zs = zs[np.isfinite(zs)]
            zr = z[ix][is_rep[ix]]; zr = zr[np.isfinite(zr)]
            if zs.size >= 50:
                try:
                    cc, loc, sc = stats.genextreme.fit(zs)
                    nlp[ix] = -np.log10(np.clip(stats.genextreme.sf(z[ix], cc, loc, sc), 1e-300, 1.0))
                except Exception:  # noqa: BLE001 - failed fit stays missing (NaN), never 0
                    pass
            if zs.size >= 50 and zr.size >= 5 and np.std(zr) > 0:
                ls = stats.norm.logpdf(z[ix], zs.mean(), zs.std())
                lr = stats.norm.logpdf(z[ix], zr.mean(), zr.std())
                post[ix] = 1.0 / (1.0 + np.exp(np.clip(ls - lr, -700, 700)))
        cdf = pd.DataFrame({"t": r[s["t"]].to_numpy(), "u": r[s["u"]].to_numpy(), "fill": r.fill.to_numpy(), "z": z, "nlp": nlp, "post": post, "c": c})
        per = cdf.groupby(["fill", "t", "u"] if s["t"] != s["u"] else ["fill", "t"])[["z", "nlp", "post", "c"]].mean().reset_index()
        blk = {}
        for f, gg in per.groupby("fill"):
            if isref(f):
                continue
            blk[f] = {}
            for k in ("z", "nlp", "post", "c"):
                v = gg.dropna(subset=[k])
                if v.empty:
                    blk[f][k] = None
                    continue
                pp = v.groupby("u" if s["t"] != s["u"] else "t")[k].agg(["sum", "count"]).reindex(persons).fillna(0)
                bs = pp["sum"].to_numpy()[bidx].sum(1) / np.maximum(pp["count"].to_numpy()[bidx].sum(1), 1e-12)
                blk[f][k] = {"mean": float(v[k].mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_targets": int(len(v))}
        res[m] = blk
        print(name, m, {f: round(blk[f]["c"]["mean"], 3) for f in ("repeat", "dataset_mean", "noise") if f in blk and blk[f].get("c")}, flush=True)
    out["datasets"][name] = res
json.dump(out, open(OUT, "w"), indent=1)
