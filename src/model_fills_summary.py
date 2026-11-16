"""Model fills on the random-person scale at f = 1/8 (aggregates only, from private rows).

For QTIM, CelebA and IMDb-WIKI the stored rows of every rung and the rows of the model fills (`model_fills_face.py`,
`model_fills_brain.py`) go through `rq4_window_size.frame` (the R_self of `cross_modality/common_frame.py`, unchanged
arithmetic and bootstrap seed). Gate: every reference rung and metric must equal `results/cross_modality/
common_f8_v1.json`; a model run on a subset of targets is summarised on that subset and marked so (no gate there).

Directional comparisons (model minus reference, in R units, paired over the cells both have, 95 % percentile
interval from 1000 multinomial bootstrap draws over the dataset's unit) are reported for the second image, the
dataset average, the mean of the 16 nearest people and the softened same person.

Usage: python src/model_fills_summary.py            writes results/model_fills_v1/model_fills_f8.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import rq4_window_size as RW  # noqa: E402

ROOT = RW.ROOT
P = RW.P
RUN = P + "model-fills-20261116-v1/"
OUT = os.path.join(ROOT, "results", "model_fills_v1", "model_fills_f8.json")
COMMON8 = os.path.join(ROOT, "results", "cross_modality", "common_f8_v1.json")
REFS = ("repeat", "dataset_mean", "lookalike16", "shrink_ref")
SHOWN = ("mse", "ssim", "tm_like", "gdt_like", "lddt")
SETS = {   # order and seeds as in the producers of common_f8_v1.json, so the reference intervals reproduce
    "QTIM": dict(paths=[P + "donor-ladder-brain-metrics-20261106-v3/rows.csv.gz"],
                 extra=[P + "donor-ladder-brain-lookalike16-20261109-v1/parts/p*.csv.gz"],
                 cell=["target", "patch", "side"], unit="target", ren={"mirror": "other_side"},
                 filt=lambda r: r[r.patch < 32], model_rows=RUN + "brain/{m}/rows.csv.gz", seed_group="brain"),
    "CelebA": dict(paths=[RW.RQ + "faces/rows_full_w10.csv.gz"], cell=["target", "patch"], unit="family",
                   ren={"mirror": "other_side"}, filt=("face", "celeba"),
                   model_rows=RUN + "faces_v2/{m}/rows.csv.gz", seed_group="faces"),
    "IMDb-WIKI": dict(paths=[RW.RQ + "faces/rows_full_w10.csv.gz"], cell=["target", "patch"], unit="family",
                      ren={"mirror": "other_side"}, filt=("face", "imdb"),
                      model_rows=RUN + "faces_v2/{m}/rows.csv.gz", seed_group="faces"),
}
MODELS = {"QTIM": ["brats24_unet"], "CelebA": ["lama_places", "lama_celebahq", "sd15_inpaint"],
          "IMDb-WIKI": ["lama_places", "lama_celebahq", "sd15_inpaint"]}


def model_rows(spec, name, model, need):
    path = spec["model_rows"].format(m=model)
    if not glob.glob(path):
        return None
    r = pd.read_csv(path, usecols=lambda c: c in need)
    r = r[r.fill == f"model_{model}"]
    f = spec.get("filt")
    if isinstance(f, tuple):
        r = r[r.src == f[1]]
    return r


def paired(r, cell, unit, m, a, b, rng):
    """Mean R difference a - b on the cells where both are scored, with the strangers of the same cells."""
    sgn, d0 = (-1.0, -1.0) if m in RW.SIM else (1.0, 0.0)
    keys = list(dict.fromkeys(cell + [unit]))
    piv = {f: r[r.fill == f].groupby(keys)[m].mean() * sgn for f in (a, b)}
    piv["s"] = r[r.fill.isin(RW.STR)].groupby(keys)[m].mean() * sgn
    j = pd.concat([v.rename(k) for k, v in piv.items()], axis=1).dropna().reset_index()
    if j.empty:
        return None
    u = j[unit].astype("category").cat.codes.to_numpy()
    nu = int(u.max() + 1)
    S = np.zeros((nu, 3))
    for k, c in enumerate((a, b, "s")):
        np.add.at(S[:, k], u, j[c].to_numpy())
    C = np.bincount(u, minlength=nu).astype(float)
    W = rng.multinomial(nu, np.full(nu, 1 / nu), size=RW.REPS).astype(float)

    def diff(w):
        n = w @ C
        ma, mb, ms = (w @ S[:, 0]) / n, (w @ S[:, 1]) / n, (w @ S[:, 2]) / n
        return (ma - mb) / (ms - d0)
    v = float(diff(np.ones((1, nu)))[0])
    lo, hi = np.nanpercentile(diff(W), [2.5, 97.5])
    return [round(v, 4), round(float(lo), 4), round(float(hi), 4), int(len(j)), nu]


def main():
    ref8 = json.load(open(COMMON8))["datasets"]
    rngs = {"brain": np.random.default_rng(RW.SEED), "faces": np.random.default_rng(RW.SEED)}
    out = {"method": __doc__.strip(), "f": "1/8", "scale": "R_self: target 0, mean of 16 random people 1; lower = closer",
           "datasets": {}}
    for name, spec in SETS.items():
        need = RW.need_cols(spec["cell"], spec["unit"], RW.ALL)
        base, why = RW.load(spec, RW.ALL)
        if base is None:
            raise SystemExit(f"{name}: {why}")
        mets = [m for m in RW.ALL if m in base.columns and base[m].notna().any()]
        # gate: the stored rows alone reproduce common_f8_v1.json (same generator stream as its producer)
        fr = RW.frame(base, spec["cell"], spec["unit"], mets, rngs[spec["seed_group"]])
        worst, worst_ci, n_cmp = 0.0, 0.0, 0
        for rung, mm in ref8[name]["R_self"].items():
            for m, v in mm.items():
                got = fr["R_self"].get(rung, {}).get(m)
                if got is None:
                    raise SystemExit(f"{name}: reference {rung}/{m} missing in the recomputation")
                worst = max(worst, abs(got[0] - v[0]))
                worst_ci = max(worst_ci, abs(got[1] - v[1]), abs(got[2] - v[2]))
                n_cmp += 1
        if worst > 1e-9:
            raise SystemExit(f"{name}: recomputed references differ from common_f8_v1.json by {worst}")
        ds = {"unit": spec["unit"], "n_units": fr["n_units"],
              "gate": {"reference_values_compared": n_cmp, "max_abs_diff_to_common_f8_v1": worst,
                       "max_abs_diff_of_interval_ends": worst_ci, "passed": True},
              "references": {k: {m: fr["R_self"][k][m] for m in SHOWN if m in fr["R_self"].get(k, {})}
                             for k in ("repeat", "other_side", "lookalike", "lookalike16", "dataset_mean", "shrink_ref",
                                       "wrong_detail", "noise") if k in fr["R_self"]},
              "models": {}}
        for model in MODELS[name]:
            mr = model_rows(spec, name, model, need | {"src"})
            if mr is None or mr.empty:
                ds["models"][model] = {"status": "not run"}
                continue
            fill = f"model_{model}"
            tset = mr[["target"]].drop_duplicates()
            n_all = base[base.fill == "dataset_mean"].target.nunique()
            sub = base.merge(mr[spec["cell"]].drop_duplicates(), on=spec["cell"])   # same targets and regions
            r = pd.concat([sub, mr.reindex(columns=sub.columns)], ignore_index=True)
            full = int(len(tset)) == int(n_all)
            saved = RW.RUNGS
            RW.RUNGS = saved + [fill]
            try:
                fm = RW.frame(r, spec["cell"], spec["unit"], mets, np.random.default_rng(RW.SEED))
            finally:
                RW.RUNGS = saved
            e = {"n_targets": int(len(tset)), "n_targets_dataset": int(n_all), "coverage": "all targets" if full
                 else "subset of targets; the reference values below are recomputed on the same subset",
                 "R_self": {m: fm["R_self"][fill][m] for m in SHOWN if m in fm["R_self"].get(fill, {})},
                 "R_self_all_metrics": fm["R_self"].get(fill, {}),
                 "references_same_targets": {k: {m: fm["R_self"][k][m] for m in SHOWN if m in fm["R_self"].get(k, {})}
                                             for k in REFS if k in fm["R_self"]},
                 "model_minus_reference": {}}
            prng = np.random.default_rng([RW.SEED, 77])
            for ref in REFS:
                if not (r.fill == ref).any():
                    continue
                e["model_minus_reference"][ref] = {m: paired(r, spec["cell"], spec["unit"], m, fill, ref, prng)
                                                   for m in SHOWN if m in mets}
            ds["models"][model] = e
            print(name, model, {m: e["R_self"][m][0] for m in e["R_self"]}, flush=True)
        out["datasets"][name] = ds
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    print(OUT, flush=True)
    size_curve()


OUT_SIZE = os.path.join(ROOT, "results", "model_fills_v1", "model_fills_size_faces.json")
SIZE_MODELS = ["lama_celebahq", "lama_places"]
SIZE_DIR = {5: "{m}_w5_n100", 10: "{m}", 16: "{m}_w16", 20: "{m}_w20_n100"}


def size_curve():
    """Faces: the model next to the RQ4 size curve (hidden region 1/16, 1/8, 1/5 and 1/4 of the face), on the F3
    subset of 100 evenly spaced targets per set, the targets of `results/rq4_window_size.json`. Gate: the reference
    values recomputed here equal that file."""
    curve = json.load(open(RW.OUT_CURVE, encoding="utf-8"))["datasets"]
    specs = RW.curve_inputs()
    out = {"method": size_curve.__doc__.strip(), "metric": "mse", "scale": "R_self: target 0, mean of 16 random "
           "people 1", "datasets": {}}
    for name, src in RW.FACE_SRC.items():
        subset = RW.face_subset_targets(src)
        out["datasets"][name] = {}
        for spec in specs[name]:
            w = spec["w"]
            base, why = RW.load(spec, ["mse"], subset)
            if base is None:
                raise SystemExit(f"{name} {w} px: {why}")
            rows, present = [base], []
            for m in SIZE_MODELS:
                path = RUN + "faces_v2/" + SIZE_DIR[w].format(m=m) + "/rows.csv.gz"
                if not os.path.exists(path):
                    continue
                mr = pd.read_csv(path, usecols=["src", "target", "family", "patch", "fill", "mse"])
                mr = mr[(mr.src == src) & mr.target.isin(subset) & (mr.fill == f"model_{m}")]
                rows.append(mr.reindex(columns=base.columns))
                present.append(m)
            saved = RW.RUNGS
            RW.RUNGS = saved + [f"model_{m}" for m in present]
            try:
                fr = RW.frame(pd.concat(rows, ignore_index=True), spec["cell"], spec["unit"], ["mse"],
                              np.random.default_rng(RW.SEED))
            finally:
                RW.RUNGS = saved
            stored = curve[name][spec["label"]]["R_self_mse"]
            worst = max(abs(fr["R_self"][k]["mse"][0] - v[0]) for k, v in stored.items())
            if worst > 1e-9:
                raise SystemExit(f"{name} {w} px: references differ from rq4_window_size.json by {worst}")
            out["datasets"][name][spec["label"]] = {
                "f": spec["f"], "window_px": w, "n_targets": len(subset), "max_abs_diff_references_to_rq4_json": worst,
                "references": {k: fr["R_self"][k]["mse"] for k in ("repeat", "shrink_ref", "lookalike", "lookalike16",
                                                                   "dataset_mean") if k in fr["R_self"]},
                "models": {m: fr["R_self"][f"model_{m}"]["mse"] for m in present}}
            print(name, spec["label"], {m: fr["R_self"][f"model_{m}"]["mse"][0] for m in present}, flush=True)
    with open(OUT_SIZE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    print(OUT_SIZE, flush=True)


if __name__ == "__main__":
    main()
