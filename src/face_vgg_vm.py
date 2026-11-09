"""Amendment F2.1 on a cloud VM: the LPIPS VGG pass for one source without copying the other source's faces.

Stages
- payload (laptop): an IMDb-only copy of the v6f face cache. X, M and emb keep their shape and row indices (so job
  names, donors and seeds are unchanged), but every row outside the source's index set (pool range, targets, retests,
  siblings) is blanked: NaN in X and emb, False in M. A fill that touched a blanked row would change MSE against v1,
  so the per-job reproduction gate stops it. Also v1 rows (keys + MSE, this source only), the face results JSON
  (aggregates, for the lambdas) and the two small files whose hashes the run records.
- pack (VM): saved parts -> one CSV of rows + one JSON of per-job reproduction residuals (no pickles leave the VM).
- unpack (laptop): CSV + JSON -> local parts, refusing jobs already saved locally unless they are the reference jobs,
  which are compared row by row against the laptop parts (keys and MSE equal, LPIPS VGG within the tolerance).

Only private folders are written; nothing here prints images, file names or identifiers.
"""
import argparse
import json
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import face_ladder_metrics as FM  # noqa: E402

VGG_TOL = 1e-5                     # LPIPS VGG, laptop vs VM CPU: float32 reductions may differ in the last bits
KEYS = ["src", "target", "family", "patch", "fill"]


def source_rows(desc, src):
    """Every X row the jobs of `src` may read: its pool range and its targets' own, retest and sibling rows."""
    d = desc["data"][src]
    rows = set(range(*d["pool"]))
    for t in d["targets"]:
        rows.update(v for v in (t["t"], t["rep"], t["sib"]) if v is not None)
    return sorted(rows)


def blank_other_rows(a, keep, fill):
    out = np.array(a, copy=True)
    drop = np.ones(len(out), bool)
    drop[keep] = False
    out[drop] = fill
    return out


def payload(version, src, out, v1_dir, face_results, fundus, betti):
    cache = os.path.join(v1_dir, f"cache{version}")
    desc = json.load(open(os.path.join(cache, "desc.json")))
    keep = source_rows(desc, src)
    desc_rows = np.load(os.path.join(cache, "M.npy"), mmap_mode="r")
    dst = os.path.join(out, "data", "projects", "identity-ladder-metric", os.path.basename(v1_dir), f"cache{version}")
    os.makedirs(dst, exist_ok=True)
    for name, fill in (("X", np.nan), ("emb", np.nan), ("M", False)):
        np.save(os.path.join(dst, f"{name}.npy"), blank_other_rows(np.load(os.path.join(cache, f"{name}.npy")),
                                                                    keep, fill))
    for f in os.listdir(cache):
        if f.startswith(src + "_") and f.endswith(".npy"):
            shutil.copy2(os.path.join(cache, f), dst)
    json.dump({"data": {src: desc["data"][src]}, "repro": {src: desc["repro"][src]}},
              open(os.path.join(dst, "desc.json"), "w"))
    v1 = pd.read_csv(os.path.join(v1_dir, f"rows{version}.csv.gz"), usecols=["src", "target", "patch", "fill", "mse"])
    v1[v1.src == src].to_csv(os.path.join(os.path.dirname(dst), f"rows{version}.csv.gz"), index=False)
    fr = os.path.join(out, "face", "results")
    os.makedirs(fr, exist_ok=True)
    shutil.copy2(os.path.join(face_results, f"donor_ladder_face{version}.json"), fr)
    ext = os.path.join(out, "ext")
    os.makedirs(ext, exist_ok=True)
    shutil.copy2(fundus, ext)
    shutil.copy2(os.path.join(betti, "BettiMatching.py"), ext)
    print(f"payload for {src}: {len(keep)} of {len(desc_rows)} image rows kept, the rest blanked; {out}")


def pack(parts, out_csv, out_json):
    rows, worst = [], {}
    for f in sorted(os.listdir(parts)):
        if f.endswith(".pkl"):
            with open(os.path.join(parts, f), "rb") as fh:     # written by this run on this VM
                r, w = pickle.load(fh)
            rows += [dict(x, job=f[:-4]) for x in r]
            worst[f[:-4]] = float(w)
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.17g")   # lossless float64 text round trip
    json.dump(worst, open(out_json, "w"), indent=1)
    print(f"packed {len(worst)} jobs, {len(rows)} rows")


def compare(a, b):
    """Rows of one job from two machines: same keys in the same order, MSE equal, LPIPS VGG within VGG_TOL."""
    a, b = pd.DataFrame(a).reset_index(drop=True), pd.DataFrame(b).reset_index(drop=True)
    same_keys = len(a) == len(b) and (a[KEYS].astype(str).values == b[KEYS].astype(str).values).all()
    if not same_keys:
        return {"same_keys": False, "passed": False}
    dm = float(np.nanmax(np.abs(a.mse.values - b.mse.values))) if len(a) else 0.0
    na, nb = a.lpips_vgg.isna().values, b.lpips_vgg.isna().values
    dv = float(np.nanmax(np.abs(a.lpips_vgg.values - b.lpips_vgg.values))) if (~na).any() else 0.0
    return {"same_keys": True, "n_rows": len(a), "n_vgg": int((~na).sum()), "max_abs_mse_diff": dm,
            "same_missing": bool((na == nb).all()), "max_abs_vgg_diff": dv,
            "passed": bool(dm == 0.0 and (na == nb).all() and dv <= VGG_TOL)}


def unpack(in_csv, in_json, parts, refs):
    r = pd.read_csv(in_csv, float_precision="round_trip")   # exact float64 text round trip
    worst = json.load(open(in_json))
    report = {}
    for job, g in r.groupby("job", sort=False):
        rows = g.drop(columns="job").to_dict("records")
        for x in rows:
            x["target"], x["family"], x["patch"] = int(x["target"]), int(x["family"]), int(x["patch"])
        local = os.path.join(parts, f"{job}.pkl")
        if os.path.exists(local):
            if job not in refs:
                raise SystemExit(f"{job} exists locally and is not a reference job; refusing to overwrite")
            with open(local, "rb") as fh:                      # written by this laptop's own run
                lr, _ = pickle.load(fh)
            report[job] = compare(lr, rows)
            continue
        with open(local + ".tmp", "wb") as fh:
            pickle.dump((rows, worst[job]), fh)
        os.replace(local + ".tmp", local)
    print(json.dumps({"imported": len(set(r.job)) - len(report), "reference": report}))
    if refs and not all(v["passed"] for v in report.values()) or set(refs) - set(report):
        raise SystemExit("reference jobs differ between laptop and VM, or are missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("payload", "pack", "unpack"))
    ap.add_argument("--version", default=None, help="face frame version, e.g. _v6f (payload)")
    ap.add_argument("--src", choices=("celeba", "imdb"))
    ap.add_argument("--out")
    ap.add_argument("--parts")
    ap.add_argument("--csv")
    ap.add_argument("--json")
    ap.add_argument("--refs", default="", help="unpack: comma-separated reference jobs to compare")
    a = ap.parse_args()
    if a.stage == "payload":
        if not (a.version and a.src and a.out):
            raise SystemExit("payload needs --version, --src and --out")
        payload(a.version, a.src, a.out, FM.PRIVATE, FM.FACE_RESULTS, os.environ["FUNDUS_METRICS_SRC"],
                os.environ["BETTI_SRC"])
    elif a.stage == "pack":
        pack(a.parts, a.csv, a.json)
    else:
        unpack(a.csv, a.json, a.parts, [j for j in a.refs.split(",") if j])


if __name__ == "__main__":
    main()
