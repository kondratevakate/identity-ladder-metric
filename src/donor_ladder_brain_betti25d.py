"""Betti matching 2.5D on the brain random cubes (PROTOCOL_donor_ladder_brain.md, Amendment 24C). Pre-QC.

Fills, donors, E and seeds come from `donor_ladder_brain_metrics.run_position_at` unchanged, with the tier23 hook
`betti25d` (`donor_ladder_brain_metrics_t23.betti_25d`: the 2D library at 3e7eca26 on the central slices wholly in
E). Cells = topology subset S: per target, 2 of the 32 random positions where the target cube lies wholly inside its
own mask on both sides, seed [20261102, target].
Usage: python src/donor_ladder_brain_betti25d.py --dataset qtim|ixi|summary --workers N [--smoke 3]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as BL  # noqa: E402
import donor_ladder_brain_lookalike16 as L  # noqa: E402
import donor_ladder_brain_metrics as M  # noqa: E402

SEED = B.SEED
PER_TARGET = 2
METRICS = ("betti_matched", "betti_unmatched")
PRIVATE = os.path.join(B.PROJ, "donor-ladder-brain-betti25d-20261110-v1")
OUT_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_betti25d.json")


def lower_is_better_in_json():
    """The JSON's lower_is_better list (checked against M.LOWER_BETTER by a test, so tests and R agree)."""
    return [m for m in METRICS if m in M.LOWER_BETTER]


def subset(whole, seed=SEED, k=PER_TARGET):
    """whole: (targets, positions) bool, target cube wholly in its own mask on both sides. Returns {target: [pos]}
    with k positions drawn per target (fewer only when fewer are whole)."""
    out = {}
    for i in range(whole.shape[0]):
        cand = np.flatnonzero(whole[i])
        if cand.size:
            out[i] = sorted(np.random.default_rng([seed, i]).choice(cand, min(k, cand.size), replace=False).tolist())
    return out


def whole_cubes(tg, pos, region_paths):
    """(targets, random positions) bool: the target's cube lies wholly inside its own mask in both frames."""
    ix = tg["index"].to_numpy()
    rand = [pi for pi, p in enumerate(pos) if p[0] == "random_patches"]
    w = np.zeros((len(tg), len(rand)), bool)
    for k, pi in enumerate(rand):
        _, _, size = pos[pi]
        m = np.load(region_paths(pi)[1], mmap_mode="r")
        sl = tuple(slice(BL.W + BL.SHELL, BL.W + BL.SHELL + s) for s in size)
        cube = np.asarray(m[ix][(slice(None), slice(None)) + sl]).astype(bool)
        w[:, k] = cube.reshape(len(ix), 2, -1).all(-1).all(-1)
    return w, rand


def run(name, workers, smoke):
    import donor_ladder_brain_ixi as I
    t, tg, region_paths, _, _ = L.dataset(name)
    pos, affine, _ = B.positions(np.random.default_rng(SEED))
    w, rand = whole_cubes(tg, pos, region_paths)
    S = subset(w)
    n_without = int(len(tg) - len(S))                       # targets with no whole cube in any random position
    if smoke:
        keep = list(range(len(tg)))
        if name == "qtim":
            rep = set(t[t.session == "ses-02"].person)
            keep = np.flatnonzero(tg.person.isin(rep).to_numpy()).tolist()
        S = {i: S[i] for i in keep if i in S}
        S = dict(list(S.items())[:smoke])
        os.environ["BETTI25D_QC"] = "1"
    by_pos = {}
    for i, ps in S.items():
        for k in ps:
            by_pos.setdefault(rand[k], set()).add(i)
    tag = f"smoke{smoke}_" if smoke else ""
    parts = os.path.join(PRIVATE, name, f"{tag}parts")
    os.makedirs(parts, exist_ok=True)
    mc = BL.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    lam = M.rung_lambdas()
    tcols = t[["person", "session"]].to_dict("list")
    tgcols = tg[["index", "person", "family", "sex", "age"]].to_dict("list")
    todo = [pi for pi in sorted(by_pos) if not os.path.exists(os.path.join(parts, f"p{pi:02d}.csv.gz"))]
    print(json.dumps({"dataset": name, "targets_in_S": len(S), "targets_without_whole_cube": n_without,
                      "cells": int(sum(len(v) for v in S.values())), "positions_to_run": len(todo)}), flush=True)
    t0 = time.time()
    jobs = [(pi, tcols, tgcols, pos, mc, lam, "betti25d", by_pos[pi],
             {"shell": BL.SHELL, "W": BL.W, "paths": region_paths(pi)}) for pi in todo]
    with ProcessPoolExecutor(workers, initializer=M._init, initargs=(affine,)) as ex:
        for k, (pi, rows) in enumerate(zip(todo, ex.map(M.run_position_at, jobs))):
            qc = [r.pop("_qc") for r in rows if "_qc" in r]
            df = pd.DataFrame(rows)
            if name == "ixi":
                df = df[~df.fill.isin(I.NA_RUNGS)]
            p = os.path.join(parts, f"p{pi:02d}.csv.gz")
            df.to_csv(p + ".tmp", index=False, compression="gzip")
            os.replace(p + ".tmp", p)
            if smoke and qc:
                np.savez_compressed(os.path.join(parts, f"qc_p{pi:02d}.npz"),
                                    **{f"r{j}": a for j, a in enumerate(qc)},
                                    fills=np.array([r["fill"] for r in rows if r.get("betti_slices", 0) > 0]))
            el = time.time() - t0
            print(f"position {pi} ({k + 1}/{len(todo)}) rows {len(df)} {el:.0f}s, "
                  f"ETA {el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min", flush=True)
    r = pd.concat([pd.read_csv(os.path.join(parts, f)) for f in sorted(os.listdir(parts)) if f.endswith(".csv.gz")],
                  ignore_index=True)
    print(json.dumps({"rows": len(r), "scored_rows": int((r.betti_slices > 0).sum()),
                      "slices_median": float(r.betti_slices.median()), "minutes": round((time.time() - t0) / 60, 1),
                      "means": r.groupby("fill")[list(METRICS)].mean().round(4)
                      .reindex(["repeat", "mirror", "stranger", "dataset_mean", "noise"]).dropna(how="all")
                      .to_dict()}), flush=True)


def summary():
    import donor_ladder_brain_ixi as I
    pos, _, _ = B.positions(np.random.default_rng(SEED))
    out = {"protocol": "docs/PROTOCOL_donor_ladder_brain.md, Amendment 24C", "library": "Betti-matching 2D 3e7eca26",
           "status": "pre-QC: affine registration only", "metrics": list(METRICS), "lower_is_better": lower_is_better_in_json(),
           "cells": "topology subset S: 2 whole random cubes per target, both sides", "datasets": {}}
    for name in ("qtim", "ixi"):
        parts = os.path.join(PRIVATE, name, "parts")
        if not os.path.isdir(parts):
            out["datasets"][name] = {"status": "not run"}
            continue
        _, tg, _, _, _ = L.dataset(name)
        r = pd.concat([pd.read_csv(os.path.join(parts, f)) for f in sorted(os.listdir(parts))
                       if f.endswith(".csv.gz")], ignore_index=True)
        fam = tg.family.to_numpy()
        views = {}
        for view, rr in M.views(r, pos).groupby("view"):
            per = rr.groupby(["target", "fill"])[list(METRICS)].mean().reset_index()
            per["family"] = fam[per.target]
            fills = sorted(f for f in per.fill.unique() if not M.is_reference_stranger(f))
            v = {"fills": {f: {m: M.boot_mean(per[per.fill == f], m) for m in METRICS} for f in fills},
                 "ladder_R": {}}
            for m in METRICS:
                R = M.ladder_R(per, m)[0] if name == "qtim" else I.other_side_R(per, m)
                if R:
                    v["ladder_R"][m] = R
            if view.endswith("/LR"):
                if name == "qtim":
                    v["tests"] = M.tests(per, list(METRICS))
                else:
                    t = M.tests(I.as_qtim(per), list(METRICS))
                    for e in t["per_metric"].values():
                        e.pop("T1", None)
                    v["tests"] = I.holm_ixi(t)
            views[view] = v
        out["datasets"][name] = {"R_anchor": "repeat" if name == "qtim" else "other_side",
                                 "n_targets": int(r.target.nunique()), "n_rows_scored": int((r.betti_slices > 0).sum()),
                                 "n_rows_missing": int((r.betti_slices == 0).sum()), "views": views}
    out["not_available"] = {"betti_3d": "Betti-Matching-3D disagreed with the 2D library (Amendment 24 B check)"}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(M.finite(out), fh, indent=1, allow_nan=False)
    print(json.dumps({d: {k: v[k] for k in ("n_targets", "n_rows_scored", "n_rows_missing")}
                      for d, v in out["datasets"].items() if "views" in v}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("qtim", "ixi", "summary"), required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.dataset == "summary":
        return summary()
    run(a.dataset, a.workers, a.smoke)


if __name__ == "__main__":
    main()
