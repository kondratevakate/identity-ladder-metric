"""Mean of 16 look-alikes, QTIM and IXI (PROTOCOL_donor_ladder_brain.md, Amendment 24 A). Pre-QC.

The fill is the masked mean of the 16 donors nearest by the scaffold distance that `run_position_at` already uses
to pick the k = 1 look-alike (no new search). This module rebuilds, per position and side, exactly the quantities of
`donor_ladder_brain_metrics.run_position_at` the fill needs (cubes, scaffold, eligible donors, dataset and cohort
means, repeat tolerance) and scores two fills: `lookalike16` and `lookalike_k1_check`, the nearest donor alone. The
check rows must equal the committed `lookalike` rows on every metric; otherwise no summary is written.
Usage: python src/donor_ladder_brain_lookalike16.py --dataset qtim|ixi|summary --workers N [--smoke 3]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as BL  # noqa: E402
import donor_ladder_brain_metrics as M  # noqa: E402

K = 16
SEED = B.SEED
QTIM_PRIVATE = os.path.join(B.PROJ, "donor-ladder-brain-lookalike16-20261109-v1")
OUT_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_lookalike16.json")
REPORT = ("lookalike16", "lookalike", "dataset_mean")


def nearest(d, k=K):
    """Indices (into d) of the k smallest finite distances, ties by index (np.argmin picks the first minimum, so
    nearest(d, 1) == [argmin(d)] whenever a finite value exists)."""
    fin = np.flatnonzero(np.isfinite(d))
    return fin[np.argsort(d[fin], kind="stable")[:k]]


def run_position(args):
    """Rows of lookalike16 and lookalike_k1_check for one position, both sides (mirrors run_position_at)."""
    pi, t, tg_cols, pos, paths, only = args
    tg = pd.DataFrame(tg_cols)
    _, start, size = pos[pi]
    vol, mask = np.load(paths[0]), np.load(paths[1]).astype(bool)
    drawn = BL.side_of(start, size, M.AFFINE_HOLDER["affine"])
    shell, W = BL.SHELL, BL.W
    n = len(tg)
    fam, sex, age = tg.family.to_numpy(), tg.sex.to_numpy(), tg.age.to_numpy()
    ix = tg["index"].to_numpy()
    row_of = {(p, s): k for k, (p, s) in enumerate(zip(t["person"], t["session"]))}
    win_sl = tuple(slice(W, W + s + 2 * shell) for s in size)
    cube_sl = tuple(slice(W + shell, W + shell + s) for s in size)
    sm = BL.shell_mask(size, shell)
    rows = []
    for side in BL.SIDES:
        fo = BL.own_frame(side, drawn)
        own = vol[:, fo]
        cube_all = own[(slice(None),) + cube_sl]
        mk_all = mask[:, fo][(slice(None),) + cube_sl]
        valid_all = mk_all.reshape(len(cube_all), -1).mean(1) >= BL.VALID
        X, MK, valid = cube_all[ix], mk_all[ix], valid_all[ix]
        win = own[(slice(None),) + win_sl][ix]
        wmask = mask[:, fo][(slice(None),) + win_sl][ix]
        C, CM = win[:, sm], wmask[:, sm]
        n_shell = int(sm.sum())
        rep_msd = {}
        for i in np.flatnonzero(valid):
            j2 = row_of.get((tg.person.iloc[i], "ses-02"))
            if j2 is not None and valid_all[j2]:
                both = MK[i] & mk_all[j2]
                if both.sum() >= 10:
                    rep_msd.setdefault(fam[i], []).append(float(np.mean((X[i][both] - cube_all[j2][both]) ** 2)))
        for i in np.flatnonzero(valid):
            if only is not None and int(i) not in only:
                continue
            others = np.flatnonzero((fam != fam[i]) & valid)
            pairs = [v for f_, vs in rep_msd.items() if f_ != fam[i] for v in vs]
            sigma = float(np.sqrt(np.mean(pairs))) if len(pairs) >= M.MIN_REP_PAIRS else None
            if others.size < 10:
                continue
            xi, ei = X[i], MK[i]
            mu_d, _ = M.masked_mean(X[others], MK[others])
            cpool = others[(sex[others] == sex[i]) & (np.abs(age[others] - age[i]) <= 2)]
            mu_c = M.masked_mean(X[cpool], MK[cpool])[0] if cpool.size else None
            d_s = M.masked_dist(C[others], CM[others], C[i], CM[i], n_shell)
            near = others[nearest(d_s)]
            if near.size == 0:
                continue
            m16, ok16 = M.masked_mean(X[near], MK[near])
            j = near[0]
            for name, f, ok in (("lookalike16", m16, ok16), ("lookalike_k1_check", X[j], MK[j])):
                s = M.score(xi, f, ei & ok, mu_d, mu_c, sigma)
                if s is None:
                    continue
                rows.append({"target": int(i), "patch": pi, "side": side, "drawn": side == drawn, "fill": name,
                             "n_donors": int(near.size if name == "lookalike16" else 1), **s})
    return rows


def dataset(name):
    """(t, tg, region paths function, private folder, committed rows loader) for 'qtim' or 'ixi'."""
    if name == "qtim":
        t = B.scans()
        return (t, t[t.session == "ses-01"].reset_index(), BL.region_paths, QTIM_PRIVATE,
                lambda: pd.read_csv(os.path.join(M.PRIVATE, "rows.csv.gz")))
    import donor_ladder_brain_ixi as I
    t, _ = I.scans()
    t = t.reset_index(drop=True)
    parts = os.path.join(I.PRIVATE, "parts")
    return (t, t.reset_index(), I.region_paths, os.path.join(I.PRIVATE, "lookalike16_parts"),
            lambda: pd.concat([pd.read_csv(os.path.join(parts, f)) for f in sorted(os.listdir(parts))],
                              ignore_index=True))


def check_rows(new, old, metrics):
    """lookalike_k1_check must equal the committed lookalike rows on every metric (same NaN pattern)."""
    a = new[new.fill == "lookalike_k1_check"].set_index(["target", "patch", "side"]).sort_index()
    b = old[old.fill == "lookalike"].set_index(["target", "patch", "side"]).sort_index()
    b = b.loc[b.index.intersection(a.index)]
    out = {"rows_check": int(len(a)), "rows_matched": int(len(b)), "max_abs_diff": {}}
    ok = len(a) == len(b) and len(a) > 0
    for m in metrics:
        x, y = a[m].to_numpy(float), b.loc[a.index, m].to_numpy(float) if len(b) == len(a) else np.array([np.nan])
        same_nan = bool(np.array_equal(np.isnan(x), np.isnan(y))) if len(b) == len(a) else False
        diff = float(np.nanmax(np.abs(x - y))) if len(b) == len(a) and np.isfinite(x - y).any() else 0.0
        out["max_abs_diff"][m] = diff
        ok &= same_nan and diff <= 1e-5
    out["ok"] = bool(ok)
    return out


def run(name, workers, smoke):
    t, tg, region_paths, private, _ = dataset(name)
    pos, affine, _ = B.positions(np.random.default_rng(SEED))
    tag = f"smoke{smoke}_" if smoke else ""
    parts = os.path.join(private, f"{tag}parts")
    os.makedirs(parts, exist_ok=True)
    only = set(range(smoke)) if smoke else None
    if smoke and name == "qtim":                              # targets with a repeat, as the Amendment 20 smoke
        rep = set(t[t.session == "ses-02"].person)
        only = set(np.flatnonzero(tg.person.isin(rep).to_numpy())[:smoke].tolist())
    tcols = t[["person", "session"]].to_dict("list")
    tgcols = tg[["index", "person", "family", "sex", "age"]].to_dict("list")
    todo = [pi for pi in range(len(pos)) if not os.path.exists(os.path.join(parts, f"p{pi:02d}.csv.gz"))]
    print(f"{name}: positions to run {len(todo)}/{len(pos)}, targets {len(tg) if only is None else len(only)}",
          flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=M._init, initargs=(affine,)) as ex:
        jobs = [(pi, tcols, tgcols, pos, region_paths(pi), only) for pi in todo]
        for k, (pi, rows) in enumerate(zip(todo, ex.map(run_position, jobs))):
            p = os.path.join(parts, f"p{pi:02d}.csv.gz")
            pd.DataFrame(rows).to_csv(p + ".tmp", index=False, compression="gzip")
            os.replace(p + ".tmp", p)
            el = time.time() - t0
            print(f"position {pi} ({k + 1}/{len(todo)}) rows {len(rows)} {el:.0f}s, "
                  f"ETA {el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min", flush=True)
    new = pd.concat([pd.read_csv(os.path.join(parts, f"p{pi:02d}.csv.gz")) for pi in range(len(pos))],
                    ignore_index=True)
    rep = check_rows(new, dataset(name)[4](), [m for m in M.METRICS if m in new.columns])
    print(json.dumps({"dataset": name, "smoke": smoke, "rows": len(new), "reproduces_k1": rep["ok"],
                      "max_abs_diff_mse": rep["max_abs_diff"].get("mse"), "minutes": round((time.time() - t0) / 60, 1),
                      "n_donors_min": int(new[new.fill == "lookalike16"].n_donors.min())}), flush=True)
    return rep


def summary():
    import donor_ladder_brain_ixi as I
    pos, _, _ = B.positions(np.random.default_rng(SEED))
    out = {"protocol": "docs/PROTOCOL_donor_ladder_brain.md, Amendment 24 A", "k": K,
           "status": "pre-QC: affine registration only", "datasets": {}}
    for name in ("qtim", "ixi"):
        t, tg, _, private, old_loader = dataset(name)
        parts = os.path.join(private, "parts")
        new = pd.concat([pd.read_csv(os.path.join(parts, f)) for f in sorted(os.listdir(parts))], ignore_index=True)
        old = old_loader()
        metrics = list(M.METRICS) if name == "qtim" else list(I.METRICS)
        rep = check_rows(new, old, metrics)
        if not rep["ok"]:
            raise SystemExit(f"{name}: the nearest of the 16 does not reproduce the committed look-alike: {rep}")
        old = old[~old.fill.isin(I.NA_RUNGS)] if name == "ixi" else old
        r = pd.concat([old, new[new.fill == "lookalike16"]], ignore_index=True)
        fam = tg.family.to_numpy()
        views = {}
        for view, rr in M.views(r, pos).groupby("view"):
            per = rr.groupby(["target", "fill"])[metrics].mean().reset_index()
            per["family"] = fam[per.target]
            v = {"fills": {f: {m: M.boot_mean(per[per.fill == f], m) for m in metrics} for f in REPORT},
                 "ladder_R": {}}
            for m in metrics:
                R = M.ladder_R(per, m)[0] if name == "qtim" else I.other_side_R(per, m)
                if R:
                    v["ladder_R"][m] = {f: R[f] for f in REPORT + ("_anchors",) if f in R}
            views[view] = v
        n16 = new[new.fill == "lookalike16"].n_donors
        rp = views["random_patches/LR"]["ladder_R"]["mse"]
        out["datasets"][name] = {
            "R_anchor": "repeat" if name == "qtim" else "other_side", "reproduction_k1": rep,
            "n_donors": {"min": int(n16.min()), "median": float(n16.median())},
            "pool_size_median": int(len(tg) - 1) if name == "ixi" else None,
            "expectation_mse_random_LR": {
                "rule": "R(lookalike16) < R(lookalike) and < R(dataset_mean) on MSE, random cubes L+R",
                "R": {f: rp[f]["R"] for f in REPORT},
                "met": bool(rp["lookalike16"]["R"] < rp["lookalike"]["R"]
                            and rp["lookalike16"]["R"] < rp["dataset_mean"]["R"])},
            "views": views}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(M.finite(out), fh, indent=1, allow_nan=False)
    print(json.dumps({k: v["expectation_mse_random_LR"] for k, v in out["datasets"].items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("qtim", "ixi", "summary"), required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.dataset == "summary":
        return summary()
    rep = run(a.dataset, a.workers, a.smoke)
    if not rep["ok"]:
        print(json.dumps(rep, indent=1))
        raise SystemExit("the nearest of the 16 does not reproduce the committed look-alike rows")


if __name__ == "__main__":
    main()
