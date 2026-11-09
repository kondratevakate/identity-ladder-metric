"""IXI brain ladder, Amendment 23 (PROTOCOL_donor_ladder_brain.md): 100 targets, other side as the R anchor. Pre-QC.

Reuses the QTIM tier-1 code unchanged: region arrays by `donor_ladder_brain_bilateral.extract`, the rungs of
`donor_ladder_brain_metrics.run_position_at` (stranger, 16 reference strangers, mean of 2, dataset mean, look-alike by
context and oracle, own other side, other side shrunk to the mean with the QTIM lambda, wrong detail, noise) and
`score`. IXI has no repeat, sibling, co-twin; those rungs are never produced and are reported as n/a.
IXI-only rungs added here (Amendment 23): same-sex age-matched mean (the 10 same-sex donors nearest in age), stranger
from the same centre and from a different centre (16 each, exact expectation = mean of the metric over the 16).
The QTIM cohort mean (same sex, age within 2 years) is replaced by the age-matched mean: its rows and acc_cohort are
not reported for IXI.
R = (d(rung) - d(other side)) / (d(strangers) - d(other side)); tests T2, T3a, T3b with reference = other side.
Rows stay in the private folder; results/donor_ladder_brain_ixi.json holds aggregates only.
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
import donor_ladder_brain_metrics as M  # noqa: E402
import ixi_prep as X  # noqa: E402

SEED = B.SEED
PRIVATE = X.PRIVATE
REGIONS = os.path.join(PRIVATE, "regions")
OUT_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_ixi.json")
N_AGE = 10
NA_RUNGS = ("repeat", "sibling", "cotwin", "cohort_mean", "shrink_repeat")
# acc_cohort needs the QTIM cohort mean (replaced here); the protein-style metrics need the repeat-scan tolerance
METRICS = tuple(m for m in M.METRICS if m != "acc_cohort" and m not in M.PROTEIN)
METRICS_NA = {"acc_cohort": "cohort mean replaced by the age-matched mean",
              **{m: "needs the repeat-scan tolerance (no repeats in IXI)" for m in M.PROTEIN}}
# the reference is the other side: renamed to the QTIM anchor names for the unchanged ladder_R and tests code
AS_QTIM = {"mirror": "repeat", "shrink_mirror": "shrink_repeat"}
TESTS_IXI = ("T2", "T3a", "T3b")      # T1 needs a repeat: n/a


def scans():
    """Prepared IXI scans that pass D7, one row each: person = family = scan key (no families in IXI)."""
    man = json.load(open(os.path.join(PRIVATE, "manifest.json"), encoding="utf-8"))["scans"]
    d7 = json.load(open(os.path.join(PRIVATE, "d7.json"), encoding="utf-8"))
    rows = [{"person": s["key"], "session": "ses-01", "family": s["key"], "sex": s["sex"], "age": s["age"],
             "centre": s["centre"], "path": os.path.join(PRIVATE, "prep", s["key"] + "_brain.nii.gz")}
            for s in man if s["ok"] and s["key"] not in d7["excluded"]
            and os.path.exists(os.path.join(PRIVATE, "prep", s["key"] + "_brain.nii.gz"))]
    return pd.DataFrame(rows), {"manifest": len(man), "d7_excluded": len(d7["excluded"]),
                                "d7_flagged": len(d7["flagged"])}


def region_paths(pi):
    return os.path.join(REGIONS, f"vol{pi}.npy"), os.path.join(REGIONS, f"mask{pi}.npy")


def age_matched(i, others, sex, age, n=N_AGE):
    """The n same-sex donors nearest in age (ties broken by index); returns (indices, median absolute age gap)."""
    same = others[sex[others] == sex[i]]
    if same.size < n:
        return None, None
    order = np.lexsort((same, np.abs(age[same] - age[i])))
    pick = same[order[:n]]
    return pick, float(np.median(np.abs(age[pick] - age[i])))


def ixi_rows(args):
    """IXI-only rungs for one position, both sides, on the same arrays and validity rules as run_position_at."""
    pi, tg_cols, pos, a = args
    tg = pd.DataFrame(tg_cols)
    _, start, size = pos[pi]
    pv, pm = region_paths(pi)
    vol, mask = np.load(pv), np.load(pm).astype(bool)
    drawn = BL.side_of(start, size, M.AFFINE_HOLDER["affine"])
    shell, W = BL.SHELL, BL.W
    sex, age, centre = tg.sex.to_numpy(), tg.age.to_numpy(), tg.centre.to_numpy()
    ix = tg["index"].to_numpy()
    cube_sl = tuple(slice(W + shell, W + shell + s) for s in size)
    rows = []
    for side in BL.SIDES:
        fo = BL.own_frame(side, drawn)
        Xc, MK = vol[:, fo][(slice(None),) + cube_sl][ix], mask[:, fo][(slice(None),) + cube_sl][ix]
        valid = MK.reshape(len(Xc), -1).mean(1) >= BL.VALID
        for i in np.flatnonzero(valid):
            others = np.flatnonzero((np.arange(len(Xc)) != i) & valid)
            if others.size < 10:
                continue
            xi, ei = Xc[i], MK[i]
            mu_d, _ = M.masked_mean(Xc[others], MK[others])
            fills = {}
            pick, gap = age_matched(i, others, sex, age)
            if pick is not None:
                m, ok = M.masked_mean(Xc[pick], MK[pick])
                fills["age_matched_mean"] = (m, ok, gap)
            for tag, pool in (("sc", others[centre[others] == centre[i]]), ("dc", others[centre[others] != centre[i]])):
                p = np.random.default_rng([SEED, i, pi, BL.SIDES.index(side), 7 if tag == "sc" else 8]).permutation(pool)
                for j, s in enumerate(p[:M.N_STRANGERS]):
                    fills[f"{tag}{j:02d}"] = (Xc[s], MK[s], None)
            for name, (f, ok, gap) in fills.items():
                s = M.score(xi, f, ei & ok, mu_d, None, None)
                if s is None:
                    continue
                rows.append({"target": int(i), "patch": pi, "side": side, "drawn": side == drawn, "fill": name,
                             "age_gap": gap if name == "age_matched_mean" else np.nan, **s})
    return rows


def stranger_expectations(r):
    """Same-centre and different-centre stranger rungs = mean of each metric over the (up to) 16 drawn strangers."""
    out = []
    for tag, name in (("sc", "stranger_same_centre"), ("dc", "stranger_other_centre")):
        g = r[r.fill.str.fullmatch(tag + r"\d\d")]
        if g.empty:
            continue
        cols = [c for c in g.columns if c not in ("target", "patch", "side", "drawn", "fill")]
        e = g.groupby(["target", "patch", "side", "drawn"])[cols].mean().reset_index()
        e["n_strangers"] = g.groupby(["target", "patch", "side", "drawn"]).size().to_numpy()
        out.append(e.assign(fill=name))
    return pd.concat(out, ignore_index=True) if out else r.iloc[:0]


def is_centre_draw(name):
    """The 16 + 16 drawn centre strangers (sc00..sc15, dc00..dc15) enter only through their means."""
    return len(name) == 4 and name[:2] in ("sc", "dc") and name[2:].isdigit()


def as_qtim(per):
    return per.assign(fill=per.fill.replace(AS_QTIM))


def other_side_R(per, m):
    """ladder_R with the other side as the anchor; centre draws dropped; names given back."""
    R, _ = M.ladder_R(as_qtim(per[~per.fill.map(is_centre_draw)]), m)
    if R is None:
        return None
    back = {v_: k for k, v_ in AS_QTIM.items()}
    R = {back.get(f, f): e for f, e in R.items()}
    a = R.pop("_anchors")
    R["_anchors"] = {"d_other_side": a["d_repeat"], "d_stranger": a["d_stranger"],
                     "n_other_side_targets": a["n_repeat_targets"]}
    return R


def qtim_other_side(view="deep/LR", metrics=("mse",)):
    """QTIM R recomputed on the other-side anchor from its existing tier-1 rows (Amendment 23, no rerun): the QTIM
    repeat and its shrunk control are dropped, the other side becomes the anchor."""
    t = B.scans()
    tg = t[t.session == "ses-01"].reset_index()
    pos, _, _ = B.positions(np.random.default_rng(SEED))
    r = pd.read_csv(os.path.join(M.PRIVATE, "rows.csv.gz"))
    rr = M.views(r[~r.fill.isin(NA_RUNGS)], pos)
    rr = rr[rr.view == view]
    per = rr.groupby(["target", "fill"])[list(metrics)].mean().reset_index()
    per["family"] = tg.family.to_numpy()[per.target]
    return {m: other_side_R(per, m) for m in metrics}


def summarise(r, tg, pos):
    rv = M.views(r, pos)
    out = {}
    for view, rr in rv.groupby("view"):
        per = rr.groupby(["target", "fill"])[list(METRICS) + list(M.DESCRIPTIVE)].mean().reset_index()
        per["family"] = per.target                                   # no families: every target is its own
        fills = sorted(f for f in per.fill.unique() if not M.is_reference_stranger(f) and not is_centre_draw(f))
        v = {"fills": {f: {m: M.boot_mean(per[per.fill == f], m) for m in METRICS + ("amplitude",)}
                       for f in fills}, "ladder_R": {}}
        q = as_qtim(per[~per.fill.map(is_centre_draw)])
        for m in METRICS:
            R = other_side_R(per, m)
            if R is not None:
                v["ladder_R"][m] = R
        if view.startswith("random_patches/LR") or view.endswith("/LR"):
            t = M.tests(q, METRICS)
            for m, e in t["per_metric"].items():
                e.pop("T1", None)
                for k in list(e):
                    e[k]["comparison"] = {"T2": "other side vs other side shrunk to the mean",
                                          "T3a": "other side vs wrong detail", "T3b": "other side vs noise"}[k]
            v["tests"] = holm_ixi(t)
        out[view] = v
    return out


def compare_qtim(ixi_R, metrics=("mse", "ssim", "pearson")):
    """E4: Spearman of the shared rungs' R (other-side anchor, deep box) between IXI and QTIM."""
    from scipy.stats import spearmanr
    q = qtim_other_side("deep/LR", metrics)
    out = {}
    for m in metrics:
        a, b = ixi_R.get(m), q.get(m)
        if not a or not b:
            out[m] = {"status": "not available"}
            continue
        shared = sorted(k for k in a if not k.startswith("_") and k in b and k != "mirror")
        rho = spearmanr([a[k]["R"] for k in shared], [b[k]["R"] for k in shared])[0]
        out[m] = {"spearman": float(rho), "n_rungs": len(shared), "rungs": shared,
                  "qtim_R": {k: b[k]["R"] for k in shared}, "qtim_anchors": b["_anchors"]}
    out["rule"] = "E4: Spearman >= 0.8 in the deep box (mse is the primary metric)"
    return out


def holm_ixi(t):
    """Holm over the T2/T3a/T3b hypotheses only (T1 is n/a in IXI and was dropped before adjusting)."""
    keys = [(m, k) for m, e in t["per_metric"].items() for k, x in e.items() if "p_one_sided" in x]
    adj = M.holm([t["per_metric"][m][k]["p_one_sided"] for m, k in keys]) if keys else []
    for (m, k), pa in zip(keys, adj):
        t["per_metric"][m][k]["p_holm"] = float(pa)
        t["per_metric"][m][k]["pass_holm_0.05"] = bool(pa < 0.05)
    t["n_hypotheses_holm"] = len(keys)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0, help="first n targets only, all scans as donors (rows to a smoke folder, no JSON)")
    a = ap.parse_args()
    t0 = time.time()
    t, counts = scans()
    t = t.reset_index(drop=True)                                # smoke: all scans are donors, first n are targets
    pos, affine, _ = B.positions(np.random.default_rng(SEED))   # the QTIM positions, same seed
    os.makedirs(REGIONS, exist_ok=True)
    BL.PRIVATE = REGIONS                                        # BL.extract writes its memmaps here
    done = os.path.join(REGIONS, f"regions_{len(t)}.done")
    if not os.path.exists(done):
        BL.extract(t, pos, affine)
        open(done, "w").close()
    mc = BL.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    lam = M.rung_lambdas()                                      # the QTIM lambdas (Amendment 23)
    tg = t.reset_index()
    tcols = t[["person", "session"]].to_dict("list")
    tgcols = tg[["index", "person", "family", "sex", "age", "centre"]].to_dict("list")
    tag = f"smoke{a.smoke}_" if a.smoke else ""
    parts = os.path.join(PRIVATE, f"{tag}parts")
    os.makedirs(parts, exist_ok=True)
    only = set(range(a.smoke)) if a.smoke else None
    todo = [pi for pi in range(len(pos)) if not os.path.exists(os.path.join(parts, f"p{pi:02d}.csv.gz"))]
    print(f"positions to run {len(todo)}/{len(pos)}, targets {len(t)}", flush=True)
    jobs = [(pi, tcols, {k: v for k, v in tgcols.items() if k != "centre"}, pos, mc, lam, None, only,
             {"shell": BL.SHELL, "W": BL.W, "paths": region_paths(pi)}) for pi in todo]
    with ProcessPoolExecutor(a.workers, initializer=M._init, initargs=(affine,)) as ex:
        for k, (pi, base, extra) in enumerate(zip(todo, ex.map(M.run_position_at, jobs),
                                                  ex.map(ixi_rows, [(pi, tgcols, pos, mc) for pi in todo]))):
            part = pd.DataFrame(base + [e for e in extra if only is None or e["target"] in only])
            part.to_csv(os.path.join(parts, f"p{pi:02d}.csv.gz.tmp"), index=False, compression="gzip")
            os.replace(os.path.join(parts, f"p{pi:02d}.csv.gz.tmp"), os.path.join(parts, f"p{pi:02d}.csv.gz"))
            el = time.time() - t0
            print(f"position {pi} ({k + 1}/{len(todo)}) rows {len(part)} {el:.0f}s, "
                  f"ETA {el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min", flush=True)
    r = pd.concat([pd.read_csv(os.path.join(parts, f"p{pi:02d}.csv.gz")) for pi in range(len(pos))], ignore_index=True)
    r = r[~r.fill.isin(NA_RUNGS)]
    r = pd.concat([r, stranger_expectations(r)], ignore_index=True)
    if a.smoke:
        print(json.dumps({"smoke_targets": a.smoke, "rows": len(r), "fills": sorted(r.fill.unique())[:40],
                          "minutes": round((time.time() - t0) / 60, 1)}))
        return
    gaps = r[r.fill == "age_matched_mean"].groupby("target")["age_gap"].first()
    out = {"status": "pre-QC: affine registration only, like QTIM",
           "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendment 23; preparation Amendment 19)",
           "R_anchor": "other_side", "seed": SEED, "metrics": list(METRICS),
           "lower_is_better": sorted(M.LOWER_BETTER), "lambda_shrink": lam,
           "not_available": {f: "n/a in IXI (no repeat, no families); never filled" for f in NA_RUNGS},
           "tests_not_available": {"T1": "needs a repeat"}, "metrics_not_available": METRICS_NA,
           "counts": dict(counts, n_targets=int(len(t)), n_positions=len(pos),
                          by_centre={c: int((t.centre == c).sum()) for c in X.CENTRES},
                          female=int((t.sex == 2).sum()), age_median=float(t.age.median()),
                          age_matched_gap_median_years=float(gaps.median())),
           "views": summarise(r, tg, pos)}
    out["comparison_qtim"] = compare_qtim(out["views"]["deep/LR"]["ladder_R"])
    out["counts"]["minutes"] = round((time.time() - t0) / 60, 1)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(M.finite(out), fh, indent=1, allow_nan=False)
    print(json.dumps(out["counts"]))


if __name__ == "__main__":
    main()
