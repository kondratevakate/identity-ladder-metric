"""Amendment P3 aggregates for the iris panel (docs/PROTOCOL_iris_metrics.md, Amendment P3).

Reads the private per-window rows and set-level files written by `iris_ladder_metrics_extra.py merge` and writes
only aggregates (per rung and per set; no subject id, no image, no per-window row) into the `amendment_p3` block of
`results/iris_ladder_metrics.json`. Scales and tests follow the panel rules (fundus panel code, same formulas):
- raw: mean over windows per person, person bootstrap (2000 resamples, panel seed);
- R_self: d / mean d of the 16 stranger anchors on the same cells (target itself = 0, strangers = 1);
- R_repeat: same with the repeat as 0;
- T1 to T3b: one-sided Wilcoxon on person means of oriented(repeat) - oriented(control), >= 10 persons;
- Holm over the 26 x 4 confirmatory family, panel p-values kept, blocked or set-level metrics as p = 1.
All new metrics are lower = closer and score 0 for the target itself (protocol, Amendment P3).

Usage (no default for the private folder version or the frame):
    python -I src/iris_ladder_metrics_p3.py --preprocessing P2 --tag full
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import iris_ladder_metrics_frame as IF  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT_JSON = os.path.join(ROOT, "results", "iris_ladder_metrics.json")
SETS = ("UTIRIS_NIR", "UTIRIS_VIS", "UPOL")
PRIMARY = "UTIRIS_NIR"
REPS = 2000
SEED = 20261102                       # panel seed (iris_ladder_metrics.SEED)
P3_METRICS = ("lpips_alex", "lpips_vgg", "betti_matched", "betti_unmatched", "iriscode_hd")   # all lower = closer
P3_CONFIRMATORY = {"lpips": "lpips_alex", "betti_matching_matched": "betti_matched",
                   "betti_matching_unmatched": "betti_unmatched"}
SET_LEVEL = ("fid", "fwd")
TESTS = {"T1": "dataset_mean", "T2": "shrink_repeat", "T3a": "wrong_detail", "T3b": "noise"}
RUNGS = ("repeat", "shrink_repeat", "fellow", "shrink_fellow", "lookalike", "lookalike16", "lookalike_oracle",
         "stranger", "mean_k2", "dataset_mean", "wrong_detail", "noise")
SET_SHARE_MAX = 0.05                  # LPIPS sensitivity: windows with at most 5% imputed input pixels
H5_MIN = 0.9


def is_anchor(f):
    return len(f) == 5 and f.startswith("str") and f[3:].isdigit()


def boot_idx(n, reps=REPS, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, size=(reps, n))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def _person_sums(g, col, persons):
    pi = {p: k for k, p in enumerate(persons)}
    agg = g.groupby("person")[col].agg(["sum", "count"])
    s, c = np.zeros(len(persons)), np.zeros(len(persons))
    for p, row in agg.iterrows():
        s[pi[p]], c[pi[p]] = row["sum"], row["count"]
    return s, c


def raw_means(r, m, persons, bidx):
    """Mean distance per rung: window mean per target, then pooled per person; person bootstrap."""
    per = r.groupby(["fill", "unit", "person"])[m].mean().reset_index().dropna()
    out = {}
    for f, g in per.groupby("fill"):
        if is_anchor(f):
            continue
        s, c = _person_sums(g, m, persons)
        bs = s[bidx].sum(1) / np.maximum(c[bidx].sum(1), 1e-12)
        out[f] = {"mean": float(s.sum() / c.sum()), "ci95": [float(np.percentile(bs, 2.5)),
                                                             float(np.percentile(bs, 97.5))],
                  "n_targets": int(c.sum()), "n_persons": int((c > 0).sum())}
    return out


def r_scale(r, m, persons, bidx, anchor0):
    """R per rung (lower = closer, d_self = 0). Stranger anchor = per-cell mean over the 16 strangers."""
    tab = r[["unit", "person", "patch", "fill", m]].dropna().rename(columns={m: "d"})
    strs = tab[tab.fill.map(is_anchor)].groupby(["unit", "person", "patch"])["d"].mean().reset_index()
    strs["fill"] = "_anchor"
    tab = pd.concat([tab[~tab.fill.map(is_anchor)], strs])
    S, C = {}, {}
    for f, g in tab.groupby("fill"):
        S[f], C[f] = _person_sums(g, "d", persons)
    if "_anchor" not in S or (anchor0 == "repeat" and "repeat" not in S):
        return None
    pt = {f: S[f].sum() / C[f].sum() for f in S}
    bs = {f: S[f][bidx].sum(1) / np.maximum(C[f][bidx].sum(1), 1e-12) for f in S}
    z_pt, z_bs = (0.0, 0.0) if anchor0 == "self" else (pt["repeat"], bs["repeat"])
    out = {}
    for f in S:
        if f == "_anchor":
            continue
        arr = (bs[f] - z_bs) / (bs["_anchor"] - z_bs)
        out[f] = {"R": float((pt[f] - z_pt) / (pt["_anchor"] - z_pt)),
                  "ci95": [float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))],
                  "n_targets": int(tab[tab.fill == f].unit.nunique()), "n_persons": int((C[f] > 0).sum())}
    return out


def paired_tests(r, m, ref="repeat"):
    """T1-T3b: control - reference per window (lower = closer, so > 0 means the reference wins), mean per target,
    then per person; one-sided Wilcoxon on person means."""
    out = {}
    for tn, ctrl in TESTS.items():
        c = ctrl.replace("repeat", ref)
        w = r[r.fill.isin([ref, c])].pivot_table(index=["unit", "person", "patch"], columns="fill",
                                                  values=m).dropna()
        if ref not in w or c not in w or w.empty:
            out[tn] = {"status": "not run (no pairs)"}
            continue
        w = w.reset_index()
        w["d"] = w[c] - w[ref]
        pt = w.groupby(["unit", "person"])["d"].mean().reset_index()
        pm = pt.groupby("person")["d"].mean()
        e = {"control": c, "n_targets": int(len(pt)), "n_persons": int(len(pm)),
             "mean_diff": float(pt.d.mean()), "share_targets_pass": float((pt.d > 0).mean())}
        if len(pm) >= 10 and np.any(pm.to_numpy() != 0):
            e["p_one_sided"] = float(stats.wilcoxon(pm.to_numpy(), alternative="greater").pvalue)
        else:
            e["status"] = "descriptive only (fewer than 10 persons)"
        out[tn] = e
    return out


def eligibility(r):
    out = {}
    for f, g in r.groupby("fill"):
        if is_anchor(f):
            continue
        gs = g[g.topo_subset.astype(bool)]
        out[f] = {"cells": int(len(g)), "cells_in_S": int(len(gs)),
                  "betti_scored": int(gs.betti_matched.notna().sum()),
                  "betti_not_scored_window_not_full": int(gs.betti_matched.isna().sum()),
                  "lpips_alex_scored": int(g.lpips_alex.notna().sum()),
                  "lpips_vgg_scored": int(g.lpips_vgg.notna().sum()),
                  "iriscode_scored": int(g.iriscode_hd.notna().sum()),
                  "lpips_cells_imputed_share_le_0.05": int((g.imputed_share <= SET_SHARE_MAX).sum())}
    return out


def set_level_block(sl):
    """Set-level FID / FWD as written by merge, plus a flag where the bootstrap interval does not contain the point
    estimate (Fréchet distances are biased upward in small or duplicated samples; resampling people with
    replacement duplicates windows, so the percentile interval is shifted and is not read as a confidence bound)."""
    out = {}
    for m in SET_LEVEL:
        out[m] = {}
        for rung, v in sl.get(m, {}).items():
            e = {k: v.get(k) for k in ("value", "windows", "persons", "ci95", "over_stranger", "over_stranger_ci95")}
            ci, oci = v.get("ci95"), v.get("over_stranger_ci95")
            e["ci_contains_point"] = bool(ci and ci[0] <= v["value"] <= ci[1])
            e["ratio_ci_contains_point"] = (bool(oci and oci[0] <= v["over_stranger"] <= oci[1])
                                            if v.get("over_stranger") is not None and rung != "stranger" else None)
            out[m][rung] = e
    return out


def summarise_set(r, sl):
    persons = np.array(sorted(r.person.astype(str).unique()))
    r = r.assign(person=r.person.astype(str))
    bidx = boot_idx(len(persons))
    s = {"counts": {"units": int(r.unit.nunique()), "persons": int(len(persons)), "rows": int(len(r)),
                    "targets_per_fill": {f: int(g.unit.nunique()) for f, g in r.groupby("fill") if not is_anchor(f)}},
         "eligibility": eligibility(r), "raw": {}, "R_self": {}, "R_repeat": {}, "tests": {}}
    for m in P3_METRICS:
        if r[m].notna().any():
            s["raw"][m] = raw_means(r, m, persons, bidx)
            s["R_self"][m] = r_scale(r, m, persons, bidx, "self")
            s["R_repeat"][m] = r_scale(r, m, persons, bidx, "repeat")
            s["tests"][m] = paired_tests(r, m)
    rs = r[r.imputed_share <= SET_SHARE_MAX]
    s["lpips_alex_R_self_imputed_share_le_0.05"] = (r_scale(rs, "lpips_alex", persons, bidx, "self")
                                                     if len(rs) else {"status": "not run (no cells)"})
    s["set_level"] = set_level_block(sl)
    return s


def holm_family(panel_tests, p3_tests, registry):
    """Holm over the 26 x 4 confirmatory family: panel p-values for the panel's computed metrics, P3 p-values for
    LPIPS and both Betti-matching parts, p = 1 for blocked and set-level metrics."""
    keys, ps, src = [], [], []
    for name, spec in registry.items():
        col = P3_CONFIRMATORY.get(name) or spec.get("column")
        for tn in TESTS:
            if name in P3_CONFIRMATORY:
                e, origin = p3_tests.get(col, {}).get(tn, {}), "amendment_p3"
            elif col:
                e, origin = panel_tests.get(col, {}).get(tn, {}), "panel"
            else:
                e, origin = {}, "blocked or set level"
            keys.append(f"{name}/{tn}")
            ps.append(e.get("p_one_sided", 1.0))
            src.append((origin, "p_one_sided" in e))
    adj = holm(ps)
    return {k: {"p_holm": float(a), "pass_holm_0.05": bool(a < 0.05), "source": o, "computed": c}
            for k, a, (o, c) in zip(keys, adj, src)}, len(ps)


def hypotheses(sets, holm_nir):
    nir = sets[PRIMARY]
    h4 = {}
    for name, col in P3_CONFIRMATORY.items():
        e = holm_nir[f"{name}/T2"]
        h4[name] = {"T2_p_holm": e["p_holm"], "T2_pass": e["pass_holm_0.05"],
                    "mean_diff_shrink_minus_repeat": nir["tests"][col]["T2"].get("mean_diff"),
                    "share_targets_pass": nir["tests"][col]["T2"].get("share_targets_pass")}
    h5 = {}
    for name, sl in ((n, nir["set_level"][n]) for n in SET_LEVEL):
        h5[name] = {}
        for rung in ("dataset_mean", "mean_k2"):
            v = sl.get(rung, {})
            ci = v.get("over_stranger_ci95")
            h5[name][rung] = {"over_stranger": v.get("over_stranger"), "over_stranger_ci95": ci,
                              "holds_point": bool(v.get("over_stranger") is not None and v["over_stranger"] >= H5_MIN),
                              "interval_excludes_0.9": bool(ci and ci[0] > H5_MIN),
                              "interval_trusted": bool(v.get("ratio_ci_contains_point"))}
    R = nir["R_self"]["lpips_alex"]
    h6 = {"R_self": {f: R[f]["R"] for f in ("repeat", "lookalike", "stranger")},
          "repeat_upper_95": R["repeat"]["ci95"][1],
          "order_holds": bool(R["repeat"]["R"] < R["lookalike"]["R"] < R["stranger"]["R"]),
          "repeat_upper_below_1": bool(R["repeat"]["ci95"][1] < 1)}
    h6["holds"] = h6["order_holds"] and h6["repeat_upper_below_1"]
    mean_vs_stranger = {}
    for s_name, s in sets.items():
        for m in P3_METRICS:
            Rm = s["R_self"].get(m)
            if Rm and "dataset_mean" in Rm and "stranger" in Rm:
                mean_vs_stranger.setdefault(m, {})[s_name] = {
                    "R_dataset_mean": Rm["dataset_mean"]["R"], "ci95": Rm["dataset_mean"]["ci95"],
                    "R_stranger": Rm["stranger"]["R"],
                    "dataset_mean_closer_than_stranger": bool(Rm["dataset_mean"]["ci95"][1] < Rm["stranger"]["R"])}
    return {
        "H4": {"definition": "LPIPS and both Betti-matching parts: the repeat scores better than the repeat pulled to "
                             "the mean (T2, Holm 26 x 4) in UTIRIS_NIR",
               "per_metric": h4, "holds": all(v["T2_pass"] for v in h4.values())},
        "H5": {"definition": "FID and FWD: dataset mean and mean of 2 over the stranger >= 0.9 in UTIRIS_NIR; read "
                             "only where the bootstrap interval excludes 0.9",
               "per_metric": h5,
               "holds_point": all(v["holds_point"] for d in h5.values() for v in d.values())},
        "H6": {"definition": "LPIPS-AlexNet: R_self order repeat < lookalike < stranger, repeat upper 95% bound "
                             "below 1, UTIRIS_NIR", **h6},
        "descriptive_dataset_mean_vs_stranger": {
            "definition": "Not a frozen hypothesis: R_self of the dataset mean against the stranger rung per metric "
                          "and set; 'closer' = upper 95% bound of the mean below the stranger's R",
            "per_metric": mean_vs_stranger},
    }


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessing", required=True, choices=IF.VERSIONS)
    ap.add_argument("--tag", required=True, help="private file tag written by merge (e.g. full)")
    a = ap.parse_args()
    if a.preprocessing in IF.REJECTED:
        raise SystemExit(f"{a.preprocessing} was rejected: {IF.REJECTED[a.preprocessing]}")
    with open(OUT_JSON, encoding="utf-8") as fh:
        res = json.load(fh)
    sets, inputs = {}, {}
    for name in SETS:
        rp = os.path.join(IF.PRIVATE, f"rows_extra_{name}_{a.preprocessing}_{a.tag}.csv.gz")
        sp = os.path.join(IF.PRIVATE, f"setlevel_extra_{name}_{a.preprocessing}_{a.tag}.json")
        r = pd.read_csv(rp, dtype={"person": str})
        with open(sp, encoding="utf-8") as fh:
            sl = json.load(fh)
        print(f"{name}: {len(r)} rows", flush=True)
        sets[name] = summarise_set(r, sl)
        inputs[name] = {"rows_sha256": sha(rp), "setlevel_sha256": sha(sp)}
    holm_by_set = {}
    for name in SETS:
        h, n_h = holm_family(res["panel"][name]["tests"]["per_metric"], sets[name]["tests"], res["metrics_confirmatory"])
        sets[name]["holm_26x4"] = h
        holm_by_set[name] = n_h
    block = {
        "status": f"computed, full run on frame {a.preprocessing}; Kate's verdict on the smoke QC sheet 8 Nov 2026 "
                  "(run on P2, lid shadow kept); set-level bootstrap intervals flagged where they exclude the point",
        "protocol": "docs/PROTOCOL_iris_metrics.md, Amendment P3",
        "frame": a.preprocessing,
        "metrics": {"lpips_alex": "LPIPS 0.1.4 AlexNet v0.1, mean over E (primary)",
                    "lpips_vgg": "LPIPS VGG (sensitivity, topology subset S)",
                    "betti_matched": "Betti matching 2D, matched part (S, full windows)",
                    "betti_unmatched": "Betti matching 2D, unmatched part (S, full windows)",
                    "iriscode_hd": "iris-code fractional Hamming distance (supplementary; not an identity encoder)",
                    "fid": "pytorch-fid 0.3.0, set level, nuclear form, ratio to stranger",
                    "fwd": "pytorchfwd 1.0.1 Haar level 2, set level, ratio to stranger"},
        "direction": "lower = closer for every P3 metric; target itself = 0",
        "scales": "raw = mean per person, person bootstrap 2000; R_self = d / mean d of 16 strangers (self 0, "
                  "strangers 1); R_repeat = repeat 0; set level: 200 person resamples",
        "still_blocked": ["topology_distance", "plausibility", "id_angle", "id_norm", "id_pfe", "id_plda",
                          "latent_pullback", "latent_fisher_rao", "latent_lddmm"],
        "inputs_private": "C:/Projects/data/projects/identity-ladder-metric/iris-ladder-metrics-20261107-v1 "
                          "(rows stay private: UTIRIS redistribution not granted)",
        "inputs_sha256": inputs,
        "code_sha256": {"iris_ladder_metrics_extra.py": sha(os.path.join(HERE, "iris_ladder_metrics_extra.py")),
                        "iris_ladder_metrics_p3.py": sha(os.path.abspath(__file__))},
        "n_hypotheses_holm": holm_by_set,
        "sets": sets,
    }
    block["hypotheses"] = hypotheses(sets, sets[PRIMARY]["holm_26x4"])
    res["amendment_p3"] = block
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False, allow_nan=False)
    hy = block["hypotheses"]
    print("H4", hy["H4"]["holds"], {k: (round(v["T2_p_holm"], 4), v["T2_pass"]) for k, v in hy["H4"]["per_metric"].items()})
    print("H5 point", hy["H5"]["holds_point"])
    print("H6", hy["H6"]["holds"], hy["H6"]["R_self"], round(hy["H6"]["repeat_upper_95"], 3))
    print("mean vs stranger", {m: {s: round(v["R_dataset_mean"], 3) for s, v in d.items()}
                               for m, d in hy["descriptive_dataset_mean_vs_stranger"]["per_metric"].items()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
