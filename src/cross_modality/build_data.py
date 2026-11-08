"""Collect aggregate identity-ladder results from both worktrees into one JSON for the exploration page.
Read-only; aggregates only (no IDs)."""
import json
import math
import sys

W1 = "C:/Projects/02_academia/studies/_worktrees/identity-ladder-cvpr2027/results/"
W2 = "C:/Projects/02_academia/studies/_worktrees/identity-ladder-bilateral/results/"
SCR = sys.argv[1]


def load(p):
    return json.load(open(p, encoding="utf-8"))


def clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def rtab(src, key="R"):
    """metric -> rung -> [R, lo, hi, n]"""
    out = {}
    for m, rr in src.items():
        out[m] = {}
        for g, v in rr.items():
            if g.startswith("_") or not isinstance(v, dict) or v.get(key) is None:
                continue
            ci = v.get("ci95") or [None, None]
            out[m][g] = [v[key], ci[0], ci[1], v.get("n_targets")]
    return out


def tests_tab(per_metric, holm):
    out = {}
    for m, tt in per_metric.items():
        out[m] = {}
        for t, v in tt.items():
            if "status" in v:
                out[m][t] = {"na": v["status"]}
                continue
            p = v.get("p_holm") if holm else v.get("p_one_sided")
            out[m][t] = {"p": p, "adj": "Holm" if holm else "unadjusted", "share": v.get("share_targets_pass"),
                         "diff": v.get("mean_diff"), "n": v.get("n_targets"), "control": v.get("control"),
                         "pass": (v.get("pass_holm_0.05") if holm else (p is not None and p < 0.05 and v.get("mean_diff", 0) > 0))}
    return out


def exp_tab(src):
    out = {}
    for m, ee in src.items():
        out[m] = {k: (None if v is None else v.get("pass")) for k, v in ee.items() if isinstance(v, dict) or v is None}
    return out


D = {}
# ---------------- brain
b = load(W1 + "donor_ladder_brain_metrics.json")
bv = b["views"]
D["QTIM"] = {"modality": "brain MRI", "status": b["status"], "qc": "pre-QC", "file": "identity-ladder-cvpr2027/results/donor_ladder_brain_metrics.json",
             "counts": b["counts"], "views": {}}
for vn, v in bv.items():
    D["QTIM"]["views"][vn] = {"R_repeat": rtab(v["ladder_R"]), "expect": exp_tab(v["expectations"]),
                              "tests": tests_tab(v["tests"]["per_metric"], True) if "tests" in v else None}
D["QTIM"]["default_view"] = "random_patches/LR"
D["QTIM"]["not_run"] = b["not_run"]
# ---------------- faces
f = load(W1 + "face_ladder_metrics.json")
for src, name in (("celeba", "CelebA"), ("imdb", "IMDb-WIKI")):
    v = f["views"][src]
    D[name] = {"modality": "face photo", "status": f["status"], "qc": "QC approved (v6f, 6 Nov)",
               "file": "identity-ladder-cvpr2027/results/face_ladder_metrics.json", "counts": f["counts"][src],
               "views": {"all": {"R_self": rtab(v["ladder_R"]), "R_repeat": rtab(v["ladder_R_retest"]),
                                 "expect": exp_tab(v["expectations"]), "tests": None}},
               "default_view": "all"}
# ---------------- fundus
u = load(W2 + "fundus_ladder_metrics.json")
for ds in ("GRAPE", "PAPILA"):
    p = u["panel"][ds]
    tests = tests_tab(p["tests"]["per_metric"], False)
    view = {"R_self": rtab(p["R_self"]), "R_repeat": rtab(p["R_repeat"]) if ds == "GRAPE" else {},
            "expect": exp_tab(p["expectations"]), "tests": tests}
    if "tests_fellow_reference" in p:
        view["tests_fellow"] = tests_tab(p["tests_fellow_reference"], False)
    D[ds] = {"modality": "fundus photo", "status": u["status"] + " (" + u["frame"] + ")", "qc": "pre-QC",
             "file": "identity-ladder-bilateral/results/fundus_ladder_metrics.json", "counts": p["counts"],
             "views": {"all": view}, "default_view": "all", "reference_rung": p["reference_rung_for_tests"]}
D["_panel"] = {k: {"group": v["group"], "column": v.get("column"), "fundus_status": v["status"], "reason": v.get("reason")}
               for k, v in u["metrics_confirmatory"].items()}
# ---------------- iris (historical, c scale)
l2 = load(W1 + "ladder2d.json")
D["_iris"] = {"ladder2d_note": l2.get("note"), "protocol": l2.get("protocol")}
codes = load(W1 + "iris_codes.json")
D["_iris"]["codes"] = {k: {"median_hamming": codes[k].get("median_hamming"), "decidability": codes[k].get("decidability_vs_other_person"),
                           "seg_qc_pass": codes[k].get("segmentation_qc_pass"), "n_images": codes[k].get("n_images")}
                       for k in ("UTIRIS_NIR", "UPOL_secondary")}
# ---------------- correlations
c = load(SCR + "/corr_all.json")
bc = load(W1 + "donor_ladder_metrics_correlation.json")
corr = {}
for name, v in c.items():
    key = "QTIM" if name.startswith("QTIM") else name
    corr[key] = {"metrics": v["metrics"], "views": v["views"]}
corr["QTIM"]["views"]["boxes (deep + parietal)"] = {k: bc["views"]["boxes"][k] for k in ("rungs", "rho", "clusters")} | {
    "n_cells": bc["views"]["boxes"]["n_cells"], "n_cells_min_pair": bc["views"]["boxes"]["n_cells"]}
D["_corr"] = corr
json.dump(clean(D), open(SCR + "/data.json", "w"), separators=(",", ":"))
print(len(json.dumps(clean(D))))
