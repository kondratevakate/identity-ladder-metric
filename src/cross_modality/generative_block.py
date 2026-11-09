"""Generative, perceptual and topology block for the cross-modality tiles (aggregates only).

Sources, read only: fundus `results/fundus_ladder_metrics_a915.json` (Amendment 9.15, identity-ladder-bilateral),
iris `results/iris_ladder_metrics.json` key `amendment_p3`, faces `results/face_ladder_metrics.json` key `panel_f2`
(LPIPS is AlexNet there), brain `results/donor_ladder_brain_metrics_t23.json` view `deep/LR` (Amendment 20, LPIPS
AlexNet on central orthogonal slices, persistence W1 over H0-H2; pre-QC). Brain R is anchored on the repeat scan
(repeat 0, strangers 1); it is an affine map of the per-rung mean distance, so the rank order is the same as on the
self scale. Relative = sibling and co-twin pooled, weighted by targets. Metrics a summary does not carry (brain Betti
3D, FID, FWD) are "not run" with the reason from that JSON; never a zero.
Per-target metrics (LPIPS AlexNet, LPIPS VGG, Betti matching matched / unmatched, intensity persistence, iris-code
Hamming distance) use R_self per rung (target 0, mean of 16 strangers 1, lower = closer). Set-level metrics (FID,
FWD) use the per-rung set value (lower = closer). The tile rank is the rung's place among the rungs present.
Writes `ranks_generative.json`, `generative_status.json` and `generative_tiles.png`.

Usage: python -I src/cross_modality/generative_block.py results/cross_modality <bilateral_worktree>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

OUT, BIL = sys.argv[1], sys.argv[2]
HERE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
COLS = ["repeat", "other_side", "relative", "lookalike_oracle", "lookalike", "lookalike16", "cohort_mean", "mean_k2",
        "dataset_mean", "stranger", "shrink_ref", "wrong_detail", "noise"]
LAB = {"repeat": "repeat image", "other_side": "other side (mirror / eye)", "relative": "relative (sibling or twin)",
       "lookalike_oracle": "best look-alike", "lookalike": "look-alike", "lookalike16": "mean of 16 look-alikes",
       "cohort_mean": "sex-age mean",
       "mean_k2": "mean of 2 strangers", "dataset_mean": "dataset mean   [T1]", "stranger": "stranger",
       "shrink_ref": "repeat shrunk to mean   [T2]", "wrong_detail": "wrong detail   [T3a]", "noise": "white noise   [T3b]"}
MET = [("lpips_alex", "LPIPS (AlexNet)", "per target"), ("lpips_vgg", "LPIPS (VGG)", "per target"),
       ("betti_matched", "Betti matching, matched", "per target"), ("betti_unmatched", "Betti matching, unmatched", "per target"),
       ("persistence", "intensity persistence", "per target"), ("iriscode_hd", "iris code Hamming distance", "per target"),
       ("fid", "FID (set level)", "set"), ("fwd", "FWD (set level)", "set")]
NOT_RUN = {"IXI": "IXI (brain Amendment 23) scores tier-1 pixel metrics only; LPIPS, Betti, persistence, FID, FWD not run"}
NA16 = {"lookalike16": {   # same reasons as common_frame.py NOT_AVAILABLE; n/a, never a value
    "QTIM": "dropped by the brain protocol (PROTOCOL_donor_ladder_brain.md, block 5: k = 4, 8, 16 and the 16-look-alike mean)",
    "CelebA": "not defined in the face protocol (PROTOCOL_face_metrics.md)",
    "IMDb-WIKI": "not defined in the face protocol (PROTOCOL_face_metrics.md)",
    "IXI": "not defined in brain Amendment 23"}}
NOT_RUN_METRICS = {"QTIM": ("betti_matched", "betti_unmatched", "fid", "fwd"),
                   "IXI": ("lpips_alex", "lpips_vgg", "betti_matched", "betti_unmatched", "persistence", "fid", "fwd")}
# Brain set level: the own-other-side and relative sets are drawn from the same scans as the target set, so their FID
# and FWD are ~0 by construction (brain Amendment 22 note); left out of the set-level ranks, never read as identity.
EXCLUDED_SET = {"QTIM": {r: "set nearly coincides with the target set by construction (Amendment 22)"
                         for r in ("mirror", "sibling", "cotwin")}}


def ren(f, has_repeat):
    f = {"fellow": "other_side", "mirror": "other_side", "sexage": "cohort_mean"}.get(f, f)
    if f == ("shrink_repeat" if has_repeat else "shrink_fellow"):
        return "shrink_ref"
    return f


def entries(R, sets, m):
    """{rung: source entry} for one metric (rungs renamed to the common frame); None if the metric is absent."""
    key = "value" if m in ("fid", "fwd") else "R"
    blk = ((sets if key == "value" else R) or {}).get(m)
    if not blk:
        return None
    has_rep = "repeat" in blk
    return {ren(f, has_rep): e for f, e in blk.items() if isinstance(e, dict) and e.get(key) is not None}


def per_rung(R, sets, m):
    """{rung: value} for one metric, lower = closer; None if the metric is absent."""
    e = entries(R, sets, m)
    return None if e is None else {r: x["value" if m in ("fid", "fwd") else "R"] for r, x in e.items()}


def value_record(m, e, stranger):
    """The number a rank comes from, with its interval and n. Per target: R with its bootstrap interval. Set level:
    value / stranger-set value, interval as the source JSON gives it (n/a when it gives none); the flag says whether
    that interval contains the point (small-sample FID bias makes it miss on fundus and iris)."""
    n = e.get("n_targets", e.get("n_cells", e.get("windows", e.get("n_windows"))))
    if m not in ("fid", "fwd"):
        ci = e.get("ci95")
        return {"R": e["R"], "ci95": ci if ci else "n/a", "n": n if n is not None else "n/a"}
    ratio = e["value"] / stranger["value"]
    ci = e.get("ratio_ci95") or e.get("over_stranger_ci95")
    rec = {"ratio_to_stranger": ratio, "value": e["value"], "stranger_value": stranger["value"],
           "ratio_ci95": ci if ci else "n/a", "n": n if n is not None else "n/a"}
    if ci:
        rec["ratio_ci_contains_point"] = bool(ci[0] <= ratio <= ci[1])
    return rec


src = {}
a915 = json.load(open(os.path.join(BIL, "results", "fundus_ladder_metrics_a915.json")))["panel"]
for d in ("GRAPE", "PAPILA"):
    g = a915[d]
    src[d] = (g["R_self"], g["amendment_9_15"]["set_level"])
iris = json.load(open(os.path.join(HERE, "results", "iris_ladder_metrics.json")))["amendment_p3"]["sets"]
for d in ("UTIRIS_NIR", "UTIRIS_VIS", "UPOL"):
    s = iris[d]
    src[d.replace("_", " ")] = (s["R_self"], s["set_level"])



def pool_relative(blk):
    """Sibling and co-twin into one 'relative' entry, mean R weighted by targets (aggregates only)."""
    out = {k: v for k, v in blk.items() if k not in ("sibling", "cotwin")}
    parts = [blk[k] for k in ("sibling", "cotwin") if isinstance(blk.get(k), dict) and blk[k].get("R") is not None]
    if parts:
        n = sum(p["n_targets"] for p in parts)
        out["relative"] = {"R": sum(p["R"] * p["n_targets"] for p in parts) / n, "n_targets": n}
    return out


def from_ladder(L, names):
    """Faces call the repeat image 'retest' and keep the target itself as rung 'self' (R = 0); both renamed / dropped."""
    fix = lambda blk: {("repeat" if k == "retest" else k): v for k, v in blk.items() if k != "self"}
    return {new: pool_relative(fix(L[old])) for old, new in names.items() if old in L}


face_all = json.load(open(os.path.join(HERE, "results", "face_ladder_metrics.json")))
face, face_vgg = face_all["panel_f2"], face_all.get("panel_f2_lpips_vgg")   # VGG: Amendment F2.1, absent until run
for key, d in (("celeba", "CelebA"), ("imdb", "IMDb-WIKI")):
    R = from_ladder(face["views"][key]["ladder_R"], {"lpips": "lpips_alex", "betti_matched": "betti_matched",
                                                    "betti_unmatched": "betti_unmatched", "persistence": "persistence"})
    if face_vgg:
        R.update(from_ladder(face_vgg["views"][key]["ladder_R"], {"lpips_vgg": "lpips_vgg"}))
    sl = {r: e for r, e in face["set_level"][key].items() if not r.startswith("_") and r != "target"}
    S = {m: pool_relative({r: {"value": e[m], "R": e[m], "n_targets": e.get("n_cells", 1)} for r, e in sl.items()
                           if e.get(m) is not None}) for m in ("fid", "fwd")}
    S = {m: {r: {"value": e["R"], "n_cells": e["n_targets"]} for r, e in b.items()} for m, b in S.items()}
    src[d] = (R, S)
brain = json.load(open(os.path.join(HERE, "results", "donor_ladder_brain_metrics_t23.json")))
R = from_ladder(brain["views"]["deep/LR"]["ladder_R"], {"lpips": "lpips_alex", "persistence_w1": "persistence"})
S = {}
b5p = os.path.join(HERE, "results", "donor_ladder_brain_metrics_t23_v5.json")   # Amendment 22: LPIPS VGG, FID, FWD
if os.path.exists(b5p):
    b5 = json.load(open(b5p))
    R.update(from_ladder(b5["views"]["deep/LR"]["ladder_R"], {"lpips_vgg": "lpips_vgg"}))
    sl = {r: e for r, e in b5["set_level"]["deep/LR"].items()
          if not r.startswith("_") and r not in EXCLUDED_SET.get("QTIM", {}) and e.get("fid") is not None}
    S = {m: {r: {"value": e[m], "n_cells": e["n_cells"]} for r, e in sl.items()} for m in ("fid", "fwd")}
    NOT_RUN_METRICS["QTIM"] = ("betti_matched", "betti_unmatched")
src["QTIM"] = (R, S)
NOT_RUN["QTIM"] = "brain: " + "; ".join(f"{k}: {v}" for k, v in brain["not_run"].items()
                                        if k == "betti_matching_3d" or (k == "fid_fwd_topology_distance" and not S))

ANCHOR = {d: "self" for d in src}
ANCHOR["QTIM"] = "repeat"   # brain Amendment 20 R: repeat scan 0, strangers 1 (an affine map of the self scale)
ranks, values = {}, {}
status = {"method": __doc__.strip(), "not_run": NOT_RUN, "not_run_metrics": NOT_RUN_METRICS, "anchor": ANCHOR, "excluded_set_level_rungs": EXCLUDED_SET,
          "present": {}}
for d, (R, S) in src.items():
    status["present"][d] = []
    for m, _, _ in MET:
        v = per_rung(R, S, m)
        if not v:
            continue
        ent = entries(R, S, m)
        if m in ("fid", "fwd") and "stranger" not in ent:
            continue
        values[f"{m}|{d}"] = {"anchor": ANCHOR[d], "scale": "ratio to stranger set" if m in ("fid", "fwd") else "R",
                              "rungs": {r: value_record(m, ent[r], ent.get("stranger")) for r in COLS if r in ent}}
        it = [(r, v[r]) for r in COLS if r in v]
        if len(it) < 3:
            continue
        order = sorted(it, key=lambda x: x[1])
        ranks[f"{m}|{d}"] = {r: i + 1 for i, (r, _) in enumerate(order)}
        status["present"][d].append(m)
json.dump(ranks, open(os.path.join(OUT, "ranks_generative.json"), "w"), indent=1)
json.dump({"method": value_record.__doc__.strip(), "anchor_note": "self: R = (d(rung) - d(self)) / (d(16 strangers) - "
           "d(self)); repeat (brain): the same with the repeat scan in place of self", "not_available": NA16,
           "not_run": NOT_RUN, "values": values}, open(os.path.join(OUT, "values_generative.json"), "w"), indent=1)
json.dump(status, open(os.path.join(OUT, "generative_status.json"), "w"), indent=1)

DS = ["QTIM", "CelebA", "IMDb-WIKI", "GRAPE", "PAPILA", "UTIRIS NIR", "UTIRIS VIS", "UPOL", "IXI"]
cmap = LinearSegmentedColormap.from_list("rank", ["#0b5d63", "#3aa6a8", "#bfe3dc", "#f6e7c8", "#f0a868", "#b7471a"])
rows = []
for m, name, kind in MET:
    blk = [(d, ranks.get(f"{m}|{d}")) for d in DS]
    if not any(r for _, r in blk):
        continue
    rows.append(("m", f"{name}  ({'lower = closer' if kind == 'per target' else 'set level, lower = closer'})"))
    rows += [("r", (d, r, m)) for d, r in blk]
    rows.append(("g", None))
H = sum(1 if k != "g" else 0.4 for k, _ in rows) + 7
fig, ax = plt.subplots(figsize=(11.5, 0.24 * H + 0.6))
for j, r in enumerate(COLS):
    ax.text(j + 0.5, -0.25, LAB[r], fontsize=7.5, rotation=50, rotation_mode="anchor", ha="left", va="bottom",
            color="#7a2e0e" if "[T" in LAB[r] else "#222", fontweight="bold" if "[T" in LAB[r] else "normal")
y = 0.0
for k, b in rows:
    if k == "g":
        y += 0.4
        continue
    if k == "m":
        ax.text(-0.2, y + 0.5, b, fontsize=8, fontweight="bold", ha="right", va="center")
        y += 1
        continue
    d, rank, m = b
    ax.text(-0.2, y + 0.46, d, fontsize=7.5, ha="right", va="center", color="#1f6f3a" if d in ("CelebA", "IMDb-WIKI") else "#8a6400")
    if not rank:
        ax.add_patch(Rectangle((0, y), len(COLS) - 0.05, 0.92, facecolor="#f4f4f4", edgecolor="#dddddd", linewidth=0.5))
        why = NOT_RUN.get(d) if m in NOT_RUN_METRICS.get(d, ()) else None
        ax.text(0.15, y + 0.46, "not run: " + why if why else "metric not defined or not computed for this dataset",
                fontsize=6.3, va="center", color="#888888")
        y += 1
        continue
    n = len(rank)
    for j, r in enumerate(COLS):
        if r not in rank:
            ax.add_patch(Rectangle((j, y), 0.95, 0.92, facecolor="none", edgecolor="#dddddd", linewidth=0.5))
            continue
        t = (rank[r] - 1) / max(1, n - 1)
        ax.add_patch(Rectangle((j, y), 0.95, 0.92, facecolor=cmap(t), edgecolor="none"))
        ax.text(j + 0.475, y + 0.46, str(rank[r]), fontsize=7, ha="center", va="center",
                color="white" if (t < 0.25 or t > 0.8) else "#222")
    y += 1
ax.text(-7.0, y + 1.8, "Same columns and colours as the main rank tiles (1 = what the metric calls closest). Per-target metrics are "
        "ranked by R_self,\nset-level FID and FWD by their per-rung set value. Fundus 9.15, iris P3, faces F2 and F2.1 (LPIPS VGG), "
        "brain Amendments 20 and 22\n(deep box, both sides; repeat-anchored R; set level leaves out own other side and relative, "
        "near 0 by construction). Grey rows: not computed, reason given. All pre-QC except faces.",
        fontsize=7, color="#333", va="center")
ax.set_xlim(-7.2, len(COLS) + 0.2)
ax.set_ylim(y + 3.2, -5.2)
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "generative_tiles.png"), dpi=170, facecolor="white")
print({d: v for d, v in status["present"].items()})
