"""RQ4 size curve and the f = 1/8 cross-dataset input (PROTOCOL_donor_ladder_general.md, Amendment 10).

Aggregates only, from private rows. R_self is the function of `src/cross_modality/common_frame.py` (copied in
`frame()` below; a test checks it against `common_all.json`): R_self = (mean d(rung) - d(self)) / (mean d(16
strangers) - d(self)), means over cells, 95% interval from 1000 multinomial bootstrap draws over the dataset's unit.

Stages:
  curve    results/rq4_window_size.json and results/figures/fig_rq4_window_size.{png,pdf}
  common8  results/cross_modality/common_f8_v1.json (faces, iris, retina at f = 1/8; brain copied from common_all)
A size whose rows do not exist yet is reported as "not run" with the path it waits for, never as a value.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
P = "C:/Projects/data/projects/identity-ladder-metric/"
RQ = P + "rq4-window-size-20261114-v1/"
OUT_CURVE = os.path.join(ROOT, "results", "rq4_window_size.json")
OUT_FIG = os.path.join(ROOT, "results", "figures", "fig_rq4_window_size")
OUT_COMMON = os.path.join(ROOT, "results", "cross_modality", "common_f8_v1.json")
COMMON_ALL = os.path.join(ROOT, "results", "cross_modality", "common_all.json")
SEED = 20261106
REPS = 1000

# --- common_frame.py definitions (unchanged) ---------------------------------------------------------------------
SIM = ["ssim", "ms_ssim", "pearson", "acc_dataset", "acc_cohort", "lddt", "lddt_rep", "tm_like", "gdt_like", "arcface_cos"]
ERR = ["mse", "rmse", "mae", "rmse_rep", "persistence"]
DB = ["psnr", "psnr_fixed"]
RAW = ["ampdev_dataset", "ampdev_cohort"]
ALL = ERR[:3] + DB + ["ssim", "ms_ssim", "pearson", "acc_dataset", "acc_cohort", "ampdev_dataset", "ampdev_cohort",
                      "lddt", "lddt_rep", "tm_like", "gdt_like", "rmse_rep", "persistence", "arcface_cos"]
STR = [f"str{j:02d}" for j in range(16)]
RUNGS = ["repeat", "other_side", "relative", "lookalike_oracle", "lookalike", "lookalike16", "cohort_mean", "mean_k2",
         "dataset_mean", "wrong_detail", "noise", "shrink_ref", "stranger_same_centre", "stranger_other_centre"]
CURVE_RUNGS = ["lookalike", "lookalike16", "lookalike_oracle", "dataset_mean", "repeat", "shrink_ref"]
LABEL = {"lookalike": "look-alike (k = 1)", "lookalike16": "mean of 16 look-alikes", "lookalike_oracle":
         "oracle look-alike", "dataset_mean": "dataset mean", "repeat": "true person (second image)",
         "shrink_ref": "true person, shrunk to the mean"}


def prep(r, ren):
    """Renames and pooled rungs exactly as common_frame.py (per dataset)."""
    if "ampdev_dataset" not in r.columns and "amplitude" in r.columns:
        r["ampdev_dataset"] = (1.0 - r["amplitude"]).abs()
    r = r.replace([np.inf, -np.inf], np.nan)
    r["fill"] = r.fill.replace(ren)
    sh = "shrink_repeat" if (r.fill == "repeat").any() else next(
        f for f in ("shrink_other_side", "shrink_mirror", "shrink_fellow") if (r.fill == f).any())
    r.loc[r.fill == sh, "fill"] = "shrink_ref"
    r.loc[r.fill.isin(["sibling", "cotwin"]), "fill"] = "relative"
    return r


def frame(r, cell, unit, mets, rng):
    """R_self and raw means per rung and metric (common_frame.py loop body, unchanged arithmetic)."""
    r = r.copy()
    for m in mets:
        if m in SIM:
            r[m] = -r[m]
    keys = list(dict.fromkeys(cell + [unit]))
    st = r[r.fill.isin(STR)].groupby(keys)[mets].mean().reset_index()
    st["fill"] = "_strangers"
    rr = pd.concat([r[r.fill.isin(RUNGS)], st], ignore_index=True)
    rr["u"] = rr[unit].astype("category").cat.codes
    nu = int(rr.u.max() + 1)
    W = rng.multinomial(nu, np.full(nu, 1 / nu), size=REPS).astype(float)
    agg = {}
    for f, g in rr.groupby("fill"):
        S = np.zeros((nu, len(mets)))
        C = np.zeros((nu, len(mets)))
        for k, m in enumerate(mets):
            gg = g[g[m].notna()]
            np.add.at(S[:, k], gg.u.to_numpy(), gg[m].to_numpy())
            np.add.at(C[:, k], gg.u.to_numpy(), 1)
        agg[f] = (S, C)
    S0, C0 = agg["_strangers"]
    res, raw = {}, {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for f in RUNGS + ["_strangers"]:
            if f not in agg:
                continue
            S, C = agg[f]
            res[f] = {}
            for k, m in enumerate(mets):
                if C[:, k].sum() == 0 or C0[:, k].sum() == 0:
                    continue
                a, ab = S[:, k].sum() / C[:, k].sum(), (W @ S[:, k]) / (W @ C[:, k])
                z, zb = S0[:, k].sum() / C0[:, k].sum(), (W @ S0[:, k]) / (W @ C0[:, k])
                if m in DB:
                    v, vb = a - z, ab - zb
                elif m in RAW:
                    v, vb = a, ab
                else:
                    d0 = -1.0 if m in SIM else 0.0
                    v, vb = (a - d0) / (z - d0), (ab - d0) / (zb - d0)
                sg = -1.0 if m in SIM else 1.0
                rlo, rhi = sorted(np.nanpercentile(sg * ab, [2.5, 97.5]))
                raw.setdefault("stranger" if f == "_strangers" else f, {})[m] = [
                    round(float(sg * a), 5), round(float(rlo), 5), round(float(rhi), 5), int((C[:, k] > 0).sum())]
                lo, hi = np.nanpercentile(vb, [2.5, 97.5])
                res[f][m] = [round(float(v), 4), round(float(lo), 4), round(float(hi), 4), int((C[:, k] > 0).sum())]
    return {"unit": unit, "n_units": nu, "metrics": mets, "R_self": res, "raw": raw}


# --- inputs ---------------------------------------------------------------------------------------------------------

def read(paths, need=None, fills=None):
    files = sorted(f for p in paths for f in glob.glob(p))
    if not files:
        return None
    use = (lambda c: c in need) if need else None
    x = pd.concat([pd.read_csv(f, usecols=use) for f in files], ignore_index=True)
    if fills is not None:
        x = x[x.fill.isin(fills)]
    return x


def need_cols(cell, unit, mets):
    return set(cell + [unit, "fill", "src", "amplitude"] + mets)


FACE_SUB = "C:/Projects/data/projects/identity-ladder-metric/face-ladder-sizes-20261108-v3/rows_v6f_w{w}_n100.csv.gz"
FACE_SRC = {"CelebA": "celeba", "IMDb-WIKI": "imdb"}
IRIS_SET = {"UTIRIS NIR": "UTIRIS_NIR", "UTIRIS VIS": "UTIRIS_VIS", "UPOL": "UPOL"}
RET_SET = {"GRAPE": ("GRAPE", "p8"), "PAPILA": ("PAPILA", "p9")}


def face_subset_targets(src):
    """The F3 subset (100 evenly spaced targets) of one source."""
    sys.path.insert(0, HERE)
    import face_ladder_metrics as FM
    desc = json.load(open(os.path.join(FM.PRIVATE, "cache_v6f", "desc.json")))["data"]
    T = desc[src]["targets"]
    return set(np.unique(np.linspace(0, len(T) - 1, 100).round().astype(int)).tolist())


def curve_inputs():
    """{dataset: [(size label, f, window, unit, cell, ren, row loader or missing reason, note)]}."""
    out = {}
    bq = P + "donor-ladder-brain-metrics-size-20261108-v1/w{c}_rows.csv.gz"
    b16 = (P + "donor-ladder-brain-metrics-20261106-v3/rows.csv.gz", P + "donor-ladder-brain-lookalike16-20261109-v1/parts/p*.csv.gz")
    cell, unit, ren = ["target", "patch", "side"], "target", {"mirror": "other_side"}
    out["QTIM"] = []
    for c, f in ((8, 1 / 16), (16, 16 / 127.0), (32, 1 / 4)):
        if c == 16:
            paths, extra = [b16[0]], [b16[1]]
        else:
            paths, extra = [bq.format(c=c)], [RQ + f"brain/parts/w{c}_p*.csv.gz"]
        out["QTIM"].append(dict(label=f"{c} mm", f=f, w=c, unit=unit, cell=cell, ren=ren, paths=paths, extra=extra,
                                filt=(lambda r: r[r.patch < 32])))
    for name, src in FACE_SRC.items():
        out[name] = []
        for w, L in ((5, None), (10, None), (16, None), (20, None)):
            if w == 16:
                paths = [P + "face-ladder-metrics-20261106-v1/rows_v6f.csv.gz"]
                extra = [P + "face-ladder-lookalike16-20261109-v1/rows_t1_v6f.csv.gz"]
            else:
                paths, extra = [FACE_SUB.format(w=w)], [RQ + f"faces/rows_sub_w{w}_n100.csv.gz"]
            out[name].append(dict(label=f"{w} px", f=w / 79.5, w=w, unit="family", cell=["target", "patch"], ren=ren,
                                  paths=paths, extra=extra, filt=("face", src), subset="F3: 100 evenly spaced targets"))
    for name, s in IRIS_SET.items():
        out[name] = []
        for w in (5, 10, 16, 20):
            if w == 16:
                paths = [P + f"iris-ladder-metrics-20261107-v1/rows_{s}_P2_full.csv.gz"]
            elif w == 10 and s == "UTIRIS_NIR":
                paths = [P + "iris-ladder-metrics-20261107-v1/rows_UTIRIS_NIR_P2_full_w10.csv.gz"]
            else:
                tag = "every4" if (w == 5 and s == "UTIRIS_NIR") else "full"   # 1/16: every 4th eye (CPU budget)
                paths = [RQ + f"iris/rows_{s}_P2_w{w}_{tag}.csv.gz"]
            out[name].append(dict(label=f"{w} px", f=w / (256 / np.pi), w=w, unit="person", cell=["unit", "patch"],
                                  ren={"fellow": "other_side"}, paths=paths, extra=None, filt=None))
    geo = os.path.join(RQ, "retina", "geometry.json")
    geo = json.load(open(geo)) if os.path.exists(geo) else {}
    for name, (s, arm) in RET_SET.items():
        out[name] = []
        L = geo.get(s, {}).get("L_px")
        ws = sorted({16} | {v["window_px"] for v in geo.get(s, {}).get("sizes", {}).values()})
        for w in ws:
            paths = ([P + f"fundus-ladder-metrics-20261106-v1/rows_{s}_{arm}_full.csv.gz"] if w == 16
                     else [RQ + f"retina/rows_{s}_{arm}_w{w}_{'every4' if w == 5 else 'full'}.csv.gz"])
            out[name].append(dict(label=f"{w} px", f=w / L if L else None, w=w, unit="person", cell=["unit", "patch"],
                                  ren={"fellow": "other_side", "sexage": "cohort_mean"}, paths=paths, extra=None,
                                  filt=None))
    return out


def load(spec, mets, subset_targets=None):
    need = need_cols(spec["cell"], spec["unit"], mets)
    r = read(spec["paths"], need)
    if r is None:
        return None, f"not run: no rows at {spec['paths'][0]}"
    if spec.get("extra"):
        x = read(spec["extra"], need, ["lookalike16"])
        if x is None:
            return None, f"not run: no lookalike16 rows at {spec['extra'][0]}"
        r = pd.concat([r[r.fill != "lookalike16"], x], ignore_index=True)
    f = spec.get("filt")
    if isinstance(f, tuple) and f[0] == "face":
        r = r[r.src == f[1]]
        if subset_targets is not None:
            r = r[r.target.isin(subset_targets)]
    elif callable(f):
        r = f(r)
    return prep(r, spec["ren"]), None


def curve():
    specs = curve_inputs()
    rng = np.random.default_rng(SEED)
    out = {"protocol": "docs/PROTOCOL_donor_ladder_general.md, Amendment 10", "metric": "mse",
           "scale": "R_self: 0 = target itself, 1 = mean of the 16 seeded strangers (common_frame.py)",
           "interval": f"95% percentile, {REPS} multinomial bootstrap draws over the unit",
           "reference_f": "1/8", "status": "brain, retina and iris provisional (registration QC gate not met)",
           "rungs": {k: LABEL[k] for k in CURVE_RUNGS}, "datasets": {}}
    for name, sizes in specs.items():
        sub = face_subset_targets(FACE_SRC[name]) if name in FACE_SRC else None
        ds = {}
        loaded = {}
        for s in sizes:
            r, why = load(s, ["mse"], sub)
            loaded[s["label"]] = (s, r, why)
        # 1/16 on every 4th eye (Amendment 10, CPU budget): the curve is drawn on those eyes at every size (paired);
        # the full-cohort values of the other sizes are kept in "all_units".
        sub16 = [r for s, r, _ in loaded.values() if r is not None and "every4" in s["paths"][0]]
        keep_units = set(sub16[0]["unit"].unique()) if sub16 else None
        for key, (s, r, why) in loaded.items():
            if r is None:
                ds[key] = {"f": s["f"], "window": s["w"], "status": why}
                continue
            views = {"all_units": r}
            if keep_units is not None:
                views = {"paired_every4": r[r["unit"].isin(keep_units)], "all_units": r}
            ds[key] = {"f": s["f"], "window": s["w"], "n_positions": int(r.patch.nunique()),
                       "subset": s.get("subset", "every 4th eye" if "every4" in s["paths"][0] else "all units")}
            for vname, rv in views.items():
                if "every4" in s["paths"][0] and vname == "all_units":
                    continue
                fr = frame(rv, s["cell"], s["unit"], ["mse"], rng)
                ds[key][vname] = {"unit": fr["unit"], "n_units": fr["n_units"],
                                  "R_self_mse": {k: fr["R_self"][k]["mse"] for k in CURVE_RUNGS
                                                 if k in fr["R_self"] and "mse" in fr["R_self"][k]},
                                  "missing_rungs": [k for k in CURVE_RUNGS if k not in fr["R_self"]]}
            first = next(iter(views))
            ds[key]["R_self_mse"] = ds[key][first]["R_self_mse"]          # the plotted (paired) values
            ds[key]["plotted"] = first
            print(name, key, first, {k: v[0] for k, v in ds[key]["R_self_mse"].items()}, flush=True)
        out["datasets"][name] = ds
    ixi = json.load(open(COMMON_ALL))["datasets"]["IXI"]["R_self"]
    out["datasets"]["IXI"] = {"16 mm": {"f": 16 / 127.0, "window": 16, "source": "results/cross_modality/common_all.json",
                                        "R_self_mse": {k: ixi[k]["mse"] for k in CURVE_RUNGS if k in ixi}},
                              "8 mm": {"status": "not run: no size geometry for IXI (no Amendment 21 run)"},
                              "32 mm": {"status": "not run: no size geometry for IXI (no Amendment 21 run)"}}
    with open(OUT_CURVE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    figure(out)
    print(OUT_CURVE, flush=True)


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"lookalike": "#d6604d", "lookalike16": "#b2182b", "lookalike_oracle": "#f4a582", "dataset_mean": "#4393c3",
           "repeat": "#1a9850", "shrink_ref": "#91cf60"}
    names = list(out["datasets"])
    ncol = 5
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.9 * nrow + 0.8), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        ds = out["datasets"][name]
        pts = sorted([(v["f"], k, v) for k, v in ds.items() if "R_self_mse" in v], key=lambda t: t[0])
        for rung in CURVE_RUNGS:
            xs = [(f, v["R_self_mse"][rung]) for f, _, v in pts if rung in v["R_self_mse"]]
            if not xs:
                continue
            x = np.array([a for a, _ in xs])
            y = np.array([b[0] for _, b in xs])
            lo = np.array([b[1] for _, b in xs])
            hi = np.array([b[2] for _, b in xs])
            ax.plot(x, y, "o-", ms=3, lw=1.1, c=col[rung], label=LABEL[rung])
            ax.vlines(x, lo, hi, color=col[rung], lw=0.8)
        ax.axvline(1 / 8, c="#555", lw=0.8, ls="--")
        ax.axhline(0, c="#bbb", lw=0.5)
        ax.axhline(1, c="#bbb", lw=0.5, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1 / 16, 1 / 8, 1 / 4])
        ax.set_xticklabels(["1/16", "1/8", "1/4"])
        ax.set_xlim(1 / 20, 1 / 3.3)
        ax.set_ylim(-0.05, 1.1)
        miss = [k for k, v in ds.items() if "R_self_mse" not in v]
        ax.set_title(name + (f"\nnot run: {', '.join(miss)}" if miss else ""), fontsize=7.5)
        ax.set_xlabel("window f = w / L (object units)", fontsize=6.5)
        ax.set_ylabel("R (MSE): 0 = target, 1 = strangers", fontsize=6.5)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(names):]:
        ax.set_axis_off()
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=7, frameon=False)
    fig.suptitle("RQ4: how close context-selected look-alikes come to the true person, by window size in object units "
                 "(dashed: f = 1/8, the summary size). Bars: 95% bootstrap interval over people / families / targets. "
                 "Brain, retina, iris provisional.", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG + ".png", dpi=170)
    fig.savefig(OUT_FIG + ".pdf")
    plt.close(fig)


def common8():
    """f = 1/8 inputs for the summary, the common_all.json format; brain entries copied (already f = 1/8)."""
    base = json.load(open(COMMON_ALL))
    rng = np.random.default_rng(SEED)
    out = {"method": base["method"], "amendment": "docs/PROTOCOL_donor_ladder_general.md, Amendment 10 (f = 1/8)",
           "not_available": base["not_available"], "metrics": ALL, "rungs": RUNGS, "kind": base["kind"],
           "window": {}, "datasets": {}}
    for name in ("QTIM", "IXI"):
        out["datasets"][name] = base["datasets"][name]
        out["window"][name] = {"window": "16 mm cube", "f": 16 / 127.0, "source": "copied from common_all.json"}
    sets = {}
    for name, src in FACE_SRC.items():
        sets[name] = dict(paths=[RQ + "faces/rows_full_w10.csv.gz"], cell=["target", "patch"], unit="family",
                          ren={"mirror": "other_side"}, filt=("face", src), w="10 px", f=10 / 79.5)
    for name, s in IRIS_SET.items():
        p = (P + "iris-ladder-metrics-20261107-v1/rows_UTIRIS_NIR_P2_full_w10.csv.gz" if s == "UTIRIS_NIR"
             else RQ + f"iris/rows_{s}_P2_w10_full.csv.gz")
        sets[name] = dict(paths=[p], cell=["unit", "patch"], unit="person", ren={"fellow": "other_side"}, filt=None,
                          w="10 px", f=10 / (256 / np.pi))
    geo = os.path.join(RQ, "retina", "geometry.json")
    geo = json.load(open(geo)) if os.path.exists(geo) else {}
    for name, (s, arm) in RET_SET.items():
        if s not in geo:
            continue
        w = geo[s]["sizes"]["1/8"]["window_px"]
        sets[name] = dict(paths=[RQ + f"retina/rows_{s}_{arm}_w{w}_full.csv.gz"], cell=["unit", "patch"],
                          unit="person", ren={"fellow": "other_side", "sexage": "cohort_mean"}, filt=None,
                          w=f"{w} px", f=w / geo[s]["L_px"], note="tier-1 metrics only (Amendment 9.15 extras off)")
    for name, s in sets.items():
        r, why = load(dict(s, extra=None), ALL)
        out["window"][name] = {"window": s["w"], "f": s["f"], **({"note": s["note"]} if "note" in s else {})}
        if r is None:
            out["datasets"][name] = {"status": why}
            print(name, why, flush=True)
            continue
        mets = [m for m in ALL if m in r.columns and r[m].notna().any()]
        out["datasets"][name] = frame(r, s["cell"], s["unit"], mets, rng)
        print(name, out["datasets"][name]["n_units"], mets, flush=True)
    with open(OUT_COMMON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    print(OUT_COMMON, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("curve", "common8"))
    a = ap.parse_args()
    curve() if a.stage == "curve" else common8()


if __name__ == "__main__":
    main()
