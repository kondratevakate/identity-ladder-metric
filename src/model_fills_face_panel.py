"""LPIPS, FID and FWD of the model fills on the face panel (PROTOCOL_face_metrics.md, Amendment F2: 16 px windows,
frame v6f), next to the stored values of the references.

The references' LPIPS rows and set-level patches exist only at the 16 px panel window (f = 0.2), so the models are
run on the same windows (`model_fills_face.py export --panel`) and scored with the panel's own functions:
`lpips_batch` (AlexNet, lpips 0.1, window at 64 px), `inception_features`, `frechet`, `fwd`. Scored cells are the
panel's: LPIPS on every (target, window) whose window is complete in the target, FID and FWD on the stored target
cells of `set_patches_v6f.npz`.

Gates before a model value is kept: (1) LPIPS of the dataset-average fill recomputed here equals the stored rows;
(2) FID and FWD of the stored random-person set recomputed here equal `set_level_v6f.json`.

Usage: python src/model_fills_face_panel.py --models lama_places lama_celebahq --torch-home DIR
Writes aggregates to results/model_fills_v1/model_fills_face_panel.json; rows stay private.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import face_ladder_metrics as FM  # noqa: E402
import face_ladder_metrics_panel as PN  # noqa: E402
import model_fills_face as MF  # noqa: E402
import rq4_window_size as RW  # noqa: E402

OUT = os.path.join(RW.ROOT, "results", "model_fills_v1", "model_fills_face_panel.json")
VERSION = MF.VERSION
SRC_NAME = {"celeba": "CelebA", "imdb": "IMDb-WIKI"}
REFS = ("repeat", "mirror", "lookalike", "dataset_mean", "shrink_repeat", "wrong_detail", "noise", "stranger")


def load_model(model):
    d = os.path.join(MF.PRIVATE, f"{model}_w16")
    meta = json.load(open(os.path.join(d, "infer_meta.json")))
    if meta["window_px"] != FM.WIN:
        raise SystemExit(f"{model}: fills at {meta['window_px']} px, the panel window is {FM.WIN} px")
    return ({s: np.load(os.path.join(d, f"{s}.npy")) for s in meta["sources"]},
            {s: {ti: a for a, ti in enumerate(m["target_index"])} for s, m in meta["sources"].items()}, meta)


def lpips_rows(models, cache, desc):
    """Per (source, target, window) with a complete target window: LPIPS of each model fill and of the dataset
    average (gate), by the panel's `lpips_batch`."""
    rows = []
    for src, d in desc.items():
        valid = np.load(os.path.join(cache, f"{src}_valid.npy"))
        pos = FM.positions(valid)
        mean, mean_ok = np.asarray(FM.WORK[f"{src}_mean"]), np.asarray(FM.WORK[f"{src}_mean_ok"])
        pairs, keys = [], []
        for ti, t in enumerate(d["targets"]):
            xt, mt = np.asarray(FM.WORK["X"][t["t"]], np.float32), np.asarray(FM.WORK["M"][t["t"]])
            for pk, (y0, x0) in enumerate(pos):
                sl = (slice(y0, y0 + FM.WIN), slice(x0, x0 + FM.WIN))
                tv = mt[sl] & valid[sl]
                if tv.mean() < FM.MIN_IN_MASK or not tv.all():
                    continue
                if mean_ok[sl].all():
                    pairs.append((xt[sl], np.asarray(mean[sl], np.float32)))
                    keys.append((src, ti, t["family"], pk, "dataset_mean"))
                for name, (fills, row, _) in models.items():
                    if ti in row[src] and np.isfinite(fills[src][row[src][ti], pk]).all():   # NaN = window not run
                        pairs.append((xt[sl], np.asarray(fills[src][row[src][ti], pk], np.float32)))
                        keys.append((src, ti, t["family"], pk, f"model_{name}"))
        vals = []
        for b in range(0, len(pairs), 512):
            vals += PN.lpips_batch(pairs[b:b + 512])
        rows += [dict(zip(("src", "target", "family", "patch", "fill"), k), lpips=v) for k, v in zip(keys, vals)]
        print(f"{src}: LPIPS on {len(pairs)} windows", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--torch-home", required=True)
    a = ap.parse_args()
    os.environ["TORCH_HOME"] = a.torch_home
    import lpips
    import torch
    torch.set_num_threads(3)
    PN.W2["lpips"], PN.W2["torch"] = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval(), torch
    cache = os.path.join(FM.PRIVATE, f"cache{VERSION}")
    FM._init(cache)
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    models = {m: load_model(m) for m in a.models}
    out = {"method": __doc__.strip(), "window": "16 px (f = 0.2), the panel window of Amendment F2",
           "gates": {}, "datasets": {SRC_NAME[s]: {"models": {}} for s in desc}}

    # ---- LPIPS, per target
    new = lpips_rows(models, cache, desc)
    ref = pd.read_csv(os.path.join(PN.PRIVATE, f"rows{VERSION}.csv.gz"),
                      usecols=["src", "target", "family", "patch", "fill", "lpips"])
    g = new[new.fill == "dataset_mean"].merge(ref[ref.fill == "dataset_mean"], on=["src", "target", "patch", "fill"],
                                              suffixes=("", "_ref"))
    n_ref = int(ref[(ref.fill == "dataset_mean") & ref.lpips.notna()].shape[0])
    d = float(np.nanmax(np.abs(g.lpips - g.lpips_ref)))
    out["gates"]["lpips_dataset_mean"] = {"rows_recomputed": int(len(g)), "stored_rows_with_lpips": n_ref,
                                          "max_abs_diff": d, "passed": bool(len(g) == n_ref and d <= 1e-5)}
    print("LPIPS gate", out["gates"]["lpips_dataset_mean"], flush=True)
    if not out["gates"]["lpips_dataset_mean"]["passed"]:
        raise SystemExit("stored LPIPS rows of the dataset average are not reproduced")
    new[new.fill != "dataset_mean"].to_csv(os.path.join(MF.PRIVATE, "panel_lpips_rows.csv.gz"), index=False)
    for src in desc:
        base = ref[(ref.src == src) & ref.lpips.notna()]
        for m in a.models:
            fill = f"model_{m}"
            own = new[(new.src == src) & (new.fill == fill)]
            sub = base.merge(own[["target"]].drop_duplicates(), on="target")
            r = pd.concat([sub, own], ignore_index=True)
            r.loc[r.fill == "shrink_repeat", "fill"] = "shrink_ref"
            saved = RW.RUNGS
            RW.RUNGS = saved + [fill, "mirror", "stranger"]
            try:
                fr = RW.frame(r, ["target", "patch"], "family", ["lpips"], np.random.default_rng(RW.SEED))
            finally:
                RW.RUNGS = saved
            e = out["datasets"][SRC_NAME[src]]["models"].setdefault(m, {})
            e["lpips_alex"] = {"R_self_model": fr["R_self"][fill]["lpips"], "n_targets": int(own.target.nunique()),
                               "R_self_references_same_targets": {k: v["lpips"] for k, v in fr["R_self"].items()
                                                                  if k != fill and "lpips" in v}}
            print(src, m, "LPIPS R", fr["R_self"][fill]["lpips"], flush=True)

    # ---- FID and FWD, set level, on the stored target cells
    z = np.load(os.path.join(PN.PRIVATE, f"set_patches{VERSION}.npz"))
    P, keys = z["patches"].astype(np.float64), [k.split("|") for k in z["keys"]]
    stored = json.load(open(os.path.join(PN.PRIVATE, f"set_level{VERSION}.json")))
    idx = {}
    for i, (s, pk, n, ti) in enumerate(keys):
        idx.setdefault((s, n), {})[(int(ti), int(pk))] = i
    for src in desc:
        tgt = idx[(src, "target")]
        cells = sorted(tgt)
        packs = {"target": P[[tgt[c] for c in cells]]}
        st = idx[(src, "stranger")]
        sc = sorted(set(st) & set(tgt))
        packs["stranger_t"], packs["stranger"] = P[[tgt[c] for c in sc]], P[[st[c] for c in sc]]
        mc = {}
        for m, (fills, row, _) in models.items():
            mc[m] = [c for c in cells if c[0] in row[src] and np.isfinite(fills[src][row[src][c[0]], c[1]]).all()]
            packs[f"model_{m}_t"] = P[[tgt[c] for c in mc[m]]]
            packs[f"model_{m}"] = np.stack([fills[src][row[src][ti], pk] for ti, pk in mc[m]]).astype(np.float16) \
                .astype(np.float64)                                   # float16 as the stored patches
        order = list(packs)
        feats = PN.inception_features(np.concatenate([packs[k] for k in order]))
        cut = np.cumsum([0] + [len(packs[k]) for k in order])
        F = {k: feats[cut[i]:cut[i + 1]] for i, k in enumerate(order)}

        def fid(a_, b_):
            return PN.frechet(F[a_].mean(0), np.cov(F[a_], rowvar=False), F[b_].mean(0), np.cov(F[b_], rowvar=False))
        s_fid = fid("stranger_t", "stranger")
        s_fwd = PN.fwd(PN.to_unit(packs["stranger_t"]), PN.to_unit(packs["stranger"]))
        ok = bool(np.isclose(s_fid, stored[src]["stranger"]["fid"], rtol=1e-6)
                  and np.isclose(s_fwd, stored[src]["stranger"]["fwd"], rtol=1e-6)
                  and len(sc) == stored[src]["stranger"]["n_cells"])
        out["gates"][f"set_level_stranger_{src}"] = {"fid": s_fid, "fid_stored": stored[src]["stranger"]["fid"],
                                                     "fwd": s_fwd, "fwd_stored": stored[src]["stranger"]["fwd"],
                                                     "n_cells": len(sc), "passed": ok}
        print("set gate", src, out["gates"][f"set_level_stranger_{src}"], flush=True)
        if not ok:
            raise SystemExit(f"{src}: stored FID / FWD of the random-person set are not reproduced")
        ds = out["datasets"][SRC_NAME[src]]
        ds["set_level_references_stored"] = {
            k: {"fid": v["fid"], "fwd": v["fwd"], "n_cells": v["n_cells"],
                "fid_over_random_person": v["fid"] / stored[src]["stranger"]["fid"],
                "fwd_over_random_person": v["fwd"] / stored[src]["stranger"]["fwd"]}
            for k, v in stored[src].items() if "fid" in v and (k in REFS or k == "_target_split_half")}
        for m in models:
            f_, w_ = fid(f"model_{m}_t", f"model_{m}"), PN.fwd(PN.to_unit(packs[f"model_{m}_t"]),
                                                              PN.to_unit(packs[f"model_{m}"]))
            ds["models"].setdefault(m, {})["set_level"] = {
                "n_cells": len(mc[m]), "fid": f_, "fwd": w_, "fid_over_random_person": f_ / s_fid,
                "fwd_over_random_person": w_ / s_fwd,
                "note": None if len(mc[m]) == len(cells) else "subset of the target cells (model run on a subset)"}
            print(src, m, ds["models"][m]["set_level"], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(PN.BM.finite(out), fh, indent=1)
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
