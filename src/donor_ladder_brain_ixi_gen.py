"""IXI LPIPS (AlexNet, VGG), intensity persistence and set-level FID / FWD (PROTOCOL_donor_ladder_brain.md,
Amendment 25). Pre-QC, affine registration only.

Cells are the topology subset S of Amendment 24C (2 whole random cubes per target, both sides). Fills, donors, E and
seeds come from `donor_ladder_brain_metrics.run_position_at` unchanged with the hook `ixi_gen`
(`donor_ladder_brain_metrics_t23.extra_ixi_gen`). R uses the IXI anchor (other side 0, 16 reference strangers 1).
Set level follows Amendment 22 with whole 16 x 16 central slices as samples. Rows and slices stay private; the
public JSON holds aggregates only. Missing is NaN, never zero.

Usage: python -I src/donor_ladder_brain_ixi_gen.py --workers N [--smoke 3]      score (resumable per position)
       python -I src/donor_ladder_brain_ixi_gen.py --workers N --set-level      FID / FWD (needs every position)
       python -I src/donor_ladder_brain_ixi_gen.py --workers 1 --summary-only   public JSON
       python -I src/donor_ladder_brain_ixi_gen.py --workers 1 --smoke 3 --qc   QC sheet of the smoke slices
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_betti25d as BT  # noqa: E402
import donor_ladder_brain_metrics as M  # noqa: E402

HOOK = "ixi_gen"
METRICS = ("lpips", "lpips_vgg", "persistence_w1")
DESCRIPTIVE = ("lpips_slices", "persistence_essential")
PRIVATE = os.path.join(B.PROJ, "ixi-gen-metrics-20261114-v1")
OUT_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_ixi_gen.json")
VIEW = "random_patches/LR"
# The own-other-side set is the target set itself (both sides are targets); its shrunk control is a blend of it.
SET_EXCLUDED = {"mirror": "other-side set = the target set by construction (both sides are targets)",
                "shrink_mirror": "blend of the other-side set, which is the target set by construction"}


def parts_dir(smoke):
    return os.path.join(PRIVATE, f"smoke{smoke}_parts" if smoke else "parts")


def cells(smoke):
    """IXI tables, positions, affine, region paths and {position: set of targets} on the topology subset S."""
    import donor_ladder_brain_lookalike16 as L
    t, tg, region_paths, _, _ = L.dataset("ixi")
    pos, affine, _ = B.positions(np.random.default_rng(B.SEED))
    w, rand = BT.whole_cubes(tg, pos, region_paths)
    S = BT.subset(w)
    if smoke:
        S = dict(list(S.items())[:smoke])
    by_pos = {}
    for i, ps in S.items():
        for k in ps:
            by_pos.setdefault(rand[k], set()).add(i)
    return t, tg, pos, affine, region_paths, S, by_pos


def split(rows):
    """Rows of run_position_at -> (scalar rows with a crop index, crops dict)."""
    scal, xc, fc, ok = [], [], [], []
    for row in rows:
        s = {k: v for k, v in row.items() if not k.startswith("_")}
        s["crop"] = -1
        if "_xc" in row:
            s["crop"] = len(xc)
            xc.append(row["_xc"])
            fc.append(row["_fc"])
            ok.append(row["_ok"])
        scal.append(s)
    n = xc[0].shape[-1] if xc else 16
    return pd.DataFrame(scal), {"xc": np.array(xc, np.float32).reshape(-1, 3, n, n),
                                "fc": np.array(fc, np.float32).reshape(-1, 3, n, n),
                                "ok": np.array(ok, bool).reshape(-1, 3)}


def run(workers, smoke):
    import nibabel as nib
    from concurrent.futures import ProcessPoolExecutor
    import donor_ladder_brain_bilateral as BL
    import donor_ladder_brain_ixi as I
    t, tg, pos, affine, region_paths, S, by_pos = cells(smoke)
    out = parts_dir(smoke)
    os.makedirs(out, exist_ok=True)
    mc = BL.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    lam = M.rung_lambdas()
    tcols = t[["person", "session"]].to_dict("list")
    tgcols = tg[["index", "person", "family", "sex", "age"]].to_dict("list")
    todo = [pi for pi in sorted(by_pos) if not os.path.exists(os.path.join(out, f"p{pi:02d}.npz"))]
    print(json.dumps({"targets_in_S": len(S), "cells": int(sum(len(v) for v in S.values())),
                      "positions": len(by_pos), "positions_to_run": len(todo)}), flush=True)
    t0 = time.time()
    jobs = [(pi, tcols, tgcols, pos, mc, lam, HOOK, by_pos[pi],
             {"shell": BL.SHELL, "W": BL.W, "paths": region_paths(pi)}) for pi in todo]
    n_cells_done, n_cells = 0, sum(len(by_pos[pi]) for pi in todo)
    with ProcessPoolExecutor(workers, initializer=M._init, initargs=(affine,)) as ex:
        for k, (pi, rows) in enumerate(zip(todo, ex.map(M.run_position_at, jobs))):
            rows = [r for r in rows if r["fill"] not in I.NA_RUNGS]
            df, crops = split(rows)
            base = os.path.join(out, f"p{pi:02d}")
            df.to_csv(base + ".csv.gz.tmp", index=False, compression="gzip")
            np.savez_compressed(base + ".tmp.npz", **crops)
            os.replace(base + ".csv.gz.tmp", base + ".csv.gz")
            os.replace(base + ".tmp.npz", base + ".npz")             # the npz is written last: it marks done
            n_cells_done += len(by_pos[pi])
            el = time.time() - t0
            eta = el / n_cells_done * (n_cells - n_cells_done)
            print(f"position {pi} ({k + 1}/{len(todo)}), cells {n_cells_done}/{n_cells}, rows {len(df)}, "
                  f"{el / 60:.1f} min, ETA {eta / 60:.0f} min ({time.strftime('%H:%M', time.localtime(time.time() + eta))})",
                  flush=True)
    timing = {"minutes": round((time.time() - t0) / 60, 2), "workers": workers, "smoke_targets": smoke,
              "cells": n_cells}
    with open(os.path.join(PRIVATE, f"timing{'_smoke' + str(smoke) if smoke else ''}.json"), "w") as fh:
        json.dump(timing, fh, indent=1)
    print(json.dumps(timing), flush=True)


def load(smoke):
    """All rows and slices; refuses when a position of S is missing."""
    _, tg, pos, _, _, S, by_pos = cells(smoke)
    out = parts_dir(smoke)
    rows, xs, fs, oks, off = [], [], [], [], 0
    for pi in sorted(by_pos):
        base = os.path.join(out, f"p{pi:02d}")
        if not os.path.exists(base + ".npz"):
            raise SystemExit(f"position {pi} missing: run the scoring first")
        r = pd.read_csv(base + ".csv.gz")
        with np.load(base + ".npz") as z:
            r.loc[r.crop >= 0, "crop"] += off
            off += len(z["ok"])
            xs.append(z["xc"])
            fs.append(z["fc"])
            oks.append(z["ok"])
        rows.append(r)
    r = pd.concat(rows, ignore_index=True)
    if r.duplicated(["target", "patch", "side", "fill"]).any():
        raise SystemExit("duplicate rows: nothing written")
    return r, (np.concatenate(xs), np.concatenate(fs), np.concatenate(oks)), tg, pos, S


def set_cells(r, ok):
    """Set-level cells: (row index, crop, axis, target, patch, side, fill) for every rung row whose slice lies wholly
    inside E; reference strangers are not rungs."""
    rr = r[(r.crop >= 0) & ~r.fill.map(M.is_reference_stranger)]
    return [(i, int(row.crop), ax, int(row.target), int(row.patch), row.side, row.fill)
            for i, row in zip(rr.index, rr.itertuples()) for ax in range(3) if ok[int(row.crop), ax]]


def target_samples(cl):
    """One target sample per (target, patch, side, axis): {key: (crop, axis)}, first occurrence."""
    tk = {}
    for _, cr, ax, tgt, p, side, _ in cl:
        tk.setdefault((tgt, p, side, ax), (cr, ax))
    return tk


def set_level(smoke, threads):
    import hashlib
    import donor_ladder_brain_metrics_t23_v5 as V5
    r, (xc, fc, ok), tg, _, _ = load(smoke)
    cl = set_cells(r, ok)
    tk = target_samples(cl)
    tlist = sorted(tk)
    timg = np.stack([xc[tk[k][0], tk[k][1]] for k in tlist])
    fimg = np.stack([fc[cr, ax] for _, cr, ax, *_ in cl])
    U = V5.to_unit(V5.upsample(np.concatenate([timg, fimg]), V5.SET_PX))
    h = hashlib.sha256(np.ascontiguousarray(U, dtype=np.float32).tobytes()).hexdigest()
    cache = os.path.join(PRIVATE, f"set_features{'_smoke' + str(smoke) if smoke else ''}.npz")
    F = None
    if os.path.exists(cache):
        with np.load(cache) as z:
            if str(z["sha256"]) == h:
                F = z["F"]
    t0 = time.time()
    if F is None:
        import torchvision_guard
        torchvision_guard.ensure()                     # Application Control may block torchvision/_C.pyd
        print(f"inception on {len(U)} samples, {threads} threads", flush=True)
        parts, step = [], 1000                         # eval-mode Inception: chunking does not change features
        for s0 in range(0, len(U), step):
            parts.append(V5.inception_features(U[s0:s0 + step], threads))
            done, el = min(s0 + step, len(U)), time.time() - t0
            print(f"inception {done}/{len(U)}, {el / 60:.1f} min, ETA {el / done * (len(U) - done) / 60:.0f} min",
                  flush=True)
        F = np.concatenate(parts)
        np.savez(cache + ".part.npz", F=F, sha256=np.array(h))
        os.replace(cache + ".part.npz", cache)
    P = V5.haar_packets(U, V5.HAAR_LEVEL)
    print(f"{len(U)} samples, inception {time.time() - t0:.0f}s", flush=True)
    tix = {k: n for n, k in enumerate(tlist)}
    nt = len(tlist)
    view = {}
    for fill in sorted({c[6] for c in cl}):
        sel = [(n, c) for n, c in enumerate(cl) if c[6] == fill]
        ia = np.array([tix[(c[3], c[4], c[5], c[2])] for _, c in sel])
        ib = np.array([nt + n for n, _ in sel])
        fam = np.array([c[3] for _, c in sel])                     # IXI: every target is its own family
        if len(np.unique(fam)) < 10:
            view[fill] = {"n_cells": len(sel), "status": "fewer than 10 targets: not computed"}
            continue
        view[fill] = V5.set_entry(F[ia], F[ib], P[ia], P[ib], fam, [M.SEED, 25, len(view)])
        if fill in SET_EXCLUDED:
            view[fill]["excluded_from_ranks"] = SET_EXCLUDED[fill]
        print(fill, {k: round(v, 4) if isinstance(v, float) else v for k, v in view[fill].items()
                     if not k.endswith("ci95")}, flush=True)
    tt = np.array([k[0] for k in tlist])
    u = np.random.default_rng([M.SEED, 25, 3000]).permutation(np.unique(tt))
    ha = np.isin(tt, u[:len(u) // 2])
    a, b = np.flatnonzero(ha), np.flatnonzero(~ha)
    view["_target_split_half"] = {"n_cells": [int(len(a)), int(len(b))], "fid": V5.frechet(F[a], F[b]),
                                  "fwd": V5.fwd(P[a], P[b]), "note": "two target-disjoint halves of the target set"}
    out = {VIEW: view}
    with open(os.path.join(PRIVATE, f"set_level{'_smoke' + str(smoke) if smoke else ''}.json"), "w") as fh:
        json.dump(M.finite(out), fh, indent=1, allow_nan=False)
    print("set level written", flush=True)
    return out


def self_R(per, m):
    """R on the self scale used by the other datasets' generative block: target itself 0, 16 strangers 1. LPIPS and
    the persistence W1 of an image with itself are exactly 0 (identical features, identical diagrams), so the anchor
    is that exact value, not a filler; it is passed to the unchanged ladder_R in the anchor slot ('repeat')."""
    if m not in METRICS:
        raise ValueError(f"self anchor defined only for {METRICS}")
    import donor_ladder_brain_ixi as I
    p = per[~per.fill.map(I.is_centre_draw)]
    anchor = p[["target", "family"]].drop_duplicates().assign(fill="repeat", **{m: 0.0})
    R, _ = M.ladder_R(pd.concat([p[["target", "family", "fill", m]], anchor], ignore_index=True), m)
    if R is None:
        return None
    R.pop("repeat", None)
    a = R.pop("_anchors")
    R["_anchors"] = {"d_self": 0.0, "d_stranger": a["d_stranger"]}
    return R


def summarise(r, pos):
    import donor_ladder_brain_ixi as I
    out = {}
    for view, rr in M.views(r, pos).groupby("view"):
        per = rr.groupby(["target", "fill"])[list(METRICS) + list(DESCRIPTIVE)].mean().reset_index()
        per["family"] = per.target
        fills = sorted(f for f in per.fill.unique() if not M.is_reference_stranger(f))
        v = {"n_rows": int(len(rr)), "missing": {m: int(rr[m].isna().sum()) for m in METRICS},
             "fills": {f: {m: M.boot_mean(per[per.fill == f], m) for m in METRICS + DESCRIPTIVE} for f in fills},
             "ladder_R": {}}
        v["ladder_R_self"] = {}
        for m in METRICS:
            R = I.other_side_R(per, m)
            if R is not None:
                v["ladder_R"][m] = R
            Rs = self_R(per, m)
            if Rs is not None:
                v["ladder_R_self"][m] = Rs
        if view.endswith("/LR"):
            t = M.tests(I.as_qtim(per), list(METRICS))
            for e in t["per_metric"].values():
                e.pop("T1", None)
            v["tests"] = I.holm_ixi(t)
        out[view] = v
    return out


def summary(smoke):
    from scipy.stats import spearmanr
    r, _, tg, pos, S = load(smoke)
    views = summarise(r, pos)
    sp = os.path.join(PRIVATE, f"set_level{'_smoke' + str(smoke) if smoke else ''}.json")
    sl = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else "not run"
    R = views[VIEW]["ladder_R"]
    shared = sorted(k for k in R.get("lpips", {}) if not k.startswith("_") and k in R.get("lpips_vgg", {}))
    e1 = (float(spearmanr([R["lpips"][k]["R"] for k in shared], [R["lpips_vgg"][k]["R"] for k in shared])[0])
          if len(shared) >= 3 else None)
    out = {"status": "pre-QC: affine registration only, MRI registration gate not met (results/registration/status.md)",
           "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendment 25)", "dataset": "IXI", "R_anchor": "other_side",
           "seed": M.SEED, "cells": "topology subset S of Amendment 24C: 2 whole random 16 mm cubes per target, both sides",
           "metrics": list(METRICS), "lower_is_better": [m for m in METRICS if m in M.LOWER_BETTER],
           "descriptive": list(DESCRIPTIVE),
           "lpips": {"nets": ["alex", "vgg"], "version": "0.1", "short_side_px": 64, "z_scale": 4.0,
                     "slices": "central orthogonal, wholly inside E (Amendments 20, 22)"},
           "persistence": "cubical sublevel (gudhi), outside E at +inf, W1 (internal p = 2) summed over H0-H2",
           "set_level_method": {"cell": "(target, cube, side, slice axis)", "sample": "whole 16 x 16 central slice, "
                                "wholly inside E, z/4 clipped to [0, 1], 64 px", "fid": "Inception-v3 pool 2048 "
                                "(pytorch-fid), exact Frechet", "fwd": "Haar packets level 2, Frechet per packet, mean",
                                "interval": "100 target bootstrap resamples, 2.5-97.5 percentiles",
                                "floor": "_target_split_half", "excluded_from_ranks": SET_EXCLUDED},
           "missing_rule": "missing is NaN, never zero; counts per view under 'missing'",
           "not_available": {f: "n/a in IXI (no repeat, no families); never filled"
                             for f in ("repeat", "sibling", "cotwin", "cohort_mean", "shrink_repeat")},
           "tests_not_available": {"T1": "needs a repeat"},
           "expectation_E1_spearman_lpips_alex_vgg": {"rho": e1, "n_rungs": len(shared), "rule": ">= 0.8"},
           "counts": {"n_targets": int(r.target.nunique()), "n_cells": int(sum(len(v) for v in S.values())),
                      "n_rows": int(len(r))},
           "views": views, "set_level": sl}
    path = OUT_JSON if not smoke else os.path.join(PRIVATE, f"summary_smoke{smoke}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(M.finite(out), fh, indent=1, allow_nan=False)
    print(f"saved {path}", flush=True)


def qc_sheet(smoke):
    """Smoke QC: for the first cell of each smoke target, the three central slices of the target and of the fills,
    and the 64 px set-level samples of the first axis. Saved in the private folder, never in git."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import donor_ladder_brain_metrics_t23_v5 as V5
    r, (xc, fc, ok), _, _, _ = load(smoke)
    show = ["mirror", "stranger", "lookalike", "lookalike_oracle", "mean_k2", "dataset_mean", "shrink_mirror",
            "wrong_detail", "noise"]
    keys = r[r.crop >= 0].drop_duplicates(["target"])[["target", "patch", "side"]].itertuples(index=False)
    keys = list(keys)
    fig, axes = plt.subplots(4 * len(keys), 1 + len(show), figsize=(1.25 * (1 + len(show)), 5.2 * len(keys)))
    for b, (tgt, p, side) in enumerate(keys):
        cell = r[(r.target == tgt) & (r.patch == p) & (r.side == side) & (r.crop >= 0)].set_index("fill")
        cr0 = int(cell.crop.iloc[0])
        cols = [("target", xc[cr0], ok[cr0])] + [(f, fc[int(cell.loc[f, "crop"])], ok[int(cell.loc[f, "crop"])])
                                                   for f in show if f in cell.index]
        for j in range(1 + len(show)):
            for ax_i in range(4):
                axes[4 * b + ax_i, j].axis("off")
        for j, (name, sl, okk) in enumerate(cols):
            lab = name if name == "target" else (f"{name}\nLP {cell.loc[name, 'lpips']:.2f} "
                                                 f"V {cell.loc[name, 'lpips_vgg']:.2f}\nW1 {cell.loc[name, 'persistence_w1']:.1f}")
            axes[4 * b, j].set_title(lab, fontsize=5.5)
            for a in range(3):
                axes[4 * b + a, j].imshow(sl[a], cmap="gray", vmin=-3, vmax=3)
                if not okk[a]:
                    axes[4 * b + a, j].set_title("not in E", fontsize=5, color="red")
            u = V5.to_unit(V5.upsample(sl[:1], V5.SET_PX))[0]
            axes[4 * b + 3, j].imshow(u, cmap="gray", vmin=0, vmax=1)
        axes[4 * b, 0].text(-20, 8, f"target {tgt}\npos {p} {side}", fontsize=6, ha="right")
    fig.suptitle("IXI Amendment 25 smoke: 3 central slices (z, clipped +-3) and the 64 px FID/FWD sample (row 4)",
                 fontsize=8)
    fig.tight_layout()
    dest = os.path.join(PRIVATE, f"qc_smoke{smoke}.png")
    fig.savefig(dest, dpi=160)
    print(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--set-level", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--qc", action="store_true")
    a = ap.parse_args()
    os.makedirs(PRIVATE, exist_ok=True)
    if a.qc:
        qc_sheet(a.smoke)
    elif a.set_level:
        set_level(a.smoke, a.workers)
    elif a.summary_only:
        summary(a.smoke)
    else:
        run(a.workers, a.smoke)


if __name__ == "__main__":
    main()
