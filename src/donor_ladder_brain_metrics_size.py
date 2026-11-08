"""Tier-1 metric panel at window sizes in object units (PROTOCOL_donor_ladder_brain.md, Amendment 21). Pre-QC.

L = cbrt(mean brain-mask volume) = 127.0 mm in the analysis space; cube edge w = f x L for f in {1/16, 1/8, 1/4} ->
8, 16, 32 mm. 16 mm is the reference (tier-1 run of Amendments 10 to 13, not redone here). This script runs the same
panel (donor_ladder_brain_metrics.run_position_at) at 8 or 32 mm on the 32 random cubes, with shell s = w/4 and the
+-8 mm block search passed explicitly. Centres keep the 16 mm draw; a centre whose window (w + 2s) is < 90% inside
the template brain mask is dropped and counted, never replaced. Region memmaps are built one position at a time (at
most workers + 1 on disk) from the whole-volume cache of Amendment 4 and deleted once the position is saved.
Rows stay private; the JSON holds aggregates only.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as BL  # noqa: E402
import donor_ladder_brain_metrics as M  # noqa: E402
import donor_ladder_brain_size as S  # noqa: E402

L_MM = 127.0                     # cbrt(mean brain-mask volume), 244 prepared scans (Amendment 21)
SIZES = {8: "1/16", 16: "1/8", 32: "1/4"}
SHELL_RATIO = 4                  # s = w / 4: the current 4 mm shell of the 16 mm cube
HALF_WINDOW = 8                  # +-8 mm block search: registration tolerance, fixed in mm (Amendment 21)
TEMPLATE_SHARE = 0.9             # rule of donor_ladder_brain.positions
PRIVATE = os.path.join(B.PROJ, "donor-ladder-brain-metrics-size-20261108-v1")
OUT_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_metrics_size.json")
REF_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_metrics.json")


def geometry(w):
    """Shell and half-window of cube edge w (mm = voxels); only the sizes of Amendment 21 are allowed."""
    if w not in SIZES:
        raise ValueError(f"cube edge {w} mm is not one of the Amendment 21 sizes {sorted(SIZES)}")
    return {"cube": w, "shell": w // SHELL_RATIO, "W": HALF_WINDOW, "f": SIZES[w], "f_exact": w / L_MM}


def sized_positions(pos, template_mask, w, shell):
    """Random cubes of edge w around the 16 mm centres, kept if >= 90% of the window is in the template mask.

    Returns (positions list with the same indices as pos, kept indices, dropped indices). Boxes are left as they are
    and never run at these sizes."""
    win = w + 2 * shell
    frac = ndimage.uniform_filter(template_mask.astype(np.float32), size=win, mode="constant")
    half = win // 2
    out, kept, dropped = [], [], []
    for pi, (region, start, size) in enumerate(pos):
        if region != "random_patches":
            out.append((region, start, size))
            continue
        c = [a + n // 2 for a, n in zip(start, size)]
        inside = all(half <= ci < n - half for ci, n in zip(c, template_mask.shape))
        ok = inside and frac[tuple(c)] >= TEMPLATE_SHARE
        out.append((region, tuple(ci - w // 2 for ci in c), (w,) * 3))
        (kept if ok else dropped).append(pi)
    return out, kept, dropped


def build_region(V, Mk, start, size, shell, W, a, pi):
    """Region memmaps (scans, frame, size + 2 (shell + W)) for one position: frame 0 the scan, 1 its mirror."""
    pad = shell + W
    lo = tuple(s0 - pad for s0 in start)
    rs = tuple(n + 2 * pad for n in size)
    pv, pm = (os.path.join(PRIVATE, "regions", f"{k}{pi:02d}.npy") for k in ("vol", "mask"))
    vol = np.lib.format.open_memmap(pv + ".tmp.npy", "w+", np.float32, (len(V), 2) + rs)
    mask = np.lib.format.open_memmap(pm + ".tmp.npy", "w+", np.uint8, (len(V), 2) + rs)
    for k in range(len(V)):
        for fr, mirror in ((0, None), (1, a)):
            vol[k, fr] = S.get_box(V[k], lo, rs, mirror)
            mask[k, fr] = S.get_box(Mk[k], lo, rs, mirror)
    vol.flush()
    mask.flush()
    del vol, mask
    os.replace(pv + ".tmp.npy", pv)
    os.replace(pm + ".tmp.npy", pm)
    return pv, pm


def rung_order(view, metric):
    """Fill -> family-bootstrap mean of one metric in a summarised view (reference strangers already excluded)."""
    out = {}
    for f, d in view["fills"].items():
        raw = (d.get(metric) or {}).get("raw")
        if raw is not None and raw.get("mean") is not None and np.isfinite(raw["mean"]):
            out[f] = raw["mean"]
    return out


def rank_stability(views_by_size, metrics):
    """Spearman correlation of the rung order between each pair of sizes, per metric, on fills present at all."""
    from scipy.stats import spearmanr
    sizes = sorted(views_by_size)
    res = {}
    for m in metrics:
        orders = {w: rung_order(views_by_size[w], m) for w in sizes}
        common = sorted(set.intersection(*(set(o) for o in orders.values())))
        res[m] = {"n_fills": len(common), "fills": common}
        for i, w1 in enumerate(sizes):
            for w2 in sizes[i + 1:]:
                if len(common) >= 3:
                    rho = spearmanr([orders[w1][f] for f in common], [orders[w2][f] for f in common]).statistic
                    res[m][f"{w1}_vs_{w2}"] = float(rho)
                else:
                    res[m][f"{w1}_vs_{w2}"] = None
    return res


def main():
    import nibabel as nib
    from nilearn.datasets import load_mni152_brain_mask
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=int, required=True, help="cube edge in mm: 8 or 32 (16 is the reference run)")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0, help="score only the first N targets that have a repeat scan")
    ap.add_argument("--positions", default="", help="comma-separated position indices (QC/smoke); default all kept")
    ap.add_argument("--stability", action="store_true", help="only recompute rank stability from the saved JSON")
    a = ap.parse_args()
    g = geometry(a.cube)
    if a.cube == 16 and not a.stability:
        raise SystemExit("16 mm is the reference run (donor_ladder_brain_metrics.py); not redone here")
    out = json.load(open(OUT_JSON, encoding="utf-8")) if os.path.exists(OUT_JSON) else {
        "status": "pre-QC: affine registration only, MRI registration gate not met (results/registration/status.md)",
        "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendment 21)", "tier": 1, "seed": M.SEED,
        "L_mm": L_MM, "L_definition": "cbrt(mean brain-mask volume), 244 prepared scans, MNI 2009a sym 1 mm (affine)",
        "reference_16mm": "results/donor_ladder_brain_metrics.json (f = 16 / 127.0 = 0.126)", "sizes": {}}
    if a.stability:
        ref = json.load(open(REF_JSON, encoding="utf-8"))["views"]["random_patches/LR"]
        views = {16: ref, **{int(k): v["views"]["random_patches/LR"] for k, v in out["sizes"].items()
                             if "views" in v}}
        out["rank_stability"] = {"view": "random_patches/LR", "order_by": "family-bootstrap mean of the raw metric",
                                 "sizes_mm": sorted(views), "metrics": rank_stability(views, M.METRICS)}
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(M.finite(out), fh, indent=1, allow_nan=False)
        print(json.dumps({m: {k: v for k, v in d.items() if "_vs_" in k}
                          for m, d in out["rank_stability"]["metrics"].items()}, indent=1))
        return
    t0 = time.time()
    t = B.scans()
    pos16, affine, _ = B.positions(np.random.default_rng(M.SEED))
    tmask = load_mni152_brain_mask(resolution=1).get_fdata() > 0
    pos, kept, dropped = sized_positions(pos16, tmask, g["cube"], g["shell"])
    sel = [int(p) for p in a.positions.split(",")] if a.positions else kept
    if set(sel) - set(kept):
        raise SystemExit(f"positions {sorted(set(sel) - set(kept))} are dropped at {a.cube} mm")
    tg = t[t.session == "ses-01"].reset_index()
    only = None
    if a.smoke:
        rep = set(t[t.session == "ses-02"].person)
        only = set(np.flatnonzero(tg.person.isin(rep).to_numpy())[:a.smoke].tolist())
    tag = f"w{a.cube}_" + (f"smoke{a.smoke}_" if a.smoke else "")
    parts = os.path.join(PRIVATE, "parts")
    for d in (parts, os.path.join(PRIVATE, "regions")):
        os.makedirs(d, exist_ok=True)
    part_path = lambda pi: os.path.join(parts, f"{tag}p{pi:02d}.csv.gz")  # noqa: E731
    todo = [pi for pi in sel if not os.path.exists(part_path(pi))]
    print(f"cube {a.cube} mm (f {g['f']}, shell {g['shell']}, search +-{g['W']}): kept {len(kept)}/32 positions, "
          f"dropped {dropped}; done {len(sel) - len(todo)}/{len(sel)}, to run {len(todo)}", flush=True)
    V, Mk = S.cache_volumes(t, affine)
    mc = BL.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    lam = M.rung_lambdas()
    tcols = t[["person", "session"]].to_dict("list")
    tgcols = tg[["index", "person", "family", "sex", "age"]].to_dict("list")
    with ProcessPoolExecutor(a.workers, initializer=M._init, initargs=(affine,)) as ex:
        running, queue, n_done = {}, list(todo), 0
        while queue or running:
            while queue and len(running) < a.workers:      # at most `workers` regions in use, one being built
                pi = queue.pop(0)
                paths = build_region(V, Mk, pos[pi][1], pos[pi][2], g["shell"], g["W"], mc, pi)
                job = (pi, tcols, tgcols, pos, mc, lam, False, only, {"shell": g["shell"], "W": g["W"], "paths": paths})
                running[ex.submit(M.run_position_at, job)] = (pi, paths)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                pi, paths = running.pop(fut)
                part = fut.result()                      # a worker error stops the run here, loudly
                pd.DataFrame(part).to_csv(part_path(pi) + ".tmp", index=False, compression="gzip")
                os.replace(part_path(pi) + ".tmp", part_path(pi))
                for p in paths:
                    os.remove(p)
                n_done += 1
                el = time.time() - t0
                eta = el / n_done * (len(todo) - n_done)
                print(f"position {pi} ({n_done}/{len(todo)}): rows {len(part)}, {el / 60:.0f} min, ETA {eta / 60:.0f} "
                      f"min ({time.strftime('%H:%M', time.localtime(time.time() + eta))})", flush=True)
    r = pd.concat([pd.read_csv(part_path(pi)) for pi in sel], ignore_index=True)
    old = os.path.join(S.PRIVATE, f"rows_p{a.cube}_s{g['shell']}.csv.gz")
    ok, rep = M.reproduce(r, t, old)
    print(json.dumps({"reproduces_size_grid": ok, "against": os.path.basename(old)}), flush=True)
    timing = {"minutes": round((time.time() - t0) / 60, 1), "workers": a.workers, "positions": len(sel),
              "smoke_targets": a.smoke}
    with open(os.path.join(PRIVATE, f"{tag}timing.json"), "w", encoding="utf-8") as fh:
        json.dump(timing, fh, indent=1)
    if a.smoke or a.positions:
        print(json.dumps({"timing": timing, "reproduction": rep}, indent=1))
        return
    if not ok:
        print(json.dumps(rep, indent=1))
        raise SystemExit("mse_allvox does not reproduce the Amendment 4 rows; no summary written")
    r.to_csv(os.path.join(PRIVATE, f"{tag}rows.csv.gz"), index=False)
    cdf = M.calibrated_c(r)
    r = M.calibrate_rows(r)
    out["sizes"][str(a.cube)] = {**g, "positions_kept": len(kept), "positions_dropped": dropped,
                                 "lambda_shrink": lam, "lambda_note": "shrink lambdas from the 16 mm reference",
                                 "reproduction": rep, "minutes": timing["minutes"],
                                 "views": M.summarise(r, cdf, tg, pos)}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(M.finite(out), fh, indent=1, allow_nan=False)
    print(f"saved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
