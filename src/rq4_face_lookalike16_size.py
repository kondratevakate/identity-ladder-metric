"""RQ4 size curve and f = 1/8 summary input, faces v6f (PROTOCOL_donor_ladder_general.md, Amendment 10; geometry of
face Amendment F3; fill of F2.2).

Stages (window from the F3 geometry, w = round_half_up(L / denom), ring = ceil(w / 2); no module default):
  sub  --denom 16|8|4 --limit 100   k = 1 look-alike and mean of 16 look-alikes on the F3 subset (the sweep's targets);
                                    gate: the recomputed k = 1 rows equal the stored F3 rows of that size.
  full --denom 8                    every tier-1 rung of F1 plus the mean of 16 look-alikes, all targets, 10 px; gate:
                                    every stored F3 subset row of that size is reproduced (all rungs, all metrics).
  qc   --limit 4                    private sheet: target, k = 1, mean of 16 and oracle at 5, 10, 16, 20 px.
Rows stay in the private folder; resumable per (source, position) job.
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import face_ladder_lookalike16 as FL16  # noqa: E402
import face_ladder_metrics as FM  # noqa: E402
import face_ladder_sizes as FS  # noqa: E402

PRIVATE = os.path.join(os.path.dirname(FM.PRIVATE), "rq4-window-size-20261114-v1", "faces")
VERSION = "_v6f"                       # the frame of F2.2 and F3; passed to every loader explicitly
KEYS = ["src", "target", "patch", "fill"]
TOL = 1e-9
_ORIG = {}


def fills_all(src, ti, d, pk, sl, ring, lam, P_, n_sub, ident=None):
    """Every F1 fill (`FM.window_fills` unchanged) plus `lookalike16` from `FL16.fills16` (which also checks that the
    rank-1 of the 16 is the k = 1 look-alike)."""
    xw, tv, out = _ORIG["window_fills"](src, ti, d, pk, sl, ring, lam, P_, n_sub, ident)
    _, _, o16 = FL16.fills16(src, ti, d, pk, sl, ring, lam, P_, n_sub, ident)
    if FL16.NEW in o16:
        out[FL16.NEW] = o16[FL16.NEW]
    return xw, tv, out


def _init(cache_dir, mode):
    from threadpoolctl import threadpool_limits
    _ORIG["blas"] = threadpool_limits(1)
    FM._init(cache_dir)
    _ORIG["window_fills"] = FM.window_fills
    FM.window_fills = FL16.fills16 if mode == "sub" else fills_all


def window_of(geo, denom):
    sizes = {src: FS.size_of(g["L_px"], denom) for src, g in geo.items()}
    if len(set(sizes.values())) != 1:
        raise SystemExit(f"sources disagree on the window at f = 1/{denom}: {sizes}")
    return next(iter(sizes.values()))


def jobs_of(cache, desc, w, rw, limit):
    lam = FM.lambdas(VERSION)
    jobs, names, targets = [], [], []
    for src, d in desc.items():
        valid = np.load(os.path.join(cache, f"{src}_valid.npy"))
        pos = FM.positions(valid, w, rw)
        sig = FM.repeat_sigma(dict(d, valid=valid), pos, cache, w)
        T = d["targets"]
        sel = np.unique(np.linspace(0, len(T) - 1, limit).round().astype(int)) if limit else np.arange(len(T))
        tl = [(int(ti), T[ti]) for ti in sel]
        targets += [(src, int(ti)) for ti in sel]
        n_sub = min(FM.POOL_SUB, d["n_pool"])
        for pk, yx in enumerate(pos):
            jobs.append((src, pk, tuple(yx), tl, lam[src], sig[pk], tuple(d["pool"]), n_sub, d["shift_px"], w, rw))
            names.append(f"{src}_p{pk:03d}")
    return jobs, names, pd.DataFrame(targets, columns=["src", "target"])


def compare(new, ref, fills, metrics):
    """Stored rows of `fills` (restricted to the targets in `new`) against the recomputed ones: every stored row
    present, equal missing pattern, max |diff| <= TOL on every metric."""
    a = new[new.fill.isin(fills)]
    b = ref[ref.fill.isin(fills)].merge(a[["src", "target"]].drop_duplicates(), on=["src", "target"])
    m = b.merge(a, on=KEYS, how="left", suffixes=("_ref", ""), indicator=True)
    out = {"fills": sorted(fills), "n_stored": int(len(b)), "n_new": int(len(a)),
           "n_matched": int((m["_merge"] == "both").sum()), "max_abs_diff": {}}
    ok = out["n_stored"] > 0 and out["n_matched"] == out["n_stored"]
    for c in metrics:
        if c not in a.columns or c + "_ref" not in m.columns:
            continue
        x, y = m[c].to_numpy(float), m[c + "_ref"].to_numpy(float)
        same = bool((np.isnan(x) == np.isnan(y)).all())
        diff = float(np.nanmax(np.abs(x - y))) if np.isfinite(x - y).any() else 0.0
        out["max_abs_diff"][c] = diff if same else "missing pattern differs"
        ok = ok and same and diff <= TOL
    out["passed"] = bool(ok)
    return out


def run(mode, denom, limit, workers, only_src=None):
    if workers > 3:
        raise SystemExit("at most 3 workers (shared CPU)")
    cache = os.path.join(FM.PRIVATE, f"cache{VERSION}")
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    if only_src:
        desc = {k: v for k, v in desc.items() if k in only_src}
    geo = FS.geometry(cache, desc)
    w, rw = window_of(geo, denom)
    tag = f"{mode}_w{w}" + (f"_n{limit}" if limit else "") + ("_" + "-".join(sorted(desc)) if only_src else "")
    jobs, names, targets = jobs_of(cache, desc, w, rw, limit)
    parts = os.path.join(PRIVATE, f"parts_{tag}")
    os.makedirs(parts, exist_ok=True)
    # pickles written by this function into the private folder only (never untrusted input)
    part = lambda k: os.path.join(parts, f"{names[k]}.pkl")  # noqa: E731
    todo = [k for k in range(len(jobs)) if not os.path.exists(part(k))]
    print(f"{mode} f = 1/{denom}: window {w} px, ring {rw} px, jobs {len(jobs)} (done {len(jobs) - len(todo)}), "
          f"targets {len(targets)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(cache, mode)) as ex:
        for n, (k, res) in enumerate(zip(todo, ex.map(FM.score_position, [jobs[k] for k in todo]))):
            with open(part(k) + ".tmp", "wb") as fh:
                pickle.dump(res, fh)
            os.replace(part(k) + ".tmp", part(k))
            if (n + 1) % max(1, len(todo) // 50) == 0 or n + 1 == len(todo):
                el = time.time() - t0
                eta = el / (n + 1) * (len(todo) - n - 1)
                print(f"job {n + 1}/{len(todo)} ({names[k]}): {el / 60:.1f} min, ETA {eta / 60:.0f} min "
                      f"({time.strftime('%H:%M', time.localtime(time.time() + eta))})", flush=True)
    rows = []
    for k in range(len(jobs)):
        with open(part(k), "rb") as fh:
            rows += pickle.load(fh)
    r = pd.DataFrame(rows)
    ref = pd.read_csv(os.path.join(FS.PRIVATE, f"rows{VERSION}_w{w}_n{limit or 100}.csv.gz"))   # n4: F3 smoke
    metrics = [c for c in FM.REPORTED if c in r.columns]
    if mode == "sub":
        gate = compare(r, ref, ["lookalike"], metrics)
    else:
        gate = compare(r, ref, sorted(set(ref.fill)), metrics)
    n16 = r[r.fill == FL16.NEW]
    gate.update({"mode": mode, "f": f"1/{denom}", "window_px": w, "ring_px": rw, "n_targets": int(len(targets)),
                 "n_lookalike16_rows": int(len(n16)), "minutes": round((time.time() - t0) / 60, 1),
                 "workers": workers, "geometry": {s: {"L_px": g["L_px"], "f_exact": w / g["L_px"]}
                                                  for s, g in geo.items()}})
    json.dump(gate, open(os.path.join(PRIVATE, f"gate_{tag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in gate.items() if k != "geometry"}), flush=True)
    if not gate["passed"]:
        raise SystemExit("the recomputed rows do not equal the stored F3 rows; nothing written")
    keep = r if mode == "full" else r[r.fill == FL16.NEW]
    keep.to_csv(os.path.join(PRIVATE, f"rows_{tag}.csv.gz"), index=False, float_format="%.17g")


def qc(limit):
    """Private sheet: per evenly spaced target, the face painted with each fill over all windows, at every size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache = os.path.join(FM.PRIVATE, f"cache{VERSION}")
    FM._init(cache)
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    geo = FS.geometry(cache, desc)
    lam = FM.lambdas(VERSION)
    sizes = [(16, *window_of(geo, 16)), (8, *window_of(geo, 8)), (0, 16, 8), (4, *window_of(geo, 4))]
    cols = (("lookalike", "look-alike k=1"), (FL16.NEW, "mean of 16"), ("lookalike_oracle", "oracle"))
    os.makedirs(PRIVATE, exist_ok=True)
    for src, d in desc.items():
        valid = np.load(os.path.join(cache, f"{src}_valid.npy"))
        T = d["targets"]
        sel = np.unique(np.linspace(0, len(T) - 1, limit).round().astype(int))
        fig, ax = plt.subplots(len(sel), 1 + 3 * len(sizes), figsize=(1.3 * (1 + 3 * len(sizes)), 1.5 * len(sel)),
                               squeeze=False)
        for i, ti in enumerate(sel):
            xt = np.asarray(FM.WORK["X"][T[ti]["t"]], np.float32)
            ax[i, 0].imshow(np.where(valid, xt, np.nan), cmap="gray", vmin=-3, vmax=3)
            for c, (denom, w, rw) in enumerate(sizes):
                pos = FM.positions(valid, w, rw)
                canv = {k: np.full((FM.SIZE, FM.SIZE), np.nan) for k, _ in cols}
                for pk, (y0, x0) in enumerate(pos):
                    sl = (slice(y0, y0 + w), slice(x0, x0 + w))
                    ring = FM.ring_mask(y0, x0, FM.SIZE, w, rw) & valid
                    P_ = FM.pool_window(src, {"pool": d["pool"]}, sl, ring)
                    if (np.asarray(FM.WORK["M"][T[ti]["t"]])[sl] & valid[sl]).mean() < FM.MIN_IN_MASK:
                        continue
                    _, tv, f = FL16.fills16(src, int(ti), T[ti], pk, sl, ring, lam[src], P_, 0)
                    j = FM.nearest(P_["win_x"], P_["win_m"], xt[sl].ravel(), tv.ravel(), P_["win_ok"])
                    if j is not None:
                        g = d["pool"][0] + j
                        f["lookalike_oracle"] = (np.asarray(FM.WORK["X"][g][sl], np.float32),
                                                 np.asarray(FM.WORK["M"][g][sl]), np.nan)
                    for k, _ in cols:
                        if k in f:
                            canv[k][sl] = np.where(tv & f[k][1], f[k][0], canv[k][sl])
                for q, (k, lab) in enumerate(cols):
                    a_ = ax[i, 1 + 3 * c + q]
                    a_.imshow(canv[k], cmap="gray", vmin=-3, vmax=3)
                    if i == 0:
                        fr = f"1/{denom}" if denom else "0.2 (ref.)"
                        a_.set_title(f"{lab}\n{w} px, f = {fr}", fontsize=5.5)
        for a_ in ax.ravel():
            a_.set_axis_off()
        ax[0, 0].set_title("target", fontsize=6)
        fig.suptitle(f"PRIVATE QC {src}{VERSION}, Amendment 10: fills painted over all windows at each size; targets "
                     f"evenly spaced; blank = not scored", fontsize=7)
        fig.tight_layout()
        path = os.path.join(PRIVATE, f"qc_{src}{VERSION}_sizes_private.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("sub", "full", "qc"))
    ap.add_argument("--denom", type=int, choices=FS.DENOMS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--src", nargs="*", help="restrict to these sources (smoke only)")
    a = ap.parse_args()
    if a.stage == "qc":
        return qc(a.limit or 4)
    if a.denom is None:
        raise SystemExit("--denom is required")
    if a.limit not in ((100, 4) if a.stage == "sub" else (0, 4)):
        raise SystemExit("sub runs on the F3 subset (--limit 100) and full on all targets; --limit 4 = smoke at 10 px")
    if a.stage == "full" and a.denom != 8:
        raise SystemExit("the full cohort runs only at f = 1/8")
    run(a.stage, a.denom, a.limit, a.workers, a.src)


if __name__ == "__main__":
    main()
