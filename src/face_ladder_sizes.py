"""Face ladder at window sizes in object units (docs/PROTOCOL_face_metrics.md, Amendment F3).

L = sqrt(mean face-mask area in the 128 px frame) over the targets and pool rows of a source; window side
w = round_half_up(f x L) for f in {1/16, 1/8, 1/4}; ring = ceil(w / 2). The tier-1 rows of `face_ladder_metrics`
are rebuilt at each size with the window and ring passed explicitly (no module default). Rows are PRIVATE.

Stages:
  geometry                      L, f of the 16 px window, sizes and positions per source (no metric)
  rows --denom 8 [--limit N]    tier-1 rows at f = 1 / denom, resumable per (source, position) job
  qc --denom 8 --limit 4        fill mosaics at that size for the smoke targets (PRIVATE PNG)
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import face_ladder_metrics as FM  # noqa: E402

PRIVATE = os.path.join(os.path.dirname(FM.PRIVATE), "face-ladder-sizes-20261108-v3")
DENOMS = (16, 8, 4)                                    # f = 1/16, 1/8, 1/4 (Amendment F3)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def extent(cache, d):
    """L = sqrt(mean face-mask area) over the source's target and pool rows; also the area inside `valid`."""
    M = np.load(os.path.join(cache, "M.npy"), mmap_mode="r")
    valid = np.load(os.path.join(cache, f"{d['_src']}_valid.npy"))
    rows = np.unique(np.r_[[t["t"] for t in d["targets"]], np.arange(*d["pool"])]).astype(int)
    a = np.asarray(M[rows]).reshape(len(rows), -1).sum(1)
    av = (np.asarray(M[rows]) & valid).reshape(len(rows), -1).sum(1)
    return {"L_px": float(np.sqrt(a.mean())), "L_in_valid_px": float(np.sqrt(av.mean())), "n_rows": int(len(rows))}


def size_of(L, denom):
    """Window side (round half up) and ring (ceil of half the side) for f = 1 / denom."""
    w = int(math.floor(L / denom + 0.5))
    return w, int(math.ceil(w / 2))


def geometry(cache, desc):
    out = {}
    for src, d in desc.items():
        d = dict(d, _src=src)
        g = extent(cache, d)
        valid = np.load(os.path.join(cache, f"{src}_valid.npy"))
        g["f_of_16px"] = 16 / g["L_px"]
        g["n_positions_16px"] = len(FM.positions(valid, FM.WIN, FM.RING))
        g["sizes"] = {}
        for k in DENOMS:
            w, rw = size_of(g["L_px"], k)
            g["sizes"][f"1/{k}"] = {"window_px": w, "ring_px": rw, "n_positions": len(FM.positions(valid, w, rw))}
        out[src] = g
    return out


def run_rows(version, denom, limit, workers):
    if workers > 3:
        raise SystemExit("at most 3 workers (shared CPU)")
    cache = os.path.join(FM.PRIVATE, f"cache{version}")
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    geo = geometry(cache, desc)
    lam = FM.lambdas(version)
    sizes = {src: size_of(geo[src]["L_px"], denom) for src in desc}
    if len(set(sizes.values())) != 1:
        raise SystemExit(f"sources disagree on the window at f = 1/{denom}: {sizes}")
    w, rw = next(iter(sizes.values()))
    tag = f"{version}_w{w}" + (f"_n{limit}" if limit else "")
    jobs, meta = [], {"version": version, "f": f"1/{denom}", "window_px": w, "ring_px": rw, "geometry": geo,
                      "sources": {}, "sha256": {"face_ladder_metrics": sha(FM.__file__), "face_ladder_sizes": sha(__file__)}}
    for src, d in desc.items():
        valid = np.load(os.path.join(cache, f"{src}_valid.npy"))
        pos = FM.positions(valid, w, rw)
        dd = dict(d, valid=valid)
        sig = FM.repeat_sigma(dd, pos, cache, w)
        T = d["targets"]
        sel = np.unique(np.linspace(0, len(T) - 1, limit).round().astype(int)) if limit else np.arange(len(T))
        tl = [(int(ti), T[ti]) for ti in sel]
        n_sub = min(FM.POOL_SUB, d["n_pool"])
        for pk, yx in enumerate(pos):
            jobs.append((src, pk, tuple(yx), tl, lam[src], sig[pk], tuple(d["pool"]), n_sub, d["shift_px"], w, rw))
        meta["sources"][src] = {"n_targets_scored": len(tl), "n_positions": len(pos), "positions": [list(p) for p in pos]}
    print(json.dumps({s: {k: v for k, v in m.items() if k != "positions"} for s, m in meta["sources"].items()}
                     | {"window_px": w, "ring_px": rw}), flush=True)
    # resumable parts: pickles written by this script into the private folder only (never untrusted input)
    parts = os.path.join(PRIVATE, f"parts{tag}")
    os.makedirs(parts, exist_ok=True)
    part = lambda k: os.path.join(parts, f"job{k:04d}.pkl")  # noqa: E731
    todo = [k for k in range(len(jobs)) if not os.path.exists(part(k))]
    print(f"jobs done {len(jobs) - len(todo)}/{len(jobs)}, to run {len(todo)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=FM._init, initargs=(cache,)) as ex:
        for n, (k, res) in enumerate(zip(todo, ex.map(FM.score_position, [jobs[k] for k in todo], chunksize=4))):
            with open(part(k) + ".tmp", "wb") as fh:
                pickle.dump(res, fh)
            os.replace(part(k) + ".tmp", part(k))
            if (n + 1) % max(1, len(todo) // 40) == 0 or n + 1 == len(todo):
                el = time.time() - t0
                eta = el / (n + 1) * (len(todo) - n - 1)
                print(f"job {n + 1}/{len(todo)}: {el / 60:.0f} min, ETA {eta / 60:.0f} min "
                      f"({time.strftime('%H:%M', time.localtime(time.time() + eta))})", flush=True)
    rows = []
    for k in range(len(jobs)):
        with open(part(k), "rb") as fh:
            rows += pickle.load(fh)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(PRIVATE, f"rows{tag}.csv.gz"), index=False)
    meta["minutes"] = round((time.time() - t0) / 60, 1)
    meta["n_rows"] = int(len(r))
    meta["summary"] = {src: FM.summarise(r[r.src == src]) for src in desc if (r.src == src).any()}
    with open(os.path.join(PRIVATE, f"meta{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(FM.BM.finite(meta), fh, indent=1, allow_nan=False)
    print("rows", len(r), "written", meta["minutes"], "min", flush=True)


def qc(version, denom, limit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import face_ladder_metrics_qc as Q
    cache = os.path.join(FM.PRIVATE, f"cache{version}")
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    FM._init(cache)
    lam = FM.lambdas(version)
    geo = geometry(cache, desc)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#c9d6e3")
    for src, d in desc.items():
        w, rw = size_of(geo[src]["L_px"], denom)
        valid = np.asarray(FM.WORK[f"{src}_valid"])
        pos = FM.positions(valid, w, rw)
        T = d["targets"]
        sel = np.unique(np.linspace(0, len(T) - 1, limit).round().astype(int))
        r = pd.read_csv(os.path.join(PRIVATE, f"rows{version}_w{w}_n{limit}.csv.gz"))
        r = r[r.src == src]
        fig, axes = plt.subplots(len(sel), len(Q.RUNGS), figsize=(1.45 * len(Q.RUNGS), 1.75 * len(sel)))
        for i, ti in enumerate(sel):
            mo = Q.mosaics(src, int(ti), T[ti], pos, lam[src], tuple(d["pool"]), min(FM.POOL_SUB, d["n_pool"]),
                           d["shift_px"], w, rw)
            for c, (k, lab) in enumerate(Q.RUNGS):
                ax = axes[i, c]
                ax.imshow(mo[k], cmap=cmap, vmin=-2.5, vmax=2.5, interpolation="nearest")
                for (y0, x0) in pos:
                    ax.add_patch(plt.Rectangle((x0 - .5, y0 - .5), w, w, fill=False, lw=0.15, ec="#e8a33d"))
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(lab, fontsize=6)
                g = r[(r.target == ti) & (r.fill == k)]
                if len(g):
                    ax.set_xlabel(f"mse {g.mse.mean():.3g}  ssim {g.ssim.mean():.3g}\n{g.patch.nunique()} windows",
                                  fontsize=4.5)
        fig.suptitle(f"PRIVATE QC {src} {version}, Amendment F3: fills at f = 1/{denom} of the face extent "
                     f"(L = {geo[src]['L_px']:.1f} px): {w} px windows, ring {rw} px, {len(pos)} positions; "
                     "blue = not scored; targets evenly spaced", fontsize=7)
        fig.tight_layout()
        out = os.path.join(PRIVATE, f"qc_fills_{src}{version}_w{w}_private.png")
        fig.savefig(out, dpi=170)
        plt.close(fig)
        print(out, flush=True)

# plain rung names, as in the paper figure (src/face_ladder_metrics_figures.py)
PLAIN = {"self": "target itself", "aug_light": "same photo, retest light", "aug_shift": "same photo, shifted",
         "retest": "another photo", "mirror": "own mirror half", "sibling": "sibling",
         "lookalike": "most similar person", "lookalike_p219": "most similar, pool 219",
         "lookalike_oracle": "best person (oracle)", "lookalike_oracle_p219": "oracle, pool 219",
         "stranger": "unrelated person", "mean_k2": "average of 2 people", "cohort_mean": "sex-age mean",
         "dataset_mean": "average of all people", "shrink_repeat": "another photo, blended to mean",
         "shrink_mirror": "mirror half, blended to mean", "wrong_detail": "wrong detail",
         "noise": "white noise, same amplitude"}
SRC_NAME = {"celeba": "CelebA", "imdb": "IMDb-WIKI"}
STAB_METRICS = ("mse", "mae", "ssim", "pearson", "lddt", "tm_like", "gdt_like")   # window-dependent tier-1 metrics
OUT_JSON = os.path.join(FM.ROOT, "results", "face_ladder_sizes.json")
OUT_FIG = os.path.join(FM.ROOT, "results", "figures", "fig_face_window_sizes")


def spearman(a, b):
    """Spearman rank correlation of two equal-length vectors (average ranks for ties); NaN if fewer than 3."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def rung_R(view, metric):
    """{rung: R} of one summarise() view and metric; rungs without a finite R are left out (never filled)."""
    lr = view.get("ladder_R", {}).get(metric, {})
    return {k: v["R"] for k, v in lr.items() if isinstance(v, dict) and v.get("R") is not None and np.isfinite(v["R"])}


def stability(version, limit):
    """Tier-1 summaries at 5, 10, 16 and 20 px on the same evenly spaced targets, and Spearman rank stability of the
    rung order (R per rung) between every pair of sizes; aggregates only (public)."""
    cache = os.path.join(FM.PRIVATE, f"cache{version}")
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    geo = geometry(cache, desc)
    w_of = {f"1/{k}": size_of(geo["celeba"]["L_px"], k)[0] for k in DENOMS}
    rows = {}
    for f, w in w_of.items():
        rows[w] = pd.read_csv(os.path.join(PRIVATE, f"rows{version}_w{w}_n{limit}.csv.gz"))
    v16 = pd.read_csv(os.path.join(FM.PRIVATE, f"rows{version}.csv.gz"))
    keep = []
    for src, d in desc.items():
        T = d["targets"]
        sel = set(np.unique(np.linspace(0, len(T) - 1, limit).round().astype(int)).tolist())
        keep.append(v16[(v16.src == src) & v16.target.isin(sel)])
    rows[16] = pd.concat(keep)
    sizes = sorted(rows)
    out = {"amendment": "F3 / F3.1 (docs/PROTOCOL_face_metrics.md)", "frame": version,
           "subset": f"{limit} evenly spaced targets per source, the same at every size", "geometry": geo,
           "window_px_by_f": w_of, "f_of_16px": {s: g["f_of_16px"] for s, g in geo.items()},
           "metrics": list(STAB_METRICS), "views": {}, "rank_stability": {}, "counts": {}}
    for src in desc:
        out["views"][src], out["rank_stability"][src], out["counts"][src] = {}, {}, {}
        for w in sizes:
            r = rows[w][rows[w].src == src]
            v = FM.summarise(r)
            out["views"][src][str(w)] = {"ladder_R": {m: v["ladder_R"].get(m) for m in STAB_METRICS},
                                         "expectations": {m: v["expectations"].get(m) for m in STAB_METRICS}}
            out["counts"][src][str(w)] = {"n_rows": int(len(r)), "n_targets": int(r.target.nunique()),
                                          "n_positions": int(r.patch.nunique())}
        for m in STAB_METRICS:
            Rs = {w: rung_R(out["views"][src][str(w)], m) for w in sizes}
            common = sorted(set.intersection(*[set(x) for x in Rs.values()]))
            pairs = {f"{a}-{b}": spearman([Rs[a][k] for k in common], [Rs[b][k] for k in common])
                     for i, a in enumerate(sizes) for b in sizes[i + 1:]}
            obj = [w for w in sizes if w != 16]
            out["rank_stability"][src][m] = {"n_rungs": len(common), "spearman": pairs,
                                             "min_over_object_unit_sizes": min(pairs[f"{a}-{b}"] for i, a in
                                                                               enumerate(obj) for b in obj[i + 1:])}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(FM.BM.finite(out), fh, indent=1, allow_nan=False)
    figure(out, sizes)
    print(OUT_JSON, flush=True)


def figure(out, sizes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ("mse", "ssim", "lddt")
    srcs = list(out["views"])
    fig, axes = plt.subplots(len(srcs), len(show), figsize=(4.2 * len(show), 3.4 * len(srcs)), squeeze=False)
    cols = {5: "#9ecae1", 10: "#4292c6", 16: "#000000", 20: "#08306b"}
    for i, src in enumerate(srcs):
        for j, m in enumerate(show):
            ax = axes[i, j]
            ref = rung_R(out["views"][src]["16"], m)
            order = sorted(ref, key=ref.get)
            for w in sizes:
                Rw = rung_R(out["views"][src][str(w)], m)
                y = [Rw.get(k, np.nan) for k in order]
                ax.plot(range(len(order)), y, "o-" if w != 16 else "s--", ms=3, lw=1, c=cols.get(w, "grey"),
                        label=f"{w} px (f = {w / out['geometry'][src]['L_px']:.3f})")
            ax.axhline(0, c="grey", lw=0.5)
            ax.axhline(1, c="grey", lw=0.5, ls=":")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([PLAIN.get(k, k) for k in order], rotation=70, ha="right", fontsize=5.5)
            rs = out["rank_stability"][src][m]
            ax.set_title(f"{SRC_NAME.get(src, src)}: {m.upper() if m == 'mse' else m}, rung order Spearman min {rs['min_over_object_unit_sizes']:.2f} "
                         f"(5, 10, 20 px)", fontsize=7)
            ax.set_ylabel("R (0 = target itself, 1 = stranger mean)", fontsize=6)
            if i == 0 and j == 0:
                ax.legend(fontsize=5.5)
    fig.suptitle("Face ladder at window sizes in units of the face extent L (Amendment F3); rungs ordered by R at "
                 "16 px; same evenly spaced targets at every size", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG + ".png", dpi=170)
    fig.savefig(OUT_FIG + ".pdf")
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["geometry", "rows", "qc", "stability"])
    ap.add_argument("--version", required=True, help="face frame version, e.g. _v6f (no default)")
    ap.add_argument("--denom", type=int, choices=DENOMS, help="f = 1 / denom (required for rows and qc)")
    ap.add_argument("--limit", type=int, default=0, help="evenly spaced targets per source (smoke)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.stage not in ("geometry", "stability") and a.denom is None:
        raise SystemExit("--denom is required")
    if a.stage == "geometry":
        cache = os.path.join(FM.PRIVATE, f"cache{a.version}")
        print(json.dumps(geometry(cache, json.load(open(os.path.join(cache, "desc.json")))["data"]), indent=1))
    elif a.stage == "stability":
        if not a.limit:
            raise SystemExit("--limit is required for stability (the sweep's limit)")
        stability(a.version, a.limit)
    elif a.stage == "rows":
        run_rows(a.version, a.denom, a.limit, a.workers)
    else:
        if not a.limit:
            raise SystemExit("--limit is required for qc (the smoke run's limit)")
        qc(a.version, a.denom, a.limit)


if __name__ == "__main__":
    main()
