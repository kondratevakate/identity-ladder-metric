"""Amendment F2.2: the rung "mean of 16 look-alikes" (`lookalike16`) on faces, v6f, CelebA and IMDb-WIKI.

The 16 donors nearest in the context ring (the F1 look-alike search, ranked instead of arg-min) are averaged with the
masked mean. Three passes reuse the existing scorers with the fill function swapped: tier-1 (`FM.score_position`),
the F2 panel (`P.score_position`, AlexNet LPIPS) and LPIPS VGG (`P.score_vgg`). Each pass recomputes the k = 1
look-alike next to the new fill; it must equal the stored rows (gates in docs/PROTOCOL_face_metrics.md, F2.2).

Stages: t1 | panel | vgg (rows, resumable parts) | qc (private sheet) | summary (sibling JSON block).
Rows stay in the private folder; only per-rung aggregates are written to results/.
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
import donor_ladder_brain_metrics as BM  # noqa: E402
import face_ladder_metrics as FM  # noqa: E402
import face_ladder_metrics_panel as P  # noqa: E402

K = 16
NEW = "lookalike16"
OUT = os.path.join(os.path.dirname(FM.PRIVATE), "face-ladder-lookalike16-20261109-v1")
KEYS = ["src", "target", "patch", "fill"]
TOL = 1e-9
TOL_NET = 1e-5                     # LPIPS: the pairs are batched differently than in the stored runs (float32 last bits)


def ranked(A, MA, b, mb, eligible, k):
    """Indices (into A) of the k rows nearest to b, by the F1 masked distance among eligible rows, ties by row order;
    fewer if fewer qualify. Its first element is `FM.nearest` (the k = 1 look-alike)."""
    d = BM.masked_dist(A, MA, b, mb, int(mb.sum()))
    d = np.where(eligible, d, np.inf)
    fin = np.flatnonzero(np.isfinite(d))
    return fin[np.argsort(d[fin], kind="stable")][:k]


def fills16(src, ti, d, pk, sl, ring, lam, P_, n_sub, ident=None):
    """Drop-in for `FM.window_fills` that returns only the k = 1 look-alike and the mean of the 16 nearest."""
    X, M, E = FM.WORK["X"], FM.WORK["M"], FM.WORK["emb"]
    p0 = P_["pool"][0]
    t = d["t"]
    xt, mt = np.asarray(X[t], np.float32), np.asarray(M[t])
    valid = np.asarray(FM.WORK[f"{src}_valid"])
    tv = mt[sl] & valid[sl]
    out = {}
    nn = ranked(P_["ring_x"], P_["ring_m"], xt[ring], mt[ring], P_["win_ok"], K)
    j1 = FM.nearest(P_["ring_x"], P_["ring_m"], xt[ring], mt[ring], P_["win_ok"])
    if (j1 is None) != (nn.size == 0) or (j1 is not None and int(nn[0]) != j1):
        raise RuntimeError(f"rank-1 of the 16 is not the k = 1 look-alike ({src} target {ti} position {pk})")
    if nn.size:
        g = p0 + int(nn[0])
        idv = FM.cos(np.asarray(E[t]), np.asarray(E[g])) if ident is None else ident(t, g)
        out["lookalike"] = (np.asarray(X[g][sl], np.float32), np.asarray(M[g][sl]), idv)
    if nn.size >= K:
        Xs = np.stack([np.asarray(X[p0 + int(j)][sl], np.float32) for j in nn])
        Ms = np.stack([np.asarray(M[p0 + int(j)][sl]) for j in nn])
        m, ok = BM.masked_mean(Xs, Ms)
        out[NEW] = (m, ok, np.nan)
    return xt[sl], tv, out


# ---------------------------------------------------------------------------------------------------- workers

def _swap():
    FM.window_fills = fills16
    P.REPRO_EXEMPT = frozenset({NEW})


_BLAS = []


def _init_t1(cache_dir):
    from threadpoolctl import threadpool_limits
    _BLAS.append(threadpool_limits(1))
    FM._init(cache_dir)
    _swap()


def _init_panel(state):
    P._init(state)
    _swap()


def _init_vgg(state):
    P._init_vgg(state)
    _swap()


def state_of(version):
    return {"cache": os.path.join(P.V1, f"cache{version}"), "fundus": P.need_env("FUNDUS_METRICS_SRC"),
            "betti": P.need_env("BETTI_SRC"), "torch_home": P.need_env("TORCH_HOME"),
            "prep": os.path.join(P.PRIVATE, f"prep{version}")}


def t1_jobs(version, limit):
    cache = os.path.join(FM.PRIVATE, f"cache{version}")
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    lam = FM.lambdas(version)
    jobs, names = [], []
    for src, d in desc.items():
        d = dict(d, valid=np.load(os.path.join(cache, f"{src}_valid.npy")))
        pos = FM.positions(d["valid"])
        sig = FM.repeat_sigma(d, pos, cache, FM.WIN)
        T = d["targets"]
        sel = np.unique(np.linspace(0, len(T) - 1, limit).round().astype(int)) if limit else np.arange(len(T))
        n_sub = min(FM.POOL_SUB, d["n_pool"])
        for pk, yx in enumerate(pos):
            for c, ch in enumerate(P.target_chunks(sel, P.CHUNK)):
                jobs.append((src, pk, tuple(yx), [(int(ti), T[ti]) for ti in ch], lam[src], sig[pk],
                             tuple(d["pool"]), n_sub, d["shift_px"], FM.WIN, FM.RING))
                names.append(f"{src}_p{pk:02d}_c{c:02d}")
    return jobs, names, cache


def run_pass(stage, version, limit, workers):
    P.ram_guard(workers)
    tag = version + (f"_n{limit}" if limit else "")
    if stage == "t1":
        jobs, names, cache = t1_jobs(version, limit)
        init, arg, fn = _init_t1, cache, FM.score_position
    else:
        st = state_of(version)
        jobs, names, _ = P.build_jobs(version, limit, st)
        init, arg, fn = (_init_panel, st, P.score_position) if stage == "panel" else (_init_vgg, st, P.score_vgg)
    parts = os.path.join(OUT, f"parts_{stage}{tag}")
    os.makedirs(parts, exist_ok=True)
    # pickles written by this function into the private folder only (never untrusted input)
    part = lambda k: os.path.join(parts, f"{names[k]}.pkl")  # noqa: E731
    todo = [k for k in range(len(jobs)) if not os.path.exists(part(k))]
    print(f"{stage}: jobs done {len(jobs) - len(todo)}/{len(jobs)}, to run {len(todo)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=init, initargs=(arg,)) as ex:
        for n, (k, res) in enumerate(zip(todo, ex.map(fn, [jobs[k] for k in todo]))):
            with open(part(k) + ".tmp", "wb") as fh:
                pickle.dump(res, fh)
            os.replace(part(k) + ".tmp", part(k))
            el = time.time() - t0
            eta = el / (n + 1) * (len(todo) - n - 1)
            print(f"{stage} job {n + 1}/{len(todo)} ({names[k]}), {el / 60:.0f} min, ETA {eta / 60:.0f} min "
                  f"({time.strftime('%H:%M', time.localtime(time.time() + eta))})", flush=True)
    rows, worst = [], 0.0
    for k in range(len(jobs)):
        with open(part(k), "rb") as fh:
            res = pickle.load(fh)
        rows += res if stage == "t1" else res[0]
        worst = max(worst, 0.0 if stage == "t1" else (res[2] if stage == "panel" else res[1]))
    r = pd.DataFrame(rows)
    targets = pd.DataFrame([(j[0], int(ti)) for j in jobs for ti, _ in j[3]],
                           columns=["src", "target"]).drop_duplicates()
    gate = gate_pass(stage, r, worst, targets)
    print(stage, json.dumps(gate), flush=True)
    if not gate["passed"]:
        raise SystemExit(f"{stage}: the recomputed k = 1 look-alike does not equal the stored rows; nothing written")
    r[r.fill == NEW].to_csv(os.path.join(OUT, f"rows_{stage}{tag}.csv.gz"), index=False, float_format="%.17g")
    json.dump(gate, open(os.path.join(OUT, f"gate_{stage}{tag}.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------------------- gates

def reference(stage):
    if stage == "t1":
        return pd.read_csv(os.path.join(FM.PRIVATE, "rows_v6f.csv.gz"))
    if stage == "panel":
        return pd.read_csv(os.path.join(P.PRIVATE, "rows_v6f.csv.gz"))
    return pd.read_csv(os.path.join(P.VGG_PRIVATE, "rows_vgg_v6f.csv.gz"))


def compare_rows(new, ref, metrics, targets):
    """Recomputed k = 1 look-alike rows against the stored ones of the scored targets (`targets`: src, target pairs
    of the jobs, so a target lost entirely is caught): every row on both sides, equal NaN pattern, max |diff|."""
    a = new[new.fill == "lookalike"]
    rl = ref[ref.fill == "lookalike"].merge(targets, on=["src", "target"])
    m = a.merge(rl, on=KEYS, how="left", suffixes=("", "_ref"), indicator=True)
    out = {"n_rows": int(len(a)), "n_stored": int(len(rl)), "n_matched": int((m["_merge"] == "both").sum()),
           "max_abs_diff": {}}
    ok = out["n_rows"] > 0 and out["n_matched"] == out["n_rows"] == out["n_stored"]
    for c in metrics:
        if c not in a or c + "_ref" not in m:
            continue
        x, y = m[c].to_numpy(float), m[c + "_ref"].to_numpy(float)
        same_nan = bool((np.isnan(x) == np.isnan(y)).all())
        diff = float(np.nanmax(np.abs(x - y))) if np.isfinite(x).any() else 0.0
        out["max_abs_diff"][c] = diff if same_nan else "missing pattern differs"
        ok = ok and same_nan and diff <= (TOL_NET if c.startswith("lpips") else TOL)
    out["passed"] = bool(ok)
    return out


def gate_pass(stage, r, worst, targets):
    ref = reference(stage)
    if stage == "t1":
        metrics = list(FM.REPORTED)
    elif stage == "panel":
        metrics = ["mse"] + [c for c in P.NEW_ROW]
    else:
        metrics = ["mse", "lpips_vgg"]
    g = compare_rows(r, ref, metrics, targets)
    g["v1_mse_residual"] = float(worst)
    g["n_new_rows"] = int((r.fill == NEW).sum())
    g["passed"] = bool(g["passed"] and worst <= 1e-5 and g["n_new_rows"] > 0)
    return g


# ---------------------------------------------------------------------------------------------------- QC

def qc(version, limit):
    """Private sheet: target, k = 1 look-alike and mean of 16 painted over all windows, evenly spaced targets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache = os.path.join(FM.PRIVATE, f"cache{version}")
    FM._init(cache)
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    lam = FM.lambdas(version)
    for src, d in desc.items():
        valid = np.load(os.path.join(cache, f"{src}_valid.npy"))
        pos = FM.positions(valid)
        T = d["targets"]
        sel = np.unique(np.linspace(0, len(T) - 1, limit).round().astype(int))
        fig, ax = plt.subplots(len(sel), 3, figsize=(6, 2.1 * len(sel)), squeeze=False)
        for i, ti in enumerate(sel):
            canv = {k: np.full((FM.SIZE, FM.SIZE), np.nan) for k in ("target", "lookalike", NEW)}
            for pk, (y0, x0) in enumerate(pos):
                sl = (slice(y0, y0 + FM.WIN), slice(x0, x0 + FM.WIN))
                ring = FM.ring_mask(y0, x0) & valid
                P_ = FM.pool_window(src, {"pool": d["pool"]}, sl, ring)
                if (np.asarray(FM.WORK["M"][T[ti]["t"]])[sl] & valid[sl]).mean() < FM.MIN_IN_MASK:
                    continue
                xw, tv, f = fills16(src, int(ti), T[ti], pk, sl, ring, lam[src], P_, 0)
                canv["target"][sl] = np.where(tv, xw, canv["target"][sl])
                for k in ("lookalike", NEW):
                    if k in f:
                        canv[k][sl] = np.where(tv & f[k][1], f[k][0], canv[k][sl])
            for j, (k, lab) in enumerate((("target", "target"), ("lookalike", "look-alike k=1"),
                                          (NEW, "mean of 16 look-alikes"))):
                ax[i, j].imshow(canv[k], cmap="gray", vmin=-3, vmax=3)
                ax[i, j].set_axis_off()
                if i == 0:
                    ax[i, j].set_title(lab, fontsize=8)
        fig.suptitle(f"PRIVATE QC {src} {version}, Amendment F2.2: fills over all {len(pos)} windows; "
                     "targets evenly spaced", fontsize=7)
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, f"qc_{src}{version}_private.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(path)


# ---------------------------------------------------------------------------------------------------- summary

def extract(view, metrics):
    return {"ladder_R": {m: view["ladder_R"][m][NEW] for m in metrics
                         if NEW in view.get("ladder_R", {}).get(m, {})},
            "fills": {m: v for m, v in view.get("fills", {}).get(NEW, {}).items() if m in metrics}}


def summary(version, limit):
    tag = version + (f"_n{limit}" if limit else "")
    new = {s: pd.read_csv(os.path.join(OUT, f"rows_{s}{tag}.csv.gz"), float_precision="round_trip")
           for s in ("t1", "panel", "vgg")}
    gates = {s: json.load(open(os.path.join(OUT, f"gate_{s}{tag}.json"))) for s in new}
    views = {}
    pub = json.load(open(FM.OUT_JSON, encoding="utf-8"))
    sanity = {}
    # tier-1 with the default self rule
    ref = reference("t1")
    for src in ("celeba", "imdb"):
        rs = pd.concat([ref[ref.src == src], new["t1"][new["t1"].src == src]], ignore_index=True)
        if limit:
            rs = rs[rs.target.isin(new["t1"][new["t1"].src == src].target.unique())]
        v = FM.summarise(rs)
        views[src] = extract(v, list(FM.REPORTED))
        if not limit:                                  # anchors unchanged: the k = 1 look-alike R equals the stored R
            a = v["ladder_R"]["mse"]["lookalike"]["R"]
            b = pub["views"][src]["ladder_R"]["mse"]["lookalike"]["R"]
            sanity[src] = {"mse_lookalike_R": a, "stored": b, "equal": bool(abs(a - b) <= 1e-12)}
    # F2 panel and VGG with the panel self rule
    BM.LOWER_BETTER.update(P.LOWER | {"lpips_vgg"})
    P.SELF_ZERO.add("lpips_vgg")
    FM.with_self = P.with_self_panel
    for stage, metrics in (("panel", [m for m in P.NEW_ROW]), ("vgg", ["lpips_vgg"])):
        ref = reference(stage)
        for src in ("celeba", "imdb"):
            rs = pd.concat([ref[ref.src == src], new[stage][new[stage].src == src]], ignore_index=True)
            if limit:
                rs = rs[rs.target.isin(new[stage][new[stage].src == src].target.unique())]
            ms = [m for m in metrics if m in new[stage] and new[stage][m].notna().any()]
            e = extract(FM.summarise(rs, ms), ms)
            views[src]["ladder_R"].update(e["ladder_R"])
            views[src]["fills"].update(e["fills"])
    block = {"amendment": "F2.2 (docs/PROTOCOL_face_metrics.md)", "rung": NEW, "k": K,
             "fill": "pixel-wise masked mean of the 16 pool donors nearest in the context ring (F1 search)",
             "rows": os.path.basename(OUT), "gates": gates, "anchor_check": sanity,
             "subset": f"{limit} evenly spaced targets per source" if limit else "all targets",
             "views": views}
    if limit:
        path = os.path.join(OUT, f"summary{tag}.json")
        json.dump(BM.finite({"rung_lookalike16": block}), open(path, "w"), indent=1, allow_nan=False)
    else:
        if pub.get("frame") != version:
            raise SystemExit(f"{FM.OUT_JSON} holds frame {pub.get('frame')}, not {version}")
        if not all(s["equal"] for s in sanity.values()):
            raise SystemExit("k = 1 look-alike R differs from the stored R; anchors changed, nothing written")
        pub["rung_lookalike16"] = block              # sibling block; every other block unchanged
        path = FM.OUT_JSON
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(BM.finite(pub), fh, indent=1, allow_nan=False)
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("t1", "panel", "vgg", "qc", "summary"))
    ap.add_argument("--version", required=True, help="face frame version, e.g. _v6f (no default)")
    ap.add_argument("--limit", type=int, default=0, help="evenly spaced targets per source (smoke)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    FM.snapshot(a.version)
    if a.stage in ("t1", "panel", "vgg"):
        run_pass(a.stage, a.version, a.limit, a.workers)
    elif a.stage == "qc":
        qc(a.version, a.limit or 4)
    else:
        summary(a.version, a.limit)


if __name__ == "__main__":
    main()
