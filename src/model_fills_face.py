"""Pretrained inpainting models on the face identity ladder at f = 1/8 (PROTOCOL_donor_ladder_general.md,
Amendment 10; frame v6f, windows and rings of rq4_face_lookalike16_size.py).

The model sees the whole 128 px frame of the target with ONE window hidden (the same window and target set the
references use) and returns that window. The fill is scored with the unchanged `FM.score_position` path as rung
`model_<name>`, next to every stored rung, so R_self uses the same 16 strangers and cells.

Grey level: the frame holds z-scored grey (0 outside the face mask). It is mapped to [0, 1] by a fixed affine
u = (clip(z, -Z, Z) + Z) / (2 Z), Z = model_fills_infer.ZCLIP, given to the model as three equal channels, and mapped back with the
inverse affine on the channel mean. The model runs at MODEL_SIZE px (bicubic up, area down); the mask is nearest.

Stages (rows and images stay in the private folder; file names and identifiers never leave it):
  export [--limit N]                        job file for `model_fills_infer.py`: the aligned z-scored grey frames of
                                            the targets and the window corners, nothing else (no names, no pool)
  infer --model M --ckpt PATH [--limit N]   local TorchScript run of the job (smoke); GPU runs call
                                            `model_fills_infer.py` on the VM and the fills are copied back
  score --model M [--limit N]               limit 4: rows of every stored rung plus model_<M>; limit 0: model_<M> and
                                            the gate rungs only. Gate: the stored rq4 rows of the same targets are
                                            reproduced exactly before anything is kept
  qc    --model M [--limit N]               private sheet: target, model fill, second image, 16 nearest, average
  summ  --model M [--limit N]               aggregate R_self per rung (no IDs), printed and written as JSON
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
import model_fills_infer as MI  # noqa: E402
import rq4_face_lookalike16_size as R4  # noqa: E402

RUN = "model-fills-20261116-v1"
JOBS = os.path.join(os.path.dirname(FM.PRIVATE), RUN, "faces")        # job files (inputs) and the v1 fills, kept
# One folder per input version (see model_fills_infer.py): v1 (leak) stays in JOBS, v2 (outline) in faces_v2,
# the current version v3 in faces_v3. Older versions are kept and never scored again.
PRIVATE = os.path.join(os.path.dirname(FM.PRIVATE), RUN, "faces_" + MI.INPUT_VERSION)
VERSION = R4.VERSION                   # "_v6f", explicit
DENOM = 8                              # f = 1/8 only (Amendment 10)
MODEL_SIZE = 256                       # local smoke only; GPU runs pass --model-size explicitly
GATE_FILLS = ("dataset_mean", "repeat")   # recomputed next to the model rows in a full run
_STATE = {}


WINDOWS = {5: 16, 10: 8, 16: None, 20: 4}   # window px -> f = 1 / denom (16 px = the F1 / F2 panel window, f = 0.2)


def setup(limit, win=None):
    """`win` None = the ladder window at f = 1/8 (10 px). Other windows are the sizes of the RQ4 curve: 5 and 20 px
    (f = 1/16 and 1/4, F3 subset of 100 targets) and 16 px (the panel window). Ring = ceil(window / 2), as F3."""
    cache = os.path.join(FM.PRIVATE, f"cache{VERSION}")
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    geo = R4.FS.geometry(cache, desc)
    if win is None:
        w, rw = R4.window_of(geo, DENOM)
    elif win == FM.WIN:
        w, rw = FM.WIN, FM.RING
    else:
        w, rw = R4.window_of(geo, WINDOWS[win])
        if w != win:
            raise SystemExit(f"f = 1/{WINDOWS[win]} is {w} px in this frame, not {win} px")
    jobs, names, targets = R4.jobs_of(cache, desc, w, rw, limit)
    return cache, desc, w, rw, jobs, names, targets


def tag_of(model, limit, win=None):
    return f"{model}" + (f"_w{win}" if win else "") + (f"_n{limit}" if limit else "")


def ref_rows_path(w, limit, win):
    """Stored rows of the references for this window and target set (read only)."""
    if win is None:
        return os.path.join(R4.PRIVATE, f"rows_full_w{w}" + (f"_n{limit}" if limit else "") + ".csv.gz")
    if win == FM.WIN and limit == 0:
        return os.path.join(FM.PRIVATE, f"rows{VERSION}.csv.gz")
    if limit == 100:
        return os.path.join(R4.FS.PRIVATE, f"rows{VERSION}_w{w}_n100.csv.gz")
    raise SystemExit(f"no stored reference rows for window {w} px with limit {limit}")


# ----------------------------------------------------------------------------------------------- export, infer

def job_path(limit, win=None, cells=False):
    return os.path.join(JOBS, f"faces_job{f'_w{win}' if win else ''}{'cells' if cells else ''}_n{limit}.npz")


def export(limit, win=None, cells=False):
    """Job file: per source the target frames (aligned z-scored grey, 0 outside the face mask), the window corners
    and the target indices. Nothing else leaves the cache. `panel`: the 16 px windows of the F1 / F2 panel
    (FM.WIN, FM.RING), where LPIPS, FID and FWD of the references are stored."""
    cache, desc, w, rw, jobs, names, targets = setup(limit, win)
    FM._init(cache)
    out = {"w": w, "size": FM.SIZE}
    for src, d in desc.items():
        valid = np.load(os.path.join(cache, f"{src}_valid.npy"))
        sel = sorted({ti for s_, ti in targets.itertuples(index=False) if s_ == src})
        out[f"X_{src}"] = np.stack([np.asarray(FM.WORK["X"][d["targets"][ti]["t"]], np.float16) for ti in sel])
        out[f"pos_{src}"] = np.asarray(FM.positions(valid, w, rw), np.int32)
        out[f"tidx_{src}"] = np.asarray(sel, np.int32)
        if cells:      # slow models: only the stored target cells of the panel's set-level metrics (FID, FWD)
            import face_ladder_metrics_panel as PN
            z = np.load(os.path.join(PN.PRIVATE, f"set_patches{VERSION}.npz"))
            row = {ti: a for a, ti in enumerate(sel)}
            cl = sorted((row[int(k[3])], int(k[1])) for k in (s_.split("|") for s_ in z["keys"])
                        if k[0] == src and k[2] == "target")
            out[f"cells_{src}"] = np.asarray(cl, np.int32)
            print(f"{src}: {len(cl)} panel cells", flush=True)
        print(f"{src}: {len(sel)} targets, {len(out[f'pos_{src}'])} windows of {w} px", flush=True)
    os.makedirs(JOBS, exist_ok=True)
    np.savez_compressed(job_path(limit, win, cells), **out)
    print(job_path(limit, win, cells), f"{os.path.getsize(job_path(limit, win, cells)) / 2 ** 20:.1f} MB",
          flush=True)


def infer(model, ckpt, limit, device, batch):
    if not os.path.exists(job_path(limit)):
        export(limit)
    MI.run(job_path(limit), model, "torchscript", ckpt, os.path.join(PRIVATE, tag_of(model, limit)), device, batch,
           MODEL_SIZE, ckpt_sha256=MI.PINNED_SHA256["big-lama.pt"])     # the local smoke uses the pinned export only


# ------------------------------------------------------------------------------------------------------- score

def _init(cache_dir, fill_dir, model, keep=None):
    from threadpoolctl import threadpool_limits
    _STATE["blas"] = threadpool_limits(1)
    FM._init(cache_dir)
    meta = json.load(open(os.path.join(fill_dir, "infer_meta.json")))
    _STATE["fills"] = {s: np.load(os.path.join(fill_dir, f"{s}.npy"), mmap_mode="r") for s in meta["sources"]}
    _STATE["row"] = {s: {ti: a for a, ti in enumerate(m["target_index"])} for s, m in meta["sources"].items()}
    _STATE["name"] = f"model_{model}"
    _STATE["keep"] = keep
    _STATE["orig"] = FM.window_fills
    FM.window_fills = fills_with_model if keep is None else fills_lean


def fills_with_model(src, ti, d, pk, sl, ring, lam, P_, n_sub, ident=None):
    """Every stored rung (F1 fills plus the mean of 16) and the model fill of this (target, window).
    The model fill is valid wherever it is finite; the target mask and valid frame are applied by the scorer."""
    xw, tv, out = _STATE["orig"](src, ti, d, pk, sl, ring, lam, P_, n_sub, ident)
    _, _, o16 = FL16.fills16(src, ti, d, pk, sl, ring, lam, P_, n_sub, ident)
    if FL16.NEW in o16:
        out[FL16.NEW] = o16[FL16.NEW]
    f = np.asarray(_STATE["fills"][src][_STATE["row"][src][ti], pk], np.float32)
    out[_STATE["name"]] = (f, np.isfinite(f), np.nan)
    if _STATE["keep"] is not None:                              # full run: model rows and the gate rungs only
        out = {k: v for k, v in out.items() if k == _STATE["name"] or k in _STATE["keep"]}
    return xw, tv, out


def fills_lean(src, ti, d, pk, sl, ring, lam, P_, n_sub, ident=None):
    """Full run: the model fill and the two gate rungs only, built by the statements of `FM.window_fills` (dataset
    average and second image); the gate then requires their rows to equal the stored ones on every target."""
    X, M, E = FM.WORK["X"], FM.WORK["M"], FM.WORK["emb"]
    t = d["t"]
    xt, mt = np.asarray(X[t], np.float32), np.asarray(M[t])
    tv = mt[sl] & np.asarray(FM.WORK[f"{src}_valid"])[sl]
    g = d["rep"]
    out = {"dataset_mean": (np.asarray(FM.WORK[f"{src}_mean"][sl]), np.asarray(FM.WORK[f"{src}_mean_ok"][sl]), np.nan),
           "repeat": (np.asarray(X[g][sl], np.float32), np.asarray(M[g][sl]),
                      FM.cos(np.asarray(E[t]), np.asarray(E[g])))}
    f = np.asarray(_STATE["fills"][src][_STATE["row"][src][ti], pk], np.float32)
    out[_STATE["name"]] = (f, np.isfinite(f), np.nan)
    return xt[sl], tv, out


def score(model, limit, workers, win=None):
    if workers > 3:
        raise SystemExit("at most 3 workers (shared CPU)")
    cache, desc, w, rw, jobs, names, targets = setup(limit, win)
    fill_dir = os.path.join(PRIVATE, tag_of(model, limit, win))
    meta = json.load(open(os.path.join(fill_dir, "infer_meta.json")))
    if meta["window_px"] != w:
        raise SystemExit(f"fills were made at {meta['window_px']} px, the ladder window is {w} px")
    if meta.get("input_version") != MI.INPUT_VERSION:        # no silent mixing of input versions (v1 leak, v2 outline)
        raise SystemExit(f"fills in {fill_dir} have input version {meta.get('input_version')}, "
                         f"this code scores {MI.INPUT_VERSION} only")
    have = {s_: set(m["target_index"]) for s_, m in meta["sources"].items()}
    jobs = [j[:3] + ([t_ for t_ in j[3] if t_[0] in have[j[0]]],) + j[4:] for j in jobs]   # targets the model ran on
    targets = targets[[ti in have[s_] for s_, ti in targets.itertuples(index=False)]]
    keep = None if limit == 4 else GATE_FILLS
    parts = os.path.join(fill_dir, "parts")
    os.makedirs(parts, exist_ok=True)
    part = lambda k: os.path.join(parts, f"{names[k]}.pkl")  # noqa: E731  (own pickles, private folder only)
    todo = [k for k in range(len(jobs)) if not os.path.exists(part(k))]
    print(f"score {model}: window {w} px, jobs {len(jobs)} (done {len(jobs) - len(todo)}), targets {len(targets)}",
          flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(cache, fill_dir, model, keep)) as ex:
        for n, (k, res) in enumerate(zip(todo, ex.map(FM.score_position, [jobs[k] for k in todo]))):
            with open(part(k) + ".tmp", "wb") as fh:
                pickle.dump(res, fh)
            os.replace(part(k) + ".tmp", part(k))
            if (n + 1) % max(1, len(todo) // 20) == 0 or n + 1 == len(todo):
                el = time.time() - t0
                print(f"job {n + 1}/{len(todo)}: {el / 60:.1f} min, ETA {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} "
                      f"min", flush=True)
    rows = []
    for k in range(len(jobs)):
        with open(part(k), "rb") as fh:
            rows += pickle.load(fh)
    r = pd.DataFrame(rows)
    ref = pd.read_csv(ref_rows_path(w, limit, win))
    metrics = [c for c in FM.REPORTED if c in r.columns and c in ref.columns]
    gate = R4.compare(r, ref, sorted(set(ref.fill)) if keep is None else sorted(keep), metrics)
    gate.update({"model": model, "window_px": w, "n_targets": int(len(targets)),
                 "input_version": meta["input_version"], "input_mode": meta.get("input_mode"),
                 "n_model_rows": int((r.fill == f"model_{model}").sum()), "minutes": round((time.time() - t0) / 60, 1)})
    json.dump(gate, open(os.path.join(fill_dir, "gate_score.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in gate.items() if k != "max_abs_diff"}), flush=True)
    if not gate["passed"]:
        raise SystemExit("stored rq4 rows not reproduced; model rows not kept")
    if keep is not None:      # full run: keep the model rows only; the other rungs stay in the stored rq4 rows
        r = r[r.fill == f"model_{model}"]
    r.to_csv(os.path.join(fill_dir, "rows.csv.gz"), index=False, float_format="%.17g")


# ------------------------------------------------------------------------------------------------- summary (R)

STR = [f"str{j:02d}" for j in range(16)]
SIM = {"ssim", "ms_ssim", "pearson", "acc_dataset", "acc_cohort", "lddt", "lddt_rep", "tm_like", "gdt_like",
       "arcface_cos"}
ERR = {"mse", "rmse", "mae", "rmse_rep"}
SHOW = ["repeat", "mirror", "lookalike16", "dataset_mean", "shrink_repeat", "wrong_detail", "noise", "stranger"]


def r_self(r, fill, m, cell):
    """common_frame.py R_self on these rows: (mean d(fill) - d(self)) / (mean d(16 strangers) - d(self)),
    means over the (target, window) cells where the fill and the strangers are both scored."""
    sgn, d_self = (-1.0, -1.0) if m in SIM else (1.0, 0.0)
    a = r[r.fill == fill].groupby(cell)[m].mean() * sgn
    s = r[r.fill.isin(STR)].groupby(cell)[m].mean() * sgn
    j = pd.concat([a.rename("a"), s.rename("s")], axis=1).dropna()
    if j.empty:
        return np.nan, 0
    return float((j.a.mean() - d_self) / (j.s.mean() - d_self)), int(len(j))


def summ(model, limit):
    fill_dir = os.path.join(PRIVATE, tag_of(model, limit))
    r = pd.read_csv(os.path.join(fill_dir, "rows.csv.gz"))
    name = f"model_{model}"
    if limit == 0:            # the other rungs of the same targets from the stored rq4 rows
        _, _, w, _, _, _, _ = setup(limit)
        ref = pd.read_csv(os.path.join(R4.PRIVATE, f"rows_full_w{w}.csv.gz"))
        r = pd.concat([ref.merge(r[["src", "target"]].drop_duplicates(), on=["src", "target"]), r], ignore_index=True)
    out = {"model": model, "f": "1/8", "scale": "R_self: target 0, mean of 16 random people 1, lower = closer",
           "unit_of_mean": "(target, window) cells", "sources": {}}
    for src in sorted(set(r.src)):
        rs = r[r.src == src]
        out["sources"][src] = {"n_targets": int(rs.target.nunique()), "R": {}}
        for m in ("mse", "ssim", "tm_like", "gdt_like", "lddt"):
            if m not in rs.columns:
                continue
            row = {f: r_self(rs, f, m, ["target", "patch"]) for f in [name] + SHOW}
            out["sources"][src]["R"][m] = {f: (round(v, 3), n) for f, (v, n) in row.items()}
    json.dump(out, open(os.path.join(fill_dir, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------------------ qc

def qc(model, limit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache, desc, w, rw, jobs, names, targets = setup(limit)
    fill_dir = os.path.join(PRIVATE, tag_of(model, limit))
    _init(cache, fill_dir, model)
    have = _STATE["row"]
    lam = FM.lambdas(VERSION)
    cols = ((f"model_{model}", f"model: {model}"), ("repeat", "same person,\nsecond image"),
            (FL16.NEW, "16 nearest people"), ("dataset_mean", "dataset average"),
            ("shrink_repeat", "same person,\nsoftened"))
    for src, d in desc.items():
        valid = np.load(os.path.join(cache, f"{src}_valid.npy"))
        pos = FM.positions(valid, w, rw)
        T = d["targets"]
        sel = sorted({ti for s_, ti in targets.itertuples(index=False) if s_ == src and ti in have[src]})
        if limit == 0:                                          # full run: 24 evenly spaced targets
            sel = [sel[i] for i in np.unique(np.linspace(0, len(sel) - 1, 24).round().astype(int))]
        fig, ax = plt.subplots(len(sel), 1 + 2 * len(cols), figsize=(1.25 * (1 + 2 * len(cols)), 1.35 * len(sel)),
                               squeeze=False)
        for i, ti in enumerate(sel):
            xt = np.asarray(FM.WORK["X"][T[ti]["t"]], np.float32)
            mt = np.asarray(FM.WORK["M"][T[ti]["t"]])
            shown = np.where(valid & mt, xt, np.nan)
            ax[i, 0].imshow(shown, cmap="gray", vmin=-3, vmax=3)
            canv = {k: np.full((FM.SIZE, FM.SIZE), np.nan) for k, _ in cols}
            for pk, (y0, x0) in enumerate(pos):
                sl = (slice(y0, y0 + w), slice(x0, x0 + w))
                if (mt[sl] & valid[sl]).mean() < FM.MIN_IN_MASK:
                    continue
                ring = FM.ring_mask(y0, x0, FM.SIZE, w, rw) & valid
                P_ = FM.pool_window(src, {"pool": d["pool"]}, sl, ring)
                P_["shift_px"] = d["shift_px"]
                _, tv, f = FM.window_fills(src, int(ti), T[ti], pk, sl, ring, lam[src], P_, 0)
                for k, _ in cols:
                    if k in f:
                        canv[k][sl] = np.where(tv & f[k][1], f[k][0], canv[k][sl])
            for q, (k, lab) in enumerate(cols):
                a_ = ax[i, 1 + 2 * q]
                a_.imshow(canv[k], cmap="gray", vmin=-3, vmax=3)
                e_ = ax[i, 2 + 2 * q]
                e_.imshow((canv[k] - shown) ** 2, cmap="magma", vmin=0, vmax=4)
                if i == 0:
                    a_.set_title(lab, fontsize=6)
                    e_.set_title("squared error", fontsize=6)
        for a_ in ax.ravel():
            a_.set_axis_off()
        ax[0, 0].set_title("target", fontsize=6)
        fig.suptitle(f"PRIVATE QC {src}{VERSION}, f = 1/8 ({w} px windows): each window filled on its own and painted "
                     f"back; blank = not scored", fontsize=7)
        fig.tight_layout()
        path = os.path.join(fill_dir, f"qc_{src}_{model}_private.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(path, flush=True)


def mean_fill(model, parts, limit=0, win=None):
    """Pixelwise mean of the fills of several runs of one stochastic model (same people and windows), stored as the
    fills of `model` so that it is scored like any other fill. A window that one run did not fill stays not run."""
    dirs = [os.path.join(PRIVATE, tag_of(m, limit, win)) for m in parts]
    metas = [json.load(open(os.path.join(d, "infer_meta.json"))) for d in dirs]
    keys = ("window_px", "input_version", "input_mode", "model_size", "every")
    for m in metas[1:]:
        same_people = all(m["sources"][s_]["target_index"] == metas[0]["sources"][s_]["target_index"]
                          for s_ in m["sources"])
        if any(m.get(k) != metas[0].get(k) for k in keys) or not same_people:
            raise SystemExit("the runs differ in window, input, model size or people; no mean is formed")
    out = os.path.join(PRIVATE, tag_of(model, limit, win))
    os.makedirs(out, exist_ok=True)
    for s_ in metas[0]["sources"]:
        arrs = [np.load(os.path.join(d, f"{s_}.npy")).astype(np.float64) for d in dirs]
        np.save(os.path.join(out, f"{s_}.npy"), np.mean(arrs, axis=0).astype(np.float32))   # NaN where any is NaN
    meta = dict(metas[0], model=model, mean_of=list(parts), samples=[m.get("sample", 0) for m in metas])
    json.dump(meta, open(os.path.join(out, "infer_meta.json"), "w"), indent=1)
    print(out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("export", "infer", "score", "qc", "summ", "mean"))
    ap.add_argument("--model")
    ap.add_argument("--parts", nargs="+", help="mean: the runs whose fills are averaged into --model")
    ap.add_argument("--ckpt")
    ap.add_argument("--limit", type=int, required=True, help="4 = smoke (rq4 n4 targets), 0 = all targets")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--win", type=int, choices=sorted(set(WINDOWS) - {10}),
                    help="window in px other than the f = 1/8 ladder window: 5, 16 or 20 (RQ4 sizes)")
    ap.add_argument("--cells", action="store_true", help="export --win 16: only the stored set-level target cells")
    a = ap.parse_args()
    if a.limit not in (0, 4, 100) or (a.limit == 100) != (a.win in (5, 20)):
        raise SystemExit("--limit 4 (smoke) or 0 (all targets) at 10 and 16 px; --limit 100 (F3 subset) at 5 and 20 px")
    if a.stage == "export":
        return export(a.limit, a.win, a.cells)
    if not a.model:
        raise SystemExit("--model is required")
    if a.stage == "infer":
        if not a.ckpt:
            raise SystemExit("--ckpt is required for infer")
        return infer(a.model, a.ckpt, a.limit, a.device, a.batch)
    if a.stage == "mean":
        if not a.parts or len(a.parts) < 2:
            raise SystemExit("--parts needs at least two runs")
        return mean_fill(a.model, a.parts, a.limit, a.win)
    if a.stage == "score":
        return score(a.model, a.limit, a.workers, a.win)
    if a.stage == "qc":
        return qc(a.model, a.limit)
    return summ(a.model, a.limit)


if __name__ == "__main__":
    main()
