"""Metric panel on the iris identity ladder (docs/PROTOCOL_iris_metrics.md). Pre-QC until Kate's visual verdict.

Iris replication of the fundus panel (`fundus_ladder_metrics.py` in the identity-ladder-bilateral worktree, general
protocol Amendment 9.14) on UTIRIS (NIR primary, visible light secondary) and UPOL (secondary). The frame is the
rubber-sheet strip of `iris_ladder_metrics_frame` (64 x 256; preprocessing version passed explicitly, P2 current). Everything that does not depend on the
frame size is imported unchanged, by explicit path, from the fundus module (FUNDUS_METRICS_SRC) and the brain metric
module (BRAIN_METRICS_SRC); no default paths; both sha256 are stored in the JSON. Re-implemented here only for the
non-square strip: window positions, the context ring, MS-SSIM context clipping, and the per-target fill loop (no
sex-age cohort exists in either dataset; the fellow eye is the mirrored strip of the other eye).

Every metric is scored only on E = window AND target mask AND fill validity AND coverage (missing is never zero).
Rows and QC sheets stay in the private folder; the public JSON holds aggregates only (no image, file name or id).
Usage:
  python src/iris_ladder_metrics.py smoke [--every N]   subset run + fill QC sheets (private), no public JSON
  python src/iris_ladder_metrics.py run                 full sets, writes results/iris_ladder_metrics.json
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import iris_ladder_metrics_frame as IF  # noqa: E402

H, W = IF.H, IF.W
WIN, RING = 16, 8                          # reference size of the committed results; the panel run sets it explicitly
WINDOWS = (16, 10)                         # 16 px = reference (f = 0.196 of D = 256 / pi); 10 px = f 1/8 (Amendment P4)
MIN_POS_VALID, MIN_RING_VALID, MIN_IN_MASK = 0.95, 0.5, 0.5
MIN_FILL_SHARE = 0.5
COVER = 0.9
N_STR, K_LOOK = 16, 16
MS_MARGIN = 24
SEED = 20261102
OUT_JSON = os.path.join(ROOT, "results", "iris_ladder_metrics.json")
STATUS = ("frame P2 (Worldcoin open-iris segmentation): Kate accepted P2 on the trial and frame QC sheets on "
          "7 Nov 2026 ('хорошо да берем'); the lid-shadow mask (z < -4 connected to the lid) was added after that "
          "and its final sheets are not yet re-shown to Kate")

FL = None                                  # fundus panel module (read only)
SRC_SHA = {}


def _load(path, modname, extra_path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    sys.path.insert(0, extra_path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def load_shared(fundus_path=None, brain_path=None):
    """Fundus and brain metric modules by explicit path (environment variables; no default)."""
    global FL
    if FL is not None:
        return FL
    fundus_path = fundus_path or os.environ.get("FUNDUS_METRICS_SRC")
    brain_path = brain_path or os.environ.get("BRAIN_METRICS_SRC")
    for p, n in ((fundus_path, "FUNDUS_METRICS_SRC"), (brain_path, "BRAIN_METRICS_SRC")):
        if not p or not os.path.isfile(p):
            raise SystemExit(f"{n} must point to the module file (no default)")
    FL = _load(fundus_path, "fundus_ladder_metrics", os.path.dirname(fundus_path))
    FL.load_bm(brain_path)
    for k, p in (("fundus_metrics_source_sha256", fundus_path), ("brain_metrics_source_sha256", brain_path)):
        SRC_SHA[k] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return FL


# ---------------------------------------------------------------------------------------------- strip geometry

def positions(valid):
    """Top-left corners of windows on a WIN grid with >= 95% of the window and >= 50% of its ring in `valid`.
    The strip is periodic in angle, but windows and rings are not wrapped across the 0 deg seam."""
    out = []
    for y in range(0, H - WIN + 1, WIN):
        for x in range(0, W - WIN + 1, WIN):
            if valid[y:y + WIN, x:x + WIN].mean() < MIN_POS_VALID:
                continue
            if valid[ring_mask(y, x)].mean() >= MIN_RING_VALID:
                out.append((y, x))
    return out


def ring_mask(y, x):
    m = np.zeros((H, W), bool)
    m[max(y - RING, 0):y + WIN + RING, max(x - RING, 0):x + WIN + RING] = True
    m[y:y + WIN, x:x + WIN] = False
    return m


def ms_ssim(x, mx, f, e, win_slice):
    """Fundus MS-SSIM (3 scales, composite of target outside the window and fill on E inside), with the context
    clipped to the strip instead of the 128 px square. Pixels of the window outside E are missing (weight 0)."""
    (ys, xs) = win_slice
    y0, y1 = max(ys.start - MS_MARGIN, 0), min(ys.stop + MS_MARGIN, H)
    x0, x1 = max(xs.start - MS_MARGIN, 0), min(xs.stop + MS_MARGIN, W)
    inwin = np.zeros((H, W), bool)
    inwin[ys, xs] = True
    comp = np.where(inwin, f, x)
    wv = (mx & ~inwin) | e
    X, C = np.nan_to_num(x[y0:y1, x0:x1].astype(np.float64)), np.nan_to_num(comp[y0:y1, x0:x1].astype(np.float64))
    Wt, S = wv[y0:y1, x0:x1].astype(np.float64), e[y0:y1, x0:x1].astype(np.float64)
    vals = []
    for s in range(len(FL.MS_WEIGHTS)):
        if s:
            X, W0 = FL._down(X, Wt)
            C, _ = FL._down(C, Wt)
            S, _ = FL._down(S, np.ones_like(S))
            Wt = (W0 >= 0.5).astype(np.float64)
        sel = (S >= 0.5) & (Wt > 0)
        if not sel.any():
            return np.nan
        lum, cs = FL.ssim_parts(X, C, Wt)
        v = (lum * cs if s == len(FL.MS_WEIGHTS) - 1 else cs)[sel]
        vals.append(max(float(np.nanmean(v)), 0.0))
    return float(np.prod(np.power(vals, FL.MS_WEIGHTS)))


def score(x, mx, f, e, ws, mu_d, sigma):
    """FL.score with this module's MS-SSIM; no cohort mean (both cohort columns are NaN: unavailable)."""
    BM = FL.BM
    out = BM.score(x[ws][None], f[ws][None], e[ws][None], mu_d[ws][None], None, sigma)
    if out is None:
        return None
    out["ampdev_dataset"] = abs(1.0 - out["amplitude"]) if np.isfinite(out["amplitude"]) else np.nan
    out["amplitude_cohort"] = np.nan
    out["ampdev_cohort"] = np.nan
    xe = x[e].astype(np.float64)
    out["mse_V"] = float(np.mean((xe - mu_d[e]) ** 2))
    out["ms_ssim"] = ms_ssim(x, mx, f, e, ws)
    out["persistence"] = FL.persistence_distance(x, f, e)
    return out


# ---------------------------------------------------------------------------------------------- fills

S_ = {}


def set_window(w):
    """Window side and context ring (ring = ceil(w / 2)) for this process and for the fundus helper it calls
    (repeat_sigma reads the fundus module's WIN). Required in every panel run; no silent default."""
    global WIN, RING
    if w not in WINDOWS:
        raise SystemExit(f"window {w} px is not in the protocol ({WINDOWS})")
    WIN, RING = int(w), int(-(-w // 2))
    if FL is not None:
        FL.WIN, FL.RING = WIN, RING
    return WIN, RING


def _init(state):
    load_shared(state.pop("fundus_path"), state.pop("brain_path"))
    set_window(state.pop("window"))
    S_.update(state)


def noise_rng(ti, pi):
    return np.random.default_rng([SEED, int(ti), int(pi), 7])


def fixed_fills(ti):
    """Fills that do not depend on the window position, and the target's leave-person-out pool."""
    X, M, XF, MF, idx = S_["X"], S_["M"], S_["XF"], S_["MF"], S_["idx"]
    person, side, rep, fel, lam = S_["person"], S_["side"], S_["repeat"], S_["fellow"], S_["lam"]
    S = S_["side_units"][side[ti]]
    pool = S[person[S] != person[ti]]
    FL.result_spec.assert_leave_person_out(person[ti], person[pool])
    Xp, Mp = X[idx[pool]], M[idx[pool]]
    mu, mu_ok = FL.masked_mean(np.nan_to_num(Xp), Mp)
    perm = np.random.default_rng([SEED, int(ti)]).permutation(pool)[:N_STR]
    fx = {}
    if rep[ti] >= 0:
        fx["repeat"] = (X[rep[ti]], M[rep[ti]])
    if fel[ti] >= 0:
        fx["fellow"] = (XF[idx[fel[ti]]], MF[idx[fel[ti]]])
    fx["stranger"] = (X[idx[perm[0]]], M[idx[perm[0]]])
    for k, q in enumerate(perm):
        fx[f"str{k:02d}"] = (X[idx[q]], M[idx[q]])
    fx["mean_k2"] = FL.masked_mean(np.nan_to_num(X[idx[perm[:2]]]), M[idx[perm[:2]]])
    fx["dataset_mean"] = (mu, mu_ok)
    for c in ("repeat", "fellow"):
        if c in fx and c in lam:
            d_, m_ = fx[c]
            fx[f"shrink_{c}"] = (FL.BM.shrink(np.nan_to_num(d_), mu, lam[c]), m_ & mu_ok)
    return fx, pool, Xp, Mp, mu, mu_ok


def window_fills(ti, pi, fx, pool, Xp, Mp, mu, mu_ok):
    """Position-dependent fills (look-alikes, oracle, wrong detail, noise) added to the fixed ones."""
    X, M, valid, idx = S_["X"], S_["M"], S_["valid"], S_["idx"]
    y, x = S_["pos"][pi]
    ws = (slice(y, y + WIN), slice(x, x + WIN))
    win = np.zeros((H, W), bool)
    win[ws] = True
    xt, mt = np.nan_to_num(X[idx[ti]]), M[idx[ti]]
    mtv = mt & valid
    fills = dict(fx)
    Xp0 = np.nan_to_num(Xp)
    elig = Mp[:, ws[0], ws[1]].reshape(len(pool), -1).mean(1) >= MIN_IN_MASK
    rmv = (ring_mask(y, x) & valid).ravel()
    nn = FL.nearest(Xp0.reshape(len(pool), -1)[:, rmv], Mp.reshape(len(pool), -1)[:, rmv], xt.ravel()[rmv],
                    mt.ravel()[rmv], elig, K_LOOK)
    if nn.size:
        fills["lookalike"] = (Xp[nn[0]], Mp[nn[0]])
    if nn.size >= K_LOOK:
        fills["lookalike16"] = FL.masked_mean(Xp0[nn], Mp[nn])
    wv = win.ravel()
    no = FL.nearest(Xp0.reshape(len(pool), -1)[:, wv], Mp.reshape(len(pool), -1)[:, wv], xt.ravel()[wv],
                    mtv.ravel()[wv], elig, 1)
    if no.size:
        fills["lookalike_oracle"] = (Xp[no[0]], Mp[no[0]])
    e0 = win & mtv & mu_ok
    st_f, st_m = fx["stranger"]
    wd = FL.BM.wrong_detail(xt, mu, np.nan_to_num(st_f), e0 & st_m)
    if wd is not None:
        fills["wrong_detail"] = (wd, e0 & st_m)
    fills["noise"] = (FL.BM.noise_fill(xt, mu, e0, noise_rng(ti, pi)), e0)
    return fills, ws, win, xt, mtv


def run_target(ti):
    """All rows of one target eye (every position, every fill)."""
    fx, pool, Xp, Mp, mu, mu_ok = fixed_fills(ti)
    person, side = S_["person"], S_["side"]
    rows = []
    for pi in range(len(S_["pos"])):
        y, x = S_["pos"][pi]
        mtv0 = S_["M"][S_["idx"][ti]] & S_["valid"]
        if mtv0[y:y + WIN, x:x + WIN].mean() < MIN_IN_MASK:
            continue
        fills, ws, win, xt, mtv = window_fills(ti, pi, fx, pool, Xp, Mp, mu, mu_ok)
        n_tw = int((win & mtv).sum())
        sigma = S_["sigma"].get((pi, person[ti])) if S_["sigma"] else None
        for fname, (f, mf) in fills.items():
            e = win & mtv & mf.astype(bool) & mu_ok
            if e.sum() < max(10, MIN_FILL_SHARE * n_tw):
                continue
            s = score(xt, mtv, np.nan_to_num(f), e, ws, mu, sigma)
            if s is None:
                continue
            rows.append({"unit": int(ti), "person": person[ti], "side": side[ti], "patch": pi, "fill": fname, **s})
    return rows


# ---------------------------------------------------------------------------------------------- per set

def whole_strip_ratios(st):
    """Squared error of the repeat and the fellow eye over the dataset mean's, summed over targets on the pixels
    valid in the target, the donor and the mean (coverage included). Used only for the shrinkage lambda."""
    X, M, idx, valid = st["X"], st["M"], st["idx"], st["valid"]
    num, den = {"repeat": 0.0, "fellow": 0.0}, {"repeat": 0.0, "fellow": 0.0}
    for ti in range(len(idx)):
        S = st["side_units"][st["side"][ti]]
        pool = S[st["person"][S] != st["person"][ti]]
        mu, mu_ok = FL.masked_mean(np.nan_to_num(X[idx[pool]]), M[idx[pool]])
        xt, mt = np.nan_to_num(X[idx[ti]]), M[idx[ti]] & valid & mu_ok
        for c in ("repeat", "fellow"):
            d, md = _donor(st, ti, c)
            if d is None:
                continue
            e = mt & md
            num[c] += float(((xt - np.nan_to_num(d)) ** 2)[e].sum())
            den[c] += float(((xt - mu) ** 2)[e].sum())
    return {c: num[c] / den[c] for c in num if den[c] > 0}


def _donor(st, ti, c):
    if c == "repeat":
        r = st["repeat"][ti]
        return (st["X"][r], st["M"][r]) if r >= 0 else (None, None)
    f = st["fellow"][ti]
    return (st["XF"][st["idx"][f]], st["MF"][st["idx"][f]]) if f >= 0 else (None, None)


def wait_for_frames(name, version, poll_s=60, max_h=6):
    """The panel never builds frames (the frame builder may be running in another process); it waits for them."""
    path = IF.cache_path(name, version).replace(".npz", "_table.csv.gz")
    t0 = time.time()
    while not os.path.isfile(path):
        if time.time() - t0 > max_h * 3600:
            raise SystemExit(f"frames for {name} {version} not found after {max_h} h; run iris_ladder_metrics_frame.py")
        print(f"  waiting for {name} {version} frames ({(time.time() - t0) / 60:.0f} min)", flush=True)
        time.sleep(poll_s)


def build(name, version):
    wait_for_frames(name, version)
    fr = IF.load_or_build(name, version)
    tab, X, M = fr["table"], fr["X"], fr["M"]
    u = IF.units(tab)
    idx = u.idx.to_numpy()
    valid = M[idx].sum(0) >= COVER * len(u)
    st = {"name": name, "X": X, "M": M, "XF": IF.mirror(X), "MF": IF.mirror(M), "idx": idx, "valid": valid,
          "person": u.person.astype(str).to_numpy(), "side": u.side.to_numpy(), "repeat": u.repeat_idx.to_numpy(),
          "fellow": u.fellow.to_numpy(), "pos": positions(valid)}
    st["side_units"] = {s: np.flatnonzero(st["side"] == s) for s in ("R", "L")}
    S_.clear()
    S_.update(st)
    st["ratio_whole_strip"] = whole_strip_ratios(st)
    st["lam"] = {c: 1.0 - r / 2.0 for c, r in st["ratio_whole_strip"].items()}
    st["sigma"] = FL.repeat_sigma(np.nan_to_num(X), M, valid, idx, st["person"], st["repeat"], st["pos"])
    st["n_images"], st["n_images_ok"] = int(len(tab)), int(tab.ok.sum())
    st["fail_reasons"] = tab.loc[~tab.ok, "reason"].astype(str).str.split(":").str[0].value_counts().to_dict()
    st["near_duplicate_repeats"] = near_duplicates(st)
    return st


def near_duplicates(st):
    """Share of repeat pairs with Pearson r > 0.9 over pixels valid in both (reported, not excluded)."""
    rs = []
    for ti in range(len(st["idx"])):
        r = st["repeat"][ti]
        if r < 0:
            continue
        a, b = st["X"][st["idx"][ti]], st["X"][r]
        m = st["M"][st["idx"][ti]] & st["M"][r]
        if m.sum() > 100:
            rs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
    rs = np.array(rs)
    return {"n_pairs": int(rs.size), "share_r_gt_0.9": float((rs > 0.9).mean()) if rs.size else None,
            "median_r": float(np.median(rs)) if rs.size else None}


def run_rows(st, targets, workers):
    state = {k: v for k, v in st.items() if k in ("X", "M", "XF", "MF", "idx", "valid", "person", "side", "repeat",
                                                     "fellow", "pos", "side_units", "lam", "sigma", "name")}
    state["fundus_path"], state["brain_path"] = os.environ["FUNDUS_METRICS_SRC"], os.environ["BRAIN_METRICS_SRC"]
    state["window"] = WIN
    rows, t0, done = [], time.time(), 0
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(state,)) as ex:
        for r in ex.map(run_target, targets, chunksize=2):
            rows.extend(r)
            done += 1
            if done % 10 == 0 or done == len(targets):
                el = time.time() - t0
                print(f"  {st['name']}: {done}/{len(targets)} targets, {el:.0f}s, ETA {el / done * (len(targets) - done):.0f}s",
                      flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- QC sheet of fills

QC_FILLS = ("repeat", "fellow", "stranger", "mean_k2", "dataset_mean", "lookalike", "lookalike16",
            "lookalike_oracle", "wrong_detail", "noise", "shrink_repeat", "shrink_fellow")
QC_LABEL = {"repeat": "repeat image", "fellow": "other eye\n(mirrored)", "stranger": "stranger", "mean_k2": "mean of 2",
            "dataset_mean": "dataset mean", "lookalike": "look-alike", "lookalike16": "mean of 16\nlook-alikes",
            "lookalike_oracle": "oracle\nlook-alike", "wrong_detail": "wrong detail", "noise": "white noise",
            "shrink_repeat": "repeat pulled\nto the mean", "shrink_fellow": "other eye pulled\nto the mean"}


def qc_sheet(st, targets, path):
    """Private: per evenly spaced target, one window with 16 px of strip context on each side and each fill pasted
    in; light blue = missing (outside the scored pixels)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S_.clear()
    S_.update(st)
    cmap = matplotlib.colormaps["gray"].copy()
    cmap.set_bad("#9ecae1")
    cols = ("target",) + QC_FILLS
    fig, ax = plt.subplots(len(targets), len(cols), figsize=(1.35 * len(cols), 1.5 * len(targets) + 0.7))
    ax = np.atleast_2d(ax)
    for r_, ti in enumerate(targets):
        fx, pool, Xp, Mp, mu, mu_ok = fixed_fills(ti)
        mtv0 = st["M"][st["idx"][ti]] & st["valid"]
        ok = [pi for pi, (y, x) in enumerate(st["pos"]) if mtv0[y:y + WIN, x:x + WIN].mean() >= MIN_IN_MASK]
        for a in ax[r_]:
            a.set_xticks([])
            a.set_yticks([])
        if not ok:
            ax[r_, 0].text(0.5, 0.5, "no position", ha="center", va="center", fontsize=7, transform=ax[r_, 0].transAxes)
            continue
        pi = ok[(r_ * 5) % len(ok)]
        fills, ws, win, xt, mtv = window_fills(ti, pi, fx, pool, Xp, Mp, mu, mu_ok)
        y, x = st["pos"][pi]
        y0, x0 = max(y - 16, 0), max(x - 16, 0)
        sl = (slice(y0, y0 + 48), slice(x0, x0 + 48))
        base = np.where(mtv, xt, np.nan)
        for c_, key in enumerate(cols):
            a = ax[r_, c_]
            if r_ == 0:
                a.set_title(QC_LABEL.get(key, key), fontsize=6.5)
            if key == "target":
                img = base
            elif key in fills:
                f, mf = fills[key]
                e = win & mtv & mf.astype(bool) & mu_ok
                img = base.copy()
                img[ws] = np.where(e[ws], np.nan_to_num(f)[ws], np.nan)
            else:
                a.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7, transform=a.transAxes)
                continue
            a.imshow(img[sl], cmap=cmap, vmin=-2.5, vmax=2.5)
            a.add_patch(plt.Rectangle((x - x0 - 0.5, y - y0 - 0.5), WIN, WIN, fill=False, ec="#d33", lw=0.6))
        ax[r_, 0].set_ylabel(f"target {r_ + 1}\n{st['side'][ti]} eye", fontsize=6)
    fig.suptitle(f"{st['name']}: fills of the metric panel (window red, 16 px context; light blue = missing). Private.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- main

def summarise(st, r):
    s = FL.summarise(st["name"], r, st)
    s["counts"].update({"images": st["n_images"], "images_segmentation_pass": st["n_images_ok"],
                        "segmentation_fail_reasons": st["fail_reasons"],
                        "eyes_with_repeat": int((st["repeat"] >= 0).sum()),
                        "eyes_with_fellow": int((st["fellow"] >= 0).sum())})
    s["ratio_whole_strip"] = st["ratio_whole_strip"]
    s["near_duplicate_repeats"] = st["near_duplicate_repeats"]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("smoke", "run"))
    ap.add_argument("--every", type=int, default=8, help="smoke: every N-th target eye")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sets", nargs="+", default=list(IF.SETS))
    ap.add_argument("--preprocessing", required=True, choices=IF.VERSIONS)
    ap.add_argument("--window", type=int, required=True, choices=WINDOWS, help="window side in px (no default)")
    a = ap.parse_args()
    if a.preprocessing in IF.REJECTED:
        raise SystemExit(f"preprocessing {a.preprocessing} was rejected: {IF.REJECTED[a.preprocessing]}")
    load_shared()
    set_window(a.window)
    sfx = "" if WIN == 16 else f"_w{WIN}"
    os.makedirs(IF.PRIVATE, exist_ok=True)
    t0 = time.time()
    results, panel = {}, {}
    for name in a.sets:
        st = build(name, a.preprocessing)
        n = len(st["idx"])
        targets = (sorted(np.concatenate([np.flatnonzero(st["side"] == s_)[::a.every] for s_ in ("R", "L")]).tolist())
                   if a.mode == "smoke" else list(range(n)))           # smoke: every N-th eye of each side
        print(f"{name}: eyes {n}, persons {len(np.unique(st['person']))}, positions {len(st['pos'])}, "
              f"lambda {({k: round(v, 3) for k, v in st['lam'].items()})}, targets {len(targets)}", flush=True)
        r = run_rows(st, targets, a.workers)
        tag = "smoke" if a.mode == "smoke" else "full"
        r.to_csv(os.path.join(IF.PRIVATE, f"rows_{name}_{a.preprocessing}_{tag}{sfx}.csv.gz"), index=False)
        if a.mode == "smoke":
            qc_sheet(st, list(np.linspace(0, n - 1, 12).round().astype(int)),
                     os.path.join(IF.PRIVATE, f"qc_fills_{name}_{a.preprocessing}{sfx}.png"))
        s = summarise(st, r)
        panel[name] = {"frame": a.preprocessing, **s}
        results[name] = {"frame": a.preprocessing, **s["mse_ratio"]}
        print(f"{name} done, rows {len(r)}, {time.time() - t0:.0f}s", flush=True)
    status = STATUS if WIN == 16 else (f"Amendment P4: window {WIN} px (f = 1/8 of D = 256 / pi), frame "
                                       f"{a.preprocessing}, sets {', '.join(a.sets)}; 16 px results are the reference")
    doc = {"protocol": "docs/PROTOCOL_iris_metrics.md", "status": status, "mode": a.mode,
           "spec": FL.result_spec.SPEC,
           "spec_meta": {"bootstrap_unit": "person", "pools": "leave-person-out", "stranger_rung": "stranger",
                         "mirror_rungs": {"fellow": "sides"}},
           "scale": ("results: squared error / squared error of the dataset mean on the same pixels (1 = mean, 2 = "
                     "stranger); panel: R_self = 0 for the target itself, 1 for the 16 strangers (Amendment F1)"),
           "frame": f"rubber-sheet iris strip {H} x {W} (radial x angular), preprocessing {a.preprocessing}",
           "primary_set": IF.PRIMARY, "window_px": WIN, "ring_px": RING, "seed": SEED, "bootstrap_reps": FL.REPS,
           **SRC_SHA,
           "metrics_confirmatory": {k: {"group": g, "column": c, "status": "computed" if c else "blocked",
                                        **({"reason": b} if b else {})} for k, (g, c, b) in confirmatory().items()},
           "metrics_supplementary": list(FL.SUPPLEMENTARY), "lower_is_better": sorted(FL.LOWER),
           "results": results, "panel": panel, "minutes": round((time.time() - t0) / 60, 1)}
    doc = FL.finite(doc)
    for flag in FL.result_spec.enforce(doc):
        print("SPEC (soft):", flag, flush=True)
    out = (OUT_JSON.replace(".json", f"{sfx}.json") if a.mode == "run"
           else os.path.join(IF.PRIVATE, f"iris_ladder_metrics_smoke{sfx}.json"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print("written", out, flush=True)


BLOCK_ID_IRIS = ("no deep iris encoder weights on this machine; the classical iris code (Hamming distance) is not an "
                 "embedding and has no angle, norm, PFE or PLDA form")
BLOCK_COHORT = "no sex or age metadata in UTIRIS or UPOL (protocol: matched mean unavailable)"
_P3 = "computed by src/iris_ladder_metrics_extra.py (Amendment P3), reported in its own block, not in this panel run"
IRIS_P3 = {"lpips": _P3, "betti_matching_matched": _P3, "betti_matching_unmatched": _P3,
           "fid": _P3 + " (set level)", "fwd": _P3 + " (set level)",
           "topology_distance": "no definition fixed in the project (Amendment P3); not invented here",
           "plausibility": "needs a trained density model of iris strips (Amendment P3)"}
BLOCK_LATENT_IRIS ="no generative iris model; LDDMM between 16 px windows is a pairwise registration (stop rule)"


def confirmatory():
    """The fundus 26-metric registry with iris reasons where they differ."""
    out = {}
    for k, (g, c, b) in FL.CONFIRMATORY.items():
        if k in ("acc_matched", "amplitude_matched"):
            c, b = None, BLOCK_COHORT
        elif k.startswith("id_"):
            b = BLOCK_ID_IRIS + ("; PFE also needs an uncertainty head" if k == "id_pfe" else
                                 "; PLDA also needs a fitted model" if k == "id_plda" else "")
        elif k.startswith("latent_"):
            b = BLOCK_LATENT_IRIS
        elif k in IRIS_P3:
            # Iris status is set here, never inherited from the fundus module (which computes these itself).
            c, b = None, IRIS_P3[k]
        out[k] = (g, c, b)
    assert len(out) == 26
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
