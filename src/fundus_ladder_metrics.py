"""Metric panel on the fundus donor ladder (general protocol, Amendment 9.14). Pre-QC.

Fundus replication of the brain (PROTOCOL_donor_ladder_brain.md, Amendments 10-14) and face
(PROTOCOL_face_metrics.md, Amendment F1) metric audits. Frame, units, fellow eyes and GRAPE other visits as
`donor_ladder_2d_facefmt` (128 px, GRAPE P8, PAPILA P9, Amendment 9.13: frame only, no pairwise registration, no
intensity fit). Tier-1 metric functions are imported unchanged from the brain audit
(`donor_ladder_brain_metrics.py`, path given by BRAIN_METRICS_SRC; no default; its sha256 is stored in the JSON).

Every metric is scored only on E = window AND target mask AND fill validity AND coverage (missing is never zero).
Rows and QC sheets stay in the private folder; the public JSON holds aggregates only (no image, file name or id).
Usage:
  python src/fundus_ladder_metrics.py smoke   [--every N]   subset run + QC sheets (private), no public JSON
  python src/fundus_ladder_metrics.py run                   full cohort, writes results/fundus_ladder_metrics.json
LADDER_EXTERNAL must point to a checkout's external/ (GRAPE metadata), as for donor_ladder_2d_facefmt.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import donor_ladder_2d as D  # noqa: E402
import donor_ladder_2d_facefmt as FF  # noqa: E402
import donor_ladder_fundus as F  # noqa: E402
import fundus_ladder_metrics_extra as EX  # noqa: E402
import result_spec  # noqa: E402

SEED = D.SEED
SIZE, WIN, RING = FF.SIZE, 16, 8
MIN_POS_VALID, MIN_RING_VALID, MIN_IN_MASK = 0.95, 0.5, 0.5
MIN_FILL_SHARE = 0.5               # a fill enters a cell only if E covers >= 50% of the target's window pixels
COVER = 0.9                        # coverage `valid`: inside the field of >= 90% of the units (face-format maps)
N_STR, K_LOOK, MIN_COHORT = 16, 16, 10
PRIVATE_V1 = os.path.join(F.DATA, "projects", "identity-ladder-metric", "fundus-ladder-metrics-20261106-v1")
PRIVATE = os.path.join(F.DATA, "projects", "identity-ladder-metric", "fundus-ladder-metrics-20261108-v2")  # 9.15
OUT_JSON = os.path.join(ROOT, "results", "fundus_ladder_metrics.json")
FACEFMT_JSON = os.path.join(ROOT, "results", "donor_ladder_2d_facefmt.json")
MS_WEIGHTS = np.array([0.0448, 0.2856, 0.3001]) / (0.0448 + 0.2856 + 0.3001)   # Wang et al. 2003, first 3 scales
MS_MARGIN = 24                     # context (px) around the window for the coarse MS-SSIM scales
REPS = 2000
TOL = 0.1

# ---------------------------------------------------------------------------------------------- brain metric functions

BM = None
BM_SHA = None


def load_bm(path=None):
    """The brain metric module, read only. The path is required (no silent default); its hash goes in the JSON."""
    global BM, BM_SHA
    if BM is not None:
        return BM
    path = path or os.environ.get("BRAIN_METRICS_SRC")
    if not path or not os.path.isfile(path):
        raise SystemExit("BRAIN_METRICS_SRC must point to donor_ladder_brain_metrics.py (cvpr2027 worktree)")
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("donor_ladder_brain_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved              # its src/ must not shadow this checkout's modules afterwards
    BM, BM_SHA = mod, hashlib.sha256(open(path, "rb").read()).hexdigest()
    return BM


# ---------------------------------------------------------------------------------------------- metric registry

# The 26 confirmatory base metrics (CVPR study set): name -> (group, row column or None, blocked reason or None)
BLOCK_ID = ("RETFound (YukunZhou/RETFound_mae_natureCFP) is gated on Hugging Face; access needs Kate's form; the "
            "identity definition and its cost get their own amendment before any result (Amendment 9.15, item 7)")
BLOCK_LATENT = ("no generative fundus model and no GPU on this machine; LDDMM between 16 px windows would be a pairwise "
                "registration (stop rule); plan and cost reported, not run (Amendment 9.15, item 8)")
BLOCK_TOPO = "no definition of Topology Distance is fixed in the project; none is invented (Amendment 9.15, item 5)"
BLOCK_PLAUS = "needs a trained density model of fundus images (Amendment 9.15, item 6)"
SET_LEVEL = {"fid", "fwd"}           # set-level metrics (Amendment 9.15): per rung, no per-target value, Holm p = 1
CONFIRMATORY = {
    "mse": ("A fidelity", "mse", None), "rmse": ("A fidelity", "rmse", None), "psnr": ("A fidelity", "psnr", None),
    "mae": ("A fidelity", "mae", None), "ssim": ("A fidelity", "ssim", None),
    "ms_ssim": ("A fidelity", "ms_ssim", None), "lpips": ("A fidelity", "lpips_alex", None),
    "acc_dataset": ("A anomaly", "acc_dataset", None), "acc_matched": ("A anomaly", "acc_cohort", None),
    "amplitude_dataset": ("A anomaly", "ampdev_dataset", None),
    "amplitude_matched": ("A anomaly", "ampdev_cohort", None),
    "lddt": ("B structure", "lddt", None), "intensity_persistence": ("B structure", "persistence", None),
    "betti_matching_matched": ("B structure", "betti_matched", None),
    "betti_matching_unmatched": ("B structure", "betti_unmatched", None),
    "id_angle": ("B identity", None, BLOCK_ID), "id_norm": ("B identity", None, BLOCK_ID),
    "id_pfe": ("B identity", None, BLOCK_ID + "; PFE also needs an uncertainty head, which RETFound has not"),
    "id_plda": ("B identity", None, BLOCK_ID + "; PLDA also needs a fitted model"),
    "latent_pullback": ("B latent", None, BLOCK_LATENT), "latent_fisher_rao": ("B latent", None, BLOCK_LATENT),
    "latent_lddmm": ("B latent", None, BLOCK_LATENT),
    "plausibility": ("C reference-free", None, BLOCK_PLAUS),
    "fid": ("C reference-free", "fid", None), "fwd": ("C reference-free", "fwd", None),
    "topology_distance": ("C reference-free", None, BLOCK_TOPO),
}
assert len(CONFIRMATORY) == 26
SUPPLEMENTARY = ("psnr_fixed", "pearson", "tm_like", "gdt_like", "lddt_rep", "rmse_rep", "lpips_vgg")
COMPUTED = tuple(c for _, c, b in CONFIRMATORY.values() if c and c not in SET_LEVEL) + SUPPLEMENTARY
LOWER = {"mse", "rmse", "mae", "rmse_rep", "persistence", "ampdev_dataset", "ampdev_cohort", "lpips_alex", "lpips_vgg",
         "betti_matched", "betti_unmatched"}
D_SELF = {**{m: 0.0 for m in LOWER}, **{m: -1.0 for m in ("ssim", "ms_ssim", "pearson", "acc_dataset", "acc_cohort",
                                                             "lddt", "tm_like", "gdt_like", "lddt_rep")}}
NO_R = {"psnr", "psnr_fixed", "ampdev_dataset", "ampdev_cohort"}   # infinite at self / no closeness orientation
DESCRIPTIVE = ("amplitude", "amplitude_cohort", "mse_V", "n_vox", "lpips_setshare", "topo_subset")
TESTS = {"T1": "dataset_mean", "T2": "shrink", "T3a": "wrong_detail", "T3b": "noise"}


def oriented(m, v):
    return -v if m in LOWER else v


def is_ref_stranger(name):
    return len(name) == 5 and name.startswith("str") and name[3:].isdigit()


# ---------------------------------------------------------------------------------------------- geometry

def positions(valid, win=WIN, ring=RING):
    """Top-left corners of windows on a `win` grid with >= 95% of the window and >= 50% of its ring in `valid`."""
    out = []
    for y in range(0, SIZE - win + 1, win):
        for x in range(0, SIZE - win + 1, win):
            if valid[y:y + win, x:x + win].mean() < MIN_POS_VALID:
                continue
            rm = ring_mask(y, x, win, ring)
            if valid[rm].mean() >= MIN_RING_VALID:
                out.append((y, x))
    return out


def ring_mask(y, x, win=WIN, ring=RING):
    m = np.zeros((SIZE, SIZE), bool)
    m[max(y - ring, 0):y + win + ring, max(x - ring, 0):x + win + ring] = True
    m[y:y + win, x:x + win] = False
    return m


# ---------------------------------------------------------------------------------------------- new metrics

def _gconv(a, w, sigma=1.5):
    return ndimage.gaussian_filter(a * w, sigma, mode="constant", truncate=3.0)


def ssim_parts(x, f, w, sigma=1.5, z_range=8.0):
    """Luminance and contrast-structure maps by normalised Gaussian convolution: only pixels with weight w enter."""
    g = ndimage.gaussian_filter(w, sigma, mode="constant", truncate=3.0)
    g = np.where(g > 1e-6, g, np.nan)
    mx, mf = _gconv(x, w) / g, _gconv(f, w) / g
    vx = _gconv(x * x, w) / g - mx ** 2
    vf = _gconv(f * f, w) / g - mf ** 2
    cxf = _gconv(x * f, w) / g - mx * mf
    c1, c2 = (0.01 * z_range) ** 2, (0.03 * z_range) ** 2
    return (2 * mx * mf + c1) / (mx ** 2 + mf ** 2 + c1), (2 * cxf + c2) / (vx + vf + c2)


def _down(a, w):
    """Masked 2x2 mean: value = weighted mean of the valid source pixels, weight = share of valid source pixels."""
    h, wd = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a, w = a[:h, :wd], w[:h, :wd]
    sw = w.reshape(h // 2, 2, wd // 2, 2).sum((1, 3))
    sa = (a * w).reshape(h // 2, 2, wd // 2, 2).sum((1, 3))
    return np.where(sw > 0, sa / np.maximum(sw, 1e-12), 0.0), sw / 4.0


def ms_ssim(x, mx, f, e, win_slice):
    """MS-SSIM of the composite (target outside the window, fill on E inside it) against the target, 3 scales, scored
    on window pixels. x, mx: target and its validity (coverage included); f: fill; e: scored pixels inside the window.
    Pixels of the window outside E are missing in the composite (weight 0)."""
    (ys, xs) = win_slice
    y0, y1 = max(ys.start - MS_MARGIN, 0), min(ys.stop + MS_MARGIN, SIZE)
    x0, x1 = max(xs.start - MS_MARGIN, 0), min(xs.stop + MS_MARGIN, SIZE)
    inwin = np.zeros((SIZE, SIZE), bool)
    inwin[ys, xs] = True
    comp = np.where(inwin, f, x)
    wv = (mx & ~inwin) | e                      # composite valid; target valid on the same pixels (E within mx)
    X, C = x[y0:y1, x0:x1].astype(np.float64), comp[y0:y1, x0:x1].astype(np.float64)
    W, S = wv[y0:y1, x0:x1].astype(np.float64), e[y0:y1, x0:x1].astype(np.float64)
    vals = []
    for s in range(len(MS_WEIGHTS)):
        if s:
            X, W0 = _down(X, W)
            C, _ = _down(C, W)
            S, _ = _down(S, np.ones_like(S))
            W = (W0 >= 0.5).astype(np.float64)
        sel = (S >= 0.5) & (W > 0)
        if not sel.any():
            return np.nan
        lum, cs = ssim_parts(X, C, W)
        v = (lum * cs if s == len(MS_WEIGHTS) - 1 else cs)[sel]
        vals.append(max(float(np.nanmean(v)), 0.0))
    return float(np.prod(np.power(vals, MS_WEIGHTS)))


def h0_diagram(v, m):
    """0-dimensional sublevel persistence of the pixels in m (4-connectivity, elder rule). The essential bar of each
    component dies at the maximum over m (bounded, so diagrams on the same pixels are comparable)."""
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return np.zeros((0, 2))
    vals = v[ys, xs].astype(np.float64)
    order = np.argsort(vals, kind="stable")
    index = -np.ones(m.shape, int)
    index[ys, xs] = np.arange(ys.size)
    parent = list(range(ys.size))
    birth = vals.copy()
    added = np.zeros(ys.size, bool)
    pairs = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    H, Wd = m.shape
    for k in order:
        added[k] = True
        y, x = ys[k], xs[k]
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < H and 0 <= nx < Wd:
                j = index[ny, nx]
                if j < 0 or not added[j]:
                    continue
                a, b = find(k), find(j)
                if a == b:
                    continue
                if birth[a] > birth[b]:
                    a, b = b, a                  # a is older (lower birth); b dies now
                if vals[k] > birth[b]:
                    pairs.append((birth[b], vals[k]))
                parent[b] = a
    top = vals.max()
    roots = {find(i) for i in range(ys.size)}
    pairs += [(birth[r], top) for r in roots]
    return np.array(pairs, np.float64).reshape(-1, 2)


def wasserstein1(A, B):
    """1-Wasserstein distance between persistence diagrams (L-infinity ground metric, diagonal allowed)."""
    na, nb = len(A), len(B)
    if na + nb == 0:
        return 0.0
    da = (A[:, 1] - A[:, 0]) / 2 if na else np.zeros(0)
    db = (B[:, 1] - B[:, 0]) / 2 if nb else np.zeros(0)
    n = na + nb
    C = np.zeros((n, n))
    if na and nb:
        C[:na, :nb] = np.maximum(np.abs(A[:, None, 0] - B[None, :, 0]), np.abs(A[:, None, 1] - B[None, :, 1]))
    C[:na, nb:] = np.inf
    C[:na, nb:][np.arange(na), np.arange(na)] = da
    C[na:, :nb] = np.inf
    C[na:, :nb][np.arange(nb), np.arange(nb)] = db
    r, c = linear_sum_assignment(C)
    return float(C[r, c].sum())


def persistence_distance(x, f, e):
    """Sublevel + superlevel H0 Wasserstein distance between target and fill, only pixels in e."""
    return (wasserstein1(h0_diagram(x, e), h0_diagram(f, e))
            + wasserstein1(h0_diagram(-x, e), h0_diagram(-f, e)))


def score(x, mx, f, e, ws, mu_d, mu_c, sigma):
    """Every computed metric of fill f against target x on the pixels e (inside the window ws)."""
    ew = e[ws][None]
    out = BM.score(x[ws][None], f[ws][None], ew, mu_d[ws][None],
                   mu_c[ws][None] if mu_c is not None else None, sigma)
    if out is None:
        return None
    xe, fe = x[e].astype(np.float64), f[e].astype(np.float64)
    out["ampdev_dataset"] = abs(1.0 - out["amplitude"]) if np.isfinite(out["amplitude"]) else np.nan
    if mu_c is not None:
        mc = mu_c[e].astype(np.float64)
        nx = np.linalg.norm(xe - mc)
        out["amplitude_cohort"] = float(np.linalg.norm(fe - mc) / nx) if nx > 0 else np.nan
    else:
        out["amplitude_cohort"] = np.nan
    out["ampdev_cohort"] = abs(1.0 - out["amplitude_cohort"]) if np.isfinite(out["amplitude_cohort"]) else np.nan
    out["mse_V"] = float(np.mean((xe - mu_d[e]) ** 2))
    out["ms_ssim"] = ms_ssim(x, mx, f, e, ws)
    out["persistence"] = persistence_distance(x, f, e)
    return out


# ---------------------------------------------------------------------------------------------- fills

def masked_mean(X, M):
    M = M.astype(np.float64)
    n = M.sum(0)
    s = (X.astype(np.float64) * M).sum(0)
    return np.where(n > 0, s / np.maximum(n, 1), 0.0).astype(np.float32), n > 0


def lambdas():
    """lambda = r = 1 - ratio / 2 of the repeat and fellow rungs in the face-format fundus JSON (Amendment 9.13)."""
    res = json.load(open(FACEFMT_JSON, encoding="utf-8"))["results"]
    return {ds: {c: 1.0 - res[ds][c]["ratio"] / 2 for c in ("repeat", "fellow") if c in res[ds]} for ds in res}


def noise_rng(ti, pi):
    return np.random.default_rng([SEED, int(ti), int(pi), 7])


def nearest(A, MA, b, mb, eligible, k):
    """Indices of the k candidates nearest to b by masked mean squared difference (>= 50% of b's pixels shared)."""
    d = BM.masked_dist(A, MA, b, mb, int(mb.sum()))
    d = np.where(eligible, d, np.inf)
    kk = min(k, int(np.isfinite(d).sum()))
    if kk == 0:
        return np.zeros(0, int)
    idx = np.argpartition(d, kk - 1)[:kk]
    return idx[np.argsort(d[idx])]


W = {}


def _init(state):
    load_bm(state.pop("bm_path"))
    models = state.pop("models_path", None)
    if state.get("extra"):
        EX.setup(models)
        EX.load_nets()
    W.update(state)


def run_target(ti):
    """All rows of one target unit (every position, every fill)."""
    return run_target_sets(ti)[0]


def run_target_sets(ti):
    """Rows of one target unit and, with the Amendment 9.15 metrics on (W['extra']), its set-level window records."""
    X, M, valid, idx = W["X"], W["M"], W["valid"], W["idx"]
    person, side, sex, age, dev = W["person"], W["side"], W["sex"], W["age"], W["dev"]
    fellow, other, pos, lam, name = W["fellow"], W["other"], W["pos"], W["lam"], W["name"]
    S = W["side_units"][side[ti]]
    pool = S[person[S] != person[ti]]
    result_spec.assert_leave_person_out(person[ti], person[pool])
    xt, mt = X[idx[ti]], M[idx[ti]]
    mtv = mt & valid
    Xp, Mp = X[idx[pool]], M[idx[pool]]
    mu, mu_ok = masked_mean(Xp, Mp)
    perm = np.random.default_rng([SEED, int(ti)]).permutation(pool)[:N_STR]
    k2, k2_ok = masked_mean(X[idx[perm[:2]]], M[idx[perm[:2]]])
    coh = pool[(sex[pool] == sex[ti]) & (np.abs(age[pool] - age[ti]) <= 5) & ((dev[pool] == dev[ti])
                                                                             if name == "GRAPE" else True)]
    if len(coh) >= MIN_COHORT:
        mc, mc_n = masked_mean(X[idx[coh]], M[idx[coh]])
        mc_ok = M[idx[coh]].sum(0) >= 0.5 * len(coh)
    else:
        mc, mc_ok = None, None
    fixed = {}
    if other[ti] >= 0:
        fixed["repeat"] = (X[other[ti]], M[other[ti]])
    if fellow[ti] >= 0:
        fixed["fellow"] = (X[idx[fellow[ti]]], M[idx[fellow[ti]]])
    fixed["stranger"] = (X[idx[perm[0]]], M[idx[perm[0]]])
    for k, q in enumerate(perm):
        fixed[f"str{k:02d}"] = (X[idx[q]], M[idx[q]])
    fixed["mean_k2"] = (k2, k2_ok)
    fixed["dataset_mean"] = (mu, mu_ok)
    if mc is not None:
        fixed["sexage"] = (mc, mc_ok)
    for c in ("repeat", "fellow"):
        if c in fixed and c in lam:
            d_, m_ = fixed[c]
            fixed[f"shrink_{c}"] = (BM.shrink(d_, mu, lam[c]), m_ & mu_ok)
    rows = []
    extra = bool(W.get("extra"))
    sets, lp = [], []                       # set-level window records; pending LPIPS jobs (row, ctx, E, in S)
    if extra:
        xnet, setpx = EX.network_input(xt, mtv)
        topo = EX.topo_cells(ti, [p for p, (y_, x_) in enumerate(pos) if mtv[y_:y_ + WIN, x_:x_ + WIN].all()], SEED)
    for pi, (y, x) in enumerate(pos):
        ws = (slice(y, y + WIN), slice(x, x + WIN))
        win = np.zeros((SIZE, SIZE), bool)
        win[ws] = True
        if mtv[ws].mean() < MIN_IN_MASK:
            continue
        in_s = extra and pi in topo
        if extra and mtv[ws].all():
            sets.append({"unit": int(ti), "person": person[ti], "patch": pi, "fill": "_target",
                         "win": xt[ws].astype(np.float16), "feat": None, "in_s": in_s})
        rm = ring_mask(y, x) & valid
        fills = dict(fixed)
        # look-alikes on this position: ring (achievable) and window (oracle), masked distances
        elig = Mp[:, ws[0], ws[1]].reshape(len(pool), -1).mean(1) >= MIN_IN_MASK
        rmv = rm.ravel()
        nn = nearest(Xp.reshape(len(pool), -1)[:, rmv], Mp.reshape(len(pool), -1)[:, rmv], xt.ravel()[rmv],
                     mt.ravel()[rmv], elig, K_LOOK)
        if nn.size:
            fills["lookalike"] = (Xp[nn[0]], Mp[nn[0]])
        if nn.size >= K_LOOK:
            fills["lookalike16"] = masked_mean(Xp[nn], Mp[nn])
        wv = win.ravel()
        no = nearest(Xp.reshape(len(pool), -1)[:, wv], Mp.reshape(len(pool), -1)[:, wv], xt.ravel()[wv],
                     mtv.ravel()[wv], elig, 1)
        if no.size:
            fills["lookalike_oracle"] = (Xp[no[0]], Mp[no[0]])
        n_tw = int((win & mtv).sum())
        e0 = win & mtv & mu_ok
        wd = BM.wrong_detail(xt, mu, fixed["stranger"][0], e0 & fixed["stranger"][1])
        if wd is not None:
            fills["wrong_detail"] = (wd, e0 & fixed["stranger"][1])
        fills["noise"] = (BM.noise_fill(xt, mu, e0, noise_rng(ti, pi)), e0)
        sigma = W["sigma"].get((pi, person[ti])) if W["sigma"] else None
        for fname, (f, mf) in fills.items():
            e = win & mtv & mf.astype(bool) & mu_ok
            if mc is not None:
                e &= mc_ok
            if e.sum() < max(10, MIN_FILL_SHARE * n_tw):        # a fill must cover half the target's window
                continue
            s = score(xt, mtv, f, e, ws, mu, mc, sigma)
            if s is None:
                continue
            row = {"unit": int(ti), "person": person[ti], "side": side[ti], "patch": pi, "fill": fname, **s}
            if extra:
                full = bool(e[ws].all())
                row["topo_subset"] = bool(in_s)
                row["betti_matched"] = row["betti_unmatched"] = np.nan
                if in_s and full:
                    row["betti_matched"], row["betti_unmatched"] = EX.betti_parts(xt[ws], f[ws])
                ctx = EX.context_box(y, x, WIN, SIZE)
                comp = xnet[ctx].copy()
                ec = e[ctx]
                comp[ec] = f[ctx][ec]
                lp.append((len(rows), xnet[ctx], comp, ec, in_s))
                row["lpips_setshare"] = float(setpx[ctx].mean())
                if full and not is_ref_stranger(fname):
                    sets.append({"unit": int(ti), "person": person[ti], "patch": pi, "fill": fname,
                                 "win": f[ws].astype(np.float16), "feat": None, "in_s": in_s})
            rows.append(row)
    if extra:
        if lp:
            refs, comps, masks = [j[1] for j in lp], [j[2] for j in lp], [j[3] for j in lp]
            va = EX.lpips_batch(refs, comps, masks, "alex")
            sel = [k for k, j in enumerate(lp) if j[4]]
            vv = EX.lpips_batch([refs[k] for k in sel], [comps[k] for k in sel], [masks[k] for k in sel], "vgg")
            vgg = dict(zip(sel, vv))
            for k, j in enumerate(lp):
                rows[j[0]]["lpips_alex"] = va[k]
                rows[j[0]]["lpips_vgg"] = vgg.get(k, np.nan)
        need = [k for k, r_ in enumerate(sets) if r_["in_s"]]
        if need:
            feats = EX.inception_features([sets[k]["win"].astype(np.float32) for k in need])
            for k, ft in zip(need, feats):
                sets[k]["feat"] = ft.astype(np.float32)
    return rows, sets


def repeat_sigma(X, M, valid, idx, person, other, pos):
    """sigma_rep per (position, target person): RMS difference of repeat pairs of the other persons (>= 5 pairs)."""
    out = {}
    have = np.flatnonzero(other >= 0)
    if have.size == 0:
        return out
    for pi, (y, x) in enumerate(pos):
        ws = (slice(y, y + WIN), slice(x, x + WIN))
        per = {}
        for i in have:
            a, b = X[idx[i]][ws], X[other[i]][ws]
            m = M[idx[i]][ws] & M[other[i]][ws] & valid[ws]
            if m.sum() < 10:
                continue
            ps = per.setdefault(person[i], [0.0, 0, 0])
            ps[0] += float(((a - b) ** 2)[m].sum())
            ps[1] += int(m.sum())
            ps[2] += 1
        tot = np.array([v[0] for v in per.values()]).sum() if per else 0.0
        cnt = np.array([v[1] for v in per.values()]).sum() if per else 0
        npair = sum(v[2] for v in per.values())
        for p in np.unique(person):
            own = per.get(p, [0.0, 0, 0])
            if npair - own[2] >= 5 and cnt - own[1] > 0:
                out[(pi, p)] = float(np.sqrt((tot - own[0]) / (cnt - own[1])))
    return out


def build(name, arm):
    t, img, fov, u, _, _, fellow, other = D.setup(name, arm)
    X, M = FF.down2(img, fov)
    idx = u.idx.to_numpy()
    cover = M[idx].sum(0)
    valid = cover >= COVER * len(u)
    pos = positions(valid)
    st = {"X": X, "M": M, "valid": valid, "idx": idx, "person": u.person.astype(str).to_numpy(),
          "side": u.side.to_numpy(), "sex": u.sex.astype(str).to_numpy(), "age": u.age.to_numpy(float),
          "dev": u.device.astype(str).to_numpy(), "fellow": fellow, "other": other, "pos": pos, "name": name,
          "lam": lambdas()[name]}
    st["side_units"] = {s: np.flatnonzero(st["side"] == s) for s in ("R", "L")}
    st["sigma"] = repeat_sigma(X, M, valid, idx, st["person"], other, pos)
    return st


def run_rows(name, arm, targets=None, workers=6, checkpoint=False):
    st = build(name, arm)
    n = len(st["idx"])
    targets = list(range(n)) if targets is None else list(targets)
    st["bm_path"] = os.environ.get("BRAIN_METRICS_SRC")
    st["models_path"] = EX.MODELS
    st["extra"] = EX.MODELS is not None
    rows, sets = [], []
    # Checkpoints (private folder): one file per finished target, so a killed run resumes where it stopped. Only
    # this pipeline's own outputs are unpickled.
    ck = os.path.join(PRIVATE, f"partial_{name}_{arm}_{len(targets)}") if checkpoint else None
    todo = targets
    if ck:
        os.makedirs(ck, exist_ok=True)
        have = {int(f[:-7]) for f in os.listdir(ck) if f.endswith(".pkl.gz")}
        for ti in sorted(have & set(targets)):
            with gzip.open(os.path.join(ck, f"{ti}.pkl.gz"), "rb") as fh:
                r, sr = pickle.load(fh)
            rows.extend(r)
            sets.extend(sr)
        todo = [ti for ti in targets if ti not in have]
        print(f"  {name}: resumed {len(targets) - len(todo)} targets from checkpoints", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(st,)) as ex:
        for ti, (r, sr) in zip(todo, ex.map(run_target_sets, todo, chunksize=1)):
            rows.extend(r)
            sets.extend(sr)
            if ck:
                tmp = os.path.join(ck, f"{ti}.tmp")
                with gzip.open(tmp, "wb") as fh:
                    pickle.dump((r, sr), fh)
                os.replace(tmp, os.path.join(ck, f"{ti}.pkl.gz"))
            done += 1
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"  {name}: {done}/{len(todo)} targets, {el / 60:.1f} min, ETA "
                      f"{el / done * (len(todo) - done) / 60:.1f} min", flush=True)
    st.pop("bm_path", None)
    st.pop("models_path", None)
    return pd.DataFrame(rows), st, sets


# ---------------------------------------------------------------------------------------------- statistics

def boot_idx(persons, reps=REPS, seed=SEED):
    return np.random.default_rng(seed).integers(0, len(persons), size=(reps, len(persons)))


def r_scale(r, m, persons, bidx, anchor0="self"):
    """R per fill with a person bootstrap. d = distance (lower = closer); anchor0 'self' (d_self, Amendment F1) or
    'repeat'. Stranger anchor = per-cell mean over the 16 strangers. Returns ({fill: {...}}, {fill: array})."""
    tab = r[["unit", "person", "patch", "fill", m]].dropna()
    tab = tab.assign(d=-oriented(m, tab[m].to_numpy()))
    tab = tab[np.isfinite(tab.d)]
    strs = tab[tab.fill.map(is_ref_stranger)].groupby(["unit", "person", "patch"])["d"].mean().reset_index()
    strs["fill"] = "_anchor"
    tab = pd.concat([tab[~tab.fill.map(is_ref_stranger)][["unit", "person", "patch", "fill", "d"]], strs])
    pi = {p: k for k, p in enumerate(persons)}
    S, C = {}, {}
    for f, g in tab.groupby("fill"):
        agg = g.groupby("person")["d"].agg(["sum", "count"])
        s_, c_ = np.zeros(len(persons)), np.zeros(len(persons))
        for p, row in agg.iterrows():
            s_[pi[p]], c_[pi[p]] = row["sum"], row["count"]
        S[f], C[f] = s_, c_
    if "_anchor" not in S or (anchor0 == "repeat" and "repeat" not in S):
        return None, None
    pt = {f: S[f].sum() / C[f].sum() for f in S}
    bs = {f: S[f][bidx].sum(1) / np.maximum(C[f][bidx].sum(1), 1e-12) for f in S}
    if anchor0 == "self":
        z_pt, z_bs = D_SELF[m], D_SELF[m]
    else:
        z_pt, z_bs = pt["repeat"], bs["repeat"]
    out, arr = {}, {}
    for f in S:
        if f == "_anchor":
            continue
        arr[f] = (bs[f] - z_bs) / (bs["_anchor"] - z_bs)
        out[f] = {"R": float((pt[f] - z_pt) / (pt["_anchor"] - z_pt)),
                  "ci95": [float(np.nanpercentile(arr[f], 2.5)), float(np.nanpercentile(arr[f], 97.5))],
                  "n_targets": int(tab[tab.fill == f].unit.nunique()), "n_persons": int((C[f] > 0).sum())}
    return out, arr


def raw_means(r, m, persons, bidx):
    out = {}
    pi = {p: k for k, p in enumerate(persons)}
    per = r.groupby(["fill", "unit", "person"])[m].mean().reset_index().dropna()
    per = per[np.isfinite(per[m])]
    for f, g in per.groupby("fill"):
        if is_ref_stranger(f):
            continue
        agg = g.groupby("person")[m].agg(["sum", "count"])
        s_, c_ = np.zeros(len(persons)), np.zeros(len(persons))
        for p, row in agg.iterrows():
            s_[pi[p]], c_[pi[p]] = row["sum"], row["count"]
        bs = s_[bidx].sum(1) / np.maximum(c_[bidx].sum(1), 1e-12)
        out[f] = {"mean": float(s_.sum() / c_.sum()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                float(np.percentile(bs, 97.5))],
                  "n_targets": int(c_.sum()), "n_persons": int((c_ > 0).sum())}
    return out


def mse_ratio_block(r, persons, side_of_unit):
    """Spec block: per rung, sum of squared error / sum of the dataset mean's squared error on the same pixels."""
    res = {}
    rr = r.assign(num=r.mse * r.n_vox, den=r.mse_V * r.n_vox)
    for f, g in rr.groupby("fill"):
        if is_ref_stranger(f):
            continue
        pp = g.groupby("person")[["num", "den"]].sum()
        per_person = {p: (row.num, row.den) for p, row in pp.iterrows()}
        n_t = int(g.unit.nunique())
        if n_t < 10:
            res.setdefault("_descriptive_only", {})[f] = {"targets": n_t}
            continue
        ratio, ci = FF.boot_ratio(per_person, np.random.default_rng(SEED))
        res[f] = {"ratio": round(ratio, 3), "ci95": [round(ci[0], 3), round(ci[1], 3)], "targets": n_t,
                  "persons": int(len(pp))}
        if f == "fellow":
            dirs = {}
            for s in ("R", "L"):
                gs = g[g.side == s]
                if len(gs):
                    dirs[f"{s} eye as target"] = {"ratio": round(float(gs.num.sum() / gs.den.sum()), 3),
                                                  "targets": int(gs.unit.nunique())}
            res[f]["directions"] = dirs
    return res


def paired_tests(r, ref, metrics, lam_ok=True):
    """T1-T3b per metric: oriented(ref) - oriented(control), mean per target over windows, then per person;
    one-sided Wilcoxon on person means (>= 10 persons). Returns {metric: {test: entry}}."""
    out = {}
    for m in metrics:
        for tn, ctrl in TESTS.items():
            if tn == "T1" and ref != "repeat":
                out.setdefault(m, {})[tn] = {"status": "not available (no repeat images)"}
                continue
            c = f"shrink_{ref}" if ctrl == "shrink" else ctrl
            w = r[r.fill.isin([ref, c])].pivot_table(index=["unit", "person", "patch"], columns="fill",
                                                      values=m).dropna()
            if ref not in w or c not in w or w.empty:
                out.setdefault(m, {})[tn] = {"status": "not run (no pairs)"}
                continue
            w = w.reset_index()
            w["d"] = oriented(m, w[ref]) - oriented(m, w[c])
            w = w[np.isfinite(w.d)]
            pt = w.groupby(["unit", "person"])["d"].mean().reset_index()
            pm = pt.groupby("person")["d"].mean()
            e = {"control": c, "n_targets": int(len(pt)), "n_persons": int(len(pm)),
                 "mean_diff": float(pt.d.mean()), "share_targets_pass": float((pt.d > 0).mean())}
            if len(pm) >= 10 and np.any(pm.to_numpy() != 0):
                e["p_one_sided"] = float(stats.wilcoxon(pm.to_numpy(), alternative="greater").pvalue)
            else:
                e["status"] = "descriptive only (fewer than 10 persons)"
            out.setdefault(m, {})[tn] = e
    return out


def holm_family(tests_by_metric):
    """Holm over all 26 x 4 confirmatory hypotheses; blocked or unavailable ones stay in the denominator as p = 1."""
    keys, ps = [], []
    for name, (_, col, _) in CONFIRMATORY.items():
        for tn in TESTS:
            e = tests_by_metric.get(col, {}).get(tn, {}) if col else {}
            keys.append((name, tn))
            ps.append(e.get("p_one_sided", 1.0))
    adj = BM.holm(ps)
    return {f"{n}/{t}": {"p_holm": float(a), "pass_holm_0.05": bool(a < 0.05),
                         "computed": "p_one_sided" in (tests_by_metric.get(CONFIRMATORY[n][1], {}).get(t, {})
                                                       if CONFIRMATORY[n][1] else {})}
            for (n, t), a in zip(keys, adj)}, len(ps)


TIES = {"wrong detail = stranger": ("wrong_detail", "stranger"), "same sex-age mean = dataset mean": ("sexage",
                                                                                                    "dataset_mean")}


def tost(arr, R):
    out = {}
    for name, (a, b) in TIES.items():
        if a not in arr or b not in arr:
            out[name] = {"status": "not available"}
            continue
        dif = arr[a] - arr[b]
        lo, hi = float(np.nanpercentile(dif, 5)), float(np.nanpercentile(dif, 95))
        out[name] = {"diff": float(R[a]["R"] - R[b]["R"]), "ci90": [lo, hi], "margin": TOL,
                     "equivalent": bool(-TOL < lo and hi < TOL)}
    return out


EXPECT_FUNDUS = {"E1 fellow eye closer than stranger": ("upper_lt", "fellow", 1.0),
                 "E3B dataset mean better than stranger": ("upper_lt", "dataset_mean", 1.0),
                 "E3B mean of 2 better than stranger": ("upper_lt", "mean_k2", 1.0),
                 "E3B averages ordered by k": ("order", ("dataset_mean", "mean_k2", "stranger"), None),
                 "E5B wrong detail not better than stranger": ("lower_ge", "wrong_detail", 1.0 - TOL),
                 "E6B white noise worse than stranger": ("lower_gt", "noise", 1.0)}


def calibrate(r, metrics):
    """Brain Amendment 12 group 2 on the rows: Tnorm z vs the 16 strangers of the same cell, EVT P-value (GEV on the
    pooled stranger z per position and side), P(same | z) (needs repeats), c = P(stranger worse) + P(tie) / 2."""
    keys = ["unit", "patch"]
    is_ref = r.fill.map(is_ref_stranger).to_numpy()
    is_rep = (r.fill == "repeat").to_numpy()
    out = {}
    for m in metrics:
        o = oriented(m, r[m].to_numpy(np.float64))
        o = np.where(np.isfinite(o), o, np.nan)
        df = r[keys + ["side", "fill"]].assign(o=o)
        ref = df[is_ref]
        g = ref.groupby(keys)["o"].agg(["mean", "std"])
        mi = pd.MultiIndex.from_frame(df[keys])
        mu, sd = g["mean"].reindex(mi).to_numpy(), g["std"].reindex(mi).to_numpy()
        z = (o - mu) / np.where(sd > 0, sd, np.nan)
        piv = ref.pivot_table(index=keys, columns="fill", values="o", aggfunc="first")
        Sm = piv.reindex(mi).to_numpy(np.float64)                  # (rows, 16) stranger values of the same cell
        n_s = np.isfinite(Sm).sum(1)
        with np.errstate(invalid="ignore"):
            c = ((Sm < o[:, None]).sum(1) + 0.5 * (Sm == o[:, None]).sum(1)) / np.where(n_s > 0, n_s, np.nan)
        c = np.where(np.isfinite(o), c, np.nan)
        nlp, post = np.full(len(df), np.nan), np.full(len(df), np.nan)
        for (_, _), ix in df.groupby(["patch", "side"]).indices.items():
            zs = z[ix][is_ref[ix]]
            zs = zs[np.isfinite(zs)]
            zr = z[ix][is_rep[ix]]
            zr = zr[np.isfinite(zr)]
            if zs.size >= 50:
                try:
                    cc, loc, sc = stats.genextreme.fit(zs)
                    nlp[ix] = -np.log10(np.clip(stats.genextreme.sf(z[ix], cc, loc, sc), 1e-300, 1.0))
                except Exception:   # noqa: BLE001 - a failed fit leaves NaN (reported as missing), never 0
                    pass
            if zs.size >= 50 and zr.size >= 5 and np.std(zr) > 0:
                ls = stats.norm.logpdf(z[ix], zs.mean(), zs.std())
                lr = stats.norm.logpdf(z[ix], zr.mean(), zr.std())
                post[ix] = 1.0 / (1.0 + np.exp(np.clip(ls - lr, -700, 700)))
        out[m] = pd.DataFrame({"unit": r.unit.to_numpy(), "person": r.person.to_numpy(), "fill": r.fill.to_numpy(),
                               "z": z, "nlp": nlp, "post": post, "c": c})
    return out


def summarise(name, r, st, sets=None):
    persons = np.array(sorted(r.person.unique()))
    bidx = boot_idx(persons)
    ref = "repeat" if (r.fill == "repeat").any() else "fellow"
    metrics = [m for m in COMPUTED if m in r.columns and r[m].notna().any()]
    s = {"reference_rung_for_tests": ref, "counts": {
        "units": int(len(st["idx"])), "persons": int(len(np.unique(st["person"]))), "positions": len(st["pos"]),
        "valid_px": int(st["valid"].sum()), "rows": int(len(r)),
        "targets_per_fill": {f: int(g.unit.nunique()) for f, g in r.groupby("fill") if not is_ref_stranger(f)}},
        "lambda_shrink": st["lam"]}
    s["mse_ratio"] = mse_ratio_block(r, persons, None)
    s["raw"], s["R_self"], s["R_repeat"], s["expectations"], s["tost"] = {}, {}, {}, {}, {}
    for m in metrics + ["amplitude", "amplitude_cohort"]:
        s["raw"][m] = raw_means(r, m, persons, bidx)
    for m in metrics:
        if m in NO_R:
            continue
        R, arr = r_scale(r, m, persons, bidx, "self")
        if R is None:
            continue
        s["R_self"][m] = R
        s["expectations"][m] = BM.expectations(R, arr, EXPECT_FUNDUS)
        s["tost"][m] = tost(arr, R)
        if ref == "repeat":
            R2, _ = r_scale(r, m, persons, bidx, "repeat")
            if R2 is not None:
                s["R_repeat"][m] = R2
    tests = paired_tests(r, ref, metrics)
    holm, n_h = holm_family(tests)
    s["tests"] = {"per_metric": tests, "holm": holm, "n_hypotheses_holm": n_h}
    if ref == "repeat" and (r.fill == "fellow").any():
        s["tests_fellow_reference"] = paired_tests(r, "fellow", metrics)
    cal = calibrate(r, [m for m in metrics if m not in ("ampdev_dataset", "ampdev_cohort")])
    s["calibrated"] = {}
    for m, cdf in cal.items():
        per = cdf.groupby(["fill", "unit", "person"])[["z", "nlp", "post", "c"]].mean().reset_index()
        blk = {}
        for f, g in per.groupby("fill"):
            if is_ref_stranger(f):
                continue
            blk[f] = {}
            for k in ("z", "nlp", "post", "c"):
                v = g.dropna(subset=[k])
                if v.empty:
                    blk[f][k] = None
                    continue
                pp = v.groupby("person")[k].agg(["sum", "count"]).reindex(persons).fillna(0)
                bs = pp["sum"].to_numpy()[bidx].sum(1) / np.maximum(pp["count"].to_numpy()[bidx].sum(1), 1e-12)
                blk[f][k] = {"mean": float(v[k].mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                   float(np.percentile(bs, 97.5))],
                             "n_targets": int(len(v))}
        s["calibrated"][m] = blk
    if ref == "repeat":
        s["position_mean0_repeat1"] = {}
        for m in metrics:
            if m in NO_R and m not in ("psnr", "psnr_fixed"):
                continue
            raw = s["raw"][m]
            if all(k in raw for k in ("repeat", "dataset_mean")):
                den = raw["repeat"]["mean"] - raw["dataset_mean"]["mean"]
                if abs(den) > 1e-12:
                    s["position_mean0_repeat1"][m] = {f: (v["mean"] - raw["dataset_mean"]["mean"]) / den
                                                      for f, v in raw.items()}
    if "topo_subset" in r.columns:
        s["amendment_9_15"] = extra_blocks(r, st, persons, bidx, sets or [])
    return s


def extra_blocks(r, st, persons, bidx, sets):
    """Amendment 9.15 reporting: eligibility per rung and reason, LPIPS sensitivity (<= 5% set pixels), set-level
    FID and FWD per rung."""
    ts = r.topo_subset.astype(bool)
    out = {"topology_subset": {"positions_per_unit": EX.TOPO_PER_UNIT,
                               "units_without_full_window": int(len(st["idx"]) - r[ts].unit.nunique())}}
    elig = {}
    for f, g in r.groupby("fill"):
        if is_ref_stranger(f):
            continue
        gs = g[g.topo_subset.astype(bool)]
        elig[f] = {"cells": int(len(g)), "cells_in_S": int(len(gs)),
                   "betti_scored": int(gs.betti_matched.notna().sum()),
                   "betti_not_scored_window_not_full": int(gs.betti_matched.isna().sum()),
                   "lpips_scored": int(g.lpips_alex.notna().sum()),
                   "lpips_cells_setshare_le_0.05": int((g.lpips_setshare <= EX.SET_SHARE_MAX).sum())}
    out["eligibility"] = elig
    rs = r[r.lpips_setshare <= EX.SET_SHARE_MAX]
    R, _ = r_scale(rs, "lpips_alex", persons, bidx, "self") if len(rs) else (None, None)
    out["lpips_alex_R_self_setshare_le_0.05"] = R if R is not None else {"status": "not run (no cells)"}
    out["set_level"] = EX.set_level(sets, persons, SEED)
    return out


def finite(o):
    if isinstance(o, dict):
        return {str(k): finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


# ---------------------------------------------------------------------------------------------- QC sheet

QC_FILLS = ("repeat", "fellow", "stranger", "mean_k2", "sexage", "dataset_mean", "lookalike", "lookalike16",
            "lookalike_oracle", "wrong_detail", "noise", "shrink_fellow")


def qc_sheet(name, st, targets, path, cache=None):
    """Private QC sheet: per evenly spaced target, the window and its context (48 px) with each fill pasted in;
    outside the scored pixels is grey. Real images: private folder only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W.clear()
    W.update(st)
    pos = st["pos"]
    cmap = matplotlib.colormaps["gray"].copy()
    cmap.set_bad("#9ecae1")                                  # missing = light blue (distinct from a white disc)
    cols = ("target",) + QC_FILLS
    fig, ax = plt.subplots(len(targets), len(cols), figsize=(1.3 * len(cols), 1.45 * len(targets) + 0.6))
    ax = np.atleast_2d(ax)
    for r_, ti in enumerate(targets):
        rows = cache[ti] if cache is not None and ti in cache else pd.DataFrame(run_target(ti))
        if cache is not None:
            cache[ti] = rows
        if rows.empty:
            continue
        pats = sorted(rows.patch.unique())
        pi = int(pats[(r_ * 5) % len(pats)])                 # positions vary down the sheet
        y, x = pos[pi]
        y0, x0 = max(y - 16, 0), max(x - 16, 0)
        sl = (slice(y0, y0 + 48), slice(x0, x0 + 48))
        xt = st["X"][st["idx"][ti]]
        mtv = st["M"][st["idx"][ti]] & st["valid"]
        fills = _fill_images(ti, pi)
        for c_, key in enumerate(cols):
            a = ax[r_, c_]
            a.set_xticks([])
            a.set_yticks([])
            if r_ == 0:
                a.set_title(key.replace("lookalike", "look-al.").replace("_", " "), fontsize=6.5)
            if key == "target":
                img = np.where(mtv, xt, np.nan)
            elif key in fills:
                f, e = fills[key]
                img = np.where(mtv, xt, np.nan)
                img[y:y + WIN, x:x + WIN] = np.where(e[y:y + WIN, x:x + WIN], f[y:y + WIN, x:x + WIN], np.nan)
            else:
                a.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7, transform=a.transAxes)
                continue
            a.imshow(img[sl], cmap=cmap, vmin=-2.5, vmax=2.5)
            n_t = max(int(mtv[y:y + WIN, x:x + WIN].sum()), 1)
            if key in fills and fills[key][1][y:y + WIN, x:x + WIN].sum() < MIN_FILL_SHARE * n_t:
                a.text(0.5, 0.5, "not scored\n(< 50% of window)", ha="center", va="center", fontsize=5.5,
                       color="#a00", transform=a.transAxes)
            a.add_patch(plt.Rectangle((x - x0 - 0.5, y - y0 - 0.5), WIN, WIN, fill=False, ec="#d33", lw=0.6))
            v = rows[(rows.patch == pi) & (rows.fill == key)]
            if key != "target" and len(v):
                v = v.iloc[0]
                a.set_xlabel(f"mse/V {v.mse / v.mse_V:.2f}\nssim {v.ssim:.2f} ms {v.ms_ssim:.2f}", fontsize=5.5)
        ax[r_, 0].set_ylabel(f"target {r_ + 1}\n{st['side'][ti]} eye, pos {pi}", fontsize=6)
    fig.suptitle(f"{name}: QC of the metric fills (window red, 48 px context; light blue = missing). Private.", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


QC_EXTRA = ("repeat", "fellow", "stranger", "lookalike", "mean_k2", "dataset_mean", "wrong_detail", "noise",
            "shrink_fellow")


def qc_extra(name, st, targets, path, cache):
    """Private QC sheet of Amendment 9.15: per evenly spaced target, one cell of its topology subset S: the 64 px
    LPIPS context as the network sees it (set pixels tinted blue), the composite per rung with LPIPS (AlexNet) and
    both Betti-matching parts; the window is outlined red."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W.clear()
    W.update(st)
    W["extra"] = True
    cols = ("network input",) + QC_EXTRA
    fig, ax = plt.subplots(len(targets), len(cols), figsize=(1.45 * len(cols), 1.6 * len(targets) + 0.6))
    ax = np.atleast_2d(ax)
    blue = np.zeros((EX.LP_CTX, EX.LP_CTX, 4))
    blue[..., 2], blue[..., 1] = 1.0, 0.55
    for r_, ti in enumerate(targets):
        rows = cache[ti] if ti in cache else pd.DataFrame(run_target(ti))
        cache[ti] = rows
        for a in ax[r_]:
            a.set_xticks([])
            a.set_yticks([])
        if rows.empty or not rows.topo_subset.astype(bool).any():
            ax[r_, 0].text(0.5, 0.5, "no full window", ha="center", va="center", fontsize=6,
                           transform=ax[r_, 0].transAxes)
            continue
        pi = int(sorted(rows[rows.topo_subset.astype(bool)].patch.unique())[0])
        y, x = st["pos"][pi]
        xt = st["X"][st["idx"][ti]]
        mtv = st["M"][st["idx"][ti]] & st["valid"]
        xnet, setpx = EX.network_input(xt, mtv)
        ctx = EX.context_box(y, x, WIN, SIZE)
        fills = _fill_images(ti, pi)
        ov = blue.copy()
        ov[..., 3] = 0.45 * setpx[ctx]
        for c_, key in enumerate(cols):
            a = ax[r_, c_]
            if r_ == 0:
                a.set_title(key.replace("lookalike", "look-al.").replace("_", " "), fontsize=6.5)
            if key == "network input":
                img = xnet[ctx]
            elif key in fills:
                f, e = fills[key]
                img = xnet[ctx].copy()
                ec = e[ctx]
                img[ec] = f[ctx][ec]
            else:
                a.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7, transform=a.transAxes)
                continue
            a.imshow(img, cmap="gray", vmin=-3, vmax=3)
            a.imshow(ov)
            a.add_patch(plt.Rectangle((x - ctx[1].start - 0.5, y - ctx[0].start - 0.5), WIN, WIN, fill=False,
                                      ec="#d33", lw=0.6))
            v = rows[(rows.patch == pi) & (rows.fill == key)]
            if key != "network input" and len(v):
                v = v.iloc[0]
                bm = (f"B m {v.betti_matched:.3f} u {v.betti_unmatched:.3f}" if np.isfinite(v.betti_matched)
                      else "Betti: window not full")
                a.set_xlabel(f"LPIPS {v.lpips_alex:.3f}\n{bm}", fontsize=5.2)
            elif key != "network input":
                a.set_xlabel("not scored", fontsize=5.5, color="#a00")
        ax[r_, 0].set_ylabel(f"target {r_ + 1}\n{st['side'][ti]} eye, pos {pi}\nset px "
                             f"{setpx[ctx].mean():.0%}", fontsize=6)
    fig.suptitle(f"{name}: QC of the Amendment 9.15 metrics (64 px LPIPS context, window red; blue = set pixels "
                 f"outside the field, never scored). Private.", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fill_images(ti, pi):
    """Rebuild the fill images of one target at one position (same rules and seeds as run_target)."""
    X, M, valid, idx = W["X"], W["M"], W["valid"], W["idx"]
    person, side = W["person"], W["side"]
    S = W["side_units"][side[ti]]
    pool = S[person[S] != person[ti]]
    xt, mt = X[idx[ti]], M[idx[ti]]
    mtv = mt & valid
    Xp, Mp = X[idx[pool]], M[idx[pool]]
    mu, mu_ok = masked_mean(Xp, Mp)
    perm = np.random.default_rng([SEED, int(ti)]).permutation(pool)[:N_STR]
    y, x = W["pos"][pi]
    win = np.zeros((SIZE, SIZE), bool)
    win[y:y + WIN, x:x + WIN] = True
    out = {}
    if W["other"][ti] >= 0:
        out["repeat"] = (X[W["other"][ti]], M[W["other"][ti]])
    if W["fellow"][ti] >= 0:
        out["fellow"] = (X[idx[W["fellow"][ti]]], M[idx[W["fellow"][ti]]])
    out["stranger"] = (X[idx[perm[0]]], M[idx[perm[0]]])
    out["mean_k2"] = masked_mean(X[idx[perm[:2]]], M[idx[perm[:2]]])
    out["dataset_mean"] = (mu, mu_ok)
    sex, age, dev = W["sex"], W["age"], W["dev"]
    coh = pool[(sex[pool] == sex[ti]) & (np.abs(age[pool] - age[ti]) <= 5) & ((dev[pool] == dev[ti])
                                                                             if W["name"] == "GRAPE" else True)]
    mc_ok = np.ones((SIZE, SIZE), bool)
    if len(coh) >= MIN_COHORT:
        mc, _ = masked_mean(X[idx[coh]], M[idx[coh]])
        mc_ok = M[idx[coh]].sum(0) >= 0.5 * len(coh)
        out["sexage"] = (mc, mc_ok)
    rm = ring_mask(y, x) & valid
    elig = Mp[:, y:y + WIN, x:x + WIN].reshape(len(pool), -1).mean(1) >= MIN_IN_MASK
    rmv = rm.ravel()
    nn = nearest(Xp.reshape(len(pool), -1)[:, rmv], Mp.reshape(len(pool), -1)[:, rmv], xt.ravel()[rmv],
                 mt.ravel()[rmv], elig, K_LOOK)
    if nn.size:
        out["lookalike"] = (Xp[nn[0]], Mp[nn[0]])
    if nn.size >= K_LOOK:
        out["lookalike16"] = masked_mean(Xp[nn], Mp[nn])
    wv = win.ravel()
    no = nearest(Xp.reshape(len(pool), -1)[:, wv], Mp.reshape(len(pool), -1)[:, wv], xt.ravel()[wv],
                 mtv.ravel()[wv], elig, 1)
    if no.size:
        out["lookalike_oracle"] = (Xp[no[0]], Mp[no[0]])
    e0 = win & mtv & mu_ok
    wd = BM.wrong_detail(xt, mu, out["stranger"][0], e0 & out["stranger"][1])
    if wd is not None:
        out["wrong_detail"] = (wd, e0 & out["stranger"][1])
    out["noise"] = (BM.noise_fill(xt, mu, e0, noise_rng(ti, pi)), e0)
    if "fellow" in out and "fellow" in W["lam"]:
        out["shrink_fellow"] = (BM.shrink(out["fellow"][0], mu, W["lam"]["fellow"]), out["fellow"][1] & mu_ok)
    return {k: (f, win & mtv & m.astype(bool) & mu_ok & mc_ok) for k, (f, m) in out.items()}


# ---------------------------------------------------------------------------------------------- main

def setup_external():
    ext = os.environ.get("LADDER_EXTERNAL")
    if not ext:
        raise SystemExit("LADDER_EXTERNAL must point to a checkout's external/ folder (GRAPE metadata)")
    F.S.GRAPE_META = os.path.join(ext, *os.path.relpath(F.S.GRAPE_META, os.path.join(
        os.path.dirname(F.S.__file__), "..", "external")).split(os.sep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("smoke", "run"))
    ap.add_argument("--every", type=int, default=10, help="smoke: every N-th target unit")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    setup_external()
    load_bm()
    EX.setup()
    EX.load_nets()                                  # main process: QC sheets
    os.makedirs(PRIVATE, exist_ok=True)
    t0 = time.time()
    results, panel = {}, {}
    for name, arm in FF.SETS:
        units = D.setup(name, arm)[3]
        n, side = len(units), units.side.to_numpy()
        targets = (sorted(np.concatenate([np.flatnonzero(side == s_)[::a.every] for s_ in ("R", "L")]).tolist())
                   if a.mode == "smoke" else None)
        r, st, sets = run_rows(name, arm, targets, a.workers, checkpoint=a.mode == "run")
        tag = "smoke" if a.mode == "smoke" else "full"
        r.to_csv(os.path.join(PRIVATE, f"rows_{name}_{arm}_{tag}.csv.gz"), index=False)
        print(name, "rows", len(r), "targets", r.unit.nunique(), "positions", len(st["pos"]),
              f"{time.time() - t0:.0f}s", flush=True)
        if a.mode == "smoke":
            qc_targets = list(np.linspace(0, n - 1, 12).round().astype(int))
            cache = {}
            st["extra"] = True
            qc_extra(name, st, qc_targets, os.path.join(PRIVATE, f"qc_extra_{name}_{arm}.png"), cache)
            qc_sheet(name, st, qc_targets, os.path.join(PRIVATE, f"qc_fills_{name}_{arm}.png"), cache)
        s = summarise(name, r, st, sets)
        panel[name] = {"arm": arm, **s}
        results[name] = {"arm": arm, **{k: v for k, v in s["mse_ratio"].items()}}
    doc = {"protocol": "docs/PROTOCOL_donor_ladder_general.md, Amendments 9.14 and 9.15", "status": "pre-QC",
           "mode": a.mode, "spec": result_spec.SPEC,
           "spec_meta": {"bootstrap_unit": "person", "pools": "leave-person-out", "stranger_rung": "stranger",
                         "mirror_rungs": {"fellow": "sides"}},
           "scale": ("results: squared error / squared error of the dataset mean on the same pixels (1 = mean, 2 = "
                     "stranger); panel: R_self = 0 for the target itself, 1 for the 16 strangers (Amendment F1)"),
           "frame": "128 px, GRAPE P8, PAPILA P9, frame only (Amendment 9.13)", "window_px": WIN, "ring_px": RING,
           "seed": SEED, "bootstrap_reps": REPS,
           "brain_metrics_source_sha256": BM_SHA,
           "metrics_confirmatory": {k: {"group": g, "column": c if c not in SET_LEVEL else None,
                                        "status": ("computed (set level, per rung; no per-target value, Holm p = 1)"
                                                   if c in SET_LEVEL else "computed" if c else "blocked"),
                                        **({"reason": b} if b else {})} for k, (g, c, b) in CONFIRMATORY.items()},
           "models": {"lpips": "lpips 0.1.4 AlexNet (VGG sensitivity on S)", "betti": "Betti-matching 2D 3e7eca26",
                      "fid": "pytorch-fid 0.3.0 pt_inception-2015-12-05", "fwd": "pytorchfwd 1.0.1 Haar level 2"},
           "metrics_supplementary": list(SUPPLEMENTARY), "lower_is_better": sorted(LOWER),
           "results": results, "panel": panel, "minutes": round((time.time() - t0) / 60, 1)}
    doc = finite(doc)
    for flag in result_spec.enforce(doc):
        print("SPEC (soft):", flag, flush=True)
    out = OUT_JSON if a.mode == "run" else os.path.join(PRIVATE, "fundus_ladder_metrics_smoke.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
