"""Tier-1 metric audit on the face donor ladder (docs/PROTOCOL_face_metrics.md). Pre-QC.

Face replication of the brain audit (PROTOCOL_donor_ladder_brain.md, Amendments 10 to 14). People, targets, pool,
repeats, siblings, coverage and cells are rebuilt exactly as `donor_ladder_face.maps` (half "all") builds them, from
the frame version given on the command line (no default: version-dependent behaviour never falls back silently).
Metrics are the brain functions, imported unchanged; a 2D window is passed as a (1, H, W) array.

Every metric is scored only on pixels inside the window, the target mask, the fill's validity and the coverage
`valid` (missing is never zero). Rows, caches and QC images stay in the private folder; the public JSON holds
aggregates only (no image, file name or identity id).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain_metrics as BM  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
FACE_SRC = os.environ.get("FACE_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "donor_ladder_face.py"))
FACE_RESULTS = os.path.join(os.path.dirname(os.path.dirname(FACE_SRC)), "results")
PRIVATE = os.path.join(os.environ.get("DATA_ROOT", r"C:\Projects\data"), "projects", "identity-ladder-metric",
                       "face-ladder-metrics-20261106-v1")
OUT_JSON = os.path.join(ROOT, "results", "face_ladder_metrics.json")

SEED = BM.SEED
SIZE = 128
WIN, RING = 16, 8                  # window (face map window, 0.4 inter-ocular distance) and its context ring
MIN_POS_VALID, MIN_RING_VALID = 0.95, 0.5
MIN_IN_MASK = 0.5                  # share of a window inside an image's own mask (and valid) to take part
STRIP = 2.5                        # face Amendment 5.4: mirror pixels with |x - 63.5| < 2.5 removed
N_STR = BM.N_STRANGERS
POOL_SUB = 219                     # QTIM pool size (brain Amendment 11), cross-modality sensitivity
SRC_CODE = {"celeba": 0, "imdb": 1}
IDENTITY = "arcface_cos"           # identity reference (higher = same person), not a tier-1 metric
METRICS = tuple(BM.METRICS)
REPORTED = METRICS + (IDENTITY,)


# Snapshot files per version (the owner session, 6 Nov 2026: v6 is safe to read; images.json, landmarks.npz and
# identity_v7.npz are being rewritten for v7 and must not be read until v7 is approved).
# v6f (owner, 6 Nov 2026): v6 selection with the final masks (segmenter + glasses); same image rows as images_v6.json.
SNAPSHOT = {"_v6": {"images": "images_v6.json", "landmarks": "landmarks_v6.npz", "identity": ("identity.npz",)},
            "_v6f": {"images": "images_v6.json", "landmarks": "landmarks_v6.npz", "identity": ("identity.npz",)}}


def snapshot(version):
    if version not in SNAPSHOT:
        raise SystemExit(f"no read-only snapshot recorded for {version}; ask the face-pipeline owner first")
    return SNAPSHOT[version]


def load_face(version):
    """The face pipeline module, read only, with its analysed version set explicitly."""
    spec = importlib.util.spec_from_file_location("donor_ladder_face", FACE_SRC)
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    F.SUF = version
    return F


# ---------------------------------------------------------------------------------------------------- io

def npz_rows(path, key, rows):
    """Rows `rows` (sorted unique ints) of array `key` in a compressed .npz, streamed so that only those rows are
    held in memory. Returns an array of shape (len(rows), *row_shape) in the stored dtype."""
    rows = np.asarray(rows, int)
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as fh:
        ver = np.lib.format.read_magic(fh)
        shape, fortran, dtype = np.lib.format._read_array_header(fh, ver)
        if fortran:
            raise ValueError(f"{key}: Fortran order not supported")
        per = int(np.prod(shape[1:])) * dtype.itemsize
        out = np.empty((len(rows),) + tuple(shape[1:]), dtype)
        want = {int(r): k for k, r in enumerate(rows)}
        for r in range(int(rows.max()) + 1 if len(rows) else 0):
            buf = fh.read(per)
            if len(buf) != per:
                raise IOError(f"{key}: truncated at row {r}")
            k = want.get(r)
            if k is not None:
                out[k] = np.frombuffer(buf, dtype).reshape(shape[1:])
    return out


def npz_small(path, key):
    with np.load(path) as z:
        return z[key]


# ---------------------------------------------------------------------------------------------------- geometry

def ring_mask(y0, x0, S=SIZE, win=WIN, ring=RING):
    m = np.zeros((S, S), bool)
    m[max(y0 - ring, 0):y0 + win + ring, max(x0 - ring, 0):x0 + win + ring] = True
    m[y0:y0 + win, x0:x0 + win] = False
    return m


def positions(valid, win=WIN, ring=RING):
    """Window corners (y0, x0) on a win-px grid with >= 95% of the window and >= 50% of its nominal ring (pixels
    outside the frame count as missing) in `valid`."""
    S = valid.shape[0]
    nominal = (win + 2 * ring) ** 2 - win * win
    out = []
    for y0 in range(0, S - win + 1, win):
        for x0 in range(0, S - win + 1, win):
            if valid[y0:y0 + win, x0:x0 + win].mean() < MIN_POS_VALID:
                continue
            if valid[ring_mask(y0, x0, S, win, ring)].sum() < MIN_RING_VALID * nominal:
                continue
            out.append((y0, x0))
    return out


def midline_keep(S=SIZE):
    return np.abs(np.arange(S) - (S - 1) / 2) >= STRIP


def mirror_fill(x, m):
    """Own mirror: flip about the frame midline; pixels whose partner is within the strip are not valid."""
    return x[:, ::-1], m[:, ::-1] & midline_keep(x.shape[1])[None, :]


ILLUM_SIGMA = 16.0                 # the face pipeline's illumination sigma (Amendment F1, aug_light)


def shift_fill(x, m, dy, dx):
    """aug_shift: the image translated by (dy, dx) px, bilinear on the masked values; a pixel is valid only if its
    whole interpolation footprint was inside the mask (no stored zero bleeds in)."""
    f = ndimage.shift(np.where(m, x, 0.0).astype(np.float64), (dy, dx), order=1, mode="constant", cval=0.0)
    w = ndimage.shift(m.astype(np.float64), (dy, dx), order=1, mode="constant", cval=0.0)
    return f.astype(np.float32), w >= 1 - 1e-6


def light_fill(xt, mt, xr, mr, sigma=ILLUM_SIGMA):
    """aug_light: the target plus the low-frequency difference between its retest and itself (mask-normalized
    Gaussian over pixels valid in both); valid where both are valid."""
    w = (mt & mr).astype(np.float64)
    num = ndimage.gaussian_filter((xr.astype(np.float64) - xt) * w, sigma, mode="constant")
    den = ndimage.gaussian_filter(w, sigma, mode="constant")
    low = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    return (xt + low).astype(np.float32), (mt & mr)


def shift_direction(src, ti):
    a = np.random.default_rng([SEED, SRC_CODE[src], ti, 7]).uniform(0, 2 * np.pi)
    return np.sin(a), np.cos(a)


def nearest(A, MA, b, mb, eligible):
    """Index (into A) of the row nearest to b over pixels valid in both (brain masked_dist rule: >= 50% of the
    target's available pixels shared), among eligible rows; None if none qualifies. Stored values outside the masks
    never enter."""
    d = BM.masked_dist(A, MA, b, mb, int(mb.sum()))
    d = np.where(eligible, d, np.inf)
    return int(np.argmin(d)) if np.isfinite(d).any() else None


def cos(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------------------------------- selection

def cell_of(d, src):
    return (d["male"], d["young"]) if src == "celeba" else (d["male"], int(d["age"] // 5))


def build(F, version, cache_dir):
    """Rebuild maps() half 'all' for both sources, write the rows the scoring needs as .npy (memory-mapped by the
    workers) and return per-source descriptors plus the whole-face reproduction of the face JSON ratios."""
    snap = snapshot(version)
    imgs = json.load(open(os.path.join(F.OUT, snap["images"]), encoding="utf-8"))
    fpath = os.path.join(F.OUT, f"frame{F.frame_suf()}.npz")
    idx = npz_small(fpath, "idx")
    with np.load(os.path.join(F.OUT, snap["landmarks"])) as lz:
        PE = np.hstack([lz["P"][idx, :2], lz["E"][idx]])
    meta = [imgs[j] for j in idx]
    pmax = F.map_pose_max()
    pose_ok = (np.abs(PE[:, 0]) < pmax) & (np.abs(PE[:, 1]) < pmax)
    kin = json.load(open(F.KIN, encoding="utf-8"))["pairs"]
    lab = json.load(open(os.path.join(os.path.dirname(F.KIN), "qualifier_labels.json"), encoding="utf-8"))
    full_sib = [(p["a"], p["b"]) for p in kin if p["relation"] == "sibling"
                and not any("half" in lab.get(q, "") or "step" in lab.get(q, "") for q in p["kinship_qualifiers"])]
    kin_people = {x for q in kin for x in (q["a"], q["b"])}
    fam_pairs = full_sib + [(q["a"], q["b"]) for q in kin if q["relation"] == "parent-child"]
    sel = {}
    need = set()
    for src in ("celeba", "imdb"):
        by_person = {}
        for i, d in enumerate(meta):
            if d["src"] == src and pose_ok[i]:
                by_person.setdefault(d["person"], []).append(i)
        best = {p: min(ii, key=lambda i: (PE[i, :2] ** 2).sum()) for p, ii in by_person.items()}
        pool_p = sorted(p for p, ii in by_person.items() if meta[ii[0]]["role"] == "pool")
        sibs = {}
        for a, b in full_sib:
            if a in by_person and b in by_person:
                sibs.setdefault(a, set()).add(b)
                sibs.setdefault(b, set()).add(a)
        targets = sorted(p for p, ii in by_person.items() if meta[ii[0]]["role"] == "target" and len(ii) >= 2
                         and (src == "celeba" or p in kin_people))
        fam = F._families(targets + pool_p, fam_pairs if src == "imdb" else [])
        sel[src] = dict(by_person=by_person, best=best, pool_p=pool_p, sibs=sibs, targets=targets, fam=fam)
        for p in targets:
            need.update(by_person[p])
            for q in sibs.get(p, ()):
                need.update(by_person[q])
        need.update(best[p] for p in pool_p)
    need = np.array(sorted(need))
    t0 = time.time()
    X16 = npz_rows(fpath, "X", need)
    Mall = npz_rows(fpath, "M", need).astype(bool)
    A = npz_rows(fpath, "A", need) if version >= "_v6" else None
    print(f"read {len(need)} frame rows in {time.time() - t0:.0f}s", flush=True)
    loc = {int(r): k for k, r in enumerate(need)}
    emb_all = load_embeddings(F, idx[need], snap["identity"])
    tmpl = npz_small(fpath, "template_shape").astype(np.float64)
    Lrows = {}

    def xf(i):
        return X16[loc[i]].astype(np.float32)

    def mk(i):
        return Mall[loc[i]]

    used, data, repro = [], {}, {}
    def use(i):          # position of a frame row in the cache
        used.append(i)
        return len(used) - 1
    for src, s in sel.items():
        by_person, best, fam = s["by_person"], s["best"], s["fam"]
        prow = [best[p] for p in s["pool_p"]]
        p0 = len(used)
        for i in prow:
            use(i)
        pool = (p0, len(used))                               # contiguous slice of the cache
        Xp = X16[[loc[i] for i in prow]].astype(np.float32)
        Mp = Mall[[loc[i] for i in prow]]
        mean, cov = F.masked_mean(Xp, Mp)
        mean = mean.astype(np.float32)
        if version >= "_v6":                                 # face Amendment 7 coverage
            valid = (A[[loc[i] for i in prow]].mean(0) >= 0.9) & (cov >= 0.5 * len(prow))
        else:
            valid = cov >= 0.9 * len(prow)
        cells = {}
        for i in prow:
            cells.setdefault(cell_of(meta[i], src), []).append(i)
        cell_mean = {}
        for c, ii in cells.items():
            if len(ii) >= 10:
                m_, n_ = F.masked_mean(X16[[loc[i] for i in ii]].astype(np.float32), Mall[[loc[i] for i in ii]])
                cell_mean[c] = (m_.astype(np.float32), n_ > 0)
        del Xp, Mp
        T, sums = [], {c: [] for c in ("repeat", "mirror", "sibling", "sexage")}
        for p in s["targets"]:
            t = best[p]
            xt, mt = xf(t), mk(t)
            others = [i for i in by_person[p] if i != t and F._corr(X16[loc[i]], X16[loc[t]], mk(i) & mt) <= F.DUP_R]
            if not others:
                continue
            rep = min(others, key=lambda i: F._pose_expr_dist(PE[i], PE[t]))
            sib = []
            if src == "imdb" and p in s["sibs"]:
                sib = [min(by_person[q], key=lambda i: F._pose_expr_dist(PE[i], PE[t])) for q in sorted(s["sibs"][p])]
            c = cell_of(meta[t], src)
            Lrows[(t, rep)] = None
            T.append({"t": use(t), "rep": use(rep), "sib": use(sib[0]) if sib else None, "fam": fam[p],
                      "cell": str(c) if c in cell_mean else None})
            vt = mt & valid
            fl = {"repeat": [(xf(rep), mk(rep))], "mirror": [mirror_fill(xt, mt)],
                  "sibling": [(xf(i), mk(i)) for i in sib]}
            if c in cell_mean:
                fl["sexage"] = [(cell_mean[c][0], valid)]
            for col, lst in fl.items():
                if lst:
                    nd = [(((xt - f) ** 2 * (vt & mf)).sum(), ((xt - mean) ** 2 * (vt & mf)).sum()) for f, mf in lst]
                    sums[col].append((float(np.mean([a for a, _ in nd])), float(np.mean([b for _, b in nd]))))
        repro[src] = {c: round(sum(a for a, _ in v) / sum(b for _, b in v), 3) for c, v in sums.items() if len(v) >= 10}
        fid = {f: k for k, f in enumerate(sorted({d["fam"] for d in T}))}
        for d in T:
            d["family"] = fid[d.pop("fam")]
        np.save(os.path.join(cache_dir, f"{src}_mean.npy"), mean)
        np.save(os.path.join(cache_dir, f"{src}_mean_ok.npy"), cov > 0)
        np.save(os.path.join(cache_dir, f"{src}_valid.npy"), valid)
        pairs = [(t_, r_) for (t_, r_) in Lrows if meta[t_]["src"] == src]
        rows_l = np.array(sorted({int(idx[i]) for pr in pairs for i in pr}))
        L = npz_rows(os.path.join(F.OUT, snap["landmarks"]), "L", rows_l).astype(np.float64)
        lk = {int(r_): k for k, r_ in enumerate(rows_l)}

        def framed(i):
            Li = L[lk[int(idx[i])]]
            return F.apply(F.affine_fit(Li, tmpl), Li)
        dist = [float(np.sqrt(((framed(t_) - framed(r_)) ** 2).sum(1).mean())) for t_, r_ in pairs]
        shift_px = float(np.median(dist))
        print(f"{src}: aug_shift amplitude {shift_px:.2f} px (median RMS landmark distance target-retest, "
              f"{len(dist)} pairs)", flush=True)
        data[src] = {"targets": T, "pool": pool, "valid": valid, "n_pool": len(prow), "shift_px": shift_px,
                     "cells": {str(c): (m_, ok_) for c, (m_, ok_) in cell_mean.items()}}
    U = np.array([loc[i] for i in used])
    np.save(os.path.join(cache_dir, "X.npy"), X16[U])
    np.save(os.path.join(cache_dir, "M.npy"), Mall[U])
    np.save(os.path.join(cache_dir, "emb.npy"), emb_all[U])
    for src, d in data.items():   # cell means stacked, referenced by index
        names = sorted(d["cells"])
        np.save(os.path.join(cache_dir, f"{src}_cells_m.npy"), np.stack([d["cells"][c][0] for c in names]))
        np.save(os.path.join(cache_dir, f"{src}_cells_ok.npy"), np.stack([d["cells"][c][1] for c in names]))
        for t in d["targets"]:
            t["cell"] = names.index(t["cell"]) if t["cell"] is not None else None
        d["cell_names"] = names
        del d["cells"]
    return data, repro


def load_embeddings(F, img_rows, names):
    """ArcFace embeddings of the given images.json rows from the face pipeline's identity file(s), NaN if absent.
    No model is run."""
    out = np.full((len(img_rows), 512), np.nan, np.float32)
    for name in names:
        p = os.path.join(F.OUT, name)
        if not os.path.exists(p):
            continue
        e = npz_small(p, "emb")
        ok = img_rows < len(e)
        cand = np.full_like(out, np.nan)
        cand[ok] = e[img_rows[ok]].astype(np.float32)
        fill = ~np.isfinite(out).all(1) & np.isfinite(cand).all(1)
        out[fill] = cand[fill]
    return out


# ---------------------------------------------------------------------------------------------------- scoring

WORK = {}


def _init(cache_dir):
    for f in os.listdir(cache_dir):
        if f.endswith(".npy"):
            WORK[f[:-4]] = np.load(os.path.join(cache_dir, f), mmap_mode="r")


def donors(src, ti, d, n_pool, n_sub):
    """Seeded donors of one target, shared by all its windows: 16 strangers (str00 = the stranger rung, str00-01 =
    mean of 2) and the pool subsample for the cross-modality sensitivity."""
    sc = SRC_CODE[src]
    perm = np.random.default_rng([SEED, sc, ti]).permutation(n_pool)
    sub = np.zeros(n_pool, bool)
    sub[np.random.default_rng([SEED, sc, ti, POOL_SUB]).choice(n_pool, n_sub, replace=False)] = True
    return perm[:N_STR], sub


def window_fills(src, ti, d, pk, sl, ring, lam, P, n_sub, ident=None):
    """All fills of one target in one window: {name: (fill window, validity, ArcFace cosine or NaN)}.
    ident(t, g), if given, replaces the ArcFace cosine of a one-image fill (t, g = cache rows of target and donor);
    used by the panel module to record donors. Default: unchanged behaviour."""
    X, M, E = WORK["X"], WORK["M"], WORK["emb"]
    p0, p1 = P["pool"]
    mean, mean_ok = WORK[f"{src}_mean"][sl], WORK[f"{src}_mean_ok"][sl]
    t = d["t"]
    xt, mt = np.asarray(X[t], np.float32), np.asarray(M[t])
    et = np.asarray(E[t])
    strs, sub = donors(src, ti, d, p1 - p0, n_sub)

    def img(g):
        idv = cos(et, np.asarray(E[g])) if ident is None else ident(t, g)
        return np.asarray(X[g][sl], np.float32), np.asarray(M[g][sl]), idv
    out = {"stranger": img(p0 + strs[0])}
    for j, s in enumerate(strs):
        out[f"str{j:02d}"] = img(p0 + s)
    Xs = np.stack([np.asarray(X[p0 + s][sl], np.float32) for s in strs[:2]])
    Ms = np.stack([np.asarray(M[p0 + s][sl]) for s in strs[:2]])
    m2, ok2 = BM.masked_mean(Xs, Ms)
    out["mean_k2"] = (m2, ok2, np.nan)
    out["dataset_mean"] = (np.asarray(mean), np.asarray(mean_ok), np.nan)
    if d["cell"] is not None:
        out["cohort_mean"] = (np.asarray(WORK[f"{src}_cells_m"][d["cell"]][sl]),
                              np.asarray(WORK[f"{src}_cells_ok"][d["cell"]][sl]), np.nan)
    out["repeat"] = img(d["rep"])
    fm, mm = mirror_fill(xt, mt)
    out["mirror"] = (fm[sl], mm[sl], np.nan)
    if d["sib"] is not None:
        out["sibling"] = img(d["sib"])
    # look-alike by context ring and oracle on the window, single person (k = 1), masked distances
    valid = np.asarray(WORK[f"{src}_valid"])
    tv = mt[sl] & valid[sl]
    elig = P["win_ok"]
    for suffix, el in (("", elig), ("_p219", elig & sub)):
        for name, A_, MA_, b_, mb_ in (("lookalike", P["ring_x"], P["ring_m"], xt[ring], mt[ring]),
                                       ("lookalike_oracle", P["win_x"], P["win_m"], xt[sl].ravel(), tv.ravel())):
            j = nearest(A_, MA_, b_, mb_, el)
            if j is not None:
                out[name + suffix] = img(p0 + j)
    lam_r, lam_m = lam["repeat"], lam["mirror"]
    xr, mr = np.asarray(X[d["rep"]], np.float32), np.asarray(M[d["rep"]])
    dy, dx = shift_direction(src, ti)
    fs, ms = shift_fill(xt, mt, dy * P["shift_px"], dx * P["shift_px"])
    out["aug_shift"] = (fs[sl], ms[sl], np.nan)
    fl, ml = light_fill(xt, mt, xr, mr)
    out["aug_light"] = (fl[sl], ml[sl], np.nan)
    out["shrink_repeat"] = (BM.shrink(out["repeat"][0], mean, lam_r), out["repeat"][1] & mean_ok, np.nan)
    out["shrink_mirror"] = (BM.shrink(out["mirror"][0], mean, lam_m), out["mirror"][1] & mean_ok, np.nan)
    xw = xt[sl]
    s0 = out["stranger"]
    e_wd = tv & mean_ok & s0[1]
    wd = BM.wrong_detail(xw[None], np.asarray(mean)[None], s0[0][None], e_wd[None])
    if wd is not None:
        out["wrong_detail"] = (wd[0], e_wd, np.nan)
    e_nz = tv & mean_ok
    nz = BM.noise_fill(xw[None], np.asarray(mean)[None], e_nz[None],
                       np.random.default_rng([SEED, SRC_CODE[src], ti, pk, 99]))
    out["noise"] = (nz[0], e_nz, np.nan)
    return xw, tv, out


def pool_window(src, P, sl, ring):
    """Pool arrays of one window: ring and window values and masks (float32 / bool) and window eligibility."""
    X, M = WORK["X"], WORK["M"]
    p0, p1 = P["pool"]
    Xp, Mp = np.asarray(X[p0:p1]), np.asarray(M[p0:p1])
    win_x = Xp[:, sl[0], sl[1]].reshape(p1 - p0, -1).astype(np.float32)
    win_m = Mp[:, sl[0], sl[1]].reshape(p1 - p0, -1)
    return {"pool": P["pool"], "ring_x": Xp[:, ring].astype(np.float32), "ring_m": Mp[:, ring],
            "win_x": win_x, "win_m": win_m, "win_ok": win_m.mean(1) >= MIN_IN_MASK}


def score_position(args):
    """Rows for one (source, window position) over the given targets."""
    src, pk, (y0, x0), targets, lam, sigmas, pool, n_sub, shift_px, win, rw = args   # window and ring explicit
    sl = (slice(y0, y0 + win), slice(x0, x0 + win))
    valid = np.asarray(WORK[f"{src}_valid"])
    ring = ring_mask(y0, x0, SIZE, win, rw) & valid
    P = pool_window(src, {"pool": pool}, sl, ring)
    P["shift_px"] = shift_px
    mu_full = np.asarray(WORK[f"{src}_mean"])
    rows = []
    for ti, d in targets:
        if (np.asarray(WORK["M"][d["t"]])[sl] & valid[sl]).mean() < MIN_IN_MASK:
            continue
        xw, tv, fills = window_fills(src, ti, d, pk, sl, ring, lam, P, n_sub)
        mu_c = np.asarray(WORK[f"{src}_cells_m"][d["cell"]][sl])[None] if d["cell"] is not None else None
        for name, (f, ok, idv) in fills.items():
            s_ = BM.score(xw[None], np.asarray(f, np.float32)[None], (tv & ok)[None], mu_full[sl][None], mu_c,
                          sigmas.get(ti))
            if s_ is None:
                continue
            rows.append({"src": src, "target": ti, "family": d["family"], "patch": pk, "fill": name,
                         IDENTITY: idv, **s_})
    return rows


def repeat_sigma(data, pos_list, cache_dir, win):
    """sigma_rep per (position, target): root mean squared target-repeat difference over pixels valid in both,
    pooled over the repeat pairs of the other families (leave-family-out, >= 5 pairs; brain Amendment 12)."""
    X = np.load(os.path.join(cache_dir, "X.npy"), mmap_mode="r")
    M = np.load(os.path.join(cache_dir, "M.npy"), mmap_mode="r")
    valid, T = data["valid"], data["targets"]
    fam = np.array([d["family"] for d in T])
    out = {}
    for pk, (y0, x0) in enumerate(pos_list):
        sl = (slice(y0, y0 + win), slice(x0, x0 + win))
        msd = np.full(len(T), np.nan)
        for k, d in enumerate(T):
            mt = M[d["t"]][sl] & valid[sl]
            both = mt & M[d["rep"]][sl]
            if mt.mean() >= MIN_IN_MASK and both.sum() >= 10:
                diff = np.asarray(X[d["t"]][sl], np.float32) - np.asarray(X[d["rep"]][sl], np.float32)
                msd[k] = float(np.mean(diff[both] ** 2))
        ok = np.isfinite(msd)
        fs = pd.Series(np.where(ok, msd, 0.0)).groupby(fam).sum()
        fn = pd.Series(ok.astype(int)).groupby(fam).sum()
        tot, n = float(np.nansum(msd)), int(ok.sum())
        out[pk] = {}
        for k in range(len(T)):
            nn = n - int(fn[fam[k]])
            if nn >= BM.MIN_REP_PAIRS:
                out[pk][k] = float(np.sqrt((tot - fs[fam[k]]) / nn))
    return out


def lambdas(version):
    """lambda = r = 1 - ratio / 2 of the repeat and mirror rungs (face results JSON, whole face), per source."""
    d = json.load(open(os.path.join(FACE_RESULTS, f"donor_ladder_face{version}.json")))["results"]
    return {s: {"repeat": 1 - d[s]["repeat"]["ratio"] / 2, "mirror": 1 - d[s]["mirror"]["ratio"] / 2} for s in d}


# ---------------------------------------------------------------------------------------------------- summary

DIST_SELF0 = {"mse", "rmse", "mae", "rmse_rep"}            # Amendment F1: value of a fill identical to the target
NO_SELF = {"psnr", "psnr_fixed"}                         # infinite on the target itself: no R_self
EXPECT_A_F = {k: (kind, tuple("retest" if f == "repeat" else f for f in fill) if isinstance(fill, tuple) else fill, t)
              for k, (kind, fill, t) in BM.EXPECT_A.items()}


def self_value(m):
    if m in NO_SELF:
        return np.nan
    return 0.0 if m in DIST_SELF0 else 1.0


def with_self(r, metrics):
    """Rows for the self-anchored scale (Amendment F1): the real retest is renamed 'retest' and an analytic 'self'
    rung is added in every (target, window) cell where the first reference stranger has a value, named 'repeat' so
    that the brain ladder functions anchor on it. Names are mapped back by `rename_back`."""
    out = r.copy()
    out.loc[out.fill == "repeat", "fill"] = "retest"
    ref = r[r.fill == "str00"].copy()
    ref["fill"] = "repeat"
    for m in metrics:
        ref[m] = np.where(ref[m].notna(), self_value(m), np.nan)
    return pd.concat([out, ref], ignore_index=True)


def rename_back(dct):
    return {("self" if k == "repeat" else k): v for k, v in dct.items()}


def ladder(per, per2, m):
    R, _ = BM.ladder_R(per.dropna(subset=[m]), m)
    if R is None:
        return None, None
    pt2, arr2 = BM.ladder_R_2way(per2.dropna(subset=[m]), m)
    for f, e in R.items():
        if f.startswith("_") or f not in arr2:
            continue
        e["ci95_family_only"] = e["ci95"]
        e["ci95"] = [float(np.nanpercentile(arr2[f], 2.5)), float(np.nanpercentile(arr2[f], 97.5))]
        e["R_cells"] = float(pt2[f])
    return R, arr2


def summarise(r, metrics=REPORTED):
    """One source: raw means; R_self (primary, Amendment F1: 0 = the target itself, 1 = mean of 16 strangers) with
    two-way (families x positions) and family-only intervals; reality-B expectations and the Amendment 14 score from
    R_self; the retest-anchored R (descriptive, `ladder_R_retest`)."""
    import donor_ladder_metrics_expected_figure as EF
    metrics = [m for m in metrics if r[m].notna().any()]
    v = {"fills": {}, "ladder_R": {}, "ladder_R_retest": {}, "expectations": {}, "expectations_realityA": {},
         "realityB_score": {}}
    per = r.groupby(["target", "family", "fill"])[metrics].mean().reset_index()
    for f in sorted(per.fill.unique()):
        if not BM.is_reference_stranger(f):
            v["fills"][f] = {m: BM.boot_mean(per[per.fill == f].dropna(subset=[m]), m) for m in metrics}
    per2 = r.groupby(["target", "family", "patch", "fill"])[metrics].mean().reset_index()
    for m in metrics:                                    # descriptive: retest = 0 (Amendments 11-14 as on the brain)
        R, _ = ladder(per, per2, m)
        if R is not None:
            v["ladder_R_retest"][m] = R
    rs = with_self(r, metrics)
    per = rs.groupby(["target", "family", "fill"])[metrics].mean().reset_index()
    per2 = rs.groupby(["target", "family", "patch", "fill"])[metrics].mean().reset_index()
    for m in metrics:
        if m in NO_SELF:
            continue
        R, arr2 = ladder(per, per2, m)
        if R is None:
            continue
        R, arr2 = rename_back(R), rename_back(arr2)
        R["_anchors"]["d_self"] = R["_anchors"].pop("d_repeat")
        R["_anchors"]["n_self_cells"] = R["_anchors"].pop("n_repeat_targets")
        v["ladder_R"][m] = R
        v["expectations"][m] = BM.expectations(R, arr2)
        v["expectations_realityA"][m] = BM.expectations(R, arr2, EXPECT_A_F)
        sc = EF.score_B(v, m)
        v["realityB_score"][m] = {"dev": sc["dev"], "rms": sc["rms"], "noise_above_1": sc["noise_above_1"],
                                  "partly_below_1": list(sc["partly_below_1"]),
                                  "oracle_le_lookalike": sc["oracle_le_lookalike"],
                                  "retest_below_1": bool(R["retest"]["ci95"][1] < 1) if "retest" in R else None}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="face frame/results version, e.g. _v6 (no default)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="evenly spaced targets per source (smoke / QC run)")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--reuse-cache", action="store_true", help="skip the frame read if the cache exists")
    a = ap.parse_args()
    if a.workers > 3:
        raise SystemExit("at most 3 workers (shared memory with other jobs)")
    F = load_face(a.version)
    cache_dir = os.path.join(PRIVATE, f"cache{a.version}")
    os.makedirs(cache_dir, exist_ok=True)
    tag = a.version + (f"_n{a.limit}" if a.limit else "")
    rows_path = os.path.join(PRIVATE, f"rows{tag}.csv.gz")
    meta_path = os.path.join(PRIVATE, f"meta{tag}.json")
    t0 = time.time()
    if not a.summary_only:
        desc_path = os.path.join(cache_dir, "desc.json")
        if a.reuse_cache and os.path.exists(desc_path):
            desc = json.load(open(desc_path))
            data, repro = desc["data"], desc["repro"]
            for src, d in data.items():
                d["valid"] = np.load(os.path.join(cache_dir, f"{src}_valid.npy"))
        else:
            data, repro = build(F, a.version, cache_dir)
            json.dump({"data": {s: {k: v for k, v in d.items() if k != "valid"} for s, d in data.items()},
                       "repro": repro}, open(desc_path, "w"))
        face_json = json.load(open(os.path.join(FACE_RESULTS, f"donor_ladder_face{a.version}.json")))["results"]
        rep_ok = {s: {c: {"here": v, "face_json": face_json[s].get(c, {}).get("ratio")} for c, v in repro[s].items()}
                  for s in repro}
        ok = all(e["here"] == e["face_json"] for s in rep_ok.values() for e in s.values())
        print(json.dumps({"reproduces_face_json": ok, "detail": rep_ok}), flush=True)
        if not ok:
            raise SystemExit("whole-face ratios do not reproduce the face results JSON; nothing scored")
        lam = lambdas(a.version)
        jobs, meta = [], {"version": a.version, "repro": rep_ok, "lambda": lam, "sources": {}}
        for src, d in data.items():
            pos_list = positions(d["valid"])
            sig = repeat_sigma(d, pos_list, cache_dir, WIN)
            T = d["targets"]
            sel = np.unique(np.linspace(0, len(T) - 1, a.limit).round().astype(int)) if a.limit else np.arange(len(T))
            tl = [(int(ti), T[ti]) for ti in sel]
            n_sub = min(POOL_SUB, d["n_pool"])
            for pk, yx in enumerate(pos_list):
                jobs.append((src, pk, tuple(yx), tl, lam[src], sig[pk], tuple(d["pool"]), n_sub, d["shift_px"], WIN,
                             RING))
            meta.setdefault("shift", {})[src] = {"shift_px": d["shift_px"]}
            meta["sources"][src] = {"n_targets": len(T), "n_targets_scored": len(tl),
                                    "n_families": len({t["family"] for t in T}), "n_pool": d["n_pool"],
                                    "n_pool_subsample": n_sub, "n_positions": len(pos_list),
                                    "positions": [list(p) for p in pos_list],
                                    "n_with_sibling": sum(t["sib"] is not None for t in T),
                                    "n_with_cohort": sum(t["cell"] is not None for t in T)}
        print(json.dumps({s: {k: v for k, v in m.items() if k != "positions"} for s, m in meta["sources"].items()}),
              flush=True)
        rows = []
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(cache_dir,)) as ex:
            for k, part in enumerate(ex.map(score_position, jobs)):
                rows.extend(part)
                print(f"position {k + 1}/{len(jobs)} rows {len(rows)} {time.time() - t0:.0f}s", flush=True)
        r = pd.DataFrame(rows)
        r.to_csv(rows_path, index=False)
        json.dump(meta, open(meta_path, "w"), indent=1)
    else:
        r = pd.read_csv(rows_path)
        meta = json.load(open(meta_path))
    out = {"status": f"face frame {a.version}: {F.qc_status()}; landmark affine to the population "
                     "template (face protocol, Limitations 3)",
           "protocol": "docs/PROTOCOL_face_metrics.md (brain Amendments 10 to 14 ported; Amendment F1 scale)", "frame": a.version,
           "tier": 1, "seed": SEED, "scale": "R_self: 0 = the target itself, 1 = mean of 16 strangers (Amendment F1)",
           "aug_shift_px": {s_: d_.get("shift_px") for s_, d_ in meta.get("shift", {}).items()}, "metrics": list(METRICS), "identity_reference": IDENTITY,
           "lower_is_better": sorted(BM.LOWER_BETTER), "window_px": WIN, "ring_px": RING,
           "pose_max_deg": 10, "lambda_shrink_by_source": meta["lambda"], "reproduction": meta["repro"],
           "subset": f"{a.limit} evenly spaced targets per source" if a.limit else "all targets",
           "counts": {s: {k: v for k, v in m.items() if k != "positions"} for s, m in meta["sources"].items()},
           "views": {}}
    for src in ("celeba", "imdb"):
        rs = r[r.src == src]
        if len(rs):
            out["views"][src] = summarise(rs)
            out["views"][src]["n_rows"] = int(len(rs))
    out["minutes"] = round((time.time() - t0) / 60, 1)
    path = OUT_JSON if not a.limit else os.path.join(PRIVATE, f"summary{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(BM.finite(out), fh, indent=1, allow_nan=False)
    print(path, out["minutes"], "min")


if __name__ == "__main__":
    main()
