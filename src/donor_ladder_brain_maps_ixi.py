"""IXI Gaussian error maps (PROTOCOL_donor_ladder_brain.md, Amendment 25). Pre-QC, affine registration only.

The QTIM maps of Amendments 5, 6, 7, 8 and 18 on the 100 IXI scans of Amendment 23:
- 2 mm maps (`donor_ladder_brain_maps`): own mirror, stranger (exact expectation over the pool) and the same-sex-
  and-age mean (QTIM rule, >= 10 donors within 2 years, else missing); local MSE = masked Gaussian mean of the voxel
  squared difference (sd = 16 mm / sqrt(12)), in units of V, the error of the leave-one-out dataset mean.
- 8 mm grid (`donor_ladder_brain_map_grid`): the mean of 16 look-alikes chosen on the 4 mm shell, Gaussian weights
  inside the 16 mm cube (sd = 16 / sqrt(12), truncated, Amendment 18); stranger and mirror for the cross-check.
Missing is never zero (the one deliberate difference from the QTIM code): a donor enters the pooled mean, the pool
variance, the look-alike mean and the shell distance only at voxels inside its own mask; a target voxel outside its
mask, or a voxel with fewer than MIN_DONORS donors, is missing. IXI has no repeats and no families: repeat and
sibling maps are n/a, every target is its own family, and the split half uses two random halves of targets.
Maps and caches stay private (`ixi-maps-20261114-v1`); the JSONs hold whole-map summaries only.

Usage: python -I src/donor_ladder_brain_maps_ixi.py fine|grid|figure [--smoke N]
"""
import argparse
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_map_grid as G  # noqa: E402
import donor_ladder_brain_maps as P  # noqa: E402
import donor_ladder_brain_size as S  # noqa: E402

PRIVATE = os.path.join(B.PROJ, "ixi-maps-20261114-v1")
OUT_FINE = os.path.join(B.ROOT, "results", "donor_ladder_brain_maps_ixi.json")
OUT_GRID = os.path.join(B.ROOT, "results", "donor_ladder_brain_map_grid_ixi.json")
MIN_DONORS = 10
FINE = ("mirror", "stranger", "cohort")
GRID = ("mirror", "stranger", "knn16_s4")
NA = {"repeat": "IXI has no repeat scans", "sibling": "IXI has no families"}


def scans(smoke=0):
    import donor_ladder_brain_ixi as I
    t, _ = I.scans()
    t = t.reset_index(drop=True)
    return t.iloc[:smoke].reset_index(drop=True) if smoke else t


def volumes(t, affine, smoke=0):
    """z-scored volumes (0 outside the own mask; the mask travels with them) and masks, cached privately."""
    S.PRIVATE = os.path.join(PRIVATE, f"smoke{smoke}" if smoke else "cache")
    os.makedirs(S.PRIVATE, exist_ok=True)
    return S.cache_volumes(t, affine)


def halves(n, seed=B.SEED):
    """Two random halves of targets (IXI has no families; QTIM splits families with the same seed)."""
    h = np.zeros(n, int)
    h[np.random.default_rng(seed).permutation(n)[: n // 2]] = 1
    return h


def pool_sums(V, M, members):
    """Per voxel: sum of v over donors whose mask covers it, sum of v^2, and their count."""
    s = np.zeros(V.shape[1:])
    q = np.zeros(V.shape[1:])
    c = np.zeros(V.shape[1:])
    for k in members:
        m = np.asarray(M[k]) > 0
        v = np.where(m, np.asarray(V[k], np.float64), 0.0)
        s += v
        q += v * v
        c += m
    return s, q, c


def loo(s, q, c, v, m):
    """Leave-one-out masked mean and variance of the pool at every voxel; NaN where < MIN_DONORS donors remain."""
    vm = np.where(m, v, 0.0)
    n = c - m
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (s - vm) / n
        var = np.maximum((q - vm * vm) / n - mu ** 2, 0)
    ok = n >= MIN_DONORS
    return np.where(ok, mu, np.nan), np.where(ok, var, np.nan), ok


def masked_mean(V, M, members):
    s, _, c = pool_sums(V, M, members)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan), c


def fine_run(t, affine, V, M, which=None):
    """Accumulated 2 mm maps; which = None (all targets) or 0 / 1 (that half only, as pool and as targets)."""
    n = len(t)
    h = halves(n)
    inpool = np.ones(n, bool) if which is None else h == which
    members = np.flatnonzero(inpool)
    mi = B.mirror_index(affine, V.shape[1])
    support = sum((P.frac(np.asarray(M[k]) > 0) >= P.VALID).astype(np.int32) for k in range(n)) >= P.SUPPORT * n
    s, q, c = pool_sums(V, M, members)
    sex, age = t.sex.to_numpy(), t.age.to_numpy()
    acc = P.Acc(support.shape)
    n_cohort = 0
    for done, i in enumerate(members):
        v = np.asarray(V[i], np.float64)
        mk = np.asarray(M[i]) > 0
        mu, var, okp = loo(s, q, c, v, mk)
        ok = (P.frac(mk) >= P.VALID) & support
        e = mk & okp
        Vi = P.local(np.nan_to_num((v - mu) ** 2), e)
        hh = int(h[i])
        acc.add("stranger", hh, Vi + P.local(np.nan_to_num(var), e), Vi, ok)
        okm = ok & (P.frac(mk[mi]) >= P.VALID)
        acc.add("mirror", hh, P.local((v - v[mi]) ** 2, mk & mk[mi]), Vi, okm)
        coh = [j for j in members if j != i and sex[j] == sex[i] and abs(age[j] - age[i]) <= P.AGE_WINDOW]
        if len(coh) >= P.COHORT_MIN:
            cm, cc = masked_mean(V, M, coh)
            ec = mk & (cc >= P.COHORT_MIN)
            acc.add("cohort", hh, P.local(np.nan_to_num((v - cm) ** 2), ec), Vi, ok)
            n_cohort += 1
        if done % 10 == 0:
            print(f"fine {'all' if which is None else 'half ' + str(which)}: target {done + 1}/{len(members)}",
                  flush=True)
    x_mm = affine[0, 0] * np.arange(0, V.shape[1], P.STEP) + affine[0, 3]
    return acc, support, np.abs(x_mm) < P.MIDLINE_MM, n_cohort


def fine(smoke):
    t0 = time.time()
    t = scans(smoke)
    _, affine, _ = B.positions(np.random.default_rng(B.SEED))
    V, M = volumes(t, affine, smoke)
    out_dir = os.path.join(PRIVATE, f"smoke{smoke}" if smoke else "maps")
    os.makedirs(out_dir, exist_ok=True)
    acc, support, midline, n_coh = fine_run(t, affine, V, M)
    hs = [fine_run(t, affine, V, M, k)[0] for k in (0, 1)]
    aff2 = affine.copy()
    aff2[:3, :3] *= P.STEP
    out = {"status": "pre-QC: affine registration only, MRI registration QC not met; exploratory maps",
           "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendment 25; QTIM method of Amendments 5, 6, 8, 16)",
           "dataset": "IXI", "n_targets": int(len(t)), "window_mm": P.WIN, "window": "masked Gaussian, sd = WIN/sqrt(12)",
           "grid_mm": P.STEP, "missing_rule": f"donor voxels outside the donor's mask never enter a mean; voxels with "
           f"fewer than {MIN_DONORS} donors are missing", "support_voxels": int(support.sum()),
           "targets_with_cohort_mean": n_coh, "not_available": NA, "maps": {}}
    v_sum = acc.a[("stranger", 0, "V")] + acc.a[("stranger", 1, "V")]
    n_str = acc.count("stranger")
    with np.errstate(invalid="ignore", divide="ignore"):
        vmap = np.where(n_str > 0, v_sum / n_str, np.nan)
    nib.save(nib.Nifti1Image(vmap.astype(np.float32), aff2), os.path.join(out_dir, "V.nii.gz"))
    for d in FINE:
        ratio = acc.ratio(d)
        keep = support & (acc.count(d) > 0)
        if d == "mirror":
            keep &= ~midline[:, None, None]
        if not keep.any():
            out["maps"][d] = {"status": "n/a", "reason": f"no target with >= {P.COHORT_MIN} same-sex donors within "
                                                          f"{P.AGE_WINDOW} years" if d == "cohort" else "no voxels"}
            continue
        ratio = np.where(keep, ratio, np.nan)
        h0, h1 = (np.where(keep, hs[k].ratio(d, (k,)), np.nan) for k in (0, 1))
        for name, arr in ((f"{d}_ratio", ratio), (f"{d}_n", acc.count(d)), (f"{d}_ratio_h0", h0),
                          (f"{d}_ratio_h1", h1)):
            nib.save(nib.Nifti1Image(arr.astype(np.float32), aff2), os.path.join(out_dir, f"{name}.nii.gz"))
        e = sum(acc.a[(d, k, "err")][keep].sum() for k in (0, 1))
        v = sum(acc.a[(d, k, "V")][keep].sum() for k in (0, 1))
        out["maps"][d] = {"pooled_ratio": float(e / v) if v > 0 else None,
                          "quantiles_5_25_50_75_95": [float(x) for x in np.nanpercentile(ratio[keep], [5, 25, 50, 75, 95])],
                          "split_half_r": P.split_half(h0, h1, keep),
                          "split_half_r_shared_pool": P.split_half(acc.ratio(d, (0,)), acc.ratio(d, (1,)), keep),
                          "max_targets_per_voxel": int(acc.count(d).max()), "voxels": int(keep.sum())}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    dest = OUT_FINE if not smoke else os.path.join(out_dir, "summary.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(B.finite(out), fh, indent=1, allow_nan=False)
    print(json.dumps(out["maps"]), flush=True)


# ---------------------------------------------------------------------------------------------------- 8 mm grid

def grid_cache(t, affine, cs, V, M, d):
    """Centre-major memmaps: every scan's 24 mm window and its mask, the mirrored 16 mm cube and its mask, and the
    QTIM validity flags (>= 95% of the cube inside the mask)."""
    paths = {k: os.path.join(d, f"grid_{k}.npy") for k in ("win", "wm", "mcube", "mm", "valid", "mvalid")}
    if os.path.exists(os.path.join(d, "grid.done")):
        return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
    n, nc = len(t), len(cs)
    shp = {"win": (np.float16, (G.WINDOW,) * 3), "wm": (np.bool_, (G.WINDOW,) * 3),
           "mcube": (np.float16, (G.CUBE,) * 3), "mm": (np.bool_, (G.CUBE,) * 3)}
    shp["valid"] = shp["mvalid"] = (np.bool_, ())
    mi = B.mirror_index(affine, V.shape[1])
    prog = os.path.join(d, "grid.progress")
    exists = all(os.path.exists(p) for p in paths.values())
    out = {k: (np.load(paths[k], mmap_mode="r+") if exists else
               np.lib.format.open_memmap(paths[k], "w+", dt, (nc, n) + s)) for k, (dt, s) in shp.items()}
    if exists and any(out[k].shape != (nc, n) + s for k, (_, s) in shp.items()):
        raise RuntimeError(f"grid cache in {d} has other shapes than {nc} centres x {n} scans; delete it to rebuild")

    def scan_parts(k):
        v, m = np.asarray(V[k]), np.asarray(M[k]) > 0
        vm, mm = v[mi], m[mi]
        f, fm = (ndimage.uniform_filter(x.astype(np.float32), G.CUBE, mode="constant") for x in (m, mm))  # box-ok: validity fraction
        return v, m, vm, mm, f[tuple(cs.T)] >= G.VALID - 1e-4, fm[tuple(cs.T)] >= G.VALID - 1e-4

    def written(k):
        """True when scan k is in the cache exactly as this code writes it (checked on 8 centres and all flags)."""
        v, m, vm, mm, va, mva = scan_parts(k)
        if not (np.array_equal(out["valid"][:, k], va) and np.array_equal(out["mvalid"][:, k], mva)):
            return False
        for ci in np.linspace(0, nc - 1, 8).astype(int):
            c = cs[ci]
            if not (np.array_equal(out["win"][ci, k], G.window_at(v, c, G.WINDOW).astype(np.float16))
                    and np.array_equal(out["wm"][ci, k], G.window_at(m, c, G.WINDOW))
                    and np.array_equal(out["mcube"][ci, k], G.window_at(vm, c, G.CUBE).astype(np.float16))
                    and np.array_equal(out["mm"][ci, k], G.window_at(mm, c, G.CUBE))):
                return False
        return True

    start = 0
    if exists and os.path.exists(prog):
        start = int(open(prog).read().strip() or 0)
    elif exists:
        # cache from before the progress file (stopped run): the first scan that is not verifiably written
        while start < n and written(start):
            start += 1
        print(f"grid cache: {start}/{n} scans verified in the existing cache, resuming there", flush=True)
    for k in range(start, n):
        v, m, vm, mm, va, mva = scan_parts(k)
        for ci, c in enumerate(cs):
            out["win"][ci, k] = G.window_at(v, c, G.WINDOW)
            out["wm"][ci, k] = G.window_at(m, c, G.WINDOW)
            out["mcube"][ci, k] = G.window_at(vm, c, G.CUBE)
            out["mm"][ci, k] = G.window_at(mm, c, G.CUBE)
        out["valid"][:, k] = va
        out["mvalid"][:, k] = mva
        for a in out.values():
            a.flush()
        with open(prog, "w") as fh:
            fh.write(str(k + 1))
        print(f"grid cache {k + 1}/{n}", flush=True)
    for a in out.values():
        a.flush()
    open(os.path.join(d, "grid.done"), "w").close()
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


def wmean(err, ok, w):
    """Gaussian-weighted mean of err over the voxels ok (weights renormalised over them); NaN if none."""
    ww = w * ok
    s = ww.sum()
    return float((np.where(ok, err, 0) * ww).sum() / s) if s > 0 else np.nan


def shell_dist(sh, shm, i, donors):
    """Mean squared shell difference over voxels valid in both shells (missing never enters)."""
    both = shm[donors] & shm[i]
    d = np.where(both, (sh[donors] - sh[i]) ** 2, 0).sum(1)
    n = both.sum(1)
    return np.where(n > 0, d / np.maximum(n, 1), np.inf)


def grid_rows(X, XM, Mc, MMc, sh4, sh4m, valid, mvalid, mirror_ok, w, pool):
    """Per target at one centre: (i, V, {fill: error}) with masked means; fills stranger, mirror, knn16_s4."""
    out = []
    for i in np.flatnonzero(valid & pool):
        others = np.flatnonzero(valid & pool & (np.arange(len(X)) != i))
        if others.size < MIN_DONORS:
            continue
        c = XM[others].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(XM[others], X[others], 0).sum(0) / c
            var = np.maximum(np.where(XM[others], X[others] ** 2, 0).sum(0) / c - mu ** 2, 0)
        e = XM[i] & (c >= MIN_DONORS)
        v = wmean((X[i] - mu) ** 2, e, w)
        if not np.isfinite(v) or v <= 0:
            continue
        r = {"stranger": wmean((X[i] - mu) ** 2 + var, e, w)}
        if mirror_ok and mvalid[i]:
            r["mirror"] = wmean((X[i] - Mc[i]) ** 2, XM[i] & MMc[i], w)
        if others.size >= G.K:
            nn = others[np.argsort(shell_dist(sh4, sh4m, i, others), kind="stable")[:G.K]]
            ck = XM[nn].sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                km = np.where(XM[nn], X[nn], 0).sum(0) / ck
            r["knn16_s4"] = wmean((X[i] - km) ** 2, XM[i] & (ck > 0), w)
        out.append((i, v, {k: x for k, x in r.items() if np.isfinite(x)}))
    return out


def grid(smoke):
    from nilearn.datasets import load_mni152_brain_mask
    t0 = time.time()
    t = scans(smoke)
    _, affine, _ = B.positions(np.random.default_rng(B.SEED))
    V, M = volumes(t, affine, smoke)
    d = os.path.join(PRIVATE, f"smoke{smoke}" if smoke else "maps")
    os.makedirs(d, exist_ok=True)
    a = int(B.mirror_index(affine, V.shape[1])[0])
    cs = G.centres(load_mni152_brain_mask(resolution=1).get_fdata() > 0, a)
    C = grid_cache(t, affine, cs, V, M, os.path.join(PRIVATE, f"smoke{smoke}" if smoke else "cache"))
    n = len(t)
    h = halves(n)
    x_mm = affine[0, 0] * cs[:, 0] + affine[0, 3]
    acc = {(f, k, q): np.zeros(len(cs)) for f in GRID for k in (0, 1) for q in ("err", "V", "n")}
    ind = {(f, k, q): np.zeros(len(cs)) for f in GRID for k in (0, 1) for q in ("err", "V")}
    vsum, vn = np.zeros(len(cs)), np.zeros(len(cs))
    allp = np.ones(n, bool)
    for ci in range(len(cs)):
        win = np.asarray(C["win"][ci], np.float32)
        wm = np.asarray(C["wm"][ci], bool)
        X, sh4 = G.parts(win, 4)
        XM, sh4m = G.parts(wm, 4)
        Mc = np.asarray(C["mcube"][ci], np.float32).reshape(n, -1)
        MMc = np.asarray(C["mm"][ci], bool).reshape(n, -1)
        valid, mvalid = np.asarray(C["valid"][ci]), np.asarray(C["mvalid"][ci])
        mok = abs(x_mm[ci]) >= P.MIDLINE_MM
        for i, v, r in grid_rows(X, XM, Mc, MMc, sh4, sh4m, valid, mvalid, mok, G.W, allp):
            vsum[ci] += v
            vn[ci] += 1
            for f, e in r.items():
                acc[(f, h[i], "err")][ci] += e
                acc[(f, h[i], "V")][ci] += v
                acc[(f, h[i], "n")][ci] += 1
        for k in (0, 1):
            for i, v, r in grid_rows(X, XM, Mc, MMc, sh4, sh4m, valid, mvalid, mok, G.W, h == k):
                for f, e in r.items():
                    ind[(f, k, "err")][ci] += e
                    ind[(f, k, "V")][ci] += v
        if ci % 200 == 0:
            el = time.time() - t0
            print(f"centre {ci + 1}/{len(cs)}, {el / 60:.1f} min", flush=True)
    shape = tuple(-(-s // G.STEP) for s in V.shape[1:])
    gi = tuple(((cs - np.array([(a // 2) % G.STEP, 0, 0])) // G.STEP).T)
    aff8 = affine.copy()
    aff8[:3, :3] *= G.STEP
    aff8[0, 3] += affine[0, 0] * ((a // 2) % G.STEP)
    out = {"status": "pre-QC: affine registration only, MRI registration QC not met; exploratory maps",
           "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendment 25; QTIM method of Amendments 7, 8, 18)",
           "dataset": "IXI", "grid_mm": G.STEP, "cube_mm": G.CUBE, "n_targets": int(n),
           "window": "Gaussian weights inside the cube, sd = CUBE/sqrt(12), truncated, renormalised over valid voxels",
           "missing_rule": "donor voxels outside the donor's mask never enter a mean or a shell distance",
           "n_centres": int(len(cs)), "not_available": NA, "maps": {}}
    vol = np.full(shape, np.nan, np.float32)
    vol[gi] = np.where(vn > 0, vsum / np.maximum(vn, 1), np.nan)
    nib.save(nib.Nifti1Image(vol, aff8), os.path.join(d, "grid_V.nii.gz"))
    for f in GRID:
        e = acc[(f, 0, "err")] + acc[(f, 1, "err")]
        v = acc[(f, 0, "V")] + acc[(f, 1, "V")]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(v > 0, e / v, np.nan)
            hv = [np.where(ind[(f, k, "V")] > 0, ind[(f, k, "err")] / ind[(f, k, "V")], np.nan) for k in (0, 1)]
        for name, arr in ((f"grid_{f}_ratio", ratio), (f"grid_{f}_ratio_h0", hv[0]), (f"grid_{f}_ratio_h1", hv[1])):
            vol = np.full(shape, np.nan, np.float32)
            vol[gi] = arr
            nib.save(nib.Nifti1Image(vol, aff8), os.path.join(d, f"{name}.nii.gz"))
        ok = np.isfinite(hv[0]) & np.isfinite(hv[1])
        fin = np.isfinite(ratio)
        out["maps"][f] = {"pooled_ratio": float(e.sum() / v.sum()) if v.sum() > 0 else None,
                          "quantiles_5_25_50_75_95": [float(x) for x in np.percentile(ratio[fin], [5, 25, 50, 75, 95])]
                          if fin.any() else None,
                          "split_half_r": float(np.corrcoef(hv[0][ok], hv[1][ok])[0, 1]) if ok.sum() > 10 else None,
                          "centres": int(fin.sum()),
                          "max_targets_per_centre": int((acc[(f, 0, "n")] + acc[(f, 1, "n")]).max())}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    dest = OUT_GRID if not smoke else os.path.join(d, "grid_summary.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(B.finite(out), fh, indent=1, allow_nan=False)
    print(json.dumps(out["maps"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("fine", "grid"))
    ap.add_argument("--smoke", type=int, default=0, help="first N scans only (targets and donors)")
    a = ap.parse_args()
    os.makedirs(PRIVATE, exist_ok=True)
    (fine if a.step == "fine" else grid)(a.smoke)


if __name__ == "__main__":
    main()
