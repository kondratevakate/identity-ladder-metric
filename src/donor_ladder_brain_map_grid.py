"""Coarse-grid ladder map with look-alikes (PROTOCOL_donor_ladder_brain.md, Amendment 7). Pre-QC, exploratory.

On an 8 mm grid of centres through the midline, every scan's 24 mm window and its mirrored 16 mm cube are cached
once (float16, centre-major memmaps). Each centre then runs the cube-ladder rules: repeat, own mirror, all
siblings, the exact stranger expectation, the same-sex-and-age mean and 16 look-alikes chosen on the 2 or 4 mm
shell, in units of the error of the leave-family-out mean. Sums are kept per family half for a split-half
reliability. Window cache and maps stay private; the JSON holds whole-map summaries only.
"""
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_maps as P  # noqa: E402
import donor_ladder_brain_size as S  # noqa: E402

STEP, CUBE, WINDOW, SUPPORT, VALID, MIN_DONORS, K = 8, 16, 24, 0.9, 0.95, 10, 16
FILLS = ("repeat", "mirror", "sibling", "stranger", "cohort", "knn16_s4", "knn16_s2")
OUT_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_map_grid.json")


def centres(template_mask, a):
    """Grid voxels (x through the midline index a / 2, step STEP) whose WINDOW cube is >= SUPPORT in the template."""
    frac = ndimage.uniform_filter(template_mask.astype(np.float32), WINDOW, mode="constant")
    axes = [np.arange((a // 2) % STEP, template_mask.shape[0], STEP)] + \
           [np.arange(0, n, STEP) for n in template_mask.shape[1:]]
    return np.array([(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2] if frac[x, y, z] >= SUPPORT])


def window_at(vol, c, size):
    """Cube of edge `size` around centre c ([c - size/2, c + size/2) per axis), zero outside the grid."""
    return S.get_box(vol, tuple(int(ci) - size // 2 for ci in c), (size,) * 3)


def parts(win, shell):
    """Cube (inner CUBE^3) and the shell of thickness `shell` around it, from a WINDOW^3 array (last 3 axes)."""
    o = (WINDOW - CUBE) // 2
    inner = (Ellipsis,) + (slice(o, o + CUBE),) * 3
    lo = o - shell
    outer = win[(Ellipsis,) + (slice(lo, lo + CUBE + 2 * shell),) * 3]
    m = np.ones((CUBE + 2 * shell,) * 3, bool)
    m[(slice(shell, shell + CUBE),) * 3] = False
    return win[inner].reshape(win.shape[:-3] + (-1,)), outer[..., m]


def cache(t, affine, cs, V, M):
    """Centre-major float16 memmaps of every scan's window and mirrored cube, and both validity flags."""
    d = P.PRIVATE
    paths = {k: os.path.join(d, f"grid_{k}.npy") for k in ("win", "mcube", "valid", "mvalid")}
    if os.path.exists(os.path.join(d, "grid.done")):
        return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
    n, nc = len(t), len(cs)
    out = {"win": np.lib.format.open_memmap(paths["win"], "w+", np.float16, (nc, n) + (WINDOW,) * 3),
           "mcube": np.lib.format.open_memmap(paths["mcube"], "w+", np.float16, (nc, n) + (CUBE,) * 3),
           "valid": np.lib.format.open_memmap(paths["valid"], "w+", np.bool_, (nc, n)),
           "mvalid": np.lib.format.open_memmap(paths["mvalid"], "w+", np.bool_, (nc, n))}
    mi = B.mirror_index(affine, V.shape[1])
    for k in range(n):
        v, m = np.asarray(V[k]), np.asarray(M[k]).astype(np.float32)
        vm = v[mi]
        f, fm = (ndimage.uniform_filter(x, CUBE, mode="constant") for x in (m, m[mi]))
        for ci, c in enumerate(cs):
            out["win"][ci, k] = window_at(v, c, WINDOW)
            out["mcube"][ci, k] = window_at(vm, c, CUBE)
        out["valid"][:, k] = f[tuple(cs.T)] >= VALID - 1e-4
        out["mvalid"][:, k] = fm[tuple(cs.T)] >= VALID - 1e-4
        if k % 25 == 0:
            print(f"cache {k + 1}/{n}", flush=True)
    for a in out.values():
        a.flush()
    open(os.path.join(d, "grid.done"), "w").close()
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


def knn_error(xi, X, shells, i, donors, k=K):
    """Error of the mean of the k donors whose shell is closest to the target's (selection on context only)."""
    d = ((shells[donors] - shells[i]) ** 2).sum(1)
    nn = donors[np.argsort(d, kind="stable")[:k]]
    return float(np.mean((xi - X[nn].mean(0)) ** 2))


def centre_rows(X, Mc, sh2, sh4, valid, mvalid, fam, sex, age, rep_cube, rep_valid, mirror_ok, pool=None):
    """Per target at one centre: dict fill -> (error, V) under the cube-ladder rules. With `pool` (bool per
    target), only those scans are targets and donors (independent half maps, Amendment 8)."""
    pool = np.ones(len(X), bool) if pool is None else pool
    out = []
    for i in np.flatnonzero(valid & pool):
        others = np.flatnonzero((fam != fam[i]) & valid & pool)
        if others.size < MIN_DONORS:
            continue
        xi = X[i]
        mu = X[others].mean(0)
        v = float(np.mean((xi - mu) ** 2))
        r = {"stranger": v + float(np.mean((X[others] - mu) ** 2))}
        coh = others[(sex[others] == sex[i]) & (np.abs(age[others] - age[i]) <= 2)]
        if coh.size >= MIN_DONORS:
            r["cohort"] = float(np.mean((xi - X[coh].mean(0)) ** 2))
        if mirror_ok and mvalid[i]:
            r["mirror"] = float(np.mean((xi - Mc[i]) ** 2))
        sibs = [j for j in np.flatnonzero((fam == fam[i]) & valid) if j != i and age[j] != age[i]]
        if sibs:
            r["sibling"] = float(np.mean([np.mean((xi - X[j]) ** 2) for j in sibs]))
        if rep_cube[i] is not None and rep_valid[i]:
            r["repeat"] = float(np.mean((xi - rep_cube[i]) ** 2))
        if others.size >= K:
            r["knn16_s4"] = knn_error(xi, X, sh4, i, others)
            r["knn16_s2"] = knn_error(xi, X, sh2, i, others)
        out.append((i, v, r))
    return out


def main():
    from nilearn.datasets import load_mni152_brain_mask
    os.makedirs(P.PRIVATE, exist_ok=True)
    t0 = time.time()
    t = B.scans()
    _, affine, _ = B.positions(np.random.default_rng(B.SEED))
    V, M = S.cache_volumes(t, affine)
    a = int(B.mirror_index(affine, V.shape[1])[0])
    cs = centres(load_mni152_brain_mask(resolution=1).get_fdata() > 0, a)
    G = cache(t, affine, cs, V, M)
    tg = t[t.session == "ses-01"].reset_index()
    ix = tg["index"].to_numpy()
    fam, sex, age = tg.family.to_numpy(), tg.sex.to_numpy(), tg.age.to_numpy()
    rep = {r.person: r.Index for r in t[t.session == "ses-02"].itertuples()}
    rep_ix = [rep.get(p) for p in tg.person]
    fams = np.unique(fam)
    half_of = dict(zip(np.random.default_rng(B.SEED).permutation(fams), np.arange(len(fams)) % 2))
    half = np.array([half_of[f] for f in fam])
    x_mm = affine[0, 0] * cs[:, 0] + affine[0, 3]
    acc = {(f, h, q): np.zeros(len(cs)) for f in FILLS for h in (0, 1) for q in ("err", "V", "n")}
    ind = {(f, h, q): np.zeros(len(cs)) for f in FILLS for h in (0, 1) for q in ("err", "V")}   # own-half pools
    vsum = np.zeros(len(cs))
    vn = np.zeros(len(cs))
    for ci in range(len(cs)):
        w = np.asarray(G["win"][ci], dtype=np.float32)
        Xall, sh2_all = parts(w, 2)
        _, sh4_all = parts(w, 4)
        X, sh2, sh4 = Xall[ix], sh2_all[ix], sh4_all[ix]
        Mc = np.asarray(G["mcube"][ci], dtype=np.float32).reshape(len(t), -1)[ix]
        valid, mvalid = np.asarray(G["valid"][ci])[ix], np.asarray(G["mvalid"][ci])[ix]
        rep_cube = [Xall[j] if j is not None else None for j in rep_ix]
        rep_valid = [bool(G["valid"][ci, j]) if j is not None else False for j in rep_ix]
        for i, v, r in centre_rows(X, Mc, sh2, sh4, valid, mvalid, fam, sex, age, rep_cube, rep_valid,
                                   abs(x_mm[ci]) >= P.MIDLINE_MM):
            vsum[ci] += v
            vn[ci] += 1
            for f, e in r.items():
                acc[(f, half[i], "err")][ci] += e
                acc[(f, half[i], "V")][ci] += v
                acc[(f, half[i], "n")][ci] += 1
        for h in (0, 1):
            for i, v, r in centre_rows(X, Mc, sh2, sh4, valid, mvalid, fam, sex, age, rep_cube, rep_valid,
                                       abs(x_mm[ci]) >= P.MIDLINE_MM, half == h):
                for f, e in r.items():
                    ind[(f, h, "err")][ci] += e
                    ind[(f, h, "V")][ci] += v
        if ci % 200 == 0:
            print(f"centre {ci + 1}/{len(cs)}", flush=True)
    shape = tuple(-(-n // STEP) for n in V.shape[1:])
    grid_idx = tuple(((cs - np.array([(a // 2) % STEP, 0, 0])) // STEP).T)
    aff8 = affine.copy()
    aff8[:3, :3] *= STEP
    aff8[0, 3] += affine[0, 0] * ((a // 2) % STEP)
    out = {"status": "pre-QC: affine registration only, MRI registration QC not met; exploratory maps",
           "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendments 7, 8)", "grid_mm": STEP, "cube_mm": CUBE,
           "n_centres": int(len(cs)), "maps": {}}
    vol = np.full(shape, np.nan, np.float32)
    vol[grid_idx] = np.where(vn > 0, vsum / np.maximum(vn, 1), np.nan)
    nib.save(nib.Nifti1Image(vol, aff8), os.path.join(P.PRIVATE, "grid_V.nii.gz"))
    for f in FILLS:
        e = acc[(f, 0, "err")] + acc[(f, 1, "err")]
        v = acc[(f, 0, "V")] + acc[(f, 1, "V")]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(v > 0, e / v, np.nan)
            shared = [np.where(acc[(f, h, "V")] > 0, acc[(f, h, "err")] / acc[(f, h, "V")], np.nan) for h in (0, 1)]
            halves = [np.where(ind[(f, h, "V")] > 0, ind[(f, h, "err")] / ind[(f, h, "V")], np.nan) for h in (0, 1)]
        vol = np.full(shape, np.nan, np.float32)
        vol[grid_idx] = ratio
        nib.save(nib.Nifti1Image(vol, aff8), os.path.join(P.PRIVATE, f"grid_{f}_ratio.nii.gz"))
        for h in (0, 1):
            hv = np.full(shape, np.nan, np.float32)
            hv[grid_idx] = halves[h]
            nib.save(nib.Nifti1Image(hv, aff8), os.path.join(P.PRIVATE, f"grid_{f}_ratio_h{h}.nii.gz"))
        oks = np.isfinite(shared[0]) & np.isfinite(shared[1])
        ok = np.isfinite(halves[0]) & np.isfinite(halves[1])
        fin = np.isfinite(ratio)
        out["maps"][f] = {"pooled_ratio": float(e.sum() / v.sum()) if v.sum() > 0 else None,
                          "quantiles_5_25_50_75_95": [float(q) for q in np.percentile(ratio[fin], [5, 25, 50, 75, 95])],
                          "split_half_r": float(np.corrcoef(halves[0][ok], halves[1][ok])[0, 1]) if ok.sum() > 10 else None,
                          "split_half_r_shared_pool": (float(np.corrcoef(shared[0][oks], shared[1][oks])[0, 1])
                                                       if oks.sum() > 10 else None),
                          "centres": int(fin.sum()), "max_targets_per_centre": int((acc[(f, 0, "n")] + acc[(f, 1, "n")]).max())}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(B.finite(out), fh, indent=1, allow_nan=False)
    print(json.dumps(out["maps"]))


if __name__ == "__main__":
    main()
