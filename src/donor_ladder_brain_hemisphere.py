"""Whole-hemisphere block search for the QTIM brain donor ladder (PROTOCOL_donor_ladder_brain.md, Amendment 3).

Pre-QC; affine only. For every position, side and target of donor_ladder_brain_bilateral, the target's context shell
is matched against every cube position of the contralateral hemisphere (the target's half of the mirrored frame) of
its own scan (`mirror_hemi`) and of the first population stranger (`stranger_hemi`). The searches are grouped by the
searched scan, so each scan is loaded once and the FFT of each half volume is reused for all kernels. Rows are added
to the bilateral rows and the public JSON is rewritten from both. Aggregates only leave the private folder.
"""
import os
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import fft

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as L  # noqa: E402

SHELL, VALID, SEED = L.SHELL, L.VALID, L.SEED
ROWS = os.path.join(L.PRIVATE, "rows_hemisphere.csv.gz")
WORKERS = 4


def half_starts(half, n, a, nx):
    """Inclusive range of cube x starts whose cube lies wholly on one side of the midline (index a / 2 = 0 mm)."""
    mid = a // 2
    return (0, mid - n + 1) if half == "L" else (mid, nx - n)


def _corr(fa, k, s, out):
    """Valid-mode cross-correlation from the FFT of the big array and a kernel: c[o] = sum_t A[o + t] k[t]."""
    return fft.irfftn(fa * np.conj(fft.rfftn(k, s, workers=WORKERS)), s, workers=WORKERS)[out]


def prepare(z, m, half, size, a):
    """Donor-side FFT terms of one frame for cubes of `size` in one half (padded crop, o2, validity)."""
    lo, hi = half_starts(half, size[0], a, z.shape[0])
    P = np.pad(z.astype(np.float64), SHELL)[lo:hi + size[0] + 2 * SHELL]   # padded index o = cube start o
    Pm = np.pad(m.astype(np.float64), SHELL)[lo:hi + size[0] + 2 * SHELL]
    s = tuple(fft.next_fast_len(d, real=True) for d in P.shape)
    kshape = tuple(n + 2 * SHELL for n in size)
    out = tuple(slice(0, d - k + 1) for d, k in zip(P.shape, kshape))
    sm = L.shell_mask(size)
    km = np.zeros(kshape)
    km[tuple(slice(SHELL, SHELL + n) for n in size)] = 1
    fp = fft.rfftn(P, s, workers=WORKERS)
    o2 = _corr(fft.rfftn(P ** 2, s, workers=WORKERS), sm.astype(np.float64), s, out)
    frac = _corr(fft.rfftn(Pm, s, workers=WORKERS), km, s, out) / np.prod(size)
    return {"P": P, "fp": fp, "s": s, "out": out, "o2": o2, "ok": frac >= VALID - 1e-4, "lo": lo, "size": size}


def search(win, prep):
    """Cube start (x, y, z on the scan grid) and cube of the best shell match of `win`; (None, None) if none valid."""
    if not prep["ok"].any():
        return None, None
    size = prep["size"]
    k = np.where(L.shell_mask(size), win, 0).astype(np.float64)
    d = prep["o2"] - 2 * _corr(prep["fp"], k, prep["s"], prep["out"])
    d[~prep["ok"]] = np.inf
    u = np.unravel_index(int(np.argmin(d)), d.shape)
    cube = prep["P"][tuple(slice(ui + SHELL, ui + SHELL + n) for ui, n in zip(u, size))]
    return (prep["lo"] + int(u[0]), int(u[1]), int(u[2])), cube


def plan(t, pos, affine):
    """Search jobs grouped by searched scan, plus per (target, patch, side) the target cube and the population mean."""
    tg = t[t.session == "ses-01"].reset_index()
    ix = tg["index"].to_numpy()
    fam = tg.family.to_numpy()
    jobs, ref = [], {}
    for pi, (_, start, size) in enumerate(pos):
        vol = np.load(L.region_paths(pi)[0], mmap_mode="r")
        mask = np.load(L.region_paths(pi)[1], mmap_mode="r")
        cube_sl = tuple(slice(L.W + SHELL, L.W + SHELL + n) for n in size)
        drawn = L.side_of(start, size, affine)
        for side in L.SIDES:
            fo = L.own_frame(side, drawn)
            X = np.stack([np.asarray(vol[k, fo][cube_sl]).ravel() for k in ix])
            valid = np.array([np.asarray(mask[k, fo][cube_sl]).mean() >= VALID for k in ix])
            for i in np.flatnonzero(valid):
                others = np.flatnonzero((fam != fam[i]) & valid)
                if others.size < 10:
                    continue
                ref[(i, pi, side)] = (X[i], X[others].mean(0))
                j = int(np.random.default_rng([SEED, i, pi, 0]).permutation(others)[0])
                for scan, cohort in ((i, "mirror_hemi"), (j, "stranger_hemi")):
                    jobs.append((int(ix[scan]), 1 - fo, drawn, tuple(size), pi, int(i), side, cohort, drawn))
        print(f"plan {pi + 1}/{len(pos)}", flush=True)
    jobs.sort(key=lambda j: j[:4])
    return tg, jobs, ref


def run(t, pos, affine):
    tg, jobs, ref = plan(t, pos, affine)
    a = L.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    ix = tg["index"].to_numpy()
    rows, frames, prep, cur = [], None, {}, None
    for n_done, (scan, frame, half, size, pi, i, side, cohort, drawn) in enumerate(jobs):
        if scan != cur:
            f = t.path.iloc[scan]
            v = nib.load(f.replace("_brain.nii.gz", ".nii.gz")).get_fdata(dtype=np.float32)
            m = nib.load(f).get_fdata(dtype=np.float32) > 0.5
            z = np.where(m, (v - v[m].mean()) / (v[m].std() + 1e-6), 0).astype(np.float32)
            mi = B.mirror_index(affine, v.shape[0])
            frames, prep, cur = [(z, m), (z[mi], m[mi])], {}, scan
        key = (frame, half, size)
        if key not in prep:
            prep = {key: prepare(*frames[frame], half, size, a)}   # jobs are sorted: keep one at a time
        vol = np.load(L.region_paths(pi)[0], mmap_mode="r")
        win = np.asarray(vol[ix[i], L.own_frame(side, drawn)][tuple(slice(L.W, L.W + n + 2 * SHELL) for n in size)])
        _, cube = search(win, prep[key])
        if cube is not None:
            xi, mu = ref[(i, pi, side)]
            fl = cube.ravel()
            rows.append((i, pi, cohort, "1", float(np.mean((xi - fl) ** 2)), float(np.mean((fl - mu) ** 2)),
                         side, side == drawn))
        if n_done % 1000 == 0:
            print(f"search {n_done + 1}/{len(jobs)}", flush=True)
    r = pd.DataFrame(rows, columns=["target", "patch", "cohort", "k", "err", "spread", "side", "drawn"])
    r["person"] = tg.family.to_numpy()[r.target]
    r["region"] = [pos[p][0] for p in r.patch]
    return r


def main():
    t0 = time.time()
    t = B.scans()
    pos, affine, n_cand = B.positions(np.random.default_rng(SEED))
    if not os.path.exists(ROWS):
        run(t, pos, affine).to_csv(ROWS, index=False)
    base = pd.read_csv(os.path.join(L.PRIVATE, "rows.csv.gz"), dtype={"k": str})
    hemi = pd.read_csv(ROWS, dtype={"k": str})
    L.write_summary(pd.concat([base, hemi], ignore_index=True), t, int((t.session == "ses-01").sum()), n_cand, t0)


if __name__ == "__main__":
    main()
