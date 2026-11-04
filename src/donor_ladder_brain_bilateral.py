"""Bilateral and look-alike variants of the QTIM brain donor ladder (PROTOCOL_donor_ladder_brain.md, Amendment 2).

Pre-QC; affine only. Each position is run on both hemispheres: the right side is the same experiment on the mirrored
scan, so every cube stays voxel-aligned. Per scan and position the run keeps a region of the scan and of its mirror
large enough for the context shell plus a +-W mm block search. The original variants of donor_ladder_brain are
re-run unchanged next to the new ones (kNN by oracle or by own mirrored cube, mirror block search, strangers on the
other side). Region arrays and rows stay in the private folder; the public JSON holds aggregates only.
"""
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_fundus as F  # noqa: E402

SEED, SHELL = B.SEED, B.SHELL
W = 8                     # block-search half-window, mm (voxels on the 1 mm grid)
VALID = 0.95              # share of cube voxels inside the brain mask
PRIVATE = os.path.join(B.PROJ, "donor-ladder-brain-bilateral-20261103-v1")
OUT_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_bilateral.json")
SIDES = ("L", "R")


def mirror_constant(affine, n):
    """A such that the mirrored voxel of x index i is A - i (asserts the reflection is a pure index flip)."""
    mi = B.mirror_index(affine, n)
    a = int(mi[0])
    if not np.array_equal(mi, a - np.arange(n)):
        raise RuntimeError("mirror is not an index flip on this grid")
    return a


def side_of(start, size, affine):
    """'L' if the cube centre lies at x < 0 mm, else 'R'."""
    return "L" if affine[0, 0] * (start[0] + size[0] / 2 - 0.5) + affine[0, 3] < 0 else "R"


def own_frame(side, drawn_side):
    """Frame 0 is the scan, frame 1 its mirror; a side is read from the scan where it was drawn."""
    return 0 if side == drawn_side else 1


def allowed_offsets(start_x, n, a, w=W):
    """Bool (2w+1,) over x offsets: the candidate's source interval may overlap the target's cube in x no more
    than the unshifted mirrored cube does. The candidate at other-frame start s+dx covers own-frame
    [a-s-dx-n+1, a-s-dx+1)."""
    def overlap(dx):
        lo, hi = a - start_x - dx - n + 1, a - start_x - dx + 1
        return max(0, min(hi, start_x + n) - max(lo, start_x))
    o0 = overlap(0)
    return np.array([overlap(dx) <= o0 for dx in range(-w, w + 1)])


def shell_mask(size, shell=None):
    shell = SHELL if shell is None else shell
    m = np.ones(tuple(n + 2 * shell for n in size), bool)
    m[tuple(slice(shell, shell + n) for n in size)] = False
    return m


def match_mask(size, shell=None, oracle=False):
    """Voxels of the window compared by the search: the shell, or the cube itself for the oracle search."""
    sm = shell_mask(size, shell)
    return ~sm if oracle else sm


def search_terms(other, other_mask, size, shell=None, oracle=False):
    """Donor-only terms of the block search: sum of other^2 under the match mask and cube validity, per offset."""
    shell = SHELL if shell is None else shell
    mm = match_mask(size, shell, oracle).astype(np.float32)
    o2 = signal.correlate(other.astype(np.float64) ** 2, mm, mode="valid", method="fft")
    k = np.zeros(mm.shape, np.float32)
    k[tuple(slice(shell, shell + n) for n in size)] = 1
    frac = signal.correlate(other_mask.astype(np.float32), k, mode="valid", method="fft") / np.prod(size)
    return o2, frac >= VALID - 1e-4


def block_search(win, other, size, o2, valid, allowed_x, shell=None, oracle=False):
    """Offset of the best match of the target window `win` within the region `other`, and the cube there.

    win: target window (size + 2 shell); other: region (size + 2 shell + 2 W); o2, valid from search_terms with
    the same shell and oracle flag. The match uses the shell, or the true cube when `oracle` (an upper bound).
    Distance = sum_mask (win - other_o)^2 up to the constant sum_mask win^2. Returns (None, None) when no offset
    is valid and allowed.
    """
    shell = SHELL if shell is None else shell
    mm = match_mask(size, shell, oracle)
    cross = signal.correlate(other.astype(np.float64), np.where(mm, win, 0).astype(np.float64),
                             mode="valid", method="fft")
    d = o2 - 2 * cross
    ok = valid & allowed_x[:, None, None]
    if not ok.any():
        return None, None
    d = np.where(ok, d, np.inf)
    o = np.unravel_index(int(np.argmin(d)), d.shape)
    cube = other[tuple(slice(oi + shell, oi + shell + n) for oi, n in zip(o, size))]
    return tuple(int(oi) - W for oi in o), cube


def region_paths(pi):
    return os.path.join(PRIVATE, f"vol{pi}.npy"), os.path.join(PRIVATE, f"mask{pi}.npy")


def extract(t, pos, affine):
    """Per position, memmapped (scans, frame, region) arrays of z-scored intensity and brain mask."""
    pad = SHELL + W
    vols, masks = [], []
    for pi, (_, _, size) in enumerate(pos):
        shape = (len(t), 2) + tuple(n + 2 * pad for n in size)
        pv, pm = region_paths(pi)
        vols.append(np.lib.format.open_memmap(pv, "w+", np.float32, shape))
        masks.append(np.lib.format.open_memmap(pm, "w+", np.uint8, shape))
    for k, f in enumerate(t.path):
        img = nib.load(f.replace("_brain.nii.gz", ".nii.gz"))
        if not np.allclose(img.affine, affine, atol=1e-3):
            raise RuntimeError("scan grid differs from the template grid")
        v = img.get_fdata(dtype=np.float32)
        m = nib.load(f).get_fdata(dtype=np.float32) > 0.5
        if v[m].std() <= 0:
            raise RuntimeError("degenerate scan: no intensity variation inside the brain mask")
        z = np.where(m, (v - v[m].mean()) / (v[m].std() + 1e-6), 0).astype(np.float32)
        mi = B.mirror_index(affine, v.shape[0])
        frames = [(np.pad(z, pad), np.pad(m, pad)), (np.pad(z[mi], pad), np.pad(m[mi], pad))]
        for pi, (_, start, size) in enumerate(pos):
            sl = tuple(slice(a, a + n + 2 * pad) for a, n in zip(start, size))  # padded coords = start - pad
            for fr, (zz, mm) in enumerate(frames):
                vols[pi][k, fr] = zz[sl]
                masks[pi][k, fr] = mm[sl]
        if k % 25 == 0:
            print(f"extract {k + 1}/{len(t)}", flush=True)
    for a in vols + masks:
        a.flush()
    open(os.path.join(PRIVATE, "regions.done"), "w").close()


def knn_rows(i, pi, xi, X, others, dist, mu, cohort, rows):
    nn = others[np.argsort(dist[others], kind="stable")]
    for k in F.KS:
        if nn.size >= k:
            f = X[nn[:k]].mean(0)
            rows.append((i, pi, cohort, k, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))


def run_side(t, tg, pi, size, vol, mask, fo, allowed_x, shell=None, oracle=False):
    """Rows (target, patch, cohort, k, err, spread) of one position and side; fo is the own frame (0 or 1).
    Regions are (size + 2 shell + 2 W) wide; `oracle` adds the oracle block searches (Amendment 4)."""
    shell = SHELL if shell is None else shell
    n = len(tg)
    fam, sex, age = tg.family.to_numpy(), tg.sex.to_numpy(), tg.age.to_numpy()
    ix = tg["index"].to_numpy()
    row_of = {(r.person, r.session): r.Index for r in t.itertuples()}
    win_sl = tuple(slice(W, W + s + 2 * shell) for s in size)
    cube_sl = tuple(slice(W + shell, W + shell + s) for s in size)
    sm = shell_mask(size, shell)
    own, other = vol[:, fo], vol[:, 1 - fo]
    cube_all = own[(slice(None),) + cube_sl].reshape(len(t), -1)
    valid_all = mask[:, fo][(slice(None),) + cube_sl].reshape(len(t), -1).mean(1) >= VALID
    X, valid = cube_all[ix], valid_all[ix]
    M = other[(slice(None),) + cube_sl][ix].reshape(n, -1)       # slice first: regions may be memmaps
    mvalid = mask[:, 1 - fo][(slice(None),) + cube_sl][ix].reshape(n, -1).mean(1) >= VALID
    win = own[(slice(None),) + win_sl][ix]
    C = win[:, sm].astype(np.float64)
    Xd = X.astype(np.float64)
    sq_c, sq_x = (C ** 2).sum(1), (Xd ** 2).sum(1)
    D_shell = sq_c[:, None] + sq_c[None] - 2 * C @ C.T
    D_oracle = sq_x[:, None] + sq_x[None] - 2 * Xd @ Xd.T
    Md = M.astype(np.float64)
    D_mirror = (Md ** 2).sum(1)[:, None] + sq_x[None] - 2 * Md @ Xd.T
    terms = {}

    def search(j, i, orc=False):
        if (j, orc) not in terms:
            terms[(j, orc)] = search_terms(other[ix[j]], mask[ix[j], 1 - fo], size, shell, orc)
        return block_search(win[i], other[ix[j]], size, *terms[(j, orc)], allowed_x, shell, orc)[1]

    rows = []
    for i in np.flatnonzero(valid):
        others = np.flatnonzero((fam != fam[i]) & valid)
        if others.size < 10:
            continue
        mu = X[others].mean(0)
        xi = X[i]
        pools = {"population": others, "cohort": others[(sex[others] == sex[i]) & (np.abs(age[others] - age[i]) <= 2)]}
        orders = {}
        for c in B.DRAWN:
            if c == "population_both":
                mo = np.flatnonzero((fam != fam[i]) & mvalid)
                src = np.concatenate([X[others], M[mo]])
                order = np.random.default_rng([SEED, i, pi, 1]).permutation(len(src))
            else:
                order = np.random.default_rng([SEED, i, pi, B.DRAWN.index(c)]).permutation(pools[c])
                src = X
            orders[c] = order
            for k in F.KS + ("all",):
                if k == "all" and c == "population_both":
                    continue
                sel = order if k == "all" else order[:k]
                if k != "all" and sel.size < k:
                    continue
                f = src[sel].mean(0)
                rows.append((i, pi, c, k, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))
        knn_rows(i, pi, xi, X, others, D_shell[i], mu, "knn", rows)
        knn_rows(i, pi, xi, X, others, D_oracle[i], mu, "knn_oracle", rows)
        single = {}
        if mvalid[i]:
            single["other_hemisphere"] = M[i]
            knn_rows(i, pi, xi, X, others, D_mirror[i], mu, "knn_mirror", rows)
        cube = search(i, i)
        if cube is not None:
            single["mirror_block"] = cube.ravel()
        s_other = [j for j in orders["population"] if mvalid[j]]
        if s_other:
            single["stranger_other_side"] = M[s_other[0]]
        cube = search(int(orders["population"][0]), i)
        if cube is not None:
            single["stranger_block"] = cube.ravel()
        if oracle:
            for c, j in (("mirror_block_oracle", i), ("stranger_block_oracle", int(orders["population"][0]))):
                cube = search(j, i, True)
                if cube is not None:
                    single[c] = cube.ravel()
        rel = np.flatnonzero((fam == fam[i]) & (np.arange(n) != i) & valid)
        sib = [j for j in rel if age[j] != age[i]]
        twin = [j for j in rel if age[j] == age[i]]
        if sib:
            single["sibling"] = X[sib[0]]
        if twin:
            single["cotwin"] = X[twin[0]]
        j2 = row_of.get((tg.person.iloc[i], "ses-02"))
        if j2 is not None and valid_all[j2]:
            single["repeat"] = cube_all[j2]
        for c, f in single.items():
            rows.append((i, pi, c, 1, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))
    return rows


def run(t, pos, affine):
    tg = t[t.session == "ses-01"].reset_index()
    a = mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    rows = []
    for pi, (region, start, size) in enumerate(pos):
        pv, pm = region_paths(pi)
        vol, mask = np.load(pv, mmap_mode="r"), np.load(pm, mmap_mode="r")
        vol, mask = np.asarray(vol), np.asarray(mask)
        drawn = side_of(start, size, affine)
        allowed_x = allowed_offsets(start[0], size[0], a)
        for side in SIDES:
            for r in run_side(t, tg, pi, size, vol, mask, own_frame(side, drawn), allowed_x):
                rows.append(r + (side, side == drawn))
        print(f"patch {pi + 1}/{len(pos)}", flush=True)
    r = pd.DataFrame(rows, columns=["target", "patch", "cohort", "k", "err", "spread", "side", "drawn"])
    r["k"] = r["k"].astype(str)
    r["person"] = tg.family.to_numpy()[r.target]
    r["region"] = [pos[p][0] for p in r.patch]
    return r, len(tg)


def views(r):
    """Rows relabelled per reported view: drawn side (orig), left, right and both sides pooled (LR)."""
    parts = [r[r.drawn].assign(region=r.region[r.drawn] + "/orig"),
             r[r.side == "L"].assign(region=r.region[r.side == "L"] + "/L"),
             r[r.side == "R"].assign(region=r.region[r.side == "R"] + "/R"),
             r.assign(region=r.region + "/LR")]
    return pd.concat(parts, ignore_index=True)


def main():
    os.makedirs(PRIVATE, exist_ok=True)
    t0 = time.time()
    t = B.scans()
    pos, affine, n_cand = B.positions(np.random.default_rng(SEED))
    if not os.path.exists(os.path.join(PRIVATE, "regions.done")):
        extract(t, pos, affine)
    r, n = run(t, pos, affine)
    r.to_csv(os.path.join(PRIVATE, "rows.csv.gz"), index=False)
    for name in ("rows_hemisphere.csv.gz", "rows_hemisphere_oracle.csv.gz"):   # Amendments 3-4, if computed
        hemi = os.path.join(PRIVATE, name)
        if os.path.exists(hemi):
            r = pd.concat([r, pd.read_csv(hemi, dtype={"k": str})], ignore_index=True)
    write_summary(r, t, n, n_cand, t0)


def write_summary(r, t, n, n_cand, t0):
    """Public aggregates of all rows (bilateral and, when present, whole-hemisphere search)."""
    out = {"status": "pre-QC: affine registration only, MRI registration QC not met",
           "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendments 2 and 3)", "seed": SEED, "patch_mm": B.PATCH,
           "shell_mm": SHELL, "block_search_halfwindow_mm": W,
           "hemisphere_search": "mirror_hemi / stranger_hemi: whole contralateral hemisphere (Amendment 3)",
           "views": {"orig": "side as drawn (reproduces donor_ladder_brain.json)", "L": "left hemisphere",
                     "R": "right hemisphere (mirrored scan)", "LR": "both sides pooled per target, pooled V"},
           "QTIM": F.summarize(views(r), n_targets=n, n_scans=len(t), n_people=int(t.person.nunique()),
                               n_families=int(t.family.nunique()), n_candidates=n_cand,
                               n_repeat=int((t.session == "ses-02").sum()))}
    out["QTIM"]["counts"]["minutes"] = round((time.time() - t0) / 60, 1)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(B.finite(out), fh, indent=1, allow_nan=False)
    print(json.dumps(out["QTIM"]["counts"]))


if __name__ == "__main__":
    main()
