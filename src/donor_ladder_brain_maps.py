"""Brain-wide variability maps of the QTIM donor ladder (PROTOCOL_donor_ladder_brain.md, Amendment 5). Pre-QC.

Local MSE in a 16 mm window around every voxel is a box filter of the voxel-wise squared difference, kept on a 2 mm
grid. Donors follow fixed rules, never a selection on the images: the person's repeat scan, own mirror at the same
place, the mean over all different-age siblings, and the exact expectation over every stranger. The unit V is the
error of the mean of all session-1 scans outside the target's family. Families are split into two halves for a
split-half reliability of each map. Aggregate maps stay private; the JSON holds whole-map summaries only.
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
import donor_ladder_brain_size as S  # noqa: E402

WIN, STEP, VALID, SUPPORT, MIDLINE_MM = 16, 2, 0.95, 0.95, 8
DONORS = ("repeat", "mirror", "sibling", "stranger", "cohort")
COHORT_MIN, AGE_WINDOW = 10, 2
PRIVATE = os.path.join(B.PROJ, "donor-ladder-brain-maps-20261104-v1")
OUT_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_maps.json")


def box(x):
    """Mean of x in the WIN-voxel cube around every voxel (zero outside the grid), sampled on the STEP grid."""
    return ndimage.uniform_filter(x.astype(np.float32), WIN, mode="constant")[::STEP, ::STEP, ::STEP]


def leave_family_out(total_s, total_q, n_total, fam_s, fam_q, n_fam):
    """Mean and voxel variance of the pool without one family, from running sums of v and v^2."""
    n = n_total - n_fam
    mu = (total_s - fam_s) / n
    return mu, np.maximum((total_q - fam_q) / n - mu ** 2, 0)


def stranger_expectation(v, mu, pool_var):
    """Exact mean over the pool of |v - v_j|^2 voxel-wise: |v - mu|^2 + the pool variance."""
    return (v - mu) ** 2 + pool_var


def cohort_mean(groups, sex, age, family):
    """Mean of the same-sex donors within AGE_WINDOW years outside the family, from (sex, age) running sums;
    None if fewer than COHORT_MIN donors. family: (sex, age, volume) of the target's family members."""
    keys = [g for g in groups if g[0] == sex and abs(g[1] - age) <= AGE_WINDOW]
    s = sum(groups[g][0].astype(np.float64) for g in keys)
    n = sum(groups[g][1] for g in keys)
    for fs, fa, fv in family:
        if fs == sex and abs(fa - age) <= AGE_WINDOW:
            s = s - fv
            n -= 1
    return None if n < COHORT_MIN else (s / n).astype(np.float32)


class Acc:
    """Sums of donor error, V and target counts per map and half, on the STEP grid."""

    def __init__(self, shape):
        self.a = {(d, h, q): np.zeros(shape) for d in DONORS for h in (0, 1) for q in ("err", "V", "n")}

    def add(self, donor, half, err, v, ok):
        self.a[(donor, half, "err")] += np.where(ok, err, 0)
        self.a[(donor, half, "V")] += np.where(ok, v, 0)
        self.a[(donor, half, "n")] += ok

    def ratio(self, donor, halves=(0, 1)):
        e = sum(self.a[(donor, h, "err")] for h in halves)
        v = sum(self.a[(donor, h, "V")] for h in halves)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(v > 0, e / v, np.nan)

    def count(self, donor):
        return self.a[(donor, 0, "n")] + self.a[(donor, 1, "n")]


def run(t, affine, V, M):
    tg = t[t.session == "ses-01"].reset_index()
    ix = tg["index"].to_numpy()
    fam = tg.family.to_numpy()
    mi = B.mirror_index(affine, V.shape[1])
    rep = {r.person: r.Index for r in t[t.session == "ses-02"].itertuples()}
    tot_s = np.zeros(V.shape[1:])
    tot_q = np.zeros(V.shape[1:])
    groups = {}                                         # (sex, age) -> [sum of volumes (float32), count]
    support = 0
    for n_done, k in enumerate(ix):
        v = np.asarray(V[k], dtype=np.float64)
        tot_s += v
        tot_q += v ** 2
        g = groups.setdefault((t.sex.iloc[k], float(t.age.iloc[k])), [np.zeros(V.shape[1:], np.float32), 0])
        g[0] += v.astype(np.float32)
        g[1] += 1
        support = support + (box(np.asarray(M[k])) >= VALID)
        if n_done % 50 == 0:
            print(f"pool {n_done + 1}/{len(ix)}", flush=True)
    support = support >= SUPPORT * len(ix)
    fams = np.unique(fam)
    half_of = dict(zip(np.random.default_rng(B.SEED).permutation(fams), np.arange(len(fams)) % 2))
    acc = Acc(support.shape)
    for n_f, f in enumerate(fams):
        members = np.flatnonzero(fam == f)
        vols = {i: np.asarray(V[ix[i]], dtype=np.float32) for i in members}
        oks = {i: box(np.asarray(M[ix[i]])) >= VALID for i in members}
        fs = sum(vols[i].astype(np.float64) for i in members)
        fq = sum(vols[i].astype(np.float64) ** 2 for i in members)
        mu, pvar = leave_family_out(tot_s, tot_q, len(ix), fs, fq, len(members))
        bp = box(pvar)
        h = half_of[f]
        for i in members:
            vi, ok = vols[i], oks[i] & support
            Vi = box((vi - mu) ** 2)
            acc.add("stranger", h, Vi + bp, Vi, ok)
            okm = ok & (box(np.asarray(M[ix[i]])[mi]) >= VALID)
            acc.add("mirror", h, box((vi - vi[mi]) ** 2), Vi, okm)
            sibs = [j for j in members if j != i and tg.age.iloc[j] != tg.age.iloc[i]]
            if sibs:
                e = sum(np.where(oks[j], box((vi - vols[j]) ** 2), 0) for j in sibs)
                n_ok = sum(oks[j].astype(np.float32) for j in sibs)
                with np.errstate(invalid="ignore", divide="ignore"):
                    acc.add("sibling", h, np.where(n_ok > 0, e / np.maximum(n_ok, 1), 0), Vi, ok & (n_ok > 0))
            cm = cohort_mean(groups, tg.sex.iloc[i], tg.age.iloc[i],
                             [(tg.sex.iloc[j], tg.age.iloc[j], vols[j]) for j in members])
            if cm is not None:
                acc.add("cohort", h, box((vi - cm) ** 2), Vi, ok)
            r = rep.get(tg.person.iloc[i])
            if r is not None:
                okr = ok & (box(np.asarray(M[r])) >= VALID)
                acc.add("repeat", h, box((vi - np.asarray(V[r], dtype=np.float32)) ** 2), Vi, okr)
        if n_f % 20 == 0:
            print(f"family {n_f + 1}/{len(fams)}", flush=True)
    x_mm = affine[0, 0] * np.arange(0, V.shape[1], STEP) + affine[0, 3]
    return acc, support, np.abs(x_mm) < MIDLINE_MM


def split_half(acc, donor, keep):
    a, b = acc.ratio(donor, (0,)), acc.ratio(donor, (1,))
    ok = keep & np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else None


def main():
    os.makedirs(PRIVATE, exist_ok=True)
    t0 = time.time()
    t = B.scans()
    _, affine, _ = B.positions(np.random.default_rng(B.SEED))
    V, M = S.cache_volumes(t, affine)
    acc, support, midline = run(t, affine, V, M)
    aff2 = affine.copy()
    aff2[:3, :3] *= STEP
    out = {"status": "pre-QC: affine registration only, MRI registration QC not met; exploratory maps",
           "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendment 5)", "window_mm": WIN, "grid_mm": STEP,
           "support_voxels": int(support.sum()), "maps": {}}
    v_mean = acc.a[("stranger", 0, "V")] + acc.a[("stranger", 1, "V")]
    n_str = acc.count("stranger")
    with np.errstate(invalid="ignore", divide="ignore"):
        vmap = np.where(n_str > 0, v_mean / n_str, np.nan)
    nib.save(nib.Nifti1Image(vmap.astype(np.float32), aff2), os.path.join(PRIVATE, "V.nii.gz"))
    for d in DONORS:
        ratio = acc.ratio(d)
        keep = support & (acc.count(d) > 0)
        if d == "mirror":
            keep &= ~midline[:, None, None]
        ratio = np.where(keep, ratio, np.nan)
        for name, arr in ((f"{d}_ratio", ratio), (f"{d}_n", acc.count(d))):
            nib.save(nib.Nifti1Image(arr.astype(np.float32), aff2), os.path.join(PRIVATE, f"{name}.nii.gz"))
        e = sum(acc.a[(d, h, "err")][keep].sum() for h in (0, 1))
        v = sum(acc.a[(d, h, "V")][keep].sum() for h in (0, 1))
        q = np.nanpercentile(ratio[keep], [5, 25, 50, 75, 95]) if keep.any() else [None] * 5
        out["maps"][d] = {"pooled_ratio": float(e / v) if v > 0 else None, "quantiles_5_25_50_75_95": [float(x) for x in q],
                          "split_half_r": split_half(acc, d, keep), "max_targets_per_voxel": int(acc.count(d).max()),
                          "voxels": int(keep.sum())}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(B.finite(out), fh, indent=1, allow_nan=False)
    print(json.dumps(out["maps"]))


if __name__ == "__main__":
    main()
