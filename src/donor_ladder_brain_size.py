"""Patch-size and shell-thickness grid of the QTIM brain donor ladder (PROTOCOL_donor_ladder_brain.md, Amendment 4).

Pre-QC; affine only. Every scan is z-scored once into a whole-volume memmap; for each configuration (cube edge p,
shell s) and position the (scans, frame, region) arrays are cut from it into a temporary memmap and run through
donor_ladder_brain_bilateral.run_side, with the oracle block searches. Random cubes keep the centres of the 16 mm
draw; the frozen boxes keep their size and vary only s. Rows stay private; the JSON holds L+R aggregates only.
"""
import gc
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as L  # noqa: E402
import donor_ladder_fundus as F  # noqa: E402

PATCHES = (8, 12, 16, 24, 32)
SHELLS = (2, 4, 8, 12)
PRIVATE = os.path.join(B.PROJ, "donor-ladder-brain-size-20261104-v1")
OUT_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_size.json")
KEEP = {("population", "all"), ("population", "1"), ("cohort", "all"), ("knn", "16"), ("knn_oracle", "16"),
        ("knn_mirror", "16")} | {(c, "1") for c in ("repeat", "other_hemisphere", "mirror_block", "mirror_block_oracle",
                                                     "sibling", "stranger_other_side", "stranger_block",
                                                     "stranger_block_oracle")}


def config_positions(pos, p):
    """Positions of one configuration: random cubes of edge p around the 16 mm centres; boxes unchanged."""
    out = []
    for region, start, size in pos:
        if region == "random_patches":
            c = [a + n // 2 for a, n in zip(start, size)]
            out.append((region, tuple(ci - p // 2 for ci in c), (p,) * 3))
        else:
            out.append((region, start, size))
    return out


def get_box(vol, lo, size, a=None):
    """Box of `size` at `lo` from a (X, Y, Z) array, zero outside the grid; with `a`, from the mirrored frame
    (voxel x of the mirror holds x index a - x of the scan)."""
    idx = [np.arange(l0, l0 + n) for l0, n in zip(lo, size)]
    if a is not None:
        idx[0] = a - idx[0]
    ok = [(i >= 0) & (i < d) for i, d in zip(idx, vol.shape)]
    out = np.zeros(size, vol.dtype)
    if all(o.any() for o in ok):
        out[np.ix_(*ok)] = vol[np.ix_(*[i[o] for i, o in zip(idx, ok)])]
    return out


def cache_volumes(t, affine):
    """Whole-volume memmaps of the z-scored scans and their brain masks (scan grid)."""
    pv, pm = os.path.join(PRIVATE, "volumes.npy"), os.path.join(PRIVATE, "masks.npy")
    if os.path.exists(os.path.join(PRIVATE, "volumes.done")):
        return np.load(pv, mmap_mode="r"), np.load(pm, mmap_mode="r")
    shape = nib.load(t.path.iloc[0]).shape
    V = np.lib.format.open_memmap(pv, "w+", np.float32, (len(t),) + shape)
    M = np.lib.format.open_memmap(pm, "w+", np.uint8, (len(t),) + shape)
    for k, f in enumerate(t.path):
        img = nib.load(f.replace("_brain.nii.gz", ".nii.gz"))
        if not np.allclose(img.affine, affine, atol=1e-3):
            raise RuntimeError("scan grid differs from the template grid")
        v = img.get_fdata(dtype=np.float32)
        m = nib.load(f).get_fdata(dtype=np.float32) > 0.5
        if v[m].std() <= 0:
            raise RuntimeError("degenerate scan: no intensity variation inside the brain mask")
        V[k] = np.where(m, (v - v[m].mean()) / (v[m].std() + 1e-6), 0)
        M[k] = m
        if k % 25 == 0:
            print(f"cache {k + 1}/{len(t)}", flush=True)
    V.flush()
    M.flush()
    open(os.path.join(PRIVATE, "volumes.done"), "w").close()
    return np.load(pv, mmap_mode="r"), np.load(pm, mmap_mode="r")


def regions(V, M, start, size, shell, a, tag):
    """Temporary memmaps (scans, frame, region) for one position, region = size + 2 (shell + W)."""
    pad = shell + L.W
    lo = tuple(s0 - pad for s0 in start)
    rs = tuple(n + 2 * pad for n in size)
    vol = np.lib.format.open_memmap(os.path.join(PRIVATE, f"tmp_vol_{tag}.npy"), "w+", np.float32, (len(V), 2) + rs)
    mask = np.lib.format.open_memmap(os.path.join(PRIVATE, f"tmp_mask_{tag}.npy"), "w+", np.uint8, (len(V), 2) + rs)
    for k in range(len(V)):
        for fr, mirror in ((0, None), (1, a)):
            vol[k, fr] = get_box(V[k], lo, rs, mirror)
            mask[k, fr] = get_box(M[k], lo, rs, mirror)
    vol.flush()
    mask.flush()
    return vol, mask


def drop_tmp():
    """Remove temporary region memmaps; one still mapped on Windows is left for the next call."""
    for fn in os.listdir(PRIVATE):
        if fn.startswith("tmp_"):
            try:
                os.remove(os.path.join(PRIVATE, fn))
            except PermissionError:
                pass


def run_config(t, pos, affine, V, M, p, s):
    tg = t[t.session == "ses-01"].reset_index()
    a = L.mirror_constant(affine, V.shape[1])
    rows = []
    for pi, (region, start, size) in enumerate(config_positions(pos, p)):
        if region != "random_patches" and p != B.PATCH:
            continue                                   # boxes vary only with the shell
        vol, mask = regions(V, M, start, size, s, a, pi)
        drawn = L.side_of(start, size, affine)
        allowed_x = L.allowed_offsets(start[0], size[0], a)
        for side in L.SIDES:
            for r in L.run_side(t, tg, pi, size, vol, mask, L.own_frame(side, drawn), allowed_x, s, True):
                rows.append(r + (side, side == drawn))
        del vol, mask
        gc.collect()
        drop_tmp()
    r = pd.DataFrame(rows, columns=["target", "patch", "cohort", "k", "err", "spread", "side", "drawn"])
    r["k"] = r["k"].astype(str)
    r["person"] = tg.family.to_numpy()[r.target]
    r["region"] = [pos[q][0] for q in r.patch]
    return r


def summary(r):
    """L+R aggregates of the kept fills (both sides pooled per target, pooled V)."""
    keep = r[[(c, k) in KEEP for c, k in zip(r.cohort, r.k)]]
    res = F.summarize(keep)["regions"]
    return {reg: {c: {k: {x: v[x] for x in ("err_over_V", "ci95", "n_targets", "n_persons")} for k, v in d.items()
                      if k != "fit"} for c, d in res[reg]["cohorts"].items()} for reg in res}


def main():
    os.makedirs(PRIVATE, exist_ok=True)
    t0 = time.time()
    t = B.scans()
    pos, affine, _ = B.positions(np.random.default_rng(B.SEED))
    V, M = cache_volumes(t, affine)
    out = {"status": "pre-QC: affine registration only, MRI registration QC not met",
           "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendment 4)", "seed": B.SEED,
           "patch_mm": list(PATCHES), "shell_mm": list(SHELLS), "block_search_halfwindow_mm": L.W,
           "view": "L+R: both sides pooled per target, pooled V", "configs": {}}
    if os.path.exists(OUT_JSON):
        out["configs"] = json.load(open(OUT_JSON, encoding="utf-8"))["configs"]
    for p in PATCHES:
        for s in SHELLS:
            key = f"p{p}_s{s}"
            f = os.path.join(PRIVATE, f"rows_{key}.csv.gz")
            if not os.path.exists(f):
                run_config(t, pos, affine, V, M, p, s).to_csv(f, index=False)
            if key not in out["configs"]:
                r = pd.read_csv(f, dtype={"k": str})
                out["configs"][key] = {"patch_mm": p, "shell_mm": s, "regions": summary(r.assign(region=r.region + "/LR"))}
                with open(OUT_JSON, "w", encoding="utf-8") as fh:
                    json.dump(B.finite(out), fh, indent=1, allow_nan=False)
            print(f"config {key} done, {round((time.time() - t0) / 60, 1)} min", flush=True)
    drop_tmp()


if __name__ == "__main__":
    main()
