"""Brain maps v2 (Kate, 15 Nov 2026): the v1 maps of donor_ladder_brain_maps.py with a validity rule that reaches
the brain edge. v1 (results/donor_ladder_brain_maps.json, donor-ladder-brain-maps-20261104-v1) is kept unchanged.

v1 kept a voxel only when 95% of the 16 mm box around it lay inside the mask, which left a gray rim several mm wide
along the brain surface. v2 keeps a grid voxel when it is itself inside the mask and the masked Gaussian that
computes the local error (sd = 16 mm / sqrt(12)) has at least half of its weight inside the mask. The local error
was already a masked Gaussian mean over valid voxels only, so nothing outside the mask enters an error and missing
stays missing (never zero). The mirror map keeps the v1 midline exclusion (|x| < 8 mm), where the mirrored side is
the same tissue. Donor rules, pools, units and the split-half design are unchanged.

Usage (repository root): python -I src/donor_ladder_brain_maps_v2.py            full run (about 80 minutes)
                         python -I src/donor_ladder_brain_maps_v2.py --smoke N  first N families, one pass, QC sheet
"""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain_maps as P  # noqa: E402

VALID_WEIGHT = 0.5


def frac_edge(m):
    """Gaussian mask weight around every grid voxel, zero where the voxel itself is outside the mask."""
    m = np.asarray(m, np.float32)
    w = ndimage.gaussian_filter(m, P.SD, mode="constant")[::P.STEP, ::P.STEP, ::P.STEP]
    return np.where(m[::P.STEP, ::P.STEP, ::P.STEP] > 0, w, 0.0).astype(np.float32)


P.frac = frac_edge
P.VALID = VALID_WEIGHT
P.PRIVATE = os.path.join(P.B.PROJ, "donor-ladder-brain-maps-20261115-v2")
P.OUT_JSON = os.path.join(P.B.ROOT, "results", "donor_ladder_brain_maps_v2.json")


def smoke(n_families):
    """One pass over the first n families (pools from the same subset), saves a private QC sheet: v1 and v2
    stranger and mirror ratio maps at the slice used in the figures (MNI z = 10 mm)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from matplotlib.colors import TwoSlopeNorm

    os.makedirs(P.PRIVATE, exist_ok=True)
    t_all = P.B.scans()
    fams = sorted(t_all.family.unique())[:n_families]
    _, affine, _ = P.B.positions(np.random.default_rng(P.B.SEED))
    V_all, M_all = P.S.cache_volumes(t_all, affine)         # cache rows follow the full table
    rows = np.flatnonzero(t_all.family.isin(fams).to_numpy())
    V, M = np.asarray(V_all[rows]), np.asarray(M_all[rows])  # in-memory subset, rows aligned with t below
    t = t_all.iloc[rows].reset_index(drop=True)
    acc, support, midline = P.run(t, affine, V, M)
    aff2 = affine.copy()
    aff2[:3, :3] *= P.STEP
    z = int(round((10 - aff2[2, 3]) / aff2[2, 2]))
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.4)
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    old_dir = os.path.join(P.B.PROJ, "donor-ladder-brain-maps-20261104-v1")
    for c, d in enumerate(("stranger", "mirror")):
        keep = support & (acc.count(d) > 0)
        if d == "mirror":
            keep &= ~midline[:, None, None]
        new = np.where(keep, acc.ratio(d), np.nan)
        old = nib.load(os.path.join(old_dir, f"{d}_ratio.nii.gz")).get_fdata()
        for r, (name, arr) in enumerate((("v1 (all families)", old), (f"v2 smoke ({n_families} families)", new))):
            ax = axes[r, c]
            ax.set_facecolor("#9a9a9a")
            ax.imshow(arr[:, :, z].T, origin="lower", cmap="RdBu_r", norm=norm, interpolation="nearest")
            ax.set_title(f"{d}: {name}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        print(d, "kept voxels v2 smoke:", int(keep.sum()), "v1:", int(np.isfinite(old).sum()), flush=True)
    fig.suptitle("Gray = no value. Brain maps v1 vs v2 validity rule, MNI z = 10 mm", fontsize=8)
    out = os.path.join(P.PRIVATE, "qc_edge_smoke.png")
    fig.savefig(out, dpi=150)
    print("QC sheet:", out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--smoke":
        smoke(int(sys.argv[2]))
    else:
        P.main()
