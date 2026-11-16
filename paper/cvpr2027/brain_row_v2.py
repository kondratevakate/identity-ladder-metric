"""Brain row cells drawn from the v2 brain maps (src/donor_ladder_brain_maps_v2.py, edge-reaching validity rule).

Display only: axial MNI z = 10 mm, the nilearn MNI152 template as underlay, ratio maps on the shared diverging
scale (0, 1, 2.4) and the dataset-mean error V on the red half of the same palette (its own scale), as in the
other rows. Gray anatomy shows where a map has no value (the mirror midline, |x| < 8 mm). The aggregate maps are
private; only these population-level images and whole-map summaries leave the private folder.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import donor_ladder_brain as B  # noqa: E402

PRIVATE = Path(B.PROJ) / "donor-ladder-brain-maps-20261115-v2"
SUMMARY = ROOT / "results" / "donor_ladder_brain_maps_v2.json"
Z_MM = 10
ALPHA = 0.85


def available():
    return SUMMARY.is_file() and (PRIVATE / "V.nii.gz").is_file()


def _slice(img, z_mm):
    inv = np.linalg.inv(img.affine)
    k = int(round((inv @ np.array([0, 0, z_mm, 1]))[2]))
    return np.asanyarray(img.dataobj, dtype=np.float32)[:, :, k].T[::-1]   # rows top to bottom, anterior up


def cells(scale=6):
    """{key: RGBA PIL image} for anatomy, repeat, mirror, stranger and V; plus {key: pooled ratio}."""
    import nibabel as nib
    from nilearn.datasets import load_mni152_template
    from nilearn.image import resample_to_img
    ref = nib.load(PRIVATE / "V.nii.gz")
    under = _slice(resample_to_img(load_mni152_template(resolution=1), ref, interpolation="continuous"), Z_MM)
    brain = under > 0.1 * np.nanmax(under)          # template tissue; interpolation noise stays outside
    lo, hi = np.nanpercentile(under[brain], [1, 99])
    gray = np.clip((under - lo) / (hi - lo), 0, 1)
    base = np.dstack([gray] * 3)
    out = {}

    def finish(rgb):
        a = np.dstack([rgb, brain.astype(np.float32)])
        ys, xs = np.nonzero(brain)
        a = a[max(ys.min() - 1, 0):ys.max() + 2, max(xs.min() - 1, 0):xs.max() + 2]
        im = Image.fromarray((a * 255).astype(np.uint8), "RGBA")
        return im.resize((im.width * scale, im.height * scale), Image.NEAREST)

    out["anatomy"] = finish(base)
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.4)
    for key, name in (("repeat", "repeat_ratio"), ("mirror", "mirror_ratio"), ("stranger", "stranger_ratio")):
        m = _slice(nib.load(PRIVATE / f"{name}.nii.gz"), Z_MM)
        col = colormaps["RdBu_r"](norm(np.nan_to_num(m, nan=1.0)))[:, :, :3]
        ok = np.isfinite(m) & brain
        rgb = np.where(ok[:, :, None], ALPHA * col + (1 - ALPHA) * base, base)
        out[key] = finish(rgb)
    v = _slice(nib.load(PRIVATE / "V.nii.gz"), Z_MM)
    v = np.where(v == 0, np.nan, v)
    ok = np.isfinite(v) & brain
    v_lo, v_hi = (float(x) for x in np.nanpercentile(v[ok], [2, 98]))   # same 2nd to 98th percentile as the other rows
    vn = Normalize(v_lo, v_hi)(np.nan_to_num(v, nan=0.0))
    col = colormaps["RdBu_r"](0.5 + np.clip(vn, 0, 1) / 2)[:, :, :3]
    out["V"] = finish(np.where(ok[:, :, None], ALPHA * col + (1 - ALPHA) * base, base))
    res = json.loads(SUMMARY.read_text(encoding="utf-8"))["maps"]
    ratios = {k: res[k]["pooled_ratio"] for k in ("repeat", "mirror", "stranger")}
    ratios["V_range"] = (v_lo, v_hi)
    return out, ratios
