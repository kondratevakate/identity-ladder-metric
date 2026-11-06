"""Known-geometry tests for groupwise fundus registration (general protocol, Amendment 9.8)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fundus_register as R  # noqa: E402

ants = pytest.importorskip("ants")


def scene(shift=(0.0, 0.0)):
    """A smooth field with a few dark 'vessels' and a bright 'disc', inside a circular mask."""
    yy, xx = np.mgrid[:96, :96].astype(float)
    x = np.exp(-((xx - 60 - shift[1]) ** 2 + (yy - 48 - shift[0]) ** 2) / 40)
    for a in (0.3, 1.2, 2.0):
        x -= 0.8 * np.exp(-(((yy - 48 - shift[0]) * np.cos(a) - (xx - 60 - shift[1]) * np.sin(a)) ** 2) / 4)
    x = ndimage.gaussian_filter(x, 1).astype(np.float32)
    m = np.hypot(xx - 48, yy - 48) < 44
    return np.where(m, x, 0), m


def test_affine_recovers_a_known_shift():
    t, tm = scene()
    s, sm = scene((4.0, -3.0))
    w, wm, _ = R.register_to(t, tm, s, sm, "R1")
    inner = tm & wm & (ndimage.distance_transform_edt(tm) > 8)
    before = np.mean((s - t)[inner & sm] ** 2)
    after = np.mean((w - t)[inner] ** 2)
    assert after < 0.2 * before


def test_warped_mask_never_extends_the_field():
    t, tm = scene()
    s, sm = scene((4.0, -3.0))
    w, wm, _ = R.register_to(t, tm, s, sm, "R1")
    assert wm.sum() <= sm.sum() * 1.02          # interpolation cannot invent valid pixels
    assert np.all(w[~wm] == 0)                   # outside stays missing (stored as 0 only with its mask)


def test_valid_mean_ignores_missing():
    X = np.array([np.ones((4, 4)), 3 * np.ones((4, 4))], np.float32)
    M = np.array([np.ones((4, 4), bool), np.zeros((4, 4), bool)])
    T, n = R.valid_mean(X, M)
    assert np.all(T == 1) and np.all(n == 1)


def test_zero_mean_images_do_not_fly_off():
    """Regression (Amendment 9.8): with ANTs' centre-of-mass start, z-scored (zero-mean) images got translations of
    ~4e10 px and empty warped masks. Starting from the identity keeps them in the frame."""
    t, tm = scene()
    s, sm = scene((2.0, 1.0))
    t = np.where(tm, (t - t[tm].mean()) / t[tm].std(), 0).astype(np.float32)
    s = np.where(sm, (s - s[sm].mean()) / s[sm].std(), 0).astype(np.float32)
    w, wm, _ = R.register_to(t, tm, s, sm, "R1")
    assert wm.sum() > 0.8 * sm.sum()


def test_bending_is_zero_for_affine_and_small_for_a_pure_shift():
    t, tm = scene()
    s, sm = scene((3.0, 0.0))
    *_, b1 = R.register_to(t, tm, s, sm, "R1")
    *_, b2 = R.register_to(t, tm, s, sm, "R2b")
    assert b1 == 0.0 and 0.0 <= b2 < 0.1           # a rigid shift needs no bending


def test_intensity_match_recovers_gain_and_offset_on_valid_pixels_only():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(32, 32)).astype(np.float32)
    m = np.ones_like(t, bool)
    m[:, :4] = False
    d = (t - 0.5) / 2.0
    d[:, :4] = 99.0                                   # filler outside the donor mask must not enter the fit
    out, (g, b) = R.intensity_match(t, m, d, m)
    assert abs(g - 2.0) < 1e-4 and abs(b - 0.5) < 1e-4
    assert np.allclose(out[m], t[m], atol=1e-4) and np.all(out[~m] == 0)


def test_register_pair_aligns_a_shifted_copy():
    t, tm = scene()
    s, sm = scene((3.0, -2.0))
    out = R.register_pair((t, tm, 0.75 * s + 0.1, sm))
    assert out[0] is not None, out
    w, wm, bend, (g, b), scale = out
    assert bend == 0.0 and 0.9 < scale < 1.1
    inner = tm & wm & (ndimage.distance_transform_edt(tm) > 8)
    assert np.mean((w - t)[inner] ** 2) < 0.2 * np.mean((s - t)[inner & sm] ** 2)


def test_register_pair_rejects_an_implausible_gain():
    t, tm = scene()
    s, sm = scene((1.0, 0.0))
    out = R.register_pair((t, tm, 0.1 * s, sm))          # needs gain ~10: implausible, counted as a failure
    assert out[0] is None and out[1].startswith("gain_")


def test_rigid_pair_never_scales():
    t, tm = scene()
    s, sm = scene((2.0, -1.0))
    out = R.register_pair((t, tm, s, sm))
    assert out[0] is not None and abs(out[4] - 1.0) < 1e-3     # rigid: |det| = 1


def test_rigid_by_vessels_recovers_a_shift_without_scaling():
    """Shifted vessels with a fixed bright spot excluded from the estimate: the rigid fit still recovers the shift and
    keeps scale 1. (This synthetic scene does not isolate the benefit of excluding the disc; the QC sheets do.)"""
    t, tm = scene()
    s, sm = scene((0.0, 4.0))
    yy, xx = np.mgrid[:96, :96]
    spot = np.hypot(xx - 60, yy - 48) < 6
    s = np.where(spot, 3.0, s).astype(np.float32)
    t = np.where(spot, 3.0, t).astype(np.float32)
    w, wm, scale = R.rigid_by_vessels(t, tm, s, sm, np.hypot(xx - 60, yy - 48) < 9)
    ring = tm & wm & ~(np.hypot(xx - 60, yy - 48) < 9) & (ndimage.distance_transform_edt(tm) > 8)
    assert abs(scale - 1) < 1e-3
    assert np.mean((w - t)[ring] ** 2) < 0.5 * np.mean((s - t)[ring & sm] ** 2)
