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
    w, wm = R.register_to(t, tm, s, sm, "R1")
    inner = tm & wm & (ndimage.distance_transform_edt(tm) > 8)
    before = np.mean((s - t)[inner & sm] ** 2)
    after = np.mean((w - t)[inner] ** 2)
    assert after < 0.2 * before


def test_warped_mask_never_extends_the_field():
    t, tm = scene()
    s, sm = scene((4.0, -3.0))
    w, wm = R.register_to(t, tm, s, sm, "R1")
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
    w, wm = R.register_to(t, tm, s, sm, "R1")
    assert wm.sum() > 0.8 * sm.sum()
