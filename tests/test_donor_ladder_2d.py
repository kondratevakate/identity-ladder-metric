"""Known-geometry checks of the 2D donor ladder under the general protocol (no images needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import donor_ladder_2d as D  # noqa: E402


def test_region_pads_outside_the_grid_as_invalid():
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    fov = np.ones((10, 10), bool)
    r, ok = D.region(img, fov, 0, 0, 4, 2)
    assert r.shape == (8, 8) and not ok[:2].any() and not ok[:, :2].any()
    assert np.array_equal(r[2:6, 2:6], img[0:4, 0:4]) and ok[2:6, 2:6].all()


def planted(rng, size, ring, off):
    other = rng.normal(size=(size + 2 * (ring + D.W),) * 2).astype(np.float32)
    win = other[D.W + off[0]:D.W + off[0] + size + 2 * ring, D.W + off[1]:D.W + off[1] + size + 2 * ring].copy()
    return other, win


def test_search_finds_the_ring_match_and_the_oracle_finds_the_patch():
    rng = np.random.default_rng(0)
    size, ring = 6, 3
    other, win = planted(rng, size, ring, (2, -3))
    true_off = (-4, 5)
    patch_at = other[D.W + true_off[0] + ring:D.W + true_off[0] + ring + size,
                     D.W + true_off[1] + ring:D.W + true_off[1] + ring + size]
    win[ring:ring + size, ring:ring + size] = patch_at
    ok = np.ones(other.shape, bool)
    by_ring = D.search(win, other, ok, size, ring)
    by_patch = D.search(win, other, ok, size, ring, oracle=True)
    want_ring = other[D.W + 2 + ring:D.W + 2 + ring + size, D.W - 3 + ring:D.W - 3 + ring + size]
    assert np.array_equal(by_ring, want_ring)
    assert np.array_equal(by_patch, patch_at)


def test_search_needs_the_patch_inside_the_field_of_view():
    rng = np.random.default_rng(1)
    size, ring = 6, 3
    other, win = planted(rng, size, ring, (0, 0))
    assert D.search(win, other, np.zeros(other.shape, bool), size, ring) is None


def test_search_treats_outside_the_field_of_view_as_missing_not_zero():
    # Target ring is all zeros (the z-score mean). A candidate whose ring lies outside the field of view (stored
    # as 0) would match it perfectly if outside counted as zero; it must be skipped instead.
    rng = np.random.default_rng(2)
    size, ring = 6, 3
    other, win = planted(rng, size, ring, (0, 0))
    win[:] = 0
    ok = np.ones(other.shape, bool)
    oy, ox = D.W - 5 + ring, D.W - 5 + ring                     # top-left of the fake candidate's ring
    other[oy - ring:oy + size + ring, ox - ring:ox + size + ring] = 0
    ok[oy - ring:oy + size + ring, ox - ring:ox + size + ring] = False
    ok[oy:oy + size, ox:ox + size] = True                         # its patch is inside, its ring is not
    got = D.search(win, other, ok, size, ring)
    assert got is not None and not np.array_equal(got, other[oy:oy + size, ox:ox + size])
    win_ok = np.ones(win.shape, bool)
    win_ok[0, 0] = False                                          # target ring pixel outside its own field of view
    assert D.search(win, other, ok, size, ring, win_ok=win_ok) is None


def test_views_split_sides_and_pool_both():
    r = pd.DataFrame({"region": ["random_patches"] * 3, "side": ["R", "L", "R"], "x": [1, 2, 3]})
    v = D.views(r)
    assert (v.region == "random_patches/R").sum() == 2 and (v.region == "random_patches/L").sum() == 1
    assert (v.region == "random_patches/LR").sum() == 3


def test_flatten_removes_smooth_illumination_and_keeps_fine_structure():
    n = 256
    yy, xx = np.mgrid[0:n, 0:n]
    illum = 0.5 + 0.4 * (xx / n) + 0.3 * np.exp(-((yy - 128) ** 2 + (xx - 128) ** 2) / (2 * 60.0 ** 2))
    fine = 1 + 0.2 * ((xx // 4 + yy // 4) % 2)          # 4 px checkerboard: structure finer than the window
    fov = (yy - 128) ** 2 + (xx - 128) ** 2 < 120 ** 2
    out = D.flatten((illum * fine).astype(np.float32), fov)
    inner = (yy - 128) ** 2 + (xx - 128) ** 2 < 80 ** 2
    ratio = out[inner] / fine[inner]
    assert ratio.std() / ratio.mean() < 0.05            # illumination gone
    assert np.corrcoef(out[inner], fine[inner])[0, 1] > 0.9   # checkerboard kept
    assert np.all(out[~fov] == 0)


def test_local_error_map_is_a_round_masked_gaussian():
    import donor_ladder_2d as D2
    x = np.zeros((101, 101), np.float32)
    x[50, 50] = 1.0
    g = D2.local(x, np.ones_like(x, bool))
    assert np.isclose(g[50, 60], g[56, 58], rtol=1e-3)          # same distance 10 px: round, not square
    m = np.zeros_like(x, bool)
    m[:, :50] = True
    y = np.where(m, 1.0, 1e6).astype(np.float32)                # garbage outside the mask must not leak in
    assert np.allclose(D2.local(y, m)[:, 10:40], 1.0, atol=1e-4)
