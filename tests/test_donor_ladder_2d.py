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


def test_views_split_sides_and_pool_both():
    r = pd.DataFrame({"region": ["random_patches"] * 3, "side": ["R", "L", "R"], "x": [1, 2, 3]})
    v = D.views(r)
    assert (v.region == "random_patches/R").sum() == 2 and (v.region == "random_patches/L").sum() == 1
    assert (v.region == "random_patches/LR").sum() == 3
