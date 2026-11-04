"""Known-geometry checks of the whole-hemisphere block search (no scans needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import donor_ladder_brain_bilateral as L  # noqa: E402
import donor_ladder_brain_hemisphere as H  # noqa: E402

SIZE = (4, 4, 4)
A, NX = 40, 41            # toy grid: midline at index 20


def test_half_starts_keep_the_cube_on_one_side():
    assert H.half_starts("L", 16, 196, 197) == (0, 83)        # cube 83..98 ends on the midline voxel
    assert H.half_starts("R", 16, 196, 197) == (98, 181)
    lo, hi = H.half_starts("L", 4, A, NX)
    assert hi + 4 - 1 == A // 2 and lo == 0


def window_at(z, start):
    return np.pad(z, L.SHELL)[tuple(slice(s, s + n + 2 * L.SHELL) for s, n in zip(start, SIZE))]


def toy(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(NX, 24, 22)).astype(np.float32), np.ones((NX, 24, 22), bool)


def test_search_finds_a_planted_window_in_the_half():
    z, m = toy()
    start = (6, 9, 7)
    win = window_at(z, start).copy()
    win[tuple(slice(L.SHELL, L.SHELL + n) for n in SIZE)] = 50.0   # the cube itself must not matter
    got, cube = H.search(win, H.prepare(z, m, "L", SIZE, A))
    assert got == start
    assert np.allclose(cube, z[tuple(slice(s, s + n) for s, n in zip(start, SIZE))])


def test_search_ignores_the_other_half_and_invalid_cubes():
    z, m = toy(1)
    far = (30, 9, 7)                                     # right half: an exact match there must be ignored
    win = window_at(z, far).copy()
    got, _ = H.search(win, H.prepare(z, m, "L", SIZE, A))
    assert got[0] + SIZE[0] - 1 <= A // 2
    got, _ = H.search(win, H.prepare(z, m, "R", SIZE, A))
    assert got == far
    m2 = m.copy()
    m2[tuple(slice(s, s + n) for s, n in zip(far, SIZE))] = False
    got, _ = H.search(win, H.prepare(z, m2, "R", SIZE, A))
    assert got != far
    assert H.search(win, H.prepare(z, np.zeros_like(m), "R", SIZE, A)) == (None, None)


def test_search_near_the_grid_edge_pads_with_zeros():
    z, m = toy(2)
    start = (0, 0, 0)                                    # shell sticks out of the grid
    got, _ = H.search(window_at(z, start), H.prepare(z, m, "L", SIZE, A))
    assert got == start
