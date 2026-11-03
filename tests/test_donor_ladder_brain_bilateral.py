"""Known-geometry checks of the bilateral and look-alike variants (no scans needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as L  # noqa: E402

AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])
AFFINE[0, 3] = -98.0
SIZE = (4, 4, 4)
PAD = L.SHELL + L.W


def test_mirror_constant_matches_mirror_index():
    a = L.mirror_constant(AFFINE, 197)
    assert a == 196
    assert np.array_equal(B.mirror_index(AFFINE, 197), a - np.arange(197))


def test_side_of_box_and_its_mirror():
    assert L.side_of((66, 0, 0), (32, 1, 1), AFFINE) == "L"   # deep box, x -32..-1 mm
    assert L.side_of((98, 0, 0), (32, 1, 1), AFFINE) == "R"   # its mirror, x 0..31 mm
    assert L.own_frame("L", "L") == 0 and L.own_frame("R", "L") == 1


def test_allowed_offsets_far_from_midline_allows_all():
    assert L.allowed_offsets(66, 16, 196).all()


def test_allowed_offsets_near_midline_blocks_moving_onto_target():
    ok = L.allowed_offsets(90, 16, 196)        # cube x -8..7 mm overlaps its mirror by 15 voxels
    assert ok[L.W] and ok[L.W - 1]
    assert not ok[L.W + 1]                     # one step further overlaps by 16, the target itself


def planted(offset, rng):
    other = rng.normal(size=tuple(n + 2 * PAD for n in SIZE)).astype(np.float32)
    sl = tuple(slice(L.W + o, L.W + o + n + 2 * L.SHELL) for o, n in zip(offset, SIZE))
    return other, other[sl].copy()


def test_block_search_finds_planted_window():
    rng = np.random.default_rng(0)
    off = (2, -3, 5)
    other, win = planted(off, rng)
    win[tuple(slice(L.SHELL, L.SHELL + n) for n in SIZE)] = 99.0   # the cube itself must not matter
    o2, valid = L.search_terms(other, np.ones(other.shape, bool), SIZE)
    got, cube = L.block_search(win, other, SIZE, o2, valid, np.ones(2 * L.W + 1, bool))
    assert got == off
    inner = tuple(slice(L.W + o + L.SHELL, L.W + o + L.SHELL + n) for o, n in zip(off, SIZE))
    assert np.array_equal(cube, other[inner])


def test_block_search_skips_invalid_and_disallowed_offsets():
    rng = np.random.default_rng(1)
    off = (2, -3, 5)
    other, win = planted(off, rng)
    m = np.ones(other.shape, bool)
    m[tuple(slice(L.W + o + L.SHELL, L.W + o + L.SHELL + n) for o, n in zip(off, SIZE))] = False
    o2, valid = L.search_terms(other, m, SIZE)
    got, _ = L.block_search(win, other, SIZE, o2, valid, np.ones(2 * L.W + 1, bool))
    assert got is not None and got != off
    o2, valid = L.search_terms(other, np.ones(other.shape, bool), SIZE)
    allowed = np.ones(2 * L.W + 1, bool)
    allowed[L.W + off[0]] = False
    got, _ = L.block_search(win, other, SIZE, o2, valid, allowed)
    assert got is not None and got[0] != off[0]
    assert L.block_search(win, other, SIZE, o2, np.zeros_like(valid), allowed) == (None, None)


def synthetic(n=14, seed=2):
    """n one-person families, session 1 only, random regions in both frames, all inside the mask."""
    rng = np.random.default_rng(seed)
    shape = tuple(s + 2 * PAD for s in SIZE)
    vol = rng.normal(size=(n, 2) + shape).astype(np.float32)
    mask = np.ones(vol.shape, np.uint8)
    t = pd.DataFrame({"person": [f"p{i}" for i in range(n)], "session": "ses-01",
                      "family": [f"f{i}" for i in range(n)], "sex": "F", "age": 25.0})
    tg = t.reset_index()
    return t, tg, vol, mask


def cube_of(region, offset=(0, 0, 0)):
    return region[tuple(slice(L.W + L.SHELL + o, L.W + L.SHELL + o + s) for o, s in zip(offset, SIZE))]


def rows_of(t, tg, vol, mask, fo=0):
    r = L.run_side(t, tg, 0, SIZE, vol, mask, fo, np.ones(2 * L.W + 1, bool))
    return pd.DataFrame(r, columns=["target", "patch", "cohort", "k", "err", "spread"])


def test_knn_oracle_finds_an_identical_donor():
    t, tg, vol, mask = synthetic()
    vol[7, 0][tuple(slice(L.W + L.SHELL, L.W + L.SHELL + s) for s in SIZE)] = cube_of(vol[0, 0])
    r = rows_of(t, tg, vol, mask)
    assert r.query("target == 0 and cohort == 'knn_oracle' and k == 1").err.iloc[0] == 0.0
    assert r.query("target == 0 and cohort == 'knn' and k == 1").err.iloc[0] > 0


def test_knn_mirror_picks_donor_matching_own_mirrored_cube():
    t, tg, vol, mask = synthetic()
    vol[0, 1][tuple(slice(L.W + L.SHELL, L.W + L.SHELL + s) for s in SIZE)] = cube_of(vol[5, 0])
    r = rows_of(t, tg, vol, mask)
    want = float(np.mean((cube_of(vol[0, 0]) - cube_of(vol[5, 0])) ** 2))
    got = r.query("target == 0 and cohort == 'knn_mirror' and k == 1").err.iloc[0]
    assert np.isclose(got, want)
    hemi = r.query("target == 0 and cohort == 'other_hemisphere'").err.iloc[0]
    assert np.isclose(hemi, want)                 # same cube: own mirror equals donor 5's cube


def test_mirror_block_recovers_a_shifted_copy_of_the_target():
    t, tg, vol, mask = synthetic()
    off = (1, 2, -1)
    win_sl = tuple(slice(L.W, L.W + s + 2 * L.SHELL) for s in SIZE)
    dst = tuple(slice(L.W + o, L.W + o + s + 2 * L.SHELL) for o, s in zip(off, SIZE))
    vol[0, 1][dst] = vol[0, 0][win_sl]
    r = rows_of(t, tg, vol, mask)
    assert r.query("target == 0 and cohort == 'mirror_block'").err.iloc[0] == 0.0
    assert r.query("target == 0 and cohort == 'other_hemisphere'").err.iloc[0] > 0
    assert r.query("target == 0 and cohort == 'stranger_block'").err.iloc[0] > 0


def test_right_side_is_the_mirrored_frame():
    t, tg, vol, mask = synthetic()
    left, right = rows_of(t, tg, vol, mask, 0), rows_of(t, tg, vol, mask, 1)
    a = left.query("target == 3 and cohort == 'other_hemisphere'").err.iloc[0]
    b = right.query("target == 3 and cohort == 'other_hemisphere'").err.iloc[0]
    assert np.isclose(a, b)                       # own-vs-mirror distance is symmetric in the two frames
    want = float(np.mean((cube_of(vol[3, 1]) - cube_of(vol[3, 0])) ** 2))
    assert np.isclose(a, want)


def test_stranger_rows_use_the_same_first_stranger():
    t, tg, vol, mask = synthetic()
    r = rows_of(t, tg, vol, mask)
    pop = r.query("target == 0 and cohort == 'population' and k == 1").err.iloc[0]
    order = np.random.default_rng([L.SEED, 0, 0, 0]).permutation(np.arange(1, len(tg)))
    j = order[0]
    assert np.isclose(pop, float(np.mean((cube_of(vol[0, 0]) - cube_of(vol[j, 0])) ** 2)))
    so = r.query("target == 0 and cohort == 'stranger_other_side'").err.iloc[0]
    assert np.isclose(so, float(np.mean((cube_of(vol[0, 0]) - cube_of(vol[j, 1])) ** 2)))
