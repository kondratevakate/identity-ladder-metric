"""Known-geometry checks of the size grid and the oracle searches (Amendment 4; no scans needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import donor_ladder_brain_bilateral as L  # noqa: E402
import donor_ladder_brain_hemisphere as H  # noqa: E402
import donor_ladder_brain_size as S  # noqa: E402

SIZE = (4, 4, 4)


def test_config_positions_keep_centres_and_boxes():
    pos = [("random_patches", (40, 50, 60), (16, 16, 16)), ("deep", (66, 115, 70), (32, 32, 24))]
    out = S.config_positions(pos, 8)
    assert out[0] == ("random_patches", (44, 54, 64), (8, 8, 8))           # same centre 48, 58, 68
    assert S.config_positions(pos, 32)[0][1] == (32, 42, 52)
    assert out[1] == pos[1]


def test_get_box_matches_slicing_pads_outside_and_mirrors():
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(20, 12, 10)).astype(np.float32)
    assert np.array_equal(S.get_box(vol, (3, 2, 1), (4, 5, 6)), vol[3:7, 2:7, 1:7])
    edge = S.get_box(vol, (-2, 0, 0), (4, 3, 3))
    assert np.all(edge[:2] == 0) and np.array_equal(edge[2:], vol[0:2, 0:3, 0:3])
    a = 19                                     # mirror of x is 19 - x on this 20-voxel grid
    mir = S.get_box(vol, (3, 2, 1), (4, 5, 6), a)
    assert np.array_equal(mir, vol[16:12:-1, 2:7, 1:7])
    assert np.array_equal(S.get_box(vol[::-1], (3, 2, 1), (4, 5, 6)), mir)


def window(other, offset, shell):
    return other[tuple(slice(L.W + o, L.W + o + n + 2 * shell) for o, n in zip(offset, SIZE))].copy()


def test_oracle_block_search_follows_the_cube_not_the_shell():
    rng = np.random.default_rng(1)
    shell = 2
    other = rng.normal(size=tuple(n + 2 * (shell + L.W) for n in SIZE)).astype(np.float32)
    true_off, shell_off = (2, -1, 3), (-3, 2, -2)
    win = window(other, shell_off, shell)                      # the shell matches shell_off exactly
    cube_sl = tuple(slice(shell, shell + n) for n in SIZE)
    win[cube_sl] = window(other, true_off, shell)[cube_sl]     # the cube matches true_off exactly
    allowed = np.ones(2 * L.W + 1, bool)
    m = np.ones(other.shape, bool)
    got_shell, _ = L.block_search(win, other, SIZE, *L.search_terms(other, m, SIZE, shell), allowed, shell)
    got_oracle, cube = L.block_search(win, other, SIZE, *L.search_terms(other, m, SIZE, shell, True), allowed,
                                      shell, True)
    assert got_shell == shell_off and got_oracle == true_off
    assert np.array_equal(cube, win[cube_sl])


def test_hemisphere_oracle_follows_the_cube():
    rng = np.random.default_rng(2)
    z, m = rng.normal(size=(41, 24, 22)).astype(np.float32), np.ones((41, 24, 22), bool)
    pad = lambda s: np.pad(z, H.SHELL)[tuple(slice(a, a + n + 2 * H.SHELL) for a, n in zip(s, SIZE))]  # noqa: E731
    shell_at, cube_at = (3, 4, 5), (12, 15, 9)
    win = pad(shell_at).copy()
    cube_sl = tuple(slice(H.SHELL, H.SHELL + n) for n in SIZE)
    win[cube_sl] = pad(cube_at)[cube_sl]
    assert H.search(win, H.prepare(z, m, "L", SIZE, 40))[0] == shell_at
    assert H.search(win, H.prepare(z, m, "L", SIZE, 40, oracle=True))[0] == cube_at


def test_run_side_with_another_shell_adds_oracle_rows():
    shell, n = 2, 14
    rng = np.random.default_rng(3)
    rs = tuple(s + 2 * (shell + L.W) for s in SIZE)
    vol = rng.normal(size=(n, 2) + rs).astype(np.float32)
    mask = np.ones(vol.shape, np.uint8)
    t = pd.DataFrame({"person": [f"p{i}" for i in range(n)], "session": "ses-01",
                      "family": [f"f{i}" for i in range(n)], "sex": "F", "age": 25.0})
    rows = L.run_side(t, t.reset_index(), 0, SIZE, vol, mask, 0, np.ones(2 * L.W + 1, bool), shell, True)
    r = pd.DataFrame(rows, columns=["target", "patch", "cohort", "k", "err", "spread"])
    assert {"mirror_block_oracle", "stranger_block_oracle", "mirror_block", "knn"} <= set(r.cohort)
    own = r[r.cohort == "mirror_block_oracle"].set_index("target").err
    assert (own <= r[r.cohort == "mirror_block"].set_index("target").err + 1e-9).all()   # oracle is a bound
