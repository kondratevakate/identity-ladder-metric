"""Known-geometry checks of the coarse-grid ladder map (Amendment 7; no scans needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import donor_ladder_brain_map_grid as G  # noqa: E402


def test_centres_pass_through_the_midline_and_need_support():
    m = np.zeros((41, 40, 40), bool)
    m[4:37, 4:36, 4:36] = True
    cs = G.centres(m, 40)                                   # midline index 20
    assert len(cs) and np.all((cs[:, 0] - 20) % G.STEP == 0)
    assert (20 in set(cs[:, 0]))
    assert G.centres(np.zeros_like(m), 40).size == 0


def test_parts_split_cube_and_shell_from_the_window():
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(60, 60, 60)).astype(np.float32)
    c = (30, 28, 31)
    win = G.window_at(vol, c, G.WINDOW)
    cube, shell4 = G.parts(win, 4)
    assert np.array_equal(cube, G.window_at(vol, c, G.CUBE).ravel())
    assert shell4.size == G.WINDOW ** 3 - G.CUBE ** 3
    _, shell2 = G.parts(win, 2)
    assert shell2.size == (G.CUBE + 4) ** 3 - G.CUBE ** 3


def test_knn_error_uses_the_closest_shells():
    rng = np.random.default_rng(1)
    X, sh = rng.normal(size=(30, 8)), rng.normal(size=(30, 5))
    sh[3:19] = sh[0] + 1e-3 * rng.normal(size=(16, 5))       # donors 3..18 share the target's context
    donors = np.arange(1, 30)
    want = float(np.mean((X[0] - X[3:19].mean(0)) ** 2))
    assert np.isclose(G.knn_error(X[0], X, sh, 0, donors, np.full(8, 1 / 8)), want)


def test_centre_rows_follow_the_cube_ladder_rules():
    rng = np.random.default_rng(2)
    n = 24
    X = rng.normal(size=(n, 10))
    Mc = rng.normal(size=(n, 10))
    sh = rng.normal(size=(n, 6))
    fam = np.array([0, 0] + list(range(1, n - 1)))          # targets 0 and 1 are siblings
    age = np.array([22.0, 25.0] + [22.0] * (n - 2))
    sex = np.array(["F"] * n)
    valid = np.ones(n, bool)
    rep_cube = [X[0] + 0.1] + [None] * (n - 1)
    U = np.full(10, 0.1)                                    # uniform weights: the plain cube mean
    rows = {i: (v, r) for i, v, r in G.centre_rows(X, Mc, sh, sh, valid, valid, fam, sex, age, rep_cube,
                                                   [True] + [False] * (n - 1), True, U)}
    v0, r0 = rows[0]
    others = np.arange(2, n)
    mu = X[others].mean(0)
    assert np.isclose(v0, np.mean((X[0] - mu) ** 2))
    assert np.isclose(r0["stranger"], np.mean([np.mean((X[0] - X[j]) ** 2) for j in others]))
    assert np.isclose(r0["sibling"], np.mean((X[0] - X[1]) ** 2))
    assert np.isclose(r0["repeat"], np.mean((X[0] - rep_cube[0]) ** 2))
    assert np.isclose(r0["mirror"], np.mean((X[0] - Mc[0]) ** 2))
    assert "knn16_s4" in r0 and "cohort" in r0
    no_mirror = G.centre_rows(X, Mc, sh, sh, valid, valid, fam, sex, age, rep_cube, [False] * n, False, U)
    assert all("mirror" not in r for _, _, r in no_mirror)


def test_centre_rows_with_a_pool_keep_targets_and_donors_inside_it():
    rng = np.random.default_rng(3)
    n = 30
    X, sh = rng.normal(size=(n, 10)), rng.normal(size=(n, 6))
    fam = np.arange(n)
    age, sex = np.full(n, 25.0), np.array(["F"] * n)
    valid = np.ones(n, bool)
    pool = np.arange(n) % 2 == 0
    rows = G.centre_rows(X, X, sh, sh, valid, valid, fam, sex, age, [None] * n, [False] * n, True, np.full(10, 0.1), pool)
    assert {i for i, _, _ in rows} == set(np.flatnonzero(pool))
    i, v, _ = rows[0]
    mu = X[[j for j in np.flatnonzero(pool) if j != i]].mean(0)
    assert np.isclose(v, np.mean((X[i] - mu) ** 2))


def test_cube_weights_are_a_round_gaussian_summing_to_one():
    w = G.cube_weights().reshape((G.CUBE,) * 3)
    assert np.isclose(w.sum(), 1.0)
    c = G.CUBE // 2
    assert w[c, c, c] > w[c, c, c + 3] > w[c, c, c + 6]               # decays from the centre
    assert np.isclose(w[c - 1 + 4, c, c], w[c, c - 1 + 4, c])           # same in every axis
    assert np.isclose(w[c + 3, c + 3, c], w[c, c + 3, c + 3])           # same at equal radius
    assert w.max() / w.min() > 50                                       # not a box


def test_centre_rows_weight_errors_inside_the_cube_and_keep_the_stranger_exact():
    rng = np.random.default_rng(4)
    n, p = 20, 10
    X = rng.normal(size=(n, p))
    fam = np.arange(n)
    age, sex = np.full(n, 25.0), np.array(["F"] * n)
    valid = np.ones(n, bool)
    w = rng.random(p)
    w /= w.sum()
    i, v, r = G.centre_rows(X, X, X[:, :3], X[:, :3], valid, valid, fam, sex, age, [None] * n, [False] * n,
                            False, w)[0]
    others = np.arange(1, n)
    assert np.isclose(v, ((X[0] - X[others].mean(0)) ** 2) @ w)
    assert np.isclose(r["stranger"], np.mean([((X[0] - X[j]) ** 2) @ w for j in others]))


def test_grid_upsampling_is_smooth_and_keeps_missing_missing():
    import nibabel as nib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper' / 'isbi2027'))
    import donor_ladder_brain_ladder_maps_figure as F
    g = np.full((6, 6, 6), np.nan, np.float32)
    g[1:5, 1:5, 1:5] = np.arange(4, dtype=np.float32)[:, None, None]        # a ramp along x, NaN border
    grid = nib.Nifti1Image(g, np.diag([8.0, 8, 8, 1]))
    ref = nib.Nifti1Image(np.zeros((24, 24, 24), np.float32), np.diag([2.0, 2, 2, 1]))
    up = np.asanyarray(F.upsample(grid, ref).dataobj)
    line = up[8:17, 12, 12]                                                   # x from 16 to 32 mm: grid 2 -> 4
    assert np.all(np.isfinite(line))
    assert len(np.unique(np.round(line, 4))) > 4                             # not a step (nearest gives 2-3 values)
    assert np.all(np.diff(line) >= -1e-5)                                     # monotone ramp, no ringing
    assert np.isnan(up[0, 12, 12])                                            # far outside the valid grid: missing
