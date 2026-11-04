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
    assert np.isclose(G.knn_error(X[0], X, sh, 0, donors), want)


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
    rows = {i: (v, r) for i, v, r in G.centre_rows(X, Mc, sh, sh, valid, valid, fam, sex, age, rep_cube,
                                                   [True] + [False] * (n - 1), True)}
    v0, r0 = rows[0]
    others = np.arange(2, n)
    mu = X[others].mean(0)
    assert np.isclose(v0, np.mean((X[0] - mu) ** 2))
    assert np.isclose(r0["stranger"], np.mean([np.mean((X[0] - X[j]) ** 2) for j in others]))
    assert np.isclose(r0["sibling"], np.mean((X[0] - X[1]) ** 2))
    assert np.isclose(r0["repeat"], np.mean((X[0] - rep_cube[0]) ** 2))
    assert np.isclose(r0["mirror"], np.mean((X[0] - Mc[0]) ** 2))
    assert "knn16_s4" in r0 and "cohort" in r0
    no_mirror = G.centre_rows(X, Mc, sh, sh, valid, valid, fam, sex, age, rep_cube, [False] * n, False)
    assert all("mirror" not in r for _, _, r in no_mirror)


def test_centre_rows_with_a_pool_keep_targets_and_donors_inside_it():
    rng = np.random.default_rng(3)
    n = 30
    X, sh = rng.normal(size=(n, 10)), rng.normal(size=(n, 6))
    fam = np.arange(n)
    age, sex = np.full(n, 25.0), np.array(["F"] * n)
    valid = np.ones(n, bool)
    pool = np.arange(n) % 2 == 0
    rows = G.centre_rows(X, X, sh, sh, valid, valid, fam, sex, age, [None] * n, [False] * n, True, pool)
    assert {i for i, _, _ in rows} == set(np.flatnonzero(pool))
    i, v, _ = rows[0]
    mu = X[[j for j in np.flatnonzero(pool) if j != i]].mean(0)
    assert np.isclose(v, np.mean((X[i] - mu) ** 2))
