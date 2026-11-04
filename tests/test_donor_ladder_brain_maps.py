"""Known-geometry checks of the variability maps (Amendment 5; no scans needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import donor_ladder_brain_maps as P  # noqa: E402


def test_box_is_the_cube_mean_on_the_step_grid():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 36, 34)).astype(np.float32)
    b = P.box(x)
    c = (20, 18, 16)                                   # a grid voxel (even indices)
    lo = [ci - P.WIN // 2 for ci in c]                 # scipy's even-size window: [c - 8, c + 7]
    want = x[tuple(slice(l0, l0 + P.WIN) for l0 in lo)].mean()
    assert np.isclose(b[tuple(ci // P.STEP for ci in c)], want, atol=1e-5)
    assert b.shape == tuple((n + 1) // P.STEP for n in x.shape)


def test_local_mse_of_two_volumes_matches_brute_force():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2, 40, 36, 34)).astype(np.float32)
    c = (18, 20, 14)
    sl = tuple(slice(ci - P.WIN // 2, ci - P.WIN // 2 + P.WIN) for ci in c)
    assert np.isclose(P.box((a - b) ** 2)[tuple(ci // 2 for ci in c)], np.mean((a[sl] - b[sl]) ** 2), atol=1e-4)


def test_leave_family_out_excludes_the_family():
    rng = np.random.default_rng(2)
    vols = rng.normal(size=(9, 5))
    fam = slice(0, 3)
    mu, var = P.leave_family_out(vols.sum(0), (vols ** 2).sum(0), 9, vols[fam].sum(0), (vols[fam] ** 2).sum(0), 3)
    assert np.allclose(mu, vols[3:].mean(0))
    assert np.allclose(var, vols[3:].var(0))


def test_stranger_expectation_equals_the_mean_over_the_pool():
    rng = np.random.default_rng(3)
    pool, v = rng.normal(size=(30, 6)), rng.normal(size=6)
    mu, var = pool.mean(0), pool.var(0)
    assert np.allclose(P.stranger_expectation(v, mu, var), ((v - pool) ** 2).mean(0))


def test_accumulator_ratio_is_sum_over_sum_and_respects_validity():
    acc = P.Acc((2,))
    acc.add("mirror", 0, np.array([2.0, 9.0]), np.array([1.0, 1.0]), np.array([True, False]))
    acc.add("mirror", 1, np.array([1.0, 3.0]), np.array([1.0, 2.0]), np.array([True, True]))
    assert np.allclose(acc.ratio("mirror"), [1.5, 1.5])
    assert np.allclose(acc.count("mirror"), [2, 1])
    assert np.allclose(acc.ratio("mirror", (0,))[0], 2.0) and np.isnan(acc.ratio("mirror", (0,))[1])


def test_cohort_mean_uses_same_sex_within_two_years_without_family():
    rng = np.random.default_rng(4)
    people = [("F", float(a), rng.normal(size=3).astype(np.float32)) for a in (20, 21, 22, 23, 25)] * 3
    people += [("M", 22.0, rng.normal(size=3).astype(np.float32))] * 4
    groups = {}
    for s, a, v in people:
        g = groups.setdefault((s, a), [np.zeros(3, np.float32), 0])
        g[0] += v
        g[1] += 1
    family = [people[1]]                                   # one F, 21 y family member
    got = P.cohort_mean(groups, "F", 22.0, family)
    pool = [v for s, a, v in people if s == "F" and abs(a - 22) <= 2]
    want = (sum(pool) - people[1][2]) / (len(pool) - 1)    # 25 y is outside the window, M excluded
    assert np.allclose(got, want, atol=1e-5)
    assert P.cohort_mean(groups, "M", 22.0, []) is None    # 4 donors < COHORT_MIN


def test_half_run_uses_only_its_own_families():
    import pandas as pd
    rng = np.random.default_rng(5)
    n, shape = 12, (20, 20, 20)
    V = rng.normal(size=(n,) + shape).astype(np.float32)
    M = np.ones((n,) + shape, np.uint8)
    t = pd.DataFrame({"person": [f"p{i}" for i in range(n)], "session": "ses-01", "family": [f"f{i:02d}" for i in range(n)],
                      "sex": "F", "age": 25.0})
    affine = np.eye(4)
    affine[0, 3] = -9.5
    fams = np.unique(t.family)
    half_of = dict(zip(np.random.default_rng(P.B.SEED).permutation(fams), np.arange(len(fams)) % 2))
    acc, support, _ = P.run(t, affine, V, M, pool_half=0)
    assert acc.a[("stranger", 1, "n")].sum() == 0 and acc.a[("stranger", 0, "n")].sum() > 0
    mine = [i for i, f in enumerate(t.family) if half_of[f] == 0]
    i = mine[0]
    mu = V[[j for j in mine if j != i]].mean(0)
    want = P.box((V[i] - mu) ** 2)
    got_total = acc.a[("stranger", 0, "V")]
    assert np.allclose(sum(P.box((V[k] - V[[j for j in mine if j != k]].mean(0)) ** 2) for k in mine)[support],
                       got_total[support], rtol=1e-4, atol=1e-4)
    assert want.shape == got_total.shape
