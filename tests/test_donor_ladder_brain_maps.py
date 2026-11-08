"""Known-geometry checks of the variability maps (Amendment 5; no scans needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import donor_ladder_brain_maps as P  # noqa: E402


def test_frac_is_the_cube_fraction_on_the_step_grid():
    m = np.zeros((40, 36, 34), bool)
    m[:20] = True
    f = P.frac(m)
    c = (20, 18, 16)                                   # cube rows [12, 27]: 8 of 16 inside
    assert np.isclose(f[tuple(ci // P.STEP for ci in c)], 0.5, atol=1e-5)
    assert f.shape == tuple((n + 1) // P.STEP for n in m.shape)


def test_local_is_a_round_masked_gaussian_mean():
    x = np.zeros((41, 41, 41), np.float32)
    x[20, 20, 20] = 1.0                                # a point: the window shape itself
    m = np.ones_like(x, bool)
    g = P.local(x, m)
    c = 10                                             # grid index of voxel 20
    # isotropic: the response at the same distance along an axis and along a diagonal (Gaussian, not a cube)
    on_axis = g[c + 3, c, c]                           # offset (6, 0, 0) voxels
    diagonal = g[c + 2, c + 2, c + 1]                  # offset (4, 4, 2): also 6 voxels away
    assert np.isclose(on_axis, diagonal, rtol=1e-3)
    # a box would be flat inside the cube and 0 outside; the Gaussian decays smoothly
    assert g[c, c, c] > g[c + 2, c, c] > g[c + 4, c, c] > 0


def test_local_ignores_voxels_outside_the_mask():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 30, 30)).astype(np.float32) ** 2
    m = np.zeros_like(x, bool)
    m[:, :15] = True
    y = x.copy()
    y[:, 15:] = 1e6                                    # garbage outside the mask must not leak in
    assert np.allclose(P.local(x, m), P.local(y, m), equal_nan=True)
    assert np.isclose(P.local(np.full_like(x, 2.0), m)[7, 3, 7], 2.0, atol=1e-5)   # constant stays constant


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
    m1 = np.ones(shape, bool)
    want = P.local((V[i] - mu) ** 2, m1)
    got_total = acc.a[("stranger", 0, "V")]
    assert np.allclose(sum(P.local((V[k] - V[[j for j in mine if j != k]].mean(0)) ** 2, m1) for k in mine)[support],
                       got_total[support], rtol=1e-4, atol=1e-4)
    assert want.shape == got_total.shape
