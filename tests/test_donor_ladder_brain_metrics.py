"""Known-geometry tests for the tier-1 brain metric panel (PROTOCOL_donor_ladder_brain.md, Amendment 10)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import donor_ladder_brain_metrics as M  # noqa: E402


def cube(seed=0, n=12):
    return np.random.default_rng(seed).normal(size=(n, n, n)).astype(np.float32)


def full(n=12):
    return np.ones((n, n, n), bool)


def test_identical_fill_is_perfect():
    x = cube()
    s = M.score(x, x.copy(), full(), np.zeros_like(x), np.zeros_like(x))
    assert s["mse"] == 0 and s["mae"] == 0
    assert s["ssim"] == pytest.approx(1.0)
    assert s["pearson"] == pytest.approx(1.0)
    assert s["lddt"] == pytest.approx(1.0)
    assert s["acc_dataset"] == pytest.approx(1.0)
    assert s["amplitude"] == pytest.approx(1.0)


def test_masked_mean_ignores_invalid_members_filler_values():
    """Missing is never zero: a member's filler 0 outside its mask must not pull the mean."""
    X = np.stack([np.full((4, 4, 4), 3.0), np.zeros((4, 4, 4))]).astype(np.float32)
    W = np.stack([np.ones((4, 4, 4), bool), np.zeros((4, 4, 4), bool)])
    W[1, 0] = True
    X[1, 0] = 5.0
    m, ok = M.masked_mean(X, W)
    assert np.allclose(m[1:], 3.0)          # only the valid member
    assert np.allclose(m[0], 4.0)           # both valid there
    assert ok.all()


def test_masked_mean_no_valid_member_is_not_scored():
    X = np.ones((2, 3, 3, 3), np.float32)
    W = np.zeros((2, 3, 3, 3), bool)
    W[:, 0] = True
    _, ok = M.masked_mean(X, W)
    assert ok[0].all() and not ok[1:].any()


def test_ssim_ignores_voxels_outside_the_mask():
    """Filler values outside e (any value) must not change SSIM."""
    x, f = cube(1), cube(2)
    e = full()
    e[:4] = False
    a = M.ssim_masked(x, f, e)
    x2, f2 = x.copy(), f.copy()
    x2[:4], f2[:4] = 0.0, 100.0
    assert M.ssim_masked(x2, f2, e) == pytest.approx(a, abs=1e-9)


def test_score_ignores_voxels_outside_the_mask():
    x, f = cube(3), cube(4)
    mu = np.zeros_like(x)
    e = full()
    e[:, :3] = False
    a = M.score(x, f, e, mu, mu, sigma=0.7)
    f2 = f.copy()
    f2[:, :3] = 0.0            # filler where the donor is invalid
    b = M.score(x, f2, e, mu, mu, sigma=0.7)
    for k in M.METRICS:
        assert a[k] == pytest.approx(b[k], abs=1e-9)


def test_lddt_is_offset_invariant_but_mse_is_not():
    x = cube(5)
    s = M.score(x, x + 1.5, full(), np.zeros_like(x), np.zeros_like(x))
    assert s["lddt"] == pytest.approx(1.0)
    assert s["mse"] == pytest.approx(2.25, rel=1e-5)


def test_lddt_offsets_are_unique_half_ball():
    offs = M._offsets(2.0)
    assert len(offs) == 16 and len(set(offs)) == 16
    assert all(tuple(-v for v in o) not in offs for o in offs)


def test_shrink_amplitude_equals_lambda():
    x, mu = cube(6), cube(7)
    y = M.shrink(x, mu, 0.4)
    s = M.score(x, y, full(), mu, mu)
    assert s["amplitude"] == pytest.approx(0.4, rel=1e-4)
    assert s["acc_dataset"] == pytest.approx(1.0, abs=1e-6)


def test_wrong_detail_is_orthogonal_with_the_same_norm():
    x, mu, s = cube(8), cube(9), cube(10)
    e = full()
    e[0] = False
    w = M.wrong_detail(x, mu, s, e)
    rx, rw = (x - mu)[e].astype(np.float64), (w - mu)[e].astype(np.float64)
    assert abs(rx @ rw) / (np.linalg.norm(rx) * np.linalg.norm(rw)) < 1e-5
    assert np.linalg.norm(rw) == pytest.approx(np.linalg.norm(rx), rel=1e-5)
    assert np.array_equal(w[~e], mu[~e])


def test_noise_fill_matches_the_individual_sd():
    x, mu = cube(11, 20), np.zeros((20, 20, 20), np.float32)
    e = np.ones_like(x, bool)
    f = M.noise_fill(x, mu, e, np.random.default_rng(0))
    assert np.std(f) == pytest.approx(np.std(x), rel=0.05)


def test_holm_matches_hand_computation():
    adj = M.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_orientation_makes_higher_better():
    assert M.oriented("mse", 2.0) == -2.0 and M.oriented("ssim", 0.5) == 0.5


def test_views_label_regions_and_sides():
    """Regression: region names are strings; the view label must concatenate without a numpy ufunc error."""
    import pandas as pd
    pos = [("random_patches", (0, 0, 0), (16, 16, 16)), ("deep", (0, 0, 0), (32, 32, 24))]
    r = pd.DataFrame({"patch": [0, 1, 1], "side": ["L", "R", "L"], "x": [1, 2, 3]})
    v = M.views(r, pos)
    assert sorted(v.view.unique()) == ["deep/L", "deep/LR", "deep/R", "random_patches/L", "random_patches/LR"]
    assert len(v) == 6


def test_position_is_none_when_the_mean_is_undefined():
    """Regression: ACC of the dataset mean is NaN; the 0 = mean / 1 = repeat position must be None, not a crash."""
    import pandas as pd
    per = pd.DataFrame({"target": [0, 0, 0], "family": ["a", "a", "a"], "fill": ["repeat", "dataset_mean", "mirror"],
                        "acc_dataset": [0.9, np.nan, 0.3]})
    assert M.boot_position(per, "acc_dataset", "mirror") is None


def test_position_of_the_anchors_is_zero_and_one():
    """Regression: the repeat and the mean themselves sit at 1 and 0 (no duplicate-column crash)."""
    import pandas as pd
    per = pd.DataFrame({"target": [0, 0, 1, 1], "family": ["a", "a", "b", "b"],
                        "fill": ["repeat", "dataset_mean"] * 2, "mse": [0.1, 0.5, 0.2, 0.6]})
    assert M.boot_position(per, "mse", "repeat")["position"] == pytest.approx(1.0)
    assert M.boot_position(per, "mse", "dataset_mean")["position"] == pytest.approx(0.0)


def test_stranger_rung_is_not_a_reference_stranger():
    """Regression: the summary dropped the 'stranger' rung together with str00..str15."""
    assert M.is_reference_stranger("str00") and M.is_reference_stranger("str15")
    assert not M.is_reference_stranger("stranger")
    assert not M.is_reference_stranger("shrink_repeat")


# ---- Amendment 11: masked scaffold distance and block search ----

def test_masked_dist_ignores_filler_outside_the_mask():
    """Regression: a candidate whose filler 0 outside its mask matches the target's 0 must not look closer."""
    rng = np.random.default_rng(0)
    b = rng.normal(size=100)
    mb = np.ones(100, bool)
    mb[:30] = False
    b[:30] = 0.0                                   # target filler outside its mask
    good = b + 0.1 * rng.normal(size=100)          # real neighbour, valid everywhere
    bad = rng.normal(size=100)
    bad[:30] = 0.0                                 # zeros exactly where the target has filler
    A = np.stack([good, bad])
    MA = np.ones((2, 100), bool)
    MA[1, :30] = False
    d = M.masked_dist(A, MA, b, mb, n_ref=100)
    assert d[0] < d[1]
    assert d[0] == pytest.approx(np.mean((good[30:] - b[30:]) ** 2))


def test_masked_dist_requires_half_shared():
    A = np.zeros((1, 10))
    MA = np.zeros((1, 10), bool)
    MA[0, :4] = True
    d = M.masked_dist(A, MA, np.zeros(10), np.ones(10, bool), n_ref=10)
    assert np.isinf(d[0])


def _region_with_target(shift, size=(4, 4, 4), shell=2, W=3, seed=1):
    rng = np.random.default_rng(seed)
    win_shape = tuple(s + 2 * shell for s in size)
    reg_shape = tuple(s + 2 * W for s in win_shape)
    region = rng.normal(size=reg_shape)
    t_win = rng.normal(size=win_shape)
    o = tuple(W + d for d in shift)
    region[tuple(slice(a, a + n) for a, n in zip(o, win_shape))] = t_win
    return t_win, region, o


def test_block_search_finds_the_planted_window():
    size, shell, W = (4, 4, 4), 2, 3
    t_win, region, o = _region_with_target((1, -2, 0), size, shell, W)
    hit = M.block_search_masked(t_win, np.ones_like(t_win, bool), region, np.ones_like(region, bool),
                                size, shell, np.ones(2 * W + 1, bool))
    assert hit is not None and hit[2] == o
    assert np.allclose(hit[0], t_win[shell:shell + 4, shell:shell + 4, shell:shell + 4])


def test_block_search_filler_outside_mask_cannot_win():
    """Regression (Amendment 9): a site outside the candidate's brain that holds values copying the target exactly
    would win if unmasked voxels counted; with masks it is excluded and the real (noisier) match wins."""
    size, shell, W = (4, 4, 4), 2, 8
    t_win, region, o = _region_with_target((-6, 0, 0), size, shell, W, seed=2)
    rng = np.random.default_rng(5)
    win_shape = t_win.shape
    region[tuple(slice(a, a + n) for a, n in zip(o, win_shape))] += 0.3 * rng.normal(size=win_shape)
    r_mask = np.ones_like(region, bool)
    fake = (W + 6, W, W)                            # x 14..22, no overlap with the planted site at x 2..10
    fwin = tuple(slice(a, a + n) for a, n in zip(fake, win_shape))
    region[fwin] = t_win                            # exact copy, but outside the candidate's brain
    r_mask[fwin] = False
    hit = M.block_search_masked(t_win, np.ones_like(t_win, bool), region, r_mask, size, shell,
                                np.ones(2 * W + 1, bool))
    assert hit is not None and hit[2] == o
    unmasked = M.block_search_masked(t_win, np.ones_like(t_win, bool), region, np.ones_like(region, bool), size,
                                     shell, np.ones(2 * W + 1, bool))
    assert unmasked[2] == fake                      # the fake site does win when the mask is ignored


def test_block_search_respects_allowed_x():
    size, shell, W = (4, 4, 4), 2, 3
    t_win, region, o = _region_with_target((2, 0, 0), size, shell, W)
    allowed = np.ones(2 * W + 1, bool)
    allowed[W + 2] = False
    hit = M.block_search_masked(t_win, np.ones_like(t_win, bool), region, np.ones_like(region, bool),
                                size, shell, allowed)
    assert hit is not None and hit[2][0] != W + 2


def test_ladder_R_anchors_repeat_zero_stranger_one():
    import pandas as pd
    rows = []
    for t in range(12):
        fam = f"f{t}"
        rows += [(t, fam, "repeat", 0.1), (t, fam, "dataset_mean", 0.5), (t, fam, "mirror", 0.7)]
        rows += [(t, fam, f"str{j:02d}", 1.1) for j in range(16)]
    per = pd.DataFrame(rows, columns=["target", "family", "fill", "mse"])
    R, arr = M.ladder_R(per, "mse", reps=50)
    assert R["repeat"]["R"] == pytest.approx(0.0)
    assert R["dataset_mean"]["R"] == pytest.approx(0.4)
    assert R["mirror"]["R"] == pytest.approx(0.6)
    e = M.expectations(R, arr)
    assert e["E1 mirror closer than stranger"]["pass"] is True
    assert e["E3B dataset mean better than stranger"]["pass"] is True        # reality B (Amendment 13)
    assert e["E2 sibling closer than stranger"] is None
    a = M.expectations(R, arr, M.EXPECT_A)
    assert a["A-E3b dataset mean not better than stranger"]["pass"] is False  # reality A, descriptive


def test_reality_B_noise_and_order():
    R = {"noise": {"R": 1.4, "ci95": [1.2, 1.6]}, "dataset_mean": {"R": 0.4, "ci95": [0.3, 0.5]},
         "mean_k2": {"R": 0.7, "ci95": [0.6, 0.8]}, "stranger": {"R": 1.0, "ci95": [0.98, 1.02]}}
    e = M.expectations(R, {})
    assert e["E6B white noise worse than stranger"]["pass"] is True
    assert e["E3B averages ordered by k"]["pass"] is True
    R["noise"] = {"R": 1.0, "ci95": [0.95, 1.05]}
    assert M.expectations(R, {})["E6B white noise worse than stranger"]["pass"] is False


def test_ladder_R_orients_similarities():
    import pandas as pd
    rows = []
    for t in range(12):
        rows += [(t, f"f{t}", "repeat", 0.9), (t, f"f{t}", "dataset_mean", 0.1)]
        rows += [(t, f"f{t}", f"str{j:02d}", 0.1) for j in range(16)]
    per = pd.DataFrame(rows, columns=["target", "family", "fill", "acc_cohort"])
    R, _ = M.ladder_R(per, "acc_cohort", reps=20)
    assert R["dataset_mean"]["R"] == pytest.approx(1.0)


# ---- Amendment 12: protein-style metrics, calibrations, two-way bootstrap ----

def test_protein_like_known_values():
    x = cube(20)
    s = M.score(x, x.copy(), full(), np.zeros_like(x), np.zeros_like(x), sigma=0.5)
    assert s["tm_like"] == pytest.approx(1.0) and s["gdt_like"] == pytest.approx(1.0)
    assert s["lddt_rep"] == pytest.approx(1.0) and s["rmse_rep"] == pytest.approx(0.0)
    s = M.score(x, x + 0.6, full(), np.zeros_like(x), np.zeros_like(x), sigma=0.5)
    assert s["tm_like"] == pytest.approx(1 / (1 + 1.2 ** 2), rel=1e-4)   # |d| = 1.2 sigma
    assert s["gdt_like"] == pytest.approx(0.5)                    # below 2 and 4 sigma, not below 0.5 and 1 sigma
    assert s["lddt_rep"] == pytest.approx(1.0)                    # offset-invariant
    assert s["rmse_rep"] == pytest.approx(1.2, rel=1e-4)


def test_protein_like_without_tolerance_is_missing_not_zero():
    x = cube(21)
    s = M.score(x, x + 1, full(), np.zeros_like(x), np.zeros_like(x), sigma=None)
    for k in M.PROTEIN:
        assert np.isnan(s[k])


def _rows_for_calibration(seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(40):
        for p_ in (0, 1):
            for j in range(16):
                rows.append({"target": t, "patch": p_, "side": "L", "fill": f"str{j:02d}",
                             **{m: rng.normal() for m in M.METRICS}})
            rows.append({"target": t, "patch": p_, "side": "L", "fill": "repeat",
                         **{m: (-4.0 if m in M.LOWER_BETTER else 4.0) + 0.3 * rng.normal() for m in M.METRICS}})
    return pd.DataFrame(rows)


def test_calibrate_rows_separates_repeat_from_strangers():
    r = M.calibrate_rows(_rows_for_calibration())
    rep = r[r.fill == "repeat"]
    st = r[r.fill.map(M.is_reference_stranger)]
    assert rep["z_ssim"].mean() > 2 and abs(st["z_ssim"].mean()) < 0.2
    assert rep["z_mse"].mean() > 2                                  # orientation: lower MSE = closer
    assert rep["post_ssim"].mean() > 0.95 and st["post_ssim"].mean() < 0.1
    assert rep["nlp_ssim"].mean() > st["nlp_ssim"].mean()


def test_ladder_R_2way_anchors():
    import pandas as pd
    rows = []
    for t in range(12):
        for p_ in range(4):
            rows += [(t, f"f{t}", p_, "repeat", 0.1), (t, f"f{t}", p_, "dataset_mean", 0.5)]
            rows += [(t, f"f{t}", p_, f"str{j:02d}", 1.1) for j in range(16)]
    per2 = pd.DataFrame(rows, columns=["target", "family", "patch", "fill", "mse"])
    pt, arr = M.ladder_R_2way(per2, "mse", reps=30)
    assert pt["repeat"] == pytest.approx(0.0) and pt["dataset_mean"] == pytest.approx(0.4)
    assert np.allclose(arr["dataset_mean"], 0.4)
