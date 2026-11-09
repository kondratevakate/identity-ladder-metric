"""Amendment 22: LPIPS VGG hook, set-level crops, exact Frechet distance and FWD for the brain boxes."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import donor_ladder_brain_metrics_t23 as T  # noqa: E402
import donor_ladder_brain_metrics_t23_v5 as V  # noqa: E402


def _scipy_frechet(A, B):
    from scipy import linalg
    ma, mb = A.mean(0), B.mean(0)
    sa, sb = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    cm = linalg.sqrtm(sa @ sb).real
    return float(((ma - mb) ** 2).sum() + np.trace(sa) + np.trace(sb) - 2 * np.trace(cm))


@pytest.mark.parametrize("n,d", [(200, 8), (30, 60)])          # d <= n (covariance path) and d > n (Gram path)
def test_frechet_is_exact_in_both_regimes(n, d):
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(n, d)), rng.normal(0.3, 1.5, size=(n + 7, d))
    assert V.frechet(A, B) == pytest.approx(_scipy_frechet(A, B), rel=1e-6, abs=1e-6)
    assert V.frechet(A, A) == pytest.approx(0.0, abs=1e-8)


def test_haar_packets_are_orthonormal_and_fwd_is_zero_for_the_same_set():
    rng = np.random.default_rng(1)
    U = rng.random((40, 16, 16))
    P = V.haar_packets(U, 2)
    assert P.shape == (40, 16, 4, 4)
    assert np.allclose((P ** 2).sum((1, 2, 3)), (U ** 2).sum((1, 2)))
    assert V.fwd(P, P) == pytest.approx(0.0, abs=1e-8)


def test_set_crops_are_central_squares_and_invalid_when_they_leave_e():
    x = np.arange(32 * 32 * 24, dtype=float).reshape(32, 32, 24)
    e = np.ones_like(x, bool)
    xc, fc, ok = T.set_crops(x, x + 1, e)
    assert xc.shape == (3, T.SET_SIDE, T.SET_SIDE) and ok.all()
    assert np.array_equal(xc[2], x[4:28, 4:28, 12])                  # central slice of axis 2, central 24 x 24
    assert np.allclose(fc - xc, 1)
    e[16, 10, 12] = False          # lies in the axis-0 slice (x = 16) and the axis-2 slice (z = 12), not in y = 16
    assert T.set_crops(x, x, e)[2].tolist() == [False, True, False]
    e2 = np.ones_like(x, bool)
    e2[0, 0, 0] = False                                              # outside every central crop: all stay valid
    assert T.set_crops(x, x, e2)[2].all()


def test_lpips_net_is_explicit_and_hooks_are_named():
    with pytest.raises(ValueError):
        T._lpips("squeeze")
    assert set(T.EXTRAS) == {"t23", "vgg_set"}
    assert T.extra_vgg_set(np.zeros((32, 32, 24)), np.zeros((32, 32, 24)), np.zeros((32, 32, 24), bool), None) == {}


def test_split_rows_keeps_crops_aligned_and_marks_rows_without_crops():
    S = T.SET_SIDE
    part = [{"target": 0, "fill": "a", "mse": 1.0, "_xc": np.zeros((3, S, S)), "_fc": np.ones((3, S, S)),
             "_ok": np.array([True, False, True])},
            {"target": 0, "fill": "b", "mse": 2.0},
            {"target": 1, "fill": "a", "mse": 3.0, "_xc": np.full((3, S, S), 2.0), "_fc": np.ones((3, S, S)),
             "_ok": np.array([True, True, True])}]
    rows, crops = V.split_rows(part)
    assert rows.crop.tolist() == [0, -1, 1] and "_xc" not in rows
    assert crops["xc"].shape == (2, 3, S, S) and crops["xc"][1, 0, 0, 0] == 2.0
    assert crops["ok"].tolist() == [[True, False, True], [True, True, True]]


def test_set_entry_reports_intervals_and_counts():
    rng = np.random.default_rng(2)
    n = 60
    F = rng.normal(size=(n, 5))
    P = V.haar_packets(rng.random((n, 8, 8)), 1)
    fam = np.repeat(np.arange(12), 5)
    e = V.set_entry(F, F + rng.normal(0, 0.1, F.shape), P, P, fam, 0)
    assert e["n_cells"] == n and e["n_families"] == 12
    assert e["fid_ci95"][0] <= e["fid_ci95"][1] and e["fwd_ci95"][0] <= e["fwd_ci95"][1]
    assert e["fwd"] == pytest.approx(0.0, abs=1e-8)                  # same packets on both sides


def test_feature_cache_is_reused_only_for_the_same_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "PRIVATE", str(tmp_path))
    calls = []
    monkeypatch.setattr(V, "inception_features", lambda U, t: calls.append(1) or U.reshape(len(U), -1).astype(float))
    U = np.random.default_rng(3).random((4, 5, 5))
    a = V.cached_features(U, 1, 32, "")
    b = V.cached_features(U, 1, 32, "")
    assert len(calls) == 1 and np.array_equal(a, b)
    V.cached_features(U + 1, 1, 32, "")                               # other samples: never the stale cache
    assert len(calls) == 2
