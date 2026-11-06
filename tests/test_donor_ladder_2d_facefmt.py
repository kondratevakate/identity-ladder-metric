"""Known-geometry tests for the fundus face-format ladder (general protocol, Amendment 9.5)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import donor_ladder_2d_facefmt as FF  # noqa: E402


def test_down2_marks_a_pixel_missing_if_any_source_pixel_is():
    img = np.ones((1, 4, 4), np.float32)
    fov = np.ones((1, 4, 4), bool)
    fov[0, 0, 0] = False
    x, m = FF.down2(img, fov)
    assert m.shape == (1, 2, 2) and not m[0, 0, 0] and m[0, 1, 1]
    assert x[0, 0, 0] == 0 and x[0, 1, 1] == 1


def test_local_error_ignores_pixels_outside_the_mask():
    a = np.zeros((FF.SIZE, FF.SIZE))
    b = np.zeros_like(a)
    b[:, :64] = 5.0                       # a large difference, but only outside the mask
    m = np.zeros(a.shape, bool)
    m[:, 64:] = True
    e = FF.local_error(a, b, m)
    assert np.nanmax(e[:, 72:]) == 0      # inside: no error leaks in from the masked-out half
    assert np.isnan(e[:, :50]).all()      # mostly outside: missing, not zero


def test_lookalike_never_picks_an_excluded_image():
    rng = np.random.default_rng(0)
    P = 30
    X = rng.normal(size=(P, FF.SIZE, FF.SIZE)).astype(np.float32)
    M = np.ones(X.shape, bool)
    valid = np.zeros((FF.SIZE, FF.SIZE), bool)
    valid[40:88, 40:88] = True
    excl = np.eye(P, dtype=bool)
    LF = FF.lookalike_fill(X[:3], M[:3], X, M, valid, excl[:3], k=1, min_have=1)
    for t in range(3):                    # with k = 1 the fill is one pool image; never the target itself
        f = LF[t][64, 64]
        assert np.isfinite(f) and not np.isclose(f, X[t, 64, 64])


def test_lookalike_finds_the_planted_twin():
    rng = np.random.default_rng(1)
    P = 20
    X = rng.normal(size=(P, FF.SIZE, FF.SIZE)).astype(np.float32)
    X[5] = X[0] + 0.01 * rng.normal(size=X[0].shape)      # pool image 5 is a near copy of target 0
    M = np.ones(X.shape, bool)
    valid = np.ones((FF.SIZE, FF.SIZE), bool)
    excl = np.eye(P, dtype=bool)
    LF = FF.lookalike_fill(X[:1], M[:1], X, M, valid, excl[:1], k=1, min_have=1)
    assert np.allclose(LF[0][40:88, 40:88], X[5][40:88, 40:88], atol=1e-4)


def test_boot_ratio_is_the_ratio_of_sums():
    pp = {"a": (1.0, 2.0), "b": (3.0, 2.0)}
    r, ci = FF.boot_ratio(pp, np.random.default_rng(0), n=200)
    assert r == 1.0 and ci[0] <= r <= ci[1]
