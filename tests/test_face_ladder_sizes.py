"""Amendment F3 geometry: window sizes in object units (no real data)."""
import os
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import face_ladder_metrics as FM  # noqa: E402
import face_ladder_sizes as S  # noqa: E402


def _valid():
    yy, xx = np.mgrid[:FM.SIZE, :FM.SIZE]
    return ((yy - 64) / 52.0) ** 2 + ((xx - 64) / 42.0) ** 2 <= 1.0       # face-like ellipse


def test_size_rounding_half_up_and_ring_ceil():
    assert S.size_of(79.55, 16) == (5, 3)
    assert S.size_of(79.55, 8) == (10, 5)
    assert S.size_of(79.55, 4) == (20, 10)
    assert S.size_of(80.0, 16) == (5, 3)            # 5.0 exactly
    assert S.size_of(72.0, 16) == (5, 3)            # 4.5 rounds up, not to even


def test_explicit_16px_geometry_reproduces_f1_positions():
    v = _valid()
    assert FM.positions(v, FM.WIN, FM.RING) == FM.positions(v)


def test_positions_satisfy_thresholds_at_every_size():
    v = _valid()
    for w, rw in ((5, 3), (10, 5), (20, 10)):
        pos = FM.positions(v, w, rw)
        assert pos, w
        nominal = (w + 2 * rw) ** 2 - w * w
        for y0, x0 in pos:
            assert y0 % w == 0 and x0 % w == 0
            assert v[y0:y0 + w, x0:x0 + w].mean() >= FM.MIN_POS_VALID
            assert v[FM.ring_mask(y0, x0, FM.SIZE, w, rw)].sum() >= FM.MIN_RING_VALID * nominal
        ring = FM.ring_mask(pos[0][0], pos[0][1], FM.SIZE, w, rw)
        assert not ring[pos[0][0]:pos[0][0] + w, pos[0][1]:pos[0][1] + w].any()


def test_spearman_and_rung_R_skip_missing():
    assert S.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert S.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert np.isnan(S.spearman([1, 2], [1, 2]))
    view = {"ladder_R": {"mse": {"a": {"R": 0.1}, "b": {"R": None}, "c": {"R": float("nan")}, "d": {"R": 1.2}}}}
    assert S.rung_R(view, "mse") == {"a": 0.1, "d": 1.2}          # missing rungs are left out, never zero


def test_every_rung_in_the_size_results_has_a_plain_label():
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "results", "face_ladder_sizes.json")
    if not os.path.exists(path):
        pytest.skip("size results not computed")
    out = json.load(open(path, encoding="utf-8"))
    rungs = {k for src in out["views"].values() for w in src.values() for k in w["ladder_R"]["mse"] if not k.startswith("_")}
    assert rungs and rungs <= set(S.PLAIN), sorted(rungs - set(S.PLAIN))
