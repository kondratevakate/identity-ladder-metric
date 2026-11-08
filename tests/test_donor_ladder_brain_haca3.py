"""Known-geometry tests for the HACA3+ wrapper (Amendment 17)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import donor_ladder_brain_haca3 as H  # noqa: E402

NATIVE = (197, 233, 189)


def test_crop_roundtrip_keeps_brain_and_marks_outside_missing():
    rng = np.random.default_rng(0)
    vol = rng.normal(size=NATIVE).astype(np.float32)
    mask = np.zeros(NATIVE, bool)
    mask[40:150, 40:190, 30:160] = True
    m = H.to_model(vol, mask)
    assert m.shape == H.MODEL_SHAPE
    back = H.from_model(m, NATIVE)
    assert np.array_equal(back[mask], vol[mask])
    # the 5 cropped x columns and 9 y rows are missing, not zero
    assert np.isnan(back[0]).all() and np.isnan(back[:, 0]).all()


def test_crop_refuses_to_cut_brain():
    vol = np.ones(NATIVE, np.float32)
    mask = np.zeros(NATIVE, bool)
    mask[0, 100, 100] = True          # x = 0 is outside the centred 192 crop
    with pytest.raises(RuntimeError):
        H.to_model(vol, mask)


def test_median_target_is_componentwise_and_ignores_order():
    per = {"a": {"theta": [0, 10], "eta": [1, 1], "norm": 5.0},
           "b": {"theta": [2, 0], "eta": [3, 1], "norm": 1.0},
           "c": {"theta": [1, 5], "eta": [2, 9], "norm": 3.0}}
    t = H.median_target(per)
    assert t["theta"] == [1, 5] and t["eta"] == [2, 1] and t["norm"] == 3.0 and t["n_scans"] == 3


def test_session_shift_uses_only_voxels_in_both_masks():
    z1 = np.zeros((10, 10, 10))
    z2 = np.full((10, 10, 10), 0.5)
    z2[:5] = 100.0                    # outside the shared mask: must not count
    m = np.zeros((10, 10, 10), bool)
    m[5:] = True
    box = (slice(0, 10),) * 3
    assert H.session_shift(z1, z2, m, box) == pytest.approx(0.5)
    assert np.isnan(H.session_shift(z1, z2, np.zeros_like(m), box))
