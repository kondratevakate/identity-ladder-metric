"""Known-answer checks for the tier 2/3 brain metrics (Amendment 20; no scans needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import donor_ladder_brain_metrics_t23 as T  # noqa: E402


def smooth(shape=(24, 24, 16), seed=0):
    from scipy import ndimage
    return ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=shape), 1.5).astype(np.float32) * 4


def test_ms_ssim_is_one_for_identity_and_drops_with_noise():
    x, e = smooth(), np.ones((24, 24, 16), bool)
    assert T.ms_ssim(x, x, e) == pytest.approx(1.0, abs=1e-9)
    noisy = x + np.random.default_rng(1).normal(size=x.shape).astype(np.float32)
    assert T.ms_ssim(x, noisy, e) < 0.9
    assert abs(sum(T.MS_WEIGHTS) - 1) < 1e-12


def test_ms_ssim_ignores_values_outside_the_mask():
    x, e = smooth(), np.ones((24, 24, 16), bool)
    e[:4] = False
    f = x.copy()
    f[:4] = 1e3                                            # garbage outside E must not matter
    assert T.ms_ssim(x, f, e) == pytest.approx(1.0, abs=1e-9)


def test_downsample_uses_only_valid_children():
    v = np.arange(8, dtype=float).reshape(2, 2, 2)
    e = np.zeros((2, 2, 2), bool)
    e[0] = True                                             # 4 valid children: 0, 1, 2, 3
    m, ok = T.downsample(v, e)
    assert ok.item() and m.item() == pytest.approx(1.5)
    e[0, 0, 0] = False
    assert not T.downsample(v, e)[1].item()


def test_persistence_is_zero_for_identity_and_ignores_outside():
    x, e = smooth((12, 12, 12)), np.ones((12, 12, 12), bool)
    e[:, :, :2] = False
    f = x.copy()
    f[:, :, :2] = -50                                       # outside E: never enters the filtration
    dx, _ = T.diagrams(x, e)
    assert T.persistence_w1(dx, T.diagrams(f, e)[0]) == pytest.approx(0.0, abs=1e-9)
    g = x + np.random.default_rng(2).normal(size=x.shape).astype(np.float32)
    assert T.persistence_w1(dx, T.diagrams(g, e)[0]) > 1.0


def test_lpips_scores_only_slices_fully_inside_the_mask():
    x, e = smooth((16, 16, 16)), np.ones((16, 16, 16), bool)
    v, n = T.lpips_25d(x, x, e)
    assert n == 3 and v == pytest.approx(0.0, abs=1e-6)
    e[8, 0, 0] = False                                      # lies only in the central x slice
    assert T.lpips_25d(x, x, e)[1] == 2
    v, n = T.lpips_25d(x, x, np.zeros_like(e))
    assert n == 0 and np.isnan(v)


def test_extra_amplitude_cohort_is_one_for_identity():
    x, e = smooth((12, 12, 12)), np.ones((12, 12, 12), bool)
    mu = np.zeros_like(x)
    out = T.extra(x, x, e, mu)
    assert out["amplitude_cohort"] == pytest.approx(1.0)
    assert np.isnan(T.extra(x, x, e, None)["amplitude_cohort"])
