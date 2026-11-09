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
    v, n = T.lpips_25d(x, x, e, net="alex")
    assert n == 3 and v == pytest.approx(0.0, abs=1e-6)
    e[8, 0, 0] = False                                      # lies only in the central x slice
    assert T.lpips_25d(x, x, e, net="alex")[1] == 2
    v, n = T.lpips_25d(x, x, np.zeros_like(e), net="alex")
    assert n == 0 and np.isnan(v)


def test_extra_amplitude_cohort_is_one_for_identity():
    x, e = smooth((12, 12, 12)), np.ones((12, 12, 12), bool)
    mu = np.zeros_like(x)
    out = T.extra(x, x, e, mu)
    assert out["amplitude_cohort"] == pytest.approx(1.0)
    assert np.isnan(T.extra(x, x, e, None)["amplitude_cohort"])


def _synthetic_rows(n_fam=12, seed=0):
    """Rows for 2 boxes x n_fam single-target families: repeat closest, noise farthest; parietal LPIPS all missing."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    base = {"repeat": 0.1, "mirror": 0.4, "dataset_mean": 0.6, "shrink_repeat": 0.3, "wrong_detail": 0.9,
            "noise": 1.5, "stranger": 1.0, **{f"str{j:02d}": 1.0 for j in range(16)}}
    rows = []
    for patch in T.FROZEN:
        for i in range(n_fam):
            for f, d in base.items():
                d = d + rng.normal(0, 0.02)
                rows.append({"target": i, "patch": patch, "side": "L", "drawn": True, "fill": f,
                             "ms_ssim": 1 - d / 2, "lpips": np.nan if patch == T.FROZEN[1] else d / 3,
                             "persistence_w1": 50 * d, "amplitude_cohort": 1.0, "lpips_slices": 0 if patch ==
                             T.FROZEN[1] else 3, "persistence_essential": 1})
    return pd.DataFrame(rows)


def test_calibrated_c_orients_lower_is_better_t23_metrics():
    r = _synthetic_rows()
    c = T.M.calibrated_c(r, T.T23)
    rep = c[(c.fill == "repeat") & (c.metric == "lpips") & (c.patch == T.FROZEN[0])].c
    noise = c[(c.fill == "noise") & (c.metric == "persistence_w1")].c
    assert (rep == 1.0).all() and (noise == 0.0).all()        # closer than all 16 strangers -> 1; farther -> 0


def test_summary_keeps_missing_lpips_missing_and_holms_over_both_boxes():
    import pandas as pd
    r = _synthetic_rows()
    tg = pd.DataFrame({"family": [f"fam{i}" for i in range(12)]})
    pos = [None] * 34
    pos[T.FROZEN[0]], pos[T.FROZEN[1]] = ("deep", (0, 0, 0), (2, 2, 2)), ("parietal", (0, 0, 0), (2, 2, 2))
    views, n_hyp = T.summarise(r, tg, pos, T.T23, T.T23_DESCRIPTIVE)
    par, deep = views["parietal/LR"], views["deep/LR"]
    assert par["missing"]["lpips"] == par["n_rows"] and deep["missing"]["lpips"] == 0
    assert par["fills"]["repeat"]["lpips"]["raw"] is None and par["ladder_R"]["lpips"] is None   # missing, not zero
    assert deep["ladder_R"]["lpips"]["repeat"]["R"] == 0.0
    assert deep["tests"]["per_metric"]["persistence_w1"]["T3b"]["pass_holm_run_0.05"]
    n_deep = sum("p_one_sided" in e for d in deep["tests"]["per_metric"].values() for e in d.values())
    n_par = sum("p_one_sided" in e for d in par["tests"]["per_metric"].values() for e in d.values())
    assert n_hyp == n_deep + n_par and n_par > 0


def test_figure_writes_a_missing_metric_as_missing_not_as_zero():
    import donor_ladder_brain_metrics_t23_figure as FIG
    view = {"ladder_R": {"lpips": None, "ms_ssim": {"repeat": {"R": 0.0, "ci95": [0.0, 0.0]},
                                                    "noise": {"R": 0.8, "ci95": [0.7, 0.9]}}}}
    assert FIG.points(view, "lpips") is None and FIG.points(view, "persistence_w1") is None
    rows = FIG.points(view, "ms_ssim")
    assert [k for k, *_ in rows] == [0, len(FIG.RUNGS) - 1]
