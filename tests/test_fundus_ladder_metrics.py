"""Known-geometry tests for the fundus metric panel (general protocol, Amendment 9.14)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fundus_ladder_metrics as FM  # noqa: E402

BRAIN = os.environ.get("BRAIN_METRICS_SRC", str(Path(__file__).resolve().parents[2] / "identity-ladder-cvpr2027" /
                                               "src" / "donor_ladder_brain_metrics.py"))


@pytest.fixture(scope="module")
def bm():
    if not os.path.isfile(BRAIN):
        pytest.skip(f"brain metric module not found at {BRAIN} (set BRAIN_METRICS_SRC)")
    return FM.load_bm(BRAIN)


def test_registry_has_26_confirmatory_metrics_and_reasons_for_blocked():
    assert len(FM.CONFIRMATORY) == 26
    for name, (group, col, reason) in FM.CONFIRMATORY.items():
        assert (col is None) == (reason is not None), name


def test_h0_diagram_two_basins():
    v = np.array([[0, 5, 1]], float)
    m = np.ones_like(v, bool)
    d = sorted(map(tuple, FM.h0_diagram(v, m)))
    assert d == [(0.0, 5.0), (1.0, 5.0)]     # the younger basin dies at the saddle 5; the essential bar ends at max


def test_h0_diagram_ignores_pixels_outside_the_mask():
    v = np.array([[0, -100, 1]], float)       # a deep filler minimum outside the mask
    m = np.array([[True, False, True]])
    d = sorted(map(tuple, FM.h0_diagram(v, m)))
    assert d == [(0.0, 1.0), (1.0, 1.0)]      # two components; the filler never enters


def test_persistence_zero_for_identical_and_ignores_filler_outside_e():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 16))
    e = np.ones((16, 16), bool)
    e[:, :4] = False
    f = x.copy()
    f[:, :4] = 50.0                           # filler outside the scored pixels
    assert FM.persistence_distance(x, f, e) == pytest.approx(0.0)
    assert FM.persistence_distance(x, x + rng.normal(size=x.shape), e) > 0


def test_wasserstein_matches_to_diagonal():
    A = np.array([[0.0, 2.0]])
    B = np.zeros((0, 2))
    assert FM.wasserstein1(A, B) == pytest.approx(1.0)


def test_ms_ssim_is_one_on_identity_and_ignores_filler_outside_e():
    rng = np.random.default_rng(1)
    x = ndimage_smooth(rng.normal(size=(FM.SIZE, FM.SIZE)))
    mx = np.ones_like(x, bool)
    ws = (slice(56, 72), slice(56, 72))
    e = np.zeros_like(mx)
    e[ws] = True
    e[56:60, 56:72] = False                   # part of the window is missing in the fill
    f = x.copy()
    f[56:60, 56:72] = 99.0                    # filler on the missing part
    assert FM.ms_ssim(x, mx, f, e, ws) == pytest.approx(1.0, abs=1e-9)
    g = x.copy()
    g[ws] = x[ws] + 1.5 * rng.normal(size=(16, 16))
    assert FM.ms_ssim(x, mx, g, e, ws) < 0.95


def ndimage_smooth(a):
    from scipy import ndimage
    return ndimage.gaussian_filter(a, 2.0)


def test_score_ignores_filler_outside_the_scored_pixels(bm):
    rng = np.random.default_rng(2)
    x = ndimage_smooth(rng.normal(size=(FM.SIZE, FM.SIZE))).astype(np.float32)
    mu = np.zeros_like(x)
    ws = (slice(32, 48), slice(32, 48))
    e = np.zeros(x.shape, bool)
    e[ws] = True
    e[32:36, 32:48] = False
    f = (x + 0.3 * rng.normal(size=x.shape)).astype(np.float32)
    f2 = f.copy()
    f2[32:36, 32:48] = 1e3                    # filler only where e is False
    a = FM.score(x, np.ones_like(e), f, e, ws, mu, None, 0.5)
    b = FM.score(x, np.ones_like(e), f2, e, ws, mu, None, 0.5)
    for k in ("mse", "ssim", "ms_ssim", "lddt", "acc_dataset", "persistence", "tm_like"):
        assert a[k] == pytest.approx(b[k]), k
    assert np.isnan(a["acc_cohort"]) and np.isnan(a["amplitude_cohort"])   # no matched mean: missing, not zero


def test_nearest_never_wins_through_filler_outside_the_mask(bm):
    n = 6
    rng = np.random.default_rng(3)
    b = rng.normal(size=20)
    mb = np.ones(20, bool)
    A = b[None] + rng.normal(scale=1.0, size=(n, 20))
    MA = np.ones((n, 20), bool)
    A[0] = b                                  # identical where valid ...
    MA[0, 5:] = False                         # ... but only 25% shared: must not be selected
    A[0, 5:] = 0.0
    A[1] = b + 0.05                           # the honest nearest candidate
    nn = FM.nearest(A, MA, b, mb, np.ones(n, bool), 1)
    assert nn[0] == 1


def test_fill_covering_less_than_half_the_window_is_not_scored(bm):
    """Regression (smoke QC, 6 Nov 2026): a stranger with no field over most of the window was scored on a few
    pixels. With synthetic units, a donor valid on 25% of every window must never produce a row."""
    n, S = 24, FM.SIZE
    rng = np.random.default_rng(4)
    X = rng.normal(size=(n, S, S)).astype(np.float32)
    M = np.ones((n, S, S), bool)
    M[1, :, : 3 * S // 4] = False             # unit 1 (the only other R-side person) sees only the right quarter
    side = np.array(["R", "L"] * (n // 2))
    side[1] = "R"
    side[0] = "R"
    person = np.array([f"p{i}" for i in range(n)])
    valid = np.ones((S, S), bool)
    st = {"X": X, "M": M, "valid": valid, "idx": np.arange(n), "person": person, "side": side,
          "sex": np.array(["F"] * n), "age": np.full(n, 60.0), "dev": np.array(["d"] * n),
          "fellow": np.full(n, -1), "other": np.full(n, -1), "pos": [(32, 32), (32, 112)], "name": "PAPILA",
          "lam": {}, "sigma": {}}
    st["side_units"] = {s: np.flatnonzero(side == s) for s in ("R", "L")}
    FM.W.clear()
    FM.W.update(st)
    rows = pd.DataFrame(FM.run_target(0))
    left = rows[rows.patch == 0]
    assert not left.empty
    assert (left.n_vox >= FM.MIN_FILL_SHARE * FM.WIN * FM.WIN).all()


def test_positions_need_coverage():
    valid = np.zeros((FM.SIZE, FM.SIZE), bool)
    valid[32:96, 32:96] = True
    pos = FM.positions(valid)
    assert pos and all(32 <= y and y + FM.WIN <= 96 and 32 <= x and x + FM.WIN <= 96 for y, x in pos)


def _rows(n_person=12):
    """Synthetic rows: per cell mse self 0, repeat 0.5, stranger 2, mean 1; fellow differs by side."""
    rows = []
    for p in range(n_person):
        for side in ("R", "L"):
            unit = 2 * p + (side == "L")
            for patch in range(3):
                for fill, v in [("repeat", 0.5), ("fellow", 1.2 if side == "R" else 1.4), ("dataset_mean", 1.0),
                                ("stranger", 2.0)] + [(f"str{k:02d}", 2.0) for k in range(16)]:
                    rows.append({"unit": unit, "person": f"p{p}", "side": side, "patch": patch, "fill": fill,
                                 "mse": v, "mse_V": 1.0, "n_vox": 100})
    return pd.DataFrame(rows)


def test_r_self_scale_and_mse_ratio_directions(bm):
    r = _rows()
    persons = np.array(sorted(r.person.unique()))
    R, _ = FM.r_scale(r, "mse", persons, FM.boot_idx(persons, reps=50))
    assert R["stranger"]["R"] == pytest.approx(1.0) and R["dataset_mean"]["R"] == pytest.approx(0.5)
    R2, _ = FM.r_scale(r, "mse", persons, FM.boot_idx(persons, reps=50), "repeat")
    assert R2["repeat"]["R"] == pytest.approx(0.0)
    blk = FM.mse_ratio_block(r, persons, None)
    d = blk["fellow"]["directions"]
    assert d["R eye as target"]["ratio"] == pytest.approx(1.2) and d["L eye as target"]["ratio"] == pytest.approx(1.4)
    assert d["R eye as target"]["targets"] + d["L eye as target"]["targets"] == blk["fellow"]["targets"]


def test_holm_keeps_blocked_endpoints_in_the_denominator(bm):
    tests = {"mse": {"T1": {"p_one_sided": 0.001}}}
    holm, n = FM.holm_family(tests)
    assert n == 104
    assert holm["mse/T1"]["p_holm"] == pytest.approx(min(1.0, 104 * 0.001))
    assert holm["lpips/T1"]["p_holm"] == 1.0 and not holm["lpips/T1"]["computed"]
