"""Amendment 10 (general protocol): one window size f = 1/8 for the cross-dataset summary, RQ4 size curve."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import rq4_window_size as R  # noqa: E402

COMMON_ALL = os.path.join(HERE, "..", "results", "cross_modality", "common_all.json")
UPOL_ROWS = R.P + "iris-ladder-metrics-20261107-v1/rows_UPOL_P2_full.csv.gz"


def _toy(rng, units=12, cells=4):
    rows = []
    for u in range(units):
        for p in range(cells):
            base = {"unit": u, "patch": p}
            for j in range(16):
                rows.append({**base, "fill": f"str{j:02d}", "mse": 2.0 + rng.normal(0, 0.1), "ssim": 0.1})
            rows.append({**base, "fill": "dataset_mean", "mse": 1.0, "ssim": 0.5})
            rows.append({**base, "fill": "lookalike16", "mse": np.nan, "ssim": np.nan})   # missing, never zero
    return pd.DataFrame(rows)


def test_frame_scale_anchors_and_missing_is_not_zero():
    r = _toy(np.random.default_rng(0))
    out = R.frame(r, ["unit", "patch"], "unit", ["mse", "ssim"], np.random.default_rng(1))
    assert out["R_self"]["_strangers"]["mse"][0] == pytest.approx(1.0)
    assert out["R_self"]["dataset_mean"]["mse"][0] == pytest.approx(1.0 / r[r.fill.str.startswith("str")].mse.mean(),
                                                                    rel=1e-3)
    # similarities: d(self) = -1, so R = (-0.5 + 1) / (-0.1 + 1)
    assert out["R_self"]["dataset_mean"]["ssim"][0] == pytest.approx(0.5 / 0.9, abs=1e-4)
    assert "mse" not in out["R_self"].get("lookalike16", {})          # all-NaN rung is absent, not 0


def test_prep_renames_like_common_frame():
    r = pd.DataFrame({"fill": ["fellow", "shrink_fellow", "sibling", "cotwin", "sexage"], "mse": [1.0] * 5})
    out = R.prep(r, {"fellow": "other_side", "sexage": "cohort_mean"})
    assert list(out.fill) == ["other_side", "shrink_ref", "relative", "relative", "cohort_mean"]


@pytest.mark.skipif(not os.path.exists(UPOL_ROWS), reason="private rows not on this machine")
def test_frame_reproduces_common_all_point_values_upol():
    """The copied R_self arithmetic gives the committed common_all.json point values (bootstrap draws differ)."""
    s = [x for x in R.curve_inputs()["UPOL"] if x["w"] == 16][0]
    r, why = R.load(s, R.ALL)
    assert why is None
    mets = [m for m in R.ALL if m in r.columns and r[m].notna().any()]
    out = R.frame(r, s["cell"], s["unit"], mets, np.random.default_rng(0))
    ref = json.load(open(COMMON_ALL))["datasets"]["UPOL"]
    assert out["n_units"] == ref["n_units"]
    for f, d in ref["R_self"].items():
        for m, v in d.items():
            assert out["R_self"][f][m][0] == pytest.approx(v[0], abs=1e-4), (f, m)
            assert out["R_self"][f][m][3] == v[3]


def test_face_window_rule_gives_the_f3_sizes():
    import face_ladder_sizes as FS
    assert [FS.size_of(79.55, k) for k in (16, 8, 4)] == [(5, 3), (10, 5), (20, 10)]


def test_iris_sizes_and_wrapper_allows_only_amendment_sizes():
    import rq4_iris_size as IR
    assert [IR.window_of(k) for k in (16, 8, 4)] == [5, 10, 20]
    assert IR.ALLOWED == (5, 10, 16, 20)


def test_brain_lookalike16_missing_when_fewer_than_16_donors():
    """run_position_geom keeps the k = 1 check but writes no lookalike16 row when fewer than 16 donors qualify."""
    import donor_ladder_brain_lookalike16 as L16
    d = np.array([np.inf] * 5 + [3.0, 1.0, 2.0])
    assert list(L16.nearest(d)) == [6, 7, 5]               # only 3 finite: nearest() returns 3, the rung needs 16
    assert L16.nearest(d).size < L16.K
