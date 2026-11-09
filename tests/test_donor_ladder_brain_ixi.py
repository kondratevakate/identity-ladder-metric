"""IXI ladder (Amendment 23): age-matched mean, centre strangers as exact expectations, other side as the anchor."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import donor_ladder_brain_ixi as I  # noqa: E402
import donor_ladder_brain_metrics as M  # noqa: E402


def test_age_matched_takes_the_ten_nearest_same_sex_donors():
    sex = np.array([1] + [1] * 12 + [2] * 5)
    age = np.array([50.0] + list(50 + np.arange(1, 13)) + [50.0] * 5)
    pick, gap = I.age_matched(0, np.arange(1, 18), sex, age)
    assert sorted(pick.tolist()) == list(range(1, 11)) and gap == 5.5
    assert I.age_matched(0, np.arange(13, 18), sex, age) == (None, None)     # fewer than 10: rung missing, not filled


def test_centre_strangers_are_the_mean_over_their_draws():
    r = pd.DataFrame({"target": [0, 0, 0, 0], "patch": 1, "side": "L", "drawn": True,
                      "fill": ["sc00", "sc01", "dc00", "stranger"], "mse": [1.0, 3.0, 5.0, 9.0]})
    e = I.stranger_expectations(r).set_index("fill")
    assert e.loc["stranger_same_centre", "mse"] == 2.0 and e.loc["stranger_same_centre", "n_strangers"] == 2
    assert e.loc["stranger_other_centre", "mse"] == 5.0


def test_other_side_is_the_r_anchor():
    rng = np.random.default_rng(0)
    rows = []
    for t in range(12):
        for f, d in (("mirror", 1.0), ("dataset_mean", 2.0), ("noise", 4.0)):
            rows.append({"target": t, "family": t, "fill": f, "mse": d + rng.normal(0, 0.01)})
        for k in range(16):
            rows.append({"target": t, "family": t, "fill": f"str{k:02d}", "mse": 3.0})
    R, _ = M.ladder_R(I.as_qtim(pd.DataFrame(rows)), "mse")
    assert abs(R["repeat"]["R"]) < 1e-9                      # the other side, renamed, is R = 0
    assert abs(R["dataset_mean"]["R"] - 0.5) < 0.02 and abs(R["noise"]["R"] - 1.5) < 0.02


def test_unavailable_rungs_and_cohort_metric_are_never_reported():
    assert "acc_cohort" not in I.METRICS and set(I.NA_RUNGS) >= {"repeat", "sibling", "cotwin", "cohort_mean"}
    assert not set(M.PROTEIN) & set(I.METRICS) and set(M.PROTEIN) <= set(I.METRICS_NA)   # need a repeat: n/a


def test_centre_draws_never_enter_the_r_table():
    rows = []
    for t in range(12):
        for f, d in (("mirror", 1.0), ("sc00", 9.0), ("stranger_same_centre", 2.0)):
            rows.append({"target": t, "family": t, "fill": f, "mse": d})
        for k in range(16):
            rows.append({"target": t, "family": t, "fill": f"str{k:02d}", "mse": 3.0})
    R = I.other_side_R(pd.DataFrame(rows), "mse")
    assert "sc00" not in R and "mirror" in R and abs(R["stranger_same_centre"]["R"] - 0.5) < 1e-9
    assert set(R["_anchors"]) == {"d_other_side", "d_stranger", "n_other_side_targets"}
