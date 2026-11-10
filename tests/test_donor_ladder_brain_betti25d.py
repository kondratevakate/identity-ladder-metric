"""Amendment 24C: 2.5D Betti matching scores only slices wholly in E (missing, never 0); subset S is seeded."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import donor_ladder_brain_betti25d as BB  # noqa: E402
import donor_ladder_brain_metrics_t23 as T  # noqa: E402

BETTI = os.path.join(os.environ.get("DATA_ROOT", r"C:\Projects\data"), "projects", "identity-ladder-metric",
                     "models-20261107", "betti-matching-2d")


def test_subset_draws_two_whole_positions_per_target_reproducibly():
    w = np.zeros((3, 32), bool)
    w[0, [1, 5, 9]] = True
    w[1, [4]] = True                                   # only one whole cube: one cell, reported
    s = BB.subset(w)
    assert len(s[0]) == 2 and set(s[0]) <= {1, 5, 9} and s[1] == [4] and 2 not in s
    assert BB.subset(w) == s


@pytest.mark.skipif(not os.path.isdir(BETTI), reason="2D Betti library not on this machine")
def test_slices_outside_e_are_missing_never_zero():
    rng = np.random.default_rng(0)
    x, f = rng.normal(size=(16, 16, 16)), rng.normal(size=(16, 16, 16))
    e = np.ones_like(x, bool)
    m, u, n, _ = T.betti_25d(x, f, e)
    assert n == 3 and m > 0 and u >= 0
    e[8, 8, 8] = False                                 # lies in all three central slices
    assert np.isnan(T.betti_25d(x, f, e)[0]) and T.betti_25d(x, f, e)[2] == 0
    e = np.ones_like(x, bool)
    e[8, 0, 0] = False                                 # only in the axis-0 central slice
    assert T.betti_25d(x, f, e)[2] == 2
    assert T.betti_25d(x, x, np.ones_like(x, bool))[:2] == (0.0, 0.0)


def test_betti_parts_are_distances_for_tests_and_r():
    import donor_ladder_brain_metrics as M
    assert set(BB.METRICS) <= M.LOWER_BETTER and set(BB.lower_is_better_in_json()) == set(BB.METRICS)
