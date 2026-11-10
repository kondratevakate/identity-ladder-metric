import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import face_ladder_lookalike16 as L  # noqa: E402
import face_ladder_metrics as FM  # noqa: E402


def test_ranking_starts_with_the_k1_lookalike_and_breaks_ties_by_pool_order():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.normal(size=(40, 30)).astype(np.float32)
        A[7] = A[3]                                    # a tie: row 3 must come before row 7
        MA = rng.random((40, 30)) > 0.1
        MA[7] = MA[3]
        b, mb = rng.normal(size=30).astype(np.float32), rng.random(30) > 0.1
        el = rng.random(40) > 0.2
        el[[3, 7]] = True
        nn = L.ranked(A, MA, b, mb, el, 16)
        assert int(nn[0]) == FM.nearest(A, MA, b, mb, el)
        assert len(nn) == min(16, int(np.isfinite(np.where(el, 0, np.inf)).sum()))
        pos = {int(j): k for k, j in enumerate(nn)}
        if 3 in pos and 7 in pos:
            assert pos[3] < pos[7]


def test_ranking_returns_fewer_when_fewer_donors_qualify():
    A, MA = np.ones((5, 4), np.float32), np.ones((5, 4), bool)
    el = np.array([True, False, True, False, False])
    assert list(L.ranked(A, MA, np.ones(4, np.float32), np.ones(4, bool), el, 16)) == [0, 2]


def _rows(fill, vals, extra=0):
    return pd.DataFrame({"src": "imdb", "target": np.arange(len(vals)) + extra, "patch": 0, "fill": fill,
                         "mse": vals})


def test_gate_needs_every_stored_row_and_equal_values():
    ref = pd.concat([_rows("lookalike", [0.1, 0.2]), _rows("stranger", [1.0, 1.0])])
    tg = pd.DataFrame({"src": "imdb", "target": [0, 1]})
    assert L.compare_rows(_rows("lookalike", [0.1, 0.2]), ref, ["mse"], tg)["passed"]
    assert not L.compare_rows(_rows("lookalike", [0.1, 0.25]), ref, ["mse"], tg)["passed"]      # value differs
    assert not L.compare_rows(_rows("lookalike", [0.1]), ref, ["mse"], tg)["passed"]            # target 1 lost
    assert L.compare_rows(_rows("lookalike", [0.1]), ref, ["mse"], tg.iloc[:1])["passed"]       # only target 0 run
