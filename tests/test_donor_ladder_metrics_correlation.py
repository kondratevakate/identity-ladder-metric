import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import donor_ladder_brain_seg as S  # noqa: E402
import donor_ladder_metrics_correlation as C  # noqa: E402


def _rows(n_cells=5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n_cells):
        base = rng.normal(size=len(C.RUNGS))
        for f, b in zip(C.RUNGS, base):
            out.append({"target": c, "patch": 0, "side": "L", "fill": f,
                        "mse": b, "rmse": np.sqrt(b - base.min() + 1),   # monotone in mse -> rho 1
                        "pearson": b,                                    # similarity equal to a distance -> -1
                        "mae": rng.normal()})
    return pd.DataFrame(out)


def test_monotone_metrics_have_rho_one_and_orientation_is_applied():
    m = ["mse", "rmse", "pearson", "mae"]
    rho, n = C.cell_rho(C.distances(_rows(), m), m, C.RUNGS, ["target", "patch", "side"])
    R = C.fisher_mean(rho)
    assert n == 5
    assert np.isclose(R[0, 1], 1, atol=1e-5)
    assert np.isclose(R[0, 2], -1, atol=1e-5)      # pearson is negated: higher pearson = closer


def test_cells_with_a_missing_rung_are_dropped():
    r = _rows()
    r = r[~((r.target == 0) & (r.fill == "noise"))]
    _, n = C.cell_rho(C.distances(r, ["mse", "mae"]), ["mse", "mae"], C.RUNGS, ["target", "patch", "side"])
    assert n == 4


def test_clusters_join_identical_metrics_only():
    R = np.array([[1, 0.99, 0.1], [0.99, 1, 0.1], [0.1, 0.1, 1]])
    groups, _ = C.cluster(R, ["a", "b", "c"], cut=0.8)
    assert groups == [["a", "b"], ["c"]]


def test_label_scores_use_only_scored_voxels():
    oracle = np.zeros((10, 10, 10), np.int16)
    oracle[:5] = 2
    oracle[5:] = 41
    pred = oracle.copy()
    pred[8:] = 2                                    # wrong, but outside the scored set
    s = np.zeros_like(oracle, bool)
    s[:8] = True
    out = S.label_scores(pred, oracle, s)
    assert out["dice_mean"] == 1.0 and out["accuracy"] == 1.0 and out["n_labels"] == 2
    s[:] = True
    assert S.label_scores(pred, oracle, s)["dice_mean"] < 1


def test_pick_is_evenly_spaced_and_keeps_ends():
    assert S.pick(list(range(17)), 5) == [0, 4, 8, 12, 16]
    assert S.pick([3, 4], 5) == [3, 4]


def test_dkt_parcels_collapse_to_aseg_cortex():
    oracle = np.zeros((10, 10, 10), np.int16)
    oracle[:5] = 1028          # left superior frontal
    oracle[5:] = 2028          # right superior frontal
    pred = oracle.copy()
    pred[:5] = 1003            # wrong left parcel, still left cortex
    s = np.ones_like(oracle, bool)
    out = S.label_both(pred, oracle, s)
    assert out["dice_mean"] == 1.0          # aseg level: cortex is cortex
    assert out["dice_dkt"] < 1.0            # DKT level sees the wrong parcel
