"""Amendment 21: window sizes in object units for the brain tier-1 panel."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import donor_ladder_brain_bilateral as BL  # noqa: E402
import donor_ladder_brain_metrics as M  # noqa: E402
import donor_ladder_brain_metrics_size as Z  # noqa: E402


def test_geometry_only_allows_the_amendment_sizes_and_scales_the_shell_not_the_search():
    assert Z.geometry(8) == {"cube": 8, "shell": 2, "W": 8, "f": "1/16", "f_exact": 8 / Z.L_MM}
    assert Z.geometry(32)["shell"] == 8 and Z.geometry(32)["W"] == 8
    assert Z.geometry(16)["shell"] == BL.SHELL         # f = 1/8 is exactly the reference configuration
    with pytest.raises(ValueError):
        Z.geometry(12)


def test_sizes_match_the_measured_object_extent():
    for w, f in Z.SIZES.items():
        num, den = map(int, f.split("/"))
        assert round(Z.L_MM * num / den) == w


def test_sized_positions_keep_centres_and_drop_windows_leaving_the_template():
    tmask = np.zeros((80, 80, 80), bool)
    tmask[10:70, 10:70, 10:70] = True
    pos = [("random_patches", (32, 32, 32), (16, 16, 16)),     # centre 40: deep inside
           ("random_patches", (6, 32, 32), (16, 16, 16)),      # centre x 14: near the template edge
           ("deep", (1, 2, 3), (4, 5, 6))]
    out8, kept8, dropped8 = Z.sized_positions(pos, tmask, 8, 2)
    out32, kept32, dropped32 = Z.sized_positions(pos, tmask, 32, 8)
    assert out8[0] == ("random_patches", (36, 36, 36), (8, 8, 8))    # same centre, start = centre - w/2
    assert out32[0][1] == (24, 24, 24)
    assert out8[2] == pos[2] and 2 not in kept8 + dropped8            # boxes untouched, never run
    assert kept32 == [0] and dropped32 == [1]                         # the 48 mm window leaves the template
    assert 0 in kept8


def test_rank_stability_is_one_for_the_same_order_and_minus_one_for_the_reverse():
    def view(vals):
        return {"fills": {f: {"mse": {"raw": {"mean": v}}} for f, v in vals.items()}}
    a = view({"repeat": 0.1, "mirror": 0.5, "stranger": 1.0, "noise": 2.0})
    b = view({"repeat": 0.2, "mirror": 0.6, "stranger": 1.5, "noise": 3.0, "extra": 9.0})
    c = view({"repeat": 2.0, "mirror": 1.0, "stranger": 0.5, "noise": 0.1})
    res = Z.rank_stability({8: a, 16: b, 32: c}, ["mse"])["mse"]
    assert res["n_fills"] == 4 and "extra" not in res["fills"]       # only fills present at every size
    assert res["8_vs_16"] == pytest.approx(1.0) and res["8_vs_32"] == pytest.approx(-1.0)


def test_rank_stability_skips_missing_values_instead_of_counting_them_as_zero():
    a = {"fills": {"repeat": {"mse": {"raw": {"mean": 0.1}}}, "noise": {"mse": {"raw": None}}}}
    res = Z.rank_stability({8: a, 32: a}, ["mse"])["mse"]
    assert res["n_fills"] == 1 and res["8_vs_32"] is None


def test_reference_entry_point_passes_the_amendment_10_geometry_explicitly(monkeypatch):
    seen = {}
    monkeypatch.setattr(M, "run_position_at", lambda args: seen.setdefault("args", args))
    M.run_position((5, None, None, None, None, None, False, None))
    geom = seen["args"][-1]
    assert geom == {"shell": BL.SHELL, "W": BL.W, "paths": BL.region_paths(5)}
