"""Amendment 24 A: the 16-look-alike mean uses the k = 1 distances; its nearest donor is the committed look-alike."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import donor_ladder_brain_lookalike16 as L  # noqa: E402


def test_nearest_starts_with_argmin_and_skips_ineligible_donors():
    rng = np.random.default_rng(0)
    d = rng.random(40)
    d[[3, 7]] = np.inf                                     # under 50% scaffold shared: never selected
    d[11] = d[5] = d.min() - 1                             # a tie: the first index wins, as np.argmin
    near = L.nearest(d)
    assert near[0] == np.argmin(d) == 5 and len(near) == 16 and not {3, 7} & set(near)
    assert np.all(np.diff(d[near]) >= 0)
    assert len(L.nearest(np.array([np.inf, 1.0, 2.0]))) == 2   # fewer eligible than 16: all of them, n reported


def test_check_rows_fails_on_any_metric_difference():
    old = pd.DataFrame({"target": [0, 1], "patch": 0, "side": "L", "fill": "lookalike", "mse": [1.0, 2.0],
                        "ssim": [0.5, np.nan]})
    new = old.assign(fill="lookalike_k1_check")
    assert L.check_rows(new, old, ["mse", "ssim"])["ok"]
    assert not L.check_rows(new.assign(mse=[1.0, 2.1]), old, ["mse", "ssim"])["ok"]
    assert not L.check_rows(new.assign(ssim=[0.5, 0.4]), old, ["mse", "ssim"])["ok"]   # NaN pattern differs
