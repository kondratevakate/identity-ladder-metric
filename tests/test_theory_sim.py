from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from theory_sim import theory_error, donor_errors, single_donor_errors, k_eff, volume_study


def test_donor_average_matches_closed_form():
    rng = np.random.default_rng(1)
    for r2 in (0., .3, .6):
        for k in (1, 4, np.inf):
            err, _ = donor_errors(rng, r2, k, n=800)
            assert abs(err.mean()-theory_error(r2, k)) < .02


def test_single_sample_doubles_the_rung_mean_error():
    assert theory_error(.4, 1) == 2*theory_error(.4, np.inf)


def test_single_donor_beats_population_mean_only_above_half():
    rng = np.random.default_rng(2)
    assert single_donor_errors(rng, .4).mean() > theory_error(0, np.inf)
    assert single_donor_errors(rng, .6).mean() < theory_error(0, np.inf)


def test_k_eff_recovers_k_from_spread():
    rng = np.random.default_rng(3)
    for r2 in (0., .3):
        for k in (1, 4, 16):
            _, spread = donor_errors(rng, r2, k, n=800)
            assert abs(k_eff(spread.mean(), r2)/k-1) < .1
    assert k_eff(.3, .3) == np.inf


def test_segmenting_the_average_loses_a_sparse_class_but_averaging_segmentations_does_not():
    s, _ = volume_study(np.random.default_rng(4), 0., n=20, kmax=8)
    assert s['seg_of_mean']['inf']['bias'] < -.9
    assert abs(s['mean_of_segs']['8']['bias']) < .1
    assert abs(s['soft']['8']['bias']) < .1
