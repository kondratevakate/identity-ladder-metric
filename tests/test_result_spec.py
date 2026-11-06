"""ladder-result-spec/1 (general protocol, section 8): unit tests of the checks, and the trigger that re-checks every
results file declaring the spec on each test run."""
import copy
import glob
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import result_spec as RS  # noqa: E402

GOOD = {
    "spec": RS.SPEC,
    "spec_meta": {"bootstrap_unit": "person", "pools": "leave-person-out", "stranger_rung": "stranger",
                  "mirror_rungs": {"fellow": "sides", "mirror": "halves"}},
    "results": {"DS": {
        "stranger": {"ratio": 2.0, "ci95": [1.9, 2.1], "targets": 100, "persons": 50},
        "fellow": {"ratio": 1.3, "ci95": [1.2, 1.4], "targets": 100, "persons": 50,
                   "directions": {"R eye as target": {"ratio": 1.29, "targets": 50},
                                  "L eye as target": {"ratio": 1.31, "targets": 50}}},
        "mirror": {"ratio": 1.1, "ci95": [1.0, 1.2], "targets": 100, "persons": 50,
                   "directions": {"upper vs reflected lower": {"ratio": 1.1, "targets": 100},
                                  "lower vs reflected upper": {"ratio": 1.12, "targets": 100}}},
        "valid_px": 1000}},
}


def hard(doc):
    return [c for c in RS.check(doc) if c[0] == "hard"]


def test_a_correct_result_passes():
    assert RS.check(copy.deepcopy(GOOD)) == []
    assert RS.enforce(copy.deepcopy(GOOD)) == []


@pytest.mark.parametrize("path, value", [
    (("spec_meta", "bootstrap_unit"), "image"),          # eyes or images are not independent units
    (("spec_meta", "pools"), "all"),                     # target's own person may sit in the pool
])
def test_wrong_unit_or_pool_fails(path, value):
    d = copy.deepcopy(GOOD)
    d[path[0]][path[1]] = value
    assert hard(d)
    with pytest.raises(RS.SpecError):
        RS.enforce(d)


def test_missing_direction_fails():
    d = copy.deepcopy(GOOD)
    del d["results"]["DS"]["fellow"]["directions"]["L eye as target"]
    assert any("2 directions" in c[3] for c in hard(d))


def test_side_directions_must_partition_the_targets():
    d = copy.deepcopy(GOOD)
    d["results"]["DS"]["fellow"]["directions"]["L eye as target"]["targets"] = 60   # 50 + 60 != 100: double count
    assert any("!= targets" in c[3] for c in hard(d))


def test_half_directions_must_cover_every_target():
    d = copy.deepcopy(GOOD)
    d["results"]["DS"]["mirror"]["directions"]["upper vs reflected lower"]["targets"] = 50
    assert hard(d)


def test_ci_must_contain_the_ratio_and_stranger_must_exist():
    d = copy.deepcopy(GOOD)
    d["results"]["DS"]["fellow"]["ci95"] = [1.4, 1.5]
    del d["results"]["DS"]["stranger"]
    msgs = " ".join(c[3] for c in hard(d))
    assert "CI" in msgs and "stranger rung missing" in msgs


def test_soft_flags_for_odd_stranger_and_asymmetric_directions():
    d = copy.deepcopy(GOOD)
    d["results"]["DS"]["stranger"].update(ratio=1.6, ci95=[1.5, 1.7])
    d["results"]["DS"]["fellow"]["directions"]["L eye as target"]["ratio"] = 1.8
    soft = RS.enforce(d)
    assert len(soft) == 2 and not hard(d)


def test_leave_person_out_assertion():
    RS.assert_leave_person_out("p1", ["p2", "p3"])
    with pytest.raises(RS.SpecError):
        RS.assert_leave_person_out("p1", ["p2", "p1"])


def test_every_results_file_that_declares_the_spec_passes():
    """The trigger: any results JSON that declares the spec is re-checked here on every test run."""
    files = [f for f in glob.glob(str(ROOT / "results" / "*.json"))
             if json.load(open(f, encoding="utf-8")).get("spec") == RS.SPEC]
    for f in files:
        RS.enforce(json.load(open(f, encoding="utf-8")))
