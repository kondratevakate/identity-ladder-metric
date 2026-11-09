import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from registration_qc_mri_manifest import rows_for


def test_repeat_unavailability_and_exposed_target_are_not_hidden():
    selection = {'set_S': [{'target': 'a', 'family': 'f'}],
                 'set_R': [{'target': 'b', 'family': 'f'}]}
    rows = rows_for(selection, 'a')
    assert len(rows) == 12
    assert sum(r['availability'] == 'required' for r in rows) == 10
    assert sum(r['status'] == 'unavailable' for r in rows) == 2
    assert all(r['phase'] == 'development' for r in rows if r['target'] == 'a')
    assert all(r['phase'] == 'validation' for r in rows if r['target'] == 'b')
    assert all(r['status'] != 'pass' for r in rows)
