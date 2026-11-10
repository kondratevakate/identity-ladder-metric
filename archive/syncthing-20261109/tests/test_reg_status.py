"""Consolidation must not turn another method's review into visual acceptance."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import reg_status
import reg_run


def test_status_preserves_comparison_and_keeps_methods_separate(tmp_path):
    shutil.copytree(ROOT / 'results/registration', tmp_path / 'results/registration')
    (tmp_path / 'docs').mkdir()
    shutil.copyfile(ROOT / 'docs/PROTOCOL_registration.md', tmp_path / 'docs/PROTOCOL_registration.md')
    frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (tmp_path / 'results/registration').glob('*') if p.is_file() and not p.name.startswith('status.')}
    state = reg_status.build(tmp_path)
    assert state['datasets']['GRAPE']['choice']['recommended'] == 'G1'
    assert state['datasets']['UPOL']['choice']['recommended'] is None
    assert state['datasets']['UTIRIS']['choice']['recommended'] is None
    assert state['development_counts']['intended'] == 1465
    assert state['development_counts']['reviewed'] == 13
    assert state['legacy_fundus_frame_counts']['reviewed'] == 0
    assert state['fundus_frame_counts']['reviewed'] == 7
    assert state['fundus_frame_counts']['pass'] == 0
    assert state['fundus_frame_counts']['fail'] == 6
    assert state['fundus_frame_counts']['unreviewable'] == 1
    assert state['mri_intended'] == 334
    assert state['mri_manifest']['unique_targets'] == 72
    assert state['mri_manifest']['unique_families'] == 68
    assert state['mri_composite']['saved_image_reproduced']
    assert not state['identity_metrics_released']
    for name, digest in frozen.items():
        assert hashlib.sha256((tmp_path / 'results/registration' / name).read_bytes()).hexdigest() == digest


def test_invalid_denominator_rejected():
    with pytest.raises(ValueError):
        reg_status.counts([dict(intended=10, reviewed=1, **{'pass': 0}, fail=0, unreviewable=0, not_reviewed=9)])


def test_scoring_runner_paused_before_reading_data(monkeypatch):
    monkeypatch.setattr(reg_run.L, 'guard', lambda: pytest.fail('Must not read source data'))
    with pytest.raises(RuntimeError, match='paused'):
        reg_run.run('GRAPE', ['G1'], 1)


def test_frozen_report_cannot_be_overwritten(tmp_path, monkeypatch):
    (tmp_path / 'manifest.json').write_text('{}')
    monkeypatch.setattr(reg_run, 'RES_DIR', str(tmp_path))
    with pytest.raises(RuntimeError, match='Frozen comparison'):
        reg_run.report()
