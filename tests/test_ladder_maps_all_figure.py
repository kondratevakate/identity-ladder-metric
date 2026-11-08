"""The all-modality figure must take brain numbers from Gaussian-window results only.

Regression for 2026-11-08: the figure drew the Gaussian brain NIfTI maps but printed ratios from the older box-window
JSON on this branch, and its footer still said "box window".
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import ladder_maps_all_figure as F  # noqa: E402


def test_box_window_json_is_refused(tmp_path, monkeypatch):
    (tmp_path / "donor_ladder_brain_maps.json").write_text(json.dumps({"window_mm": 16, "maps": {}}))
    monkeypatch.setattr(F, "BRAIN_RESULTS", str(tmp_path))
    with pytest.raises(ValueError):
        F.load_brain_json("donor_ladder_brain_maps.json")


def test_gaussian_window_json_is_accepted(tmp_path, monkeypatch):
    d = {"window": "masked Gaussian, sd = WIN/sqrt(12)", "maps": {"repeat": {"pooled_ratio": 0.3}}}
    (tmp_path / "donor_ladder_brain_maps.json").write_text(json.dumps(d))
    monkeypatch.setattr(F, "BRAIN_RESULTS", str(tmp_path))
    assert F.load_brain_json("donor_ladder_brain_maps.json")["maps"]["repeat"]["pooled_ratio"] == 0.3


def test_footer_does_not_claim_box_window():
    src = open(F.__file__, encoding="utf-8").read()
    assert "box window" not in src.replace("box window).", "")
    assert "not yet switched to the Gaussian" not in src
