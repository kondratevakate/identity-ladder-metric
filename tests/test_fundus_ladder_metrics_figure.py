"""The Amendment 9.15 figure reads only the aggregate, skips absent rungs and never draws a missing value as zero."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import fundus_ladder_metrics_figure as FIG  # noqa: E402


def _panel():
    R = {"repeat": {"R": 0.3, "ci95": [0.2, 0.4]}, "stranger": {"R": 1.0, "ci95": [0.9, 1.1]}}
    sl = {"stranger": {"ratio_to_stranger": 1.0}, "fellow": {"ratio_to_stranger": 0.0},
          "repeat": {"status": "not run (fewer than 50 windows)"}}
    return {"R_self": {m: R for m, _ in FIG.PER_TARGET},
            "amendment_9_15": {"set_level": {"fid": sl, "fwd": sl}}}


def test_absent_rungs_are_skipped_not_zero():
    p = _panel()
    xs = [x for x, *_ in FIG.per_target_points(p, "lpips_alex")]
    assert xs == [FIG.ORDER.index("repeat"), FIG.ORDER.index("stranger")]
    sp = dict(FIG.set_points(p, "fid"))
    assert FIG.ORDER.index("repeat") not in sp            # 'not run' stays absent
    assert sp[FIG.ORDER.index("fellow")] == 0.0            # clamped value kept, labelled in the figure


def test_draw_writes_png_and_pdf(tmp_path):
    src = tmp_path / "a.json"
    src.write_text(json.dumps({"panel": {ds: _panel() for ds in FIG.DATASETS}}), encoding="utf-8")
    out = FIG.draw(str(src), str(tmp_path / "fig"))
    assert os.path.getsize(out + ".png") > 0 and os.path.getsize(out + ".pdf") > 0
