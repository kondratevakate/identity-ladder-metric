"""IXI preparation (Amendment 23): the same image, boxes and ANTs calls as QTIM; eligibility and sampling rules."""
import ast
import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import ixi_prep as X  # noqa: E402

RC = os.path.join(os.path.dirname(__file__), "..", "src", "real_cases.py")


def _rc_source():
    return open(RC, encoding="utf-8").read()


def _func(src, name):
    return next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)


def _ants_calls(fn):
    """Every ants.* call with its keyword arguments, as source text, in order."""
    return [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "ants" and n.func.attr != "image_read"]


def test_prepare_uses_exactly_the_qtim_ants_calls():
    qtim = _ants_calls(_func(_rc_source(), "prepare_scan"))
    ixi = _ants_calls(_func(inspect.getsource(X), "prepare_scan"))
    assert ixi == qtim and len(qtim) >= 6


def test_synthstrip_image_and_call_match_qtim():
    src = _rc_source()
    assert X.SYNTHSTRIP_IMAGE in src
    assert "mri_synthstrip -i \"/bids/$rel\" -m \"/out/${key}_mask.nii.gz\" -t %d" in inspect.getsource(X.strip)


def test_boxes_and_box_mask_match_real_cases():
    src = _rc_source()
    for name, (box, mirror) in X.BOXES.items():
        assert repr(box) in src and repr(mirror) in src
    assert ast.unparse(_func(src, "box_mask").body[1:]) == ast.unparse(_func(inspect.getsource(X), "box_mask").body[1:])
    m = X.box_mask((4, 1, 1), np.eye(4), ((1, 3), (0, 1), (0, 1)))
    assert m[:, 0, 0].tolist() == [False, True, True, False]          # half-open, as QTIM


def test_eligibility_excludes_disagreeing_and_missing_rows():
    xls = pd.DataFrame({"IXI_ID": [1, 2, 2, 3, 4, 4, 5], "SEX_ID (1=m, 2=f)": [1, 2, 2, 1, 1, 2, 1],
                        "AGE": [30, 40, 40.5, np.nan, 50, 50, 60]})
    files = {i: (f"f{i}", "Guys") for i in (1, 2, 3, 4, 5, 6)}
    rows, c = X.eligible(xls, files)
    assert [r["id"] for r in rows] == [1, 2, 5]
    assert (c["missing_sex_or_age"], c["rows_disagree"], c["no_spreadsheet_row"]) == (1, 1, 1)


def test_sample_is_proportional_by_centre_and_reproducible():
    rng = np.random.default_rng(0)
    rows = [{"id": i, "centre": c, "age": float(rng.uniform(20, 86)), "sex": 1, "file": ""}
            for i, c in enumerate(["Guys"] * 313 + ["HH"] * 180 + ["IOP"] * 68)]
    s = X.stratified_sample(rows)
    assert len(s) == 100 and len({r["id"] for r in s}) == 100
    n = {c: sum(r["centre"] == c for r in s) for c in X.CENTRES}
    assert n == {"Guys": 56, "HH": 32, "IOP": 12}
    assert [r["id"] for r in s] == [r["id"] for r in X.stratified_sample(rows)]


def test_vm_stages_have_no_default_threads_or_workers(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ixi_prep.py", "prepare", "--run", "/x", "--threads", "2"])
    with pytest.raises(SystemExit):
        X.main()


def test_d7_excludes_on_the_deep_box_only_as_qtim():
    base = {"template_corr": 0.5, "deep_box_in_own_brain": 1.0, "deep_mirror_in_own_brain": 1.0,
            "parietal_box_in_own_brain": 1.0, "parietal_mirror_in_own_brain": 1.0}
    metas = [dict(base, scan=f"ixi_{k:03d}") for k in range(10)]
    metas[1]["parietal_box_in_own_brain"] = 0.94          # handled per cube by the ladder, as in QTIM
    metas[2]["deep_box_in_own_brain"] = 0.94
    metas[3] = {"scan": "ixi_003", "error": "RuntimeError: x"}
    d = X.d7_flags(metas)
    assert set(d["excluded"]) == {"ixi_002", "ixi_003"}
    assert "box_in_own_brain" in inspect.getsource(X.d7_flags) and "box_in_own_brain" in _rc_source()
