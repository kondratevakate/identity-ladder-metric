"""Block checkpoints of the RQ4 wrappers: resume loads saved blocks, recomputes nothing, keeps row order and values."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rq4_checkpoint import blocked_rows  # noqa: E402

CALLS = []


def fn(t):
    CALLS.append(t)
    return [{"unit": t, "patch": p, "mse": np.float64(t) / 3 + p * 1e-17} for p in range(2)]


def test_resume_after_kill_recomputes_only_missing_blocks(tmp_path):
    targets = list(range(10))
    with ThreadPoolExecutor(2) as ex:
        full = blocked_rows(ex, fn, targets, str(tmp_path / "a"), "t", block=4)
    # simulate a kill after the first two blocks: copy them into a fresh folder
    part = tmp_path / "b"
    part.mkdir()
    for f in ("manifest.json", "b00000.pkl", "b00004.pkl"):
        (part / f).write_bytes((tmp_path / "a" / f).read_bytes())
    CALLS.clear()
    with ThreadPoolExecutor(2) as ex:
        resumed = blocked_rows(ex, fn, targets, str(part), "t", block=4)
    assert sorted(CALLS) == [8, 9]
    assert resumed.equals(full)
    assert len(full) == 20 and full.unit.tolist() == sorted(full.unit.tolist())


def test_other_targets_refuse_saved_parts(tmp_path):
    with ThreadPoolExecutor(1) as ex:
        blocked_rows(ex, fn, [0, 1, 2], str(tmp_path), "t", block=2)
        with pytest.raises(SystemExit):
            blocked_rows(ex, fn, [0, 1, 3], str(tmp_path), "t", block=2)
