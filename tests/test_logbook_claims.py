"""Every number the CVPR paper prints for a claim equals its result file and is still in the cited .tex section."""
import os
import sys

import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "logbook"))
import claims as C  # noqa: E402

CHECKS = [(c["id"], k) for c in C.CLAIMS for k in c["checks"]]


@pytest.mark.parametrize("cid,chk", CHECKS, ids=[f"{c}-{k[0]}" for c, k in CHECKS])
def test_printed_number_equals_result_file(cid, chk):
    r = C.run_check(chk)
    if r["error"] and chk[4].startswith("bilateral:"):
        pytest.skip(f"fundus worktree not available: {r['error']}")
    assert r["error"] is None, r["error"]
    assert r["match"], f"{cid} {r['label']}: text {r['text']} vs file {r['value']} ({r['source']} {r['key']})"
    assert r["in_text"] is not False, f"{cid} {r['label']}: '{r['text']}' no longer in {r['tex']}"


def test_claim_ids_unique_and_labels_short():
    ids = [c["id"] for c in C.CLAIMS]
    assert len(ids) == len(set(ids))
    for c in C.CLAIMS:   # long labels break Trackio page folders on Windows (MAX_PATH)
        assert len(c["label"]) <= 40, c["label"]
        assert c["status"] in ("verified", "provisional (pre-QC)", "not run"), c["status"]
