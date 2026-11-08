"""Drawn local-error maps use a masked Gaussian window, never a square box (brain lesson L8, rule STUDIES-019).

A box (uniform) filter is allowed only for validity fractions: the line assigns to a name containing "frac" or
"valid", or carries an explicit "# box-ok: <reason>". Scans the tracked .py files of this repository.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOX = re.compile(r"uniform_filter|box_filter|boxcar")  # box-ok: the pattern this regression test searches for
BOX_OK = re.compile(r"#\s*box-ok:|\b\w*(frac|valid)\w*\s*(,\s*\w+\s*)*=")


def box_windows(lines):
    return [l.strip() for l in lines if BOX.search(l) and not BOX_OK.search(l)]


def test_rule_catches_a_box_map_and_spares_validity_fractions():
    assert box_windows(["err = ndimage.uniform_filter(e, 16)"])  # box-ok: a deliberate offender to test the detector
    assert not box_windows(["frac = ndimage.uniform_filter(m, 16)", "x = uniform_filter(m, 16)  # box-ok: validity"])


def test_no_square_windows_in_map_code():
    r = subprocess.run(["git", "-C", str(ROOT), "grep", "-nE", BOX.pattern, "--", "*.py", ":!tests/"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    hits = [h for h in r.stdout.splitlines() if box_windows([h.split(":", 2)[-1]])]
    assert not hits, "square windows in map code (use a masked Gaussian, or '# box-ok: <reason>'):\n" + "\n".join(hits)
