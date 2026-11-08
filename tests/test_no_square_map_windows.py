"""No square windows in map code (general protocol section 8; brain lesson L8, rule STUDIES-019).

Every box filter in a tracked source file must carry `# box-ok: <reason>` on the same line (allowed only for
validity fractions and ladder patch definitions); drawn maps use a masked Gaussian window with the box's standard
deviation (width / sqrt(12)). Files owned by another session that still draw box maps are listed in KNOWN with their
owner; the list must shrink, never silently grow.
"""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOX = re.compile(r"uniform_filter|box_filter|boxcar")  # box-ok: detector pattern
OK = re.compile(r"#\s*box-ok:\s*\S")
KNOWN = {}  # donor_ladder_brain_maps.py synced to the Gaussian version of cvpr2027-paper on 2026-11-08


def offenders(lines_by_file):
    out = {}
    for path, lines in lines_by_file.items():
        hits = [f"{path}:{n}: {line.strip()[:120]}" for n, line in enumerate(lines, 1)
                if BOX.search(line) and not OK.search(line)]
        if hits:
            out[path] = hits
    return out


def tracked_sources():
    files = subprocess.run(["git", "-C", str(REPO), "ls-files", "*.py"], capture_output=True, text=True,
                           check=True).stdout.split()
    return {f: (REPO / f).read_text(encoding="utf-8", errors="replace").splitlines()
            for f in files if (REPO / f).exists() and not f.startswith("tests/")}


def test_detector_catches_an_unmarked_box_and_accepts_a_marked_one():
    src = {"a.py": ["e = ndimage.uniform_filter(d ** 2, 16)",  # box-ok: test fixture
                    "frac = ndimage.uniform_filter(m, 16)  # box-ok: validity fraction"]}
    hits = offenders(src)
    assert list(hits) == ["a.py"] and len(hits["a.py"]) == 1 and ":1:" in hits["a.py"][0]


def test_no_unmarked_box_windows_outside_known_owner_files():
    hits = offenders(tracked_sources())
    new = {f: h for f, h in hits.items() if f not in KNOWN}
    assert not new, "square windows in map code (use a masked Gaussian or mark '# box-ok: <reason>'):\n" + \
        "\n".join(h for v in new.values() for h in v)


def test_known_list_is_current():
    hits = offenders(tracked_sources())
    stale = [f for f in KNOWN if f not in hits]
    assert not stale, f"fixed or removed, drop from KNOWN: {stale}"


def test_fundus_map_window_is_round_and_ignores_filler_outside_the_mask():
    import sys
    import numpy as np
    sys.path.insert(0, str(REPO / "src"))
    import donor_ladder_2d as D
    n = 129
    x = np.zeros((n, n))
    x[64, 64] = 1.0                                   # an impulse: a box draws a square, a Gaussian a disc
    m = np.ones((n, n), bool)
    g = D.gauss(x, m)
    assert np.isclose(g[64, 84], g[76, 80], rtol=0.02)   # both 20 px from the impulse (box: on vs off the square)
    assert g[64, 84] > 0 and g[84, 84] < g[64, 84]      # the corner at 28 px is lower; a 48 px box would be flat
    y = np.full((n, n), 50.0)                         # filler outside the mask must not leak in
    mm = np.zeros((n, n), bool)
    mm[:, 70:] = True
    y[:, 70:] = 1.0
    assert np.allclose(D.gauss(y, mm)[:, 75:], 1.0)
