"""Known-geometry tests for the H1 harmonised fundus frame (general protocol, Amendment 9.9)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fundus_preprocess as P  # noqa: E402


def marker_image(x, y, size=512):
    rgb = np.full((size, size, 3), 0.3, np.float32)
    rgb[y - 1:y + 2, x - 1:x + 2] = 1.0
    return rgb, np.ones((size, size), bool)


def brightest(c):
    y, x = np.unravel_index(np.argmax(c[..., 0]), c.shape[:2])
    return x, y


def test_disc_lands_on_the_centre_for_both_eyes():
    rgb, ins = marker_image(300, 200)
    c, _, _ = P.frame_h1(rgb, ins, "R", (300.0, 200.0), 0.8)
    assert np.allclose(brightest(c), P.H1_CENTRE, atol=1)
    left = rgb[:, ::-1]                               # the same eye seen as a left eye: disc at x = 511 - 300
    c, _, _ = P.frame_h1(left, ins, "L", (300.0, 200.0), 0.8)   # disc given in oriented coordinates
    assert np.allclose(brightest(c), P.H1_CENTRE, atol=1)


def test_scale_is_applied_around_the_disc():
    rgb, ins = marker_image(300, 200)
    rgb[200, 340] = 0.9                               # 40 px right of the disc
    c, _, _ = P.frame_h1(rgb, ins, "R", (300.0, 200.0), 0.5)
    row = c[256, :, 0]
    peaks = np.flatnonzero(row > 0.85)
    assert np.any(np.abs(peaks - (256 + 20)) <= 1)    # 40 px x 0.5


def test_outside_the_photo_is_missing():
    rgb = np.full((512, 512, 3), 0.3, np.float32)
    ins = np.zeros((512, 512), bool)
    ins[:, 100:400] = True
    _, inside, circle = P.frame_h1(rgb, ins, "R", (256.0, 256.0), 1.0)
    assert not inside[:, :90].any() and inside[256, 256]
    assert circle[256, 256] and not circle[0, 0]


def test_load_square_marks_padding_missing(tmp_path):
    from PIL import Image
    Image.fromarray(np.full((50, 100, 3), 200, np.uint8)).save(tmp_path / "x.png")
    rgb, inside, k, off = P.load_square(str(tmp_path / "x.png"), size=100)
    assert k == 1.0 and tuple(off) == (0, 25)
    assert not inside[:20].any() and inside[30:70].all() and not inside[80:].any()
