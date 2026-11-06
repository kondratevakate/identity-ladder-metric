"""Known-geometry tests for the P9 full-field canvas (general protocol, Amendment 9.2)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fundus_preprocess as P  # noqa: E402

K = 512 / P.P9_RAW


def photo(disc_offset):
    """Synthetic k-scaled PAPILA photo: field circle (diameter 2377 raw) cut by the sensor, a 3x3 marker at the disc."""
    h, w = int(round(1934 * K)), int(round(2576 * K))
    yy, xx = np.mgrid[:h, :w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    img = np.zeros((h, w, 3), np.float32)
    img[np.hypot(xx - cx, yy - cy) <= 2377 * K / 2] = 0.5
    d = np.array([cx + disc_offset[0], cy + disc_offset[1]])
    dy, dx = int(round(d[1])), int(round(d[0]))
    img[dy - 1:dy + 2, dx - 1:dx + 2] = 1.0
    return img, np.array([dx, dy], float)


def test_p9_keeps_the_whole_field_and_centres_the_disc():
    for off in ((0, 0), (80, -40), (-36, 30), (82, 34), (-37, -50)):                 # within the 1-99% spread of disc offsets (k-scaled)
        img, disc = photo(off)
        c, inside = P.canvas_p9(img, disc, "R")
        f = c[..., 0] > 0.1
        assert abs(f.sum() / (img[..., 0] > 0.1).sum() - 1) < 0.005    # no acquired field lost (edge interpolation)
        assert not (f[0].any() or f[-1].any() or f[:, 0].any() or f[:, -1].any())
        y, x = np.unravel_index(np.argmax(c[..., 0]), c.shape[:2])
        assert abs(x - P.P9_CENTRE[0]) <= 1 and abs(y - P.P9_CENTRE[1]) <= 1
        assert not (f & ~inside).any()                             # every field pixel is marked as photo


def test_p9_reflects_left_eyes_and_keeps_the_disc_at_the_centre():
    left, disc_l = photo((-60, 10))                            # a left eye: disc on the other side of the photo
    right, disc_r = photo((60, 10))                            # the matching right eye
    c, _ = P.canvas_p9(left, disc_l, "L")
    y, x = np.unravel_index(np.argmax(c[..., 0]), c.shape[:2])
    assert abs(x - P.P9_CENTRE[0]) <= 1 and abs(y - P.P9_CENTRE[1]) <= 1
    cr, _ = P.canvas_p9(right, disc_r, "R")
    assert abs((c[..., 0] > 0.1).sum() / (cr[..., 0] > 0.1).sum() - 1) < 0.005


def test_p9_outside_the_photo_is_marked_missing():
    img, disc = photo((0, 0))
    _, inside = P.canvas_p9(img, disc, "R")
    assert not inside[0, 0] and not inside[0, 256]             # corners and the strips above the sensor
    assert inside[247, 278]
