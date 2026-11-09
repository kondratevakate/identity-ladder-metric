import sys
from pathlib import Path
import numpy as np
from skimage.transform import warp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import ladder2d_register as R
import registration_qc_fundus_frame as F


def synthetic_fundus(disc_x=360, fovea_x=210, y=250):
    """Field of view with a bright disc and a dark fovea at known positions."""
    yy, xx = np.mgrid[:512, :512]
    rgb = np.zeros((512, 512, 3), np.float32)
    fov = np.hypot(xx - 256, yy - 256) < 240
    rgb[fov] = (0.6, 0.3, 0.1)
    disc = np.hypot(xx - disc_x, yy - y) < 25
    rgb[disc] = (1.0, 0.9, 0.5)
    fovea = np.hypot(xx - fovea_x, yy - y) < 20
    rgb[fovea, 1] = 0.05
    rgb[~fov] = 0
    return rgb


def test_frame_transform_matches_scoring_chain():
    rgb = synthetic_fundus()
    t = F.frame_transform(*R.fundus_landmarks(rgb))
    ours = warp(rgb, t, order=1, mode='constant', cval=0.0).astype(np.float32)
    assert np.allclose(ours, R.fundus_canonical(rgb))


def test_landmarks_land_on_targets():
    rgb = synthetic_fundus()
    disc, fovea = R.fundus_landmarks(rgb)
    t = F.frame_transform(disc, fovea)
    assert np.allclose(t.inverse(np.array([disc, fovea])), [R.DISC_R, R.FOVEA_R], atol=1e-6)


def test_fellow_eye_reflection_puts_disc_on_anchor_side():
    # A left-disc image (fellow eye) reflected after canonicalization lands where a right-disc anchor does.
    left = synthetic_fundus(disc_x=152, fovea_x=302)
    disc, fovea = R.fundus_landmarks(left)
    xy = F.flip_x(F.frame_transform(disc, fovea).inverse(np.array([disc, fovea])))
    assert np.allclose(xy, [R.DISC_R, R.FOVEA_R], atol=1e-6)


def test_box_round_trip_and_reflected_source():
    rgb = synthetic_fundus()
    t = F.frame_transform(*R.fundus_landmarks(rgb))
    box = F.BOXES['metric']
    src = F.box_to_source(box, t)
    back = t.inverse(src)
    assert np.allclose(back[0], box[:2], atol=1e-6) and np.allclose(back[2], box[2:], atol=1e-6)
    # Displaying a reflected input mirrors the mapped box, nothing else.
    assert np.allclose(F.box_to_source(box, t, flip_source=True), F.flip_x(src))


def test_reflected_input_is_reflected_back(tmp_path):
    from PIL import Image
    rgb = synthetic_fundus()
    p = tmp_path / 'f.png'
    Image.fromarray((rgb * 255).astype(np.uint8)).save(p)
    plain = F.prepare(p)
    reflected = F.prepare(p, reflected_input=True)
    assert np.allclose(plain[1][0], reflected[1][0]) and np.allclose(plain[3], reflected[3])


def test_boxes_inside_canvas():
    for x0, y0, x1, y1 in F.BOXES.values():
        assert 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512


def test_review_uses_exact_shared_g1_pixels_and_raw_reflection(tmp_path):
    from PIL import Image
    import reg_methods as M
    import ladder2d as L
    path = tmp_path / 'synthetic.png'
    Image.fromarray((synthetic_fundus(disc_x=152, fovea_x=302) * 255).astype(np.uint8)).save(path)
    raw = L.load_base(str(path), 512, True)
    points = np.array([[100., 200.], [255., 255.], [330., 300.]])
    for fellow in (False, True):
        result = M.g1(raw[:, ::-1].copy() if fellow else raw)
        _, _, mapping, output = F.prepare(path, fellow=fellow)
        assert np.array_equal(output, result.out)
        source = result.phi(points)
        if fellow:
            source = F.flip_x(source)
        assert np.allclose(mapping(points), source)
        assert np.allclose(mapping.inverse(source), points)
