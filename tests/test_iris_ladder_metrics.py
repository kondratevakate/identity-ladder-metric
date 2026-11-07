"""Known-geometry tests for the iris metric panel (docs/PROTOCOL_iris_metrics.md). Synthetic data only."""
import os
import sys

import numpy as np
import pytest
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import iris_ladder_metrics_frame as IF  # noqa: E402

FUNDUS = os.path.normpath(os.path.join(ROOT, "..", "identity-ladder-bilateral", "src", "fundus_ladder_metrics.py"))
BRAIN = os.path.join(ROOT, "src", "donor_ladder_brain_metrics.py")


def eye(w=320, h=240, cx=160.0, cy=120.0, rp=25.0, ri=70.0, seed=0, reflex=True):
    """Synthetic eye: dark pupil, textured iris (radial fibres), bright sclera; optional ring reflex in the pupil."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    r = np.hypot(xx - cx, yy - cy)
    th = np.arctan2(cy - yy, xx - cx)
    rng = np.random.default_rng(seed)
    fib = sum(rng.uniform(5, 15) * np.cos(k * th + rng.uniform(0, 6.28)) for k in (7, 13, 29, 41))
    img = np.where(r < rp, 30.0, np.where(r < ri, 110.0 + fib, 210.0))
    if reflex:
        img = np.where((r > 4) & (r < 7), 250.0, img)
    return img


def test_pupil_found_through_a_ring_reflex():
    img = eye()
    s = IF.segment(img, "UTIRIS_NIR")
    px, py, rp = s["pupil"]
    ix, iy, ri = s["iris"]
    assert abs(px - 160) <= 2 and abs(py - 120) <= 2 and abs(rp - 25) <= 2.5
    assert abs(ri - 70) <= 4 and s["offset"] < 0.1


def test_mirror_is_the_strip_of_the_flipped_image():
    """mirror() of the strip equals the strip sampled from the left-right flipped image (exact column re-index)."""
    img = eye(reflex=False)
    w = img.shape[1]
    seg = {"pupil": (150.0, 118.0, 25.0), "iris": (152.0, 119.0, 70.0)}
    segf = {"pupil": (w - 1 - 150.0, 118.0, 25.0), "iris": (w - 1 - 152.0, 119.0, 70.0)}
    xs, ys, _ = IF.strip_coords(seg, 1.0)
    xf, yf, _ = IF.strip_coords(segf, 1.0)
    a = ndimage.map_coordinates(img, [ys, xs], order=1)
    b = ndimage.map_coordinates(img[:, ::-1], [yf, xf], order=1)
    np.testing.assert_allclose(IF.mirror(a), b, atol=1e-6)
    np.testing.assert_array_equal(IF.mirror(IF.mirror(a)), a)


def test_eyelid_masked_outward_and_reflections_masked():
    rng = np.random.default_rng(1)
    v = rng.normal(100, 10, (IF.H, IF.W))
    inside = np.ones_like(v, bool)
    th = np.degrees(np.linspace(0, 2 * np.pi, IF.W, endpoint=False))
    up = (th >= 70) & (th <= 110)
    v[40:, up] = 220.0                       # eyelid skin from radial row 40 outward in the upper sector
    v[10:13, 200:203] = 255.0                # specular spot outside the eyelid sectors
    valid, shares = IF.occlusion_mask(v, inside)
    assert not valid[40:, up].any()
    assert not valid[10:13, 200:203].any()
    assert valid[:35, up].mean() > 0.95      # iris above the lid edge kept
    assert shares["eyelid"] > 0


def test_positions_respect_validity():
    import iris_ladder_metrics as IL
    valid = np.ones((IF.H, IF.W), bool)
    valid[:, :20] = False
    for y, x in IL.positions(valid):
        assert valid[y:y + IL.WIN, x:x + IL.WIN].mean() >= IL.MIN_POS_VALID


@pytest.mark.skipif(not (os.path.isfile(FUNDUS) and os.path.isfile(BRAIN)), reason="shared metric modules not found")
def test_filler_outside_mask_never_wins_lookalike_and_is_never_scored():
    """A pool image identical to the target inside the window but whose mask excludes the window (stored values
    there are a filler) must not become the oracle look-alike, and stored zeros never enter a score."""
    import iris_ladder_metrics as IL
    IL.load_shared(FUNDUS, BRAIN)
    rng = np.random.default_rng(3)
    n = 24
    X = rng.normal(0, 1, (n, IF.H, IF.W)).astype(np.float32)
    M = np.ones((n, IF.H, IF.W), bool)
    person = np.array([f"p{i // 2:02d}" for i in range(n)])
    side = np.array(["R", "L"] * (n // 2))
    idx = np.arange(n)
    # pool image 2 (person p01, side R): copy of the target 0 inside window (0, 0), but masked there
    X[2, :16, :16] = X[0, :16, :16]
    M[2, :16, :16] = False
    valid = np.ones((IF.H, IF.W), bool)
    st = {"name": "synthetic", "X": X, "M": M, "XF": IF.mirror(X), "MF": IF.mirror(M), "idx": idx, "valid": valid,
          "person": person, "side": side, "repeat": -np.ones(n, int), "fellow": np.array([i + 1 if i % 2 == 0 else i - 1 for i in range(n)]),
          "pos": [(0, 0)], "lam": {"fellow": 0.3}, "sigma": {}}
    st["side_units"] = {s: np.flatnonzero(side == s) for s in ("R", "L")}
    IL.S_.clear()
    IL.S_.update(st)
    fx, pool, Xp, Mp, mu, mu_ok = IL.fixed_fills(0)
    fills, *_ = IL.window_fills(0, 0, fx, pool, Xp, Mp, mu, mu_ok)
    oracle = fills["lookalike_oracle"][0]
    assert not np.allclose(oracle[:16, :16], X[0, :16, :16])
    rows = IL.run_target(0)
    assert rows and all(np.isfinite(r["mse"]) for r in rows)
    assert not any(r["fill"] == "lookalike_oracle" and r["mse"] == 0 for r in rows)


@pytest.mark.skipif(not (os.path.isfile(FUNDUS) and os.path.isfile(BRAIN)), reason="shared metric modules not found")
def test_identical_fill_scores_perfectly():
    import iris_ladder_metrics as IL
    IL.load_shared(FUNDUS, BRAIN)
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (IF.H, IF.W))
    m = np.ones_like(x, bool)
    ws = (slice(16, 32), slice(32, 48))
    e = np.zeros_like(m)
    e[ws] = True
    s = IL.score(x, m, x.copy(), e, ws, np.zeros_like(x), None)
    assert s["mse"] == 0 and s["ssim"] == pytest.approx(1.0) and s["ms_ssim"] == pytest.approx(1.0)
    assert s["persistence"] == pytest.approx(0.0)


def test_preprocessing_version_is_required_and_checked():
    with pytest.raises(TypeError):
        IF.cache_path("UPOL")                       # no default version
    with pytest.raises(ValueError):
        IF.cache_path("UPOL", "P9")
    assert IF.cache_path("UPOL", "P2").endswith("frames_UPOL_P2.npz")
    assert "P1" in IF.REJECTED


def test_fit_circle_recovers_a_partly_occluded_circle_with_outliers():
    import iris_ladder_metrics_models as MM
    rng = np.random.default_rng(5)
    th = rng.uniform(-0.6, 0.6, 200)                 # only the lateral arcs are seen (eyelids cover the rest)
    th = np.concatenate([th, th + np.pi])
    x, y = 300 + 80 * np.cos(th) + rng.normal(0, 0.5, th.size), 200 + 80 * np.sin(th) + rng.normal(0, 0.5, th.size)
    x[:20] += 40                                     # outliers (eyelid points)
    cx, cy, r, kept = MM.fit_circle(x, y)
    assert abs(cx - 300) < 1.5 and abs(cy - 200) < 1.5 and abs(r - 80) < 1.5 and kept < 0.97


def test_iso_padding_round_trip():
    import iris_ladder_metrics_models as MM
    from PIL import Image
    im = Image.fromarray(np.zeros((576, 768), np.uint8))   # UPOL size, 4:3 exactly
    _, s, px, py = MM.to_iso(im)
    assert (px, py) == (0.0, 0.0) and s == pytest.approx(640 / 768)
    im = Image.fromarray(np.zeros((1360, 2048), np.uint8))  # UTIRIS visible: wider than 4:3, padded vertically
    iso, s, px, py = MM.to_iso(im)
    assert iso.size == (640, 480) and px == 0 and py > 0
    # a point in the original maps to ISO and back
    x, y = 1000.0, 700.0
    assert ((x + px) * s) / s - px == pytest.approx(x) and ((y + py) * s) / s - py == pytest.approx(y)


@pytest.mark.skipif(not os.path.isfile(r"C:\Projects\data\projects\identity-ladder-metric\iris-models-20261107\worldcoin\iris_semseg_upp_scse_mobilenetv2.onnx"),
                    reason="Worldcoin model not present")
def test_worldcoin_wrapper_runs_and_masks_padding():
    import iris_ladder_metrics_models as MM
    from PIL import Image
    wc = MM.Worldcoin()
    maps, s, px, py = wc.iso(Image.fromarray(np.full((1360, 2048), 120, np.uint8)))
    assert set(maps) == set(MM.WC_CLASSES)
    assert all(m.shape == (480, 640) for m in maps.values())
    assert np.isnan(maps["iris"][0, 0]) and np.isfinite(maps["iris"][240, 320])   # padding is missing, not zero


def test_p2_mask_removes_lid_shadow_but_keeps_isolated_dark_crypts():
    rng = np.random.default_rng(6)
    v = rng.normal(100, 8, (IF.H, IF.W))
    inside = np.ones_like(v, bool)
    mp = {"iris": np.ones_like(v), "pupil": np.zeros_like(v), "eyelashes": np.zeros_like(v)}
    mp["iris"][40:, 60:70] = 0.0             # predicted upper lid (outer radius)
    v[30:40, 55:75] = 30.0                   # dark lid shadow touching it, labelled iris by the model
    v[5:8, 200:203] = 30.0                   # isolated dark crypt near the pupil
    valid, refl, shadow, iris_vis = IF.p2_mask(v, mp, inside)
    assert shadow[30:40, 55:75].all() and not valid[30:40, 55:75].any()
    assert valid[5:8, 200:203].all()         # crypt is anatomy, kept
    assert not valid[40:, 60:70].any()       # lid itself not iris
    assert valid[:20, 120:180].mean() > 0.95


def test_twenty_six_confirmatory_metrics_and_cohort_unavailable():
    if not (os.path.isfile(FUNDUS) and os.path.isfile(BRAIN)):
        pytest.skip("shared metric modules not found")
    import iris_ladder_metrics as IL
    IL.load_shared(FUNDUS, BRAIN)
    c = IL.confirmatory()
    assert len(c) == 26
    assert c["acc_matched"][1] is None and c["amplitude_matched"][1] is None
    assert {k for k, v in c.items() if v[1]} == {"mse", "rmse", "psnr", "mae", "ssim", "ms_ssim", "acc_dataset",
                                                  "amplitude_dataset", "lddt", "intensity_persistence"}
