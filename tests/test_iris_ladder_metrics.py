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


# ---------------------------------------------------------------- donor curve and maps (iris_ladder_metrics_ladder)

def _toy_set(n_people=12, seed=0):
    """Toy strips: each person has an own texture; the repeat is the same texture plus small noise."""
    rng = np.random.default_rng(seed)
    H, W = IF.H, IF.W
    X, M, person, side, rep_img = [], [], [], [], []
    for p in range(n_people):
        for s in ("R", "L"):
            base = rng.normal(size=(H, W))
            X.append(base)
            X.append(base + 0.01 * rng.normal(size=(H, W)))
            M += [np.ones((H, W), bool)] * 2
    X, M = np.array(X), np.array(M)
    idx = np.arange(0, len(X), 2)
    person = np.repeat([f"p{p}" for p in range(n_people)], 2)
    side = np.tile(["R", "L"], n_people)
    fellow = np.arange(len(idx)) ^ 1
    import iris_ladder_metrics as IM
    valid = np.ones((H, W), bool)
    return {"X": X, "M": M, "XF": IF.mirror(X), "MF": IF.mirror(M), "idx": idx, "valid": valid, "person": person,
            "side": side, "repeat": idx + 1, "fellow": fellow, "pos": IM.positions(valid)[:4], "ratio": 0.45}


def test_local_map_ignores_missing_and_wraps_in_angle():
    import iris_ladder_metrics_ladder as LD
    x = np.full((IF.H, IF.W), 1.0)
    m = np.ones_like(x, bool)
    m[:, 100:110] = False
    x[:, 100:110] = 0.0                      # filler under the mask must not pull the mean down
    out = LD.local(x, m)
    assert np.allclose(out[np.isfinite(out)], 1.0)
    y = np.zeros((IF.H, IF.W))
    y[:, 0] = 1.0
    sm = LD.local(y, np.ones_like(y, bool))
    assert abs(sm[30, 2] - sm[30, IF.W - 2]) < 1e-9 and sm[30, IF.W - 2] > 0.01   # 0 deg seam is not an edge


def test_curve_stranger_about_two_repeat_near_zero():
    import iris_ladder_metrics_ladder as LD
    st = _toy_set()
    c = LD.summarise_curve(LD.curve_rows(st))
    assert abs(c["population"]["all"]["err_over_V"] - 1.0) < 1e-9
    assert 1.7 < c["population"]["1"]["err_over_V"] < 2.3
    assert c["other_photo"]["1"]["err_over_V"] < 0.01
    assert "16" not in c["population"]       # pool of 11 people: k = 16 is not drawn, not padded
    assert c["population"]["8"]["err_over_V"] < c["population"]["2"]["err_over_V"]


def test_maps_repeat_low_stranger_two():
    import iris_ladder_metrics_ladder as LD
    out, res = LD.maps(_toy_set())
    assert res["other_photo"]["median_ratio"] < 0.01
    assert 1.8 < res["stranger"]["median_ratio"] < 2.2


def test_lookalike_map_between_repeat_and_stranger():
    import iris_ladder_metrics_ladder as LD
    out, res = LD.maps(_toy_set(n_people=20))
    r = res["lookalike16"]["median_ratio"]
    assert r is not None and 0.9 < r < 1.3          # 16 unrelated textures: about 1 + 1/16 of the mean


# ---------------------------------------------------------------- Amendment P3 metrics (iris_ladder_metrics_extra)

BETTI = os.path.join(r"C:\Projects\data", "projects", "identity-ladder-metric", "models-20261107", "betti-matching-2d")


def test_impute_keeps_valid_pixels_and_fills_only_outside():
    import iris_ladder_metrics_extra as EX
    rng = np.random.default_rng(1)
    x = rng.normal(size=(IF.H, IF.W))
    m = np.ones_like(x, bool)
    m[10:20, 30:40] = False
    out = EX.impute(x, m)
    assert np.array_equal(out[m], x[m]) and np.isfinite(out).all()


def test_iriscode_hd_zero_for_identical_and_high_for_noise():
    import iris_ladder_metrics_extra as EX
    import iris_codes as IC
    rng = np.random.default_rng(2)
    x = ndimage.gaussian_filter(rng.normal(size=(IF.H, IF.W)), 2, mode="wrap")
    m = np.ones_like(x, bool)
    code, valid = IC.log_gabor_code(x)
    e = np.zeros_like(m)
    e[16:32, 96:112] = True
    assert EX.code_hd(code, valid, x.copy(), m, e, 16, 96) == 0.0
    comp = x.copy()
    comp[16:32, 96:112] = rng.normal(size=(16, 16))
    assert EX.code_hd(code, valid, comp, m, e, 16, 96) > 0.2


def test_betti_parts_zero_for_identical_window():
    if not os.path.isdir(BETTI):
        pytest.skip("Betti-matching code not found")
    sys.path.insert(0, BETTI)
    import iris_ladder_metrics_extra as EX
    rng = np.random.default_rng(3)
    a = EX.to_unit(rng.normal(size=(16, 16)))
    assert EX.betti_parts(a, a.copy()) == (0.0, 0.0)
    m, u = EX.betti_parts(a, EX.to_unit(rng.normal(size=(16, 16))))
    assert m + u > 0


def test_edge_margin_drops_samples_near_missing_and_pupil_end_and_wraps():
    """P2e: kept samples are more than 3 samples from any missing sample and from the pupil end (row 0 side), the
    limbus end is not an edge, the angle wraps, and a dark rim along a mask edge is removed while the z-score is
    redone inside the kept samples (Kate 8 Nov 2026: margin from the dark mask edge, as for the brain)."""
    m = np.ones((IF.H, IF.W), bool)
    m[:, 0:10] = False                                   # a lid gap at the angle seam
    k = IF.edge_margin(m, 3)
    assert not k[:3].any() and k[3:, 20].all()          # rows 0..2 dropped (row -1 counts as missing)
    assert k[-1, 20]                                      # limbus end kept
    assert not k[:, 10:13].any() and k[10:, 13].all()    # right of the gap
    assert not k[:, 253:].any() and k[10:, 252].all()    # left of the gap, through the wrap
    assert not (k & ~m).any()
    x = np.where(m, 0.0, np.nan)
    x[:, 10:12] = -5.0                                    # dark rim along the gap
    rng = np.random.default_rng(0)
    x = np.where(m, x + rng.normal(0, 1, x.shape), np.nan).astype(np.float32)
    tab = __import__("pandas").DataFrame({"ok": [True]})
    out = IF.with_edge_margin({"table": tab, "X": x[None], "M": m[None]}, 3)
    xm, mm = out["X"][0], out["M"][0]
    assert np.nanmin(xm) > -5.0 and np.isnan(xm[~mm]).all()
    assert abs(np.nanmean(xm)) < 1e-5 and abs(np.nanstd(xm) - 1) < 1e-4
    with pytest.raises(ValueError):
        IF.process("UPOL", None, "P2e")


def test_shards_cover_every_target_once_and_patches_round_trip(tmp_path):
    import iris_ladder_metrics_extra as IX
    n = 157
    parts = [IX.shard_targets(n, i, 6) for i in range(6)]
    assert sorted(np.concatenate(parts).tolist()) == list(range(n))
    with pytest.raises(ValueError):
        IX.parse_shard("6/6")
    rng = np.random.default_rng(0)
    pp = [("stranger", (3, 7), True, rng.random((16, 16))), ("repeat", (5, 0), False, rng.random((16, 16)))]
    IX.save_patches(tmp_path / "p.npz", pp)
    back = IX.load_patches(tmp_path / "p.npz")
    assert [(r, u, s) for r, u, s, _ in back] == [(r, u, s) for r, u, s, _ in pp]
    assert all(np.allclose(a, b[3], atol=1e-6) for (*_x, a), b in zip(back, pp))


def test_p3_aggregates_r_scale_tests_and_holm_on_toy_rows():
    import pandas as pd

    import iris_ladder_metrics_p3 as P3
    rows = []
    for p in range(12):
        for f, d in [("repeat", 0.2), ("shrink_repeat", 0.3), ("dataset_mean", 0.5), ("wrong_detail", 0.9),
                     ("noise", 1.2), ("stranger", 1.0)] + [(f"str{k:02d}", 1.0) for k in range(16)]:
            rows.append({"unit": p, "person": f"{p:03d}", "patch": 0, "fill": f, "m": d + 0.01 * p})
    r = pd.DataFrame(rows)
    persons = np.array(sorted(r.person.unique()))
    bidx = P3.boot_idx(len(persons), reps=50)
    R = P3.r_scale(r, "m", persons, bidx, "self")
    assert abs(R["stranger"]["R"] - 1.0) < 1e-9
    assert R["repeat"]["R"] < R["dataset_mean"]["R"] < 1
    t = P3.paired_tests(r, "m")
    assert t["T2"]["share_targets_pass"] == 1.0 and t["T2"]["p_one_sided"] < 0.01
    adj = P3.holm([0.01, 0.04, 1.0])
    assert np.allclose(adj, [0.03, 0.08, 1.0])
    assert "str00" not in P3.raw_means(r, "m", persons, bidx)


def test_window_is_explicit_and_10px_grid_keeps_validity_rules():
    import iris_ladder_metrics as IL

    old = (IL.WIN, IL.RING)
    try:
        with pytest.raises(SystemExit):
            IL.set_window(12)
        assert IL.set_window(10) == (10, 5)
        valid = np.ones((IL.H, IL.W), bool)
        valid[:, :40] = False
        pos = IL.positions(valid)
        assert pos and all(y % 10 == 0 and x % 10 == 0 for y, x in pos)
        assert all(valid[y:y + 10, x:x + 10].mean() >= IL.MIN_POS_VALID for y, x in pos)
        ring = IL.ring_mask(20, 100)
        assert ring.sum() == (10 + 2 * 5) ** 2 - 100
    finally:
        IL.WIN, IL.RING = old


def test_window_comparison_flags_order_changes_beyond_tolerance():
    import iris_ladder_metrics_window as WC

    R16 = {"mse": {"repeat": {"R": 0.2}, "lookalike": {"R": 0.5}, "stranger": {"R": 1.0}, "dataset_mean": {"R": 0.5}}}
    R10 = {"mse": {"repeat": {"R": 0.25, "ci95": [0.2, 0.3]}, "lookalike": {"R": 0.55, "ci95": [0.5, 0.6]},
                   "stranger": {"R": 1.0, "ci95": [1, 1]}, "dataset_mean": {"R": 1.3, "ci95": [1.2, 1.4]}}}
    c = WC.compare(R16, R10)["mse"]
    assert [x["pair"] for x in c["order_changes_beyond_0.1"]] == ["dataset_mean vs stranger"]
    assert c["spearman_rung_order"] > 0.5
