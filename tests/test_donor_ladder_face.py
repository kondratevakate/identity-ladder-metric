"""Known-geometry tests for the face donor ladder (docs/PROTOCOL_donor_ladder_face.md)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import donor_ladder_face as f  # noqa: E402

f.SUF = "_v6"  # tests exercise the current version unless they set another one


def test_similarity_recovers_known_transform():
    s, th, t = 1.7, np.deg2rad(12), np.array([5.0, -3.0])
    R = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    src = np.array([[10.0, 20.0], [50.0, 22.0], [30.0, 60.0]])
    M = f.similarity(src, src @ R.T + t)
    assert np.allclose(M[:, :2], R) and np.allclose(M[:, 2], t)


def test_canonical_eyes_are_mirror_images():
    pts = np.array([f.EYE_L, f.EYE_R, f.MOUTH])
    img = np.zeros((f.SIZE, f.SIZE))
    for x, y in pts[:2].astype(int):
        img[y, x] = 1
    m = f.mirror(img)
    assert m[int(f.EYE_L[1]), int(f.EYE_R[0]) - 1] == 1 or m[int(f.EYE_L[1]), int(f.EYE_L[0])] == 1
    assert np.array_equal(f.mirror(f.mirror(img)), img)
    assert f.EYE_L[0] + f.EYE_R[0] == 2 * f.MOUTH[0]  # eyes symmetric about the mouth's vertical line


def test_box_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 20))
    w, p = 4, 2
    pad = np.pad(x, p)
    brute = np.array([[pad[i:i + w, j:j + w].mean() for j in range(20)] for i in range(20)])
    assert np.allclose(f.box(x, w), brute)


def test_local_error_zero_for_identical_and_two_for_strangers():
    rng = np.random.default_rng(1)
    mask = np.ones((32, 32), bool)
    a = rng.normal(size=(32, 32))
    e0, _ = f.local_error(a, a, mask)
    assert np.allclose(e0, 0)
    # population mean 0, unit variance: stranger error / mean error -> 2
    tg = rng.normal(size=(400, 32, 32))
    st = rng.normal(size=(400, 32, 32))
    errs = np.array([f.local_error(t, s, mask)[0] for t, s in zip(tg, st)])
    verr = np.array([f.local_error(t, np.zeros_like(t), mask)[0] for t in tg])
    r = f.ratio_map(errs, None, verr)
    assert abs(np.nanmean(r) - 2) < 0.05


def test_lookalike_fill_reproduces_target_when_pool_holds_copies():
    rng = np.random.default_rng(2)
    from scipy.ndimage import gaussian_filter
    t = gaussian_filter(rng.normal(size=(f.SIZE, f.SIZE)), 3).astype(np.float32)
    others = gaussian_filter(rng.normal(size=(40, f.SIZE, f.SIZE)), (0, 3, 3)).astype(np.float32) * 5
    pool = np.concatenate([np.repeat(t[None], f.K_LOOK, 0), others])
    valid = np.zeros((f.SIZE, f.SIZE), bool)
    valid[20:108, 20:108] = True
    F = f.lookalike_fill(t[None], pool, valid)[0]
    ok = np.isfinite(F)
    assert ok[40:88, 40:88].all() and np.allclose(F[ok], t[ok], atol=1e-4)


def test_person_outliers_drops_the_foreign_image():
    rng = np.random.default_rng(3)
    base = rng.normal(size=512)
    own = base + 0.6 * rng.normal(size=(6, 512))
    other = rng.normal(size=(1, 512))
    E = np.vstack([own, other])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    assert f.person_outliers(E) == [6]
    two = np.vstack([own[0], other[0]])
    two /= np.linalg.norm(two, axis=1, keepdims=True)
    assert f.person_outliers(two) == [0, 1]


def test_masked_lookalike_fill_ignores_pool_holes():
    """Amendment 7 erratum: holes in pool masks (stored as 0) must not enter the fill or pull it towards 0."""
    rng = np.random.default_rng(1)
    S = f.SIZE
    target = rng.normal(size=(S, S)).astype(np.float32)
    copies = np.repeat(target[None], 16, 0)
    Mc = rng.random((16, S, S)) > 0.2  # each copy misses 20% of pixels
    others = rng.normal(size=(40, S, S)).astype(np.float32)
    Xp = np.concatenate([copies * Mc, others]).astype(np.float32)
    Mp = np.concatenate([Mc, np.ones((40, S, S), bool)])
    valid = np.ones((S, S), bool)
    fill = f.lookalike_fill(target[None], Xp, valid, Mt=np.ones((1, S, S), bool), Mp=Mp)[0]
    ok = np.isfinite(fill)
    assert ok.mean() > 0.9
    assert np.abs(fill[ok] - target[ok]).max() < 1e-4
    # the unmasked call reproduces the bug: it shrinks towards 0
    old = f.lookalike_fill(target[None], Xp, valid)[0]
    o = np.isfinite(old)
    assert np.abs(old[o]).mean() < 0.9 * np.abs(target[o]).mean()


def test_lens_mask_is_mirror_symmetric():
    S = f.SIZE
    tmpl = np.zeros((478, 2))
    tmpl[f.IRIS_A] = (44, 52)
    tmpl[f.IRIS_B] = (84, 52)
    m = f.lens_mask(tmpl)
    assert m.sum() > 0
    assert (m == m[:, ::-1]).all()


def test_colour_flags_mark_a_dark_patch_not_skin():
    S = f.SIZE
    ref = np.array([60, 15, 15], np.float32)
    lab = np.repeat(ref[:, None, None], S, 1).repeat(S, 2) + np.random.default_rng(0).normal(0, 1, (3, S, S)).astype(np.float32)
    lab[:, 80:100, 50:80] = np.array([25, 2, 2], np.float32)[:, None, None]  # beard-like patch
    d = f.colour_distance(lab, ref)
    flags = f.colour_flags(d, np.ones((S, S), bool), thr=10)
    assert flags[82:98, 52:78].all()
    assert flags[:60].sum() == 0


def test_local_error_from_v3_is_gaussian_not_box():
    """A point error must spread as a smooth bump (Amendment 4.7), not as a flat 16 px square; scripts that import
    the module must set f.SUF, because the window depends on it."""
    S = f.SIZE
    a = np.zeros((S, S), np.float32)
    b = a.copy()
    b[64, 64] = 1.0
    old = f.SUF
    try:
        f.SUF = "_v6"
        e, _ = f.local_error(a, b, np.ones((S, S), bool))
        row = e[64, 50:79]
        assert np.argmax(row) == 14 and row[14] > 2 * row[6]  # peaked, not a plateau
        f.SUF = ""
        e0, _ = f.local_error(a, b, np.ones((S, S), bool))
        assert np.isclose(e0[60, 60], e0[64, 64])  # the v1 box: flat inside the square
    finally:
        f.SUF = old


def test_version_dependent_code_refuses_an_unset_version():
    """Regression (6 Nov 2026): QC scripts imported the module without setting SUF and silently got the v1 box window
    on v6 data. An unset version must raise, not fall back."""
    S = f.SIZE
    old = f.SUF
    try:
        f.SUF = None
        try:
            f.local_error(np.zeros((S, S)), np.zeros((S, S)), np.ones((S, S), bool))
        except RuntimeError:
            pass
        else:
            raise AssertionError("local_error ran with an unset version")
    finally:
        f.SUF = old


def test_beard_features_are_per_image_and_sized():
    """Amendment 7c: features of one image do not depend on any other image (2D filters only)."""
    rng = np.random.default_rng(3)
    S = f.SIZE
    x = rng.normal(size=(S, S)).astype(np.float32)
    m = np.ones((S, S), bool)
    a = f.beard_features(x, m)
    assert a.shape == (24 * 40 + 6 * 10,)
    assert np.allclose(a, f.beard_features(x.copy(), m))


def test_landmark_residual_ignores_affine_and_sees_shape_change():
    """Amendment 8.1: the fit-failure QC uses geometry only; any affine copy of the template has residual 0."""
    rng = np.random.default_rng(5)
    tmpl = rng.uniform(20, 108, size=(478, 2))
    A = np.array([[1.3, 0.2, 5.0], [-0.1, 0.9, -7.0]])
    moved = tmpl @ A[:, :2].T + A[:, 2]
    assert f.landmark_residual(moved, tmpl) < 1e-6
    bent = moved.copy()
    bent[:100] += rng.normal(0, 5, size=(100, 2))
    assert f.landmark_residual(bent, tmpl) > 0.01


def test_mask_rules_are_an_explicit_table_not_string_order():
    # '_v6f' sorts after '_v6c'; it must get the final masks (Amendment 10), not the withdrawn beard detector
    assert f.mask_rules("_v5") is None
    assert f.mask_rules("_v6") == {"symmetric": False, "beard": False, "glasses_only": False}
    assert f.mask_rules("_v6b")["symmetric"] and not f.mask_rules("_v6b")["beard"]
    assert f.mask_rules("_v6c")["beard"]
    for suf in ("_v6f", "_v7", "_v7_p15"):
        assert f.mask_rules(suf) == {"symmetric": False, "beard": False, "glasses_only": True}


def test_qc_status_is_approved_only_for_listed_versions():
    old = f.SUF
    try:
        f.SUF = "_v6f"
        assert f.qc_status().startswith("visual QC approved")
        f.SUF = "_v6"
        assert f.qc_status().startswith("pre-QC")
    finally:
        f.SUF = old
