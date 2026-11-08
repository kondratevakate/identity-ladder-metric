"""Known-geometry tests for the face metric audit (docs/PROTOCOL_face_metrics.md)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import face_ladder_metrics as FM  # noqa: E402
import donor_ladder_brain_metrics as BM  # noqa: E402

S = FM.SIZE


def test_positions_full_valid_drops_only_corners():
    # corner windows keep 320 of 768 nominal ring pixels (< 50%); edges keep 512 (>= 50%)
    pos = FM.positions(np.ones((S, S), bool))
    assert len(pos) == 64 - 4
    assert (0, 0) not in pos and (0, 16) in pos and (112, 112) not in pos


def test_positions_need_95_percent_valid():
    v = np.ones((S, S), bool)
    v[48:64, 48:52] = False                          # 25% of one window missing
    assert (48, 48) not in FM.positions(v)
    v = np.ones((S, S), bool)
    v[48, 48:60] = False                             # 12 of 256 pixels missing (4.7%)
    assert (48, 48) in FM.positions(v)


def test_mirror_flips_and_drops_midline_strip():
    x = np.tile(np.arange(S, dtype=np.float32), (S, 1))
    f, m = FM.mirror_fill(x, np.ones((S, S), bool))
    assert f[0, 0] == S - 1 and f[0, S - 1] == 0
    bad = np.flatnonzero(~m[0])
    assert list(bad) == [62, 63, 64, 65]             # |x - 63.5| < 2.5


def test_nearest_ignores_filler_outside_masks():
    rng = np.random.default_rng(0)
    b = rng.normal(size=200)
    mb = np.ones(200, bool)
    good = b + rng.normal(scale=0.9, size=200)       # moderately close inside its whole mask
    filler = b.copy()                                # identical values outside its mask ...
    mf = np.zeros(200, bool)
    mf[:110] = True                                  # ... 55% inside its mask, and there it is off by 1
    filler[:110] = b[:110] + 1.0
    A, MA = np.stack([good, filler]), np.stack([np.ones(200, bool), mf])
    assert FM.nearest(A, MA, b, mb, np.ones(2, bool)) == 0
    # if the masks were ignored the filler would win
    assert np.mean((filler - b) ** 2) < np.mean((good - b) ** 2)
    assert FM.nearest(A, np.ones_like(MA), b, mb, np.ones(2, bool)) == 1


def test_nearest_requires_half_of_target_pixels_shared():
    b, mb = np.zeros(100), np.ones(100, bool)
    A = np.zeros((1, 100))
    MA = np.zeros((1, 100), bool)
    MA[0, :49] = True
    assert FM.nearest(A, MA, b, mb, np.ones(1, bool)) is None
    MA[0, :50] = True
    assert FM.nearest(A, MA, b, mb, np.ones(1, bool)) == 0


def test_score_2d_as_singleton_3d():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 16, 16)).astype(np.float32)
    e = np.ones((1, 16, 16), bool)
    mu = np.zeros_like(x)
    s = BM.score(x, x.copy(), e, mu, None, 1.0)
    assert s["mse"] == 0 and s["ssim"] == pytest.approx(1.0) and s["lddt"] == pytest.approx(1.0)
    f = x + 0.5
    s = BM.score(x, f, e, mu, None, 1.0)
    assert s["mse"] == pytest.approx(0.25, rel=1e-5) and s["mae"] == pytest.approx(0.5, rel=1e-5)
    assert s["lddt"] == pytest.approx(1.0)           # lDDT ignores a constant offset
    # the singleton axis adds no pairs: lDDT pairs counted equal the 2D count
    n2d = sum(((np.abs(np.array(o[1:])) ** 2).sum() <= 4) for o in BM.OFFSETS if o[0] == 0)
    assert n2d == len([o for o in BM.OFFSETS if o[0] == 0])


def test_score_ignores_values_outside_e():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 16, 16)).astype(np.float32)
    f = x + rng.normal(scale=0.5, size=x.shape).astype(np.float32)
    e = np.ones_like(x, bool)
    e[0, :, :6] = False
    mu = np.zeros_like(x)
    a = BM.score(x, f, e, mu, None, 0.7)
    g = f.copy()
    g[~e] = 1e6
    b = BM.score(x, g, e, mu, None, 0.7)
    for k in BM.METRICS:
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12, nan_ok=True), k


def _fake_cache(tmp_path, n_pool=30, filler=True):
    """One target, its repeat, a pool. Pool person 0 stores the target's exact values outside its (small) mask
    (would win any search that counts them); pool person 1 is a true near match inside a full mask."""
    rng = np.random.default_rng(3)
    xt = rng.normal(size=(S, S)).astype(np.float32)
    pool = rng.normal(size=(n_pool, S, S)).astype(np.float32)
    pm = np.ones((n_pool, S, S), bool)
    pool[1] = xt + rng.normal(scale=0.4, size=(S, S))
    if filler:
        pool[0] = xt
        pm[0] = False
        pm[0][:, :40] = True                           # mask covers only the left third
        pool[0][:, :40] = xt[:, :40] + 4.0             # far from the target where valid
    rep = xt + rng.normal(scale=0.2, size=(S, S)).astype(np.float32)
    X = np.concatenate([pool, xt[None], rep[None]]).astype(np.float16)
    M = np.concatenate([pm, np.ones((2, S, S), bool)])
    d = str(tmp_path)
    np.save(os.path.join(d, "X.npy"), X)
    np.save(os.path.join(d, "M.npy"), M)
    np.save(os.path.join(d, "emb.npy"), np.full((len(X), 512), np.nan, np.float32))
    mean, ok = BM.masked_mean(X[:n_pool].astype(np.float32), M[:n_pool])
    np.save(os.path.join(d, "celeba_mean.npy"), mean)
    np.save(os.path.join(d, "celeba_mean_ok.npy"), ok)
    np.save(os.path.join(d, "celeba_valid.npy"), np.ones((S, S), bool))
    np.save(os.path.join(d, "celeba_cells_m.npy"), mean[None])
    np.save(os.path.join(d, "celeba_cells_ok.npy"), ok[None])
    tgt = {"t": n_pool, "rep": n_pool + 1, "sib": None, "family": 0, "cell": 0}
    return d, tgt, n_pool


def test_pipeline_lookalike_not_won_by_filler(tmp_path):
    d, tgt, n_pool = _fake_cache(tmp_path)
    FM.WORK.clear()
    FM._init(d)
    lam = {"repeat": 0.5, "mirror": 0.5}
    for yx in ((48, 64), (64, 96)):                   # windows where pool person 0 is masked out
        rows = FM.score_position(("celeba", 0, yx, [(0, tgt)], lam, {0: 0.3}, (0, n_pool), 10, 1.5, FM.WIN, FM.RING))
        by = {r["fill"]: r for r in rows}
        # the oracle and the ring look-alike pick the true near match (mse about 0.16), never the filler
        assert by["lookalike_oracle"]["mse"] == pytest.approx(0.16, rel=0.35)
        assert by["lookalike"]["mse"] == pytest.approx(0.16, rel=0.35)
        strangers = np.mean([r["mse"] for r in rows if BM.is_reference_stranger(r["fill"])])
        assert by["repeat"]["mse"] < by["lookalike"]["mse"] < strangers
        assert by["dataset_mean"]["mse"] < strangers
        assert by["noise"]["mse"] > by["dataset_mean"]["mse"]


def test_mean_of_two_is_masked(tmp_path):
    d, tgt, n_pool = _fake_cache(tmp_path, filler=False)
    FM.WORK.clear()
    FM._init(d)
    strs, _ = FM.donors("celeba", 0, tgt, n_pool, 10)
    X, M = np.load(os.path.join(d, "X.npy")), np.load(os.path.join(d, "M.npy"))
    M[strs[1]][:, :] = False
    M[strs[1]][:16, :16] = True
    FM.WORK.clear()                                  # release the memory maps before rewriting (Windows)
    import gc
    gc.collect()
    np.save(os.path.join(d, "M.npy"), M)
    FM._init(d)
    sl = (slice(16, 32), slice(16, 32))
    P = FM.pool_window("celeba", {"pool": (0, n_pool)}, sl, FM.ring_mask(16, 16))
    P["shift_px"] = 1.5
    _, _, fills = FM.window_fills("celeba", 0, tgt, 0, sl, FM.ring_mask(16, 16), {"repeat": .5, "mirror": .5}, P, 10)
    m2 = fills["mean_k2"][0]
    np.testing.assert_allclose(m2, X[strs[0]][sl].astype(np.float32), rtol=1e-6)   # only member 0 valid here


def test_npz_rows_streams_selected_rows(tmp_path):
    a = np.arange(5 * 3 * 2, dtype=np.float16).reshape(5, 3, 2)
    p = os.path.join(tmp_path, "z.npz")
    np.savez_compressed(p, X=a)
    np.testing.assert_array_equal(FM.npz_rows(p, "X", [1, 4]), a[[1, 4]])


def test_version_is_required():
    with pytest.raises(SystemExit):
        sys.argv = ["face_ladder_metrics.py"]
        FM.main()


@pytest.mark.skipif(not os.path.exists(FM.FACE_SRC), reason="face pipeline source not available")
def test_qc_status_follows_face_version():
    # Regression: the summary status was a hard-coded "pre-QC" string and stayed wrong after Kate approved v6f.
    assert "approved" in FM.load_face("_v6f").qc_status()
    assert "pre-QC" in FM.load_face("_v6").qc_status()


def test_shift_fill_integer_shift_exact_and_no_bleed():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(S, S)).astype(np.float32)
    m = np.zeros((S, S), bool)
    m[20:100, 30:90] = True
    f, ok = FM.shift_fill(x, m, 2, -3)
    np.testing.assert_allclose(f[22:102, 27:87][ok[22:102, 27:87]], x[20:100, 30:90][ok[22:102, 27:87]], atol=1e-5)
    assert ok.sum() == m.sum()
    f, ok = FM.shift_fill(x, m, 0.5, 0.0)            # half pixel: border rows touching the mask edge are invalid
    assert ok.sum() == m.sum() - 60
    assert np.all(np.abs(f[ok]) < 10)


def test_light_fill_transfers_low_frequency_difference_only_on_joint_mask():
    rng = np.random.default_rng(5)
    xt = rng.normal(size=(S, S)).astype(np.float32)
    mt = np.ones((S, S), bool)
    mr = np.ones((S, S), bool)
    mr[:, :20] = False
    xr = xt + 0.7                                   # retest = target + a constant light offset
    xr[:, :20] = 99.0                               # stored filler outside the retest mask never enters
    f, ok = FM.light_fill(xt, mt, xr, mr)
    assert not ok[:, :20].any()
    np.testing.assert_allclose(f[ok] - xt[ok], 0.7, atol=1e-5)


def test_self_scale_zero_for_copy_and_one_for_strangers():
    import pandas as pd
    rows = []
    rng = np.random.default_rng(6)
    for t in range(30):
        for pk in range(3):
            base = {"src": "celeba", "target": t, "family": t, "patch": pk}
            for j in range(16):
                rows.append({**base, "fill": f"str{j:02d}", "mse": 2.0 + rng.normal(scale=0.1), "ssim": 0.2})
            rows.append({**base, "fill": "repeat", "mse": 1.3, "ssim": 0.5})
            rows.append({**base, "fill": "dataset_mean", "mse": 1.0, "ssim": 0.6})
            rows.append({**base, "fill": "copy", "mse": 0.0, "ssim": 1.0})
    r = pd.DataFrame(rows)
    rs = FM.with_self(r, ["mse", "ssim"])
    per = rs.groupby(["target", "family", "fill"])[["mse", "ssim"]].mean().reset_index()
    per2 = rs.groupby(["target", "family", "patch", "fill"])[["mse", "ssim"]].mean().reset_index()
    for m in ("mse", "ssim"):
        R, _ = FM.ladder(per, per2, m)
        R = FM.rename_back(R)
        assert R["copy"]["R"] == pytest.approx(0.0, abs=1e-9)
        assert "retest" in R and "self" not in R or R["self"]["R"] == pytest.approx(0.0)
    R = FM.rename_back(FM.ladder(per, per2, "mse")[0])
    assert R["dataset_mean"]["R"] == pytest.approx(0.5, abs=0.02)
    assert R["retest"]["R"] == pytest.approx(0.65, abs=0.02)
    assert np.isnan(FM.self_value("psnr"))
