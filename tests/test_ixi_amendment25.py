"""Amendment 25 (IXI LPIPS, persistence, FID/FWD and Gaussian maps): missing is never zero, the set-level cells are
paired and exclude reference strangers, the self anchor is exact, and the torchvision guard leaves a normal import
alone."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import donor_ladder_brain_ixi_gen as XG  # noqa: E402
import donor_ladder_brain_maps_ixi as XM  # noqa: E402
import donor_ladder_brain_metrics_t23 as T  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_hook_registered_and_whole_slices_flag_slices_outside_e():
    assert T.EXTRAS["ixi_gen"] is T.extra_ixi_gen
    x = np.arange(16 ** 3, dtype=np.float32).reshape(16, 16, 16)
    e = np.ones_like(x, bool)
    e[8, 0, 0] = False                                  # in the first central slice (axis 0) and nowhere else central
    xs, fs, ok = T.whole_slices(x, x + 1, e)
    assert xs.shape == fs.shape == (3, 16, 16)
    assert ok.tolist() == [False, True, True]
    assert np.array_equal(xs[1], x[:, 8]) and np.array_equal(fs[2], x[:, :, 8] + 1)
    with pytest.raises(ValueError):
        T.whole_slices(np.zeros((16, 16, 8)), np.zeros((16, 16, 8)), np.ones((16, 16, 8), bool))


def test_lpips_is_missing_not_zero_when_no_slice_lies_in_e():
    x = np.random.default_rng(0).normal(size=(16, 16, 16))
    e = np.ones_like(x, bool)
    e[8, 8, 8] = False                                  # the shared centre voxel: all three central slices leave E
    v, n = T.lpips_25d(x, x, e, net="alex")
    assert n == 0 and np.isnan(v)


def test_set_cells_skip_reference_strangers_and_invalid_axes():
    r = pd.DataFrame({"target": [0, 0, 0, 1], "patch": [3, 3, 3, 3], "side": ["L", "L", "L", "R"],
                      "fill": ["mirror", "str00", "noise", "mirror"], "crop": [0, 1, 2, 3]})
    ok = np.array([[True, True, True], [True, True, True], [True, False, True], [False, False, False]])
    cl = XG.set_cells(r, ok)
    assert {c[6] for c in cl} == {"mirror", "noise"}
    assert sorted((c[6], c[2]) for c in cl) == [("mirror", 0), ("mirror", 1), ("mirror", 2), ("noise", 0), ("noise", 2)]
    tk = XG.target_samples(cl)                          # one target sample per (target, cube, side, axis)
    assert sorted(tk) == [(0, 3, "L", 0), (0, 3, "L", 1), (0, 3, "L", 2)]


def test_self_anchor_is_the_exact_zero_distance():
    rows = []
    for t in range(12):
        rows += [{"target": t, "fill": "mirror", "lpips": 0.2 + 0.01 * t},
                 {"target": t, "fill": "dataset_mean", "lpips": 0.5}]
        rows += [{"target": t, "fill": f"str{j:02d}", "lpips": 0.4} for j in range(16)]
    per = pd.DataFrame(rows).assign(family=lambda d: d.target)
    R = XG.self_R(per, "lpips")
    mean_mirror = np.mean([0.2 + 0.01 * t for t in range(12)])
    assert R["mirror"]["R"] == pytest.approx(mean_mirror / 0.4)
    assert R["dataset_mean"]["R"] == pytest.approx(1.25)
    assert "repeat" not in R and R["_anchors"]["d_self"] == 0.0
    with pytest.raises(ValueError):
        XG.self_R(per, "mse")


def test_loo_pool_ignores_voxels_outside_each_donors_mask():
    """Missing is never zero (maps): a donor's out-of-mask voxel (stored as 0 in the z cache) never enters the pool."""
    n, shp = 15, (4,)
    V = np.full((n,) + shp, 2.0)
    Mk = np.ones((n,) + shp, np.uint8)
    V[1:4, 0], Mk[1:4, 0] = 0.0, 0                      # three donors miss voxel 0
    s, q, c = XM.pool_sums(V, Mk, range(n))
    mu, var, ok = XM.loo(s, q, c, V[0], Mk[0] > 0)
    assert mu[0] == pytest.approx(2.0) and var[0] == pytest.approx(0.0) and ok[0]
    assert c[0] == n - 3
    Mk[1:6, 1] = 0                                      # 10 scans cover voxel 1: leave-one-out 9 < MIN_DONORS
    s, q, c = XM.pool_sums(V, Mk, range(n))
    mu, _, ok = XM.loo(s, q, c, V[0], Mk[0] > 0)
    assert XM.MIN_DONORS == 10 and not ok[1] and np.isnan(mu[1])


def test_grid_rows_masked_means_and_shell_distance():
    rng = np.random.default_rng(1)
    n, k = 20, 64
    X = rng.normal(size=(n, k)).astype(np.float32)
    XM_ = np.ones((n, k), bool)
    X2, M2 = X.copy(), XM_.copy()
    X2[5, :10], M2[5, :10] = 0.0, False                 # donor 5 misses 10 voxels (zero-filled in the cache)
    w = np.full(k, 1.0 / k)
    sh = rng.normal(size=(n, 30))
    shm = np.ones((n, 30), bool)
    args = (np.zeros((n, k), np.float32), np.zeros((n, k), bool), sh, shm, np.ones(n, bool), np.zeros(n, bool), False,
            w, np.ones(n, bool))
    a = {i: r for i, _, r in XM.grid_rows(X2, M2, *args)}
    # recompute target 0's stranger error by hand with the masked mean
    others = np.arange(1, n)
    mu = np.array([X2[others][M2[others][:, v], v].mean() for v in range(k)])
    var = np.array([X2[others][M2[others][:, v], v].var() for v in range(k)])
    assert a[0]["stranger"] == pytest.approx(float(((X[0] - mu) ** 2 + var).mean()), rel=1e-5)
    d = XM.shell_dist(np.array([[1.0, 5.0], [1.0, 0.0]]), np.array([[True, True], [True, False]]), 0, np.array([1]))
    assert d[0] == 0.0                                  # the donor's missing shell voxel does not count as 0


def test_wmean_renormalises_over_valid_voxels():
    w = np.array([0.25, 0.25, 0.5])
    assert XM.wmean(np.array([1.0, 3.0, 99.0]), np.array([True, True, False]), w) == pytest.approx(2.0)
    assert np.isnan(XM.wmean(np.ones(3), np.zeros(3, bool), w))


def test_torchvision_guard_imports():
    import torchvision_guard
    tv = torchvision_guard.ensure()
    assert hasattr(tv, "models")


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "results", "donor_ladder_brain_ixi_gen.json")),
                    reason="Amendment 25 not summarised yet")
def test_public_json_is_aggregate_only_and_complete():
    import json
    d = json.load(open(os.path.join(ROOT, "results", "donor_ladder_brain_ixi_gen.json"), encoding="utf-8"))
    txt = json.dumps(d)
    assert "IXI0" not in txt and "_brain.nii" not in txt  # no scan keys or paths
    v = d["views"]["random_patches/LR"]
    for m in ("lpips", "lpips_vgg", "persistence_w1"):
        assert v["ladder_R"][m]["stranger"]["R"] is not None and v["ladder_R_self"][m]["mirror"]["R"] is not None
    sl = d["set_level"]["random_patches/LR"]
    assert sl["stranger"]["fid"] is not None and "excluded_from_ranks" in sl["mirror"]


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "results", "cross_modality", "generative_status.json")),
                    reason="generative block not run")
def test_generative_block_has_ixi_for_every_amendment25_metric():
    import json
    st = json.load(open(os.path.join(ROOT, "results", "cross_modality", "generative_status.json"), encoding="utf-8"))
    if not os.path.exists(os.path.join(ROOT, "results", "donor_ladder_brain_ixi_gen.json")):
        pytest.skip("Amendment 25 not summarised yet")
    assert {"lpips_alex", "lpips_vgg", "persistence", "fid", "fwd"} <= set(st["present"]["IXI"])
    assert "IXI" not in st["not_run"]


def test_grid_cache_resumes_a_stopped_build_without_touching_verified_scans(tmp_path):
    """A build stopped part way (no progress file, no grid.done) resumes at the first scan that is not verifiably
    written; earlier scans are kept, the broken one is rewritten, the result equals a fresh build."""
    rng = np.random.default_rng(3)
    n, s = 4, 40
    V = rng.normal(size=(n, s, s, s)).astype(np.float32)
    M = np.ones((n, s, s, s), np.uint8)
    M[:, :4] = 0
    aff = np.diag([-1.0, 1.0, 1.0, 1.0])
    aff[0, 3] = (s - 1) / 2
    cs = np.array([[20, 20, 20], [16, 22, 18], [24, 18, 21]])
    (tmp_path / "ref").mkdir()
    (tmp_path / "stopped").mkdir()
    ref = {k: np.array(a) for k, a in XM.grid_cache(list(range(n)), aff, cs, V, M, str(tmp_path / "ref")).items()}
    d = tmp_path / "stopped"
    first = XM.grid_cache(list(range(n)), aff, cs, V, M, str(d))
    del first
    os.remove(d / "grid.done")
    os.remove(d / "grid.progress")
    w = np.load(d / "grid_win.npy", mmap_mode="r+")
    w[:, 2] = 0                                         # scan 2 was being written when the run stopped
    w.flush()
    del w
    out = XM.grid_cache(list(range(n)), aff, cs, V, M, str(d))
    assert (d / "grid.done").exists() and open(d / "grid.progress").read() == str(n)
    for k in ref:
        assert np.array_equal(np.asarray(out[k]), ref[k]), k
