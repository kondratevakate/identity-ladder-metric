"""Known-geometry tests for the Amendment F2 face panel metrics (no real data)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import face_ladder_metrics_panel as P  # noqa: E402

BETTI = os.environ.get("BETTI_SRC")


def _img(seed, n=16):
    return np.random.default_rng(seed).normal(size=(n, n))


def test_registry_has_26_and_blocked_reasons():
    assert len(P.REGISTRY) == 26
    blocked = [k for k, (_, c, why) in P.REGISTRY.items() if why]
    assert set(blocked) == {"id_pfe", "latent_pullback", "latent_fisher_rao", "latent_lddmm"}
    assert all(c is None for k, (_, c, why) in P.REGISTRY.items() if why)


def test_haar_packets_orthonormal_and_shape():
    a = np.random.default_rng(0).normal(size=(5, 16, 16))
    p = P.haar_packets(a, 2)
    assert p.shape == (5, 16, 4, 4)
    assert np.allclose((p ** 2).sum((1, 2, 3)), (a ** 2).sum((1, 2)))
    c = np.ones((1, 16, 16))
    q = P.haar_packets(c, 2)
    assert np.allclose(q[0, 0], 4.0) and np.allclose(q[0, 1:], 0.0)   # constant image: only the LL-LL packet


def test_fwd_and_frechet_zero_on_identical_sets():
    A = np.random.default_rng(1).uniform(size=(200, 16, 16))
    assert abs(P.fwd(A, A)) < 1e-6
    B = np.clip(A + 0.3, 0, 1)
    assert P.fwd(A, B) > 0.01


def test_topology_distance_zero_on_identical_and_positive_on_scaled():
    F = np.random.default_rng(2).normal(size=(100, 8))
    assert P.topology_distance(F, F) == pytest.approx(0.0)
    assert P.topology_distance(F, 2 * F) > 0.1
    d = P.mst_deaths(F)
    assert len(d) == 99 and np.all(np.diff(d) >= 0)


def test_gauss_nll_marginal_ignores_missing_pixels():
    rng = np.random.default_rng(3)
    mean, cov = np.zeros(256), np.eye(256)
    v, ok = rng.normal(size=(16, 16)), np.ones((16, 16), bool)
    ok[:4] = False
    a = P.gauss_nll(v, ok, mean, cov)
    v2 = v.copy()
    v2[:4] = 1e6                                     # a filler outside the scored pixels changes nothing
    assert P.gauss_nll(v2, ok, mean, cov) == pytest.approx(a)
    assert a == pytest.approx(0.5 * np.mean(v[ok] ** 2) + 0.5 * np.log(2 * np.pi))


def test_ampdev_zero_for_identical_and_filler_outside_e():
    x, mu = _img(4), _img(5)
    e = np.ones((16, 16), bool)
    e[:, :3] = False
    assert P.ampdev(x, x, mu, e) == pytest.approx(0.0)
    f = x.copy()
    f[:, :3] = 99.0
    assert P.ampdev(x, f, mu, e) == pytest.approx(0.0)


def test_plda_llr_prefers_same_person():
    rng = np.random.default_rng(6)
    d, n_p, n_i = 6, 300, 3
    y = rng.normal(scale=2.0, size=(n_p, d))
    Z = np.repeat(y, n_i, 0) + rng.normal(scale=0.5, size=(n_p * n_i, d))
    person = np.repeat(np.arange(n_p), n_i)
    B, W, n_multi, n_people = P.plda_fit(Z, person)
    assert n_multi == n_p and np.allclose(np.diag(W), 0.25, atol=0.05)
    S = P.plda_scorer(B, W)
    same = np.mean([P.plda_llr(S, Z[3 * k], Z[3 * k + 1]) for k in range(50)])
    diff = np.mean([P.plda_llr(S, Z[3 * k], Z[3 * (k + 1)]) for k in range(50)])
    assert same > 0 > diff
    assert np.isnan(P.plda_llr(S, np.full(d, np.nan), Z[0]))


@pytest.mark.skipif(not BETTI, reason="BETTI_SRC not set")
def test_betti_parts_identical_zero_sum_equals_loss_and_ignores_outside_e():
    sys.path.insert(0, BETTI)
    import BettiMatching as B
    x, f = _img(7) * 2, _img(8) * 2
    e = np.ones((16, 16), bool)
    e[:2] = False
    m0, u0, _ = P.betti_parts(B, x, x, e)
    assert m0 == pytest.approx(0.0) and u0 == pytest.approx(0.0)
    m, u, bm = P.betti_parts(B, x, f, e)
    assert m + u == pytest.approx(float(bm.loss()))
    f2 = f.copy()
    f2[:2] = -50.0
    assert P.betti_parts(B, x, f2, e)[:2] == pytest.approx((m, u))


def test_self_values_cover_new_metrics():
    assert P.SELF_ZERO | set(P.SELF_PER_ROW) == P.LOWER | {"id_plda"}
    assert set(P.NEW_ROW) - P.LOWER == {"ms_ssim", "id_plda"}


def test_target_chunks_cover_each_target_once_in_order():
    sel = list(range(0, 921))
    ch = P.target_chunks(sel, P.CHUNK)
    assert all(0 < len(c) <= P.CHUNK for c in ch)
    assert [t for c in ch for t in c] == sel
    assert P.target_chunks([3, 7], 50) == [[3, 7]] and P.target_chunks([], 50) == []


def test_vgg_scored_set_gate_requires_exactly_the_alexnet_rows():
    import pandas as pd
    f2 = pd.DataFrame({"src": ["celeba"] * 3, "target": [1, 1, 2], "patch": [0, 1, 0], "fill": ["mirror"] * 3,
                       "lpips": [0.1, np.nan, 0.2]})
    same = f2.rename(columns={"lpips": "lpips_vgg"})
    assert P.scored_set_gate(same, f2)["passed"]
    extra = same.copy()
    extra.loc[1, "lpips_vgg"] = 0.3                    # VGG scored a row AlexNet left missing
    g = P.scored_set_gate(extra, f2)
    assert not g["passed"] and g["only_vgg"] == 1
    lost = same.copy()
    lost.loc[0, "lpips_vgg"] = np.nan                  # VGG lost a row AlexNet scored
    g = P.scored_set_gate(lost, f2)
    assert not g["passed"] and g["only_alexnet"] == 1
    empty = same.assign(lpips_vgg=np.nan)
    assert not P.scored_set_gate(empty, f2.assign(lpips=np.nan))["passed"]


def test_rung_spearman_skips_metadata_and_missing():
    Ra = {"a": {"R": 0.1}, "b": {"R": 0.5}, "c": {"R": 0.9}, "d": {"R": np.nan}, "_anchors": {"R": 1.0}}
    Rb = {"a": {"R": 0.2}, "b": {"R": 0.4}, "c": {"R": 1.1}, "d": {"R": 0.3}, "_anchors": {"R": -5.0}}
    out = P.spearman_rungs(Ra, Rb)
    assert out["n_rungs"] == 3 and out["spearman"] == pytest.approx(1.0)
    assert P.spearman_rungs({"a": {"R": 1.0}}, {"a": {"R": 1.0}})["spearman"] is None


def test_vgg_outputs_are_a_new_folder_and_f2_keeps_its_own():
    assert P.VGG_PRIVATE != P.PRIVATE and os.path.basename(P.VGG_PRIVATE).endswith("-vgg")
    assert "lpips_vgg" not in P.NEW_ROW and "lpips_vgg" not in P.LOWER


def test_ram_guard_refuses_workers_that_do_not_fit():
    # 9 Nov: 6 workers x 3.3 GB on 15 GB free starved the workstation; the guard must refuse that start
    with pytest.raises(SystemExit):
        P.ram_guard(6, free_gb=15.0)
    assert P.ram_guard(2, free_gb=12.0) == 12.0
