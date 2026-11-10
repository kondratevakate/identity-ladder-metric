"""Checks on the committed cross-modality aggregates (results/cross_modality), no private data needed."""
import json
import os

import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "cross_modality")
SRC = os.path.join(ROOT, "src", "cross_modality")


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not built")
    return json.load(open(path, encoding="utf-8"))


def test_every_aggregate_has_its_generating_script():
    made_by = {"common_all.json": "common_frame.py", "tests_all.json": "tests_uniform.py", "cal_faces.json": "cal_uniform.py",
               "cal_brain.json": "cal_uniform.py", "cal_c_all.json": "cal_tiles.py", "ranks_all.json": "rank_tiles.py",
               "agreement.json": "agreement.py", "corr_all.json": "explore_corr.py", "data.json": "build_data.py",
               "dataset_card.json": "dataset_card.py", "ranks_generative.json": "generative_block.py", "values_generative.json": "generative_block.py", "fig_metric_summary_R.json": "summary_figure.py", "fig_metric_summary_paper_R.json": "summary_figure.py",
               "generative_status.json": "generative_block.py"}
    for f in os.listdir(OUT):
        if f.endswith(".json"):
            assert f in made_by, f"{f} has no known generating script"
            assert os.path.exists(os.path.join(SRC, made_by[f])), made_by[f]


def test_dataset_card_never_writes_zero_for_missing():
    rows = load("dataset_card.json")["rows"]
    papila = next(r for r in rows if r["dataset"] == "PAPILA")
    assert papila["ratio_repeat"] == "n/a"          # PAPILA has no repeat images
    for r in rows:
        for k, v in r.items():
            if k.startswith("ratio_") or k == "V_dataset_mean_mse":
                assert v == "n/a" or v > 0, (r["dataset"], k, v)
        if r["ratio_stranger"] != "n/a":
            assert 1.5 < r["ratio_stranger"] < 2.5   # a stranger sits near 2 x the dataset-mean error


def test_generative_ranks_are_permutations_and_not_run_is_absent():
    ranks = load("ranks_generative.json")
    status = load("generative_status.json")
    for key, rank in ranks.items():
        assert sorted(rank.values()) == list(range(1, len(rank) + 1)), key
        metric, dataset = key.split("|")
        assert metric not in status["not_run_metrics"].get(dataset, ()), key
        assert "self" not in rank and "retest" not in rank, key   # the target itself is the 0 anchor, not a rung


def test_generative_brain_and_faces_present():
    status = load("generative_status.json")
    for d in ("QTIM", "CelebA", "IMDb-WIKI"):
        assert "lpips_alex" in status["present"][d], d
    for r in ("other_side", "relative"):   # brain sets that coincide with the target set by construction
        assert r not in load("ranks_generative.json").get("fid|QTIM", {}), r
    assert "betti_matched" in status["present"]["QTIM"]   # Betti 2.5D (c710455); 3D not available


def test_tile_ranks_are_permutations():
    for key, rank in load("ranks_all.json").items():
        assert sorted(rank.values()) == list(range(1, len(rank) + 1)), key


def test_every_tile_column_has_a_drawn_thumbnail():
    import numpy as np
    keys = set(np.load(os.path.join(OUT, "drawn_face_thumbs.npz")).files)
    for col in ("repeat", "other_side", "relative", "lookalike_oracle", "lookalike", "lookalike16", "cohort_mean",
                "mean_k2", "dataset_mean", "stranger", "stranger_same_centre", "stranger_other_centre", "shrink_ref",
                "wrong_detail", "noise"):
        assert col in keys, col


def test_lookalike16_is_na_with_reason_never_a_value():
    c = load("common_all.json")
    na = c["not_available"]["lookalike16"]
    for d in ("CelebA", "IMDb-WIKI"):
        assert d in na and na[d]
        assert "lookalike16" not in c["datasets"][d]["R_self"], d
    for d in ("QTIM", "IXI", "GRAPE", "PAPILA", "UTIRIS NIR", "UTIRIS VIS", "UPOL"):
        assert "lookalike16" in c["datasets"][d]["R_self"], d


def test_generative_values_match_ranks_and_mark_anchor():
    v = load("values_generative.json")["values"]
    for key, rank in load("ranks_generative.json").items():
        assert set(v[key]["rungs"]) == set(rank), key
        want = "repeat" if key.endswith("|QTIM") else ("other_side" if key.endswith("|IXI") else "self")
        assert v[key]["anchor"] == want, key
        for r, rec in v[key]["rungs"].items():
            assert rec.get("ci95", rec.get("ratio_ci95")) is not None and rec["n"] is not None, (key, r)


def test_ixi_in_frame_with_na_rungs_never_values():
    c = load("common_all.json")
    raw = c["datasets"]["IXI"]["raw"]
    for r in ("repeat", "relative", "cohort_mean"):
        assert r not in raw, r
        assert c["not_available"][r]["IXI"], r
    assert "other_side" in raw and "stranger" in raw
    card = {r["dataset"]: r for r in load("dataset_card.json")["rows"]}
    assert card["IXI"]["ratio_repeat"] == "n/a"
    assert 1.5 < card["IXI"]["ratio_stranger"] < 2.5


def test_ixi_generative_not_run_with_reason():
    v = load("values_generative.json")
    assert v["not_run"]["IXI"]
    assert not [k for k in v["values"] if k.endswith("|IXI") and not k.startswith("betti_")]


def test_centre_strangers_ixi_only_and_na_elsewhere():
    c = load("common_all.json")
    for r in ("stranger_same_centre", "stranger_other_centre"):
        assert r in c["datasets"]["IXI"]["R_self"], r
        for d in c["datasets"]:
            if d != "IXI":
                assert r not in c["datasets"][d]["raw"] and c["not_available"][r][d], (r, d)


def _mean_R(rows, rung):
    vals = []
    for row in rows:
        ref = "repeat" if "repeat" in row else ("other_side" if "other_side" in row else None)
        if ref is None:
            continue
        src = ref if rung == "ref" else rung
        if src in row and row[src] is not None:
            vals.append(row[src])
    return sum(vals) / len(vals) if vals else None


def test_summary_figure_values_equal_the_tables():
    """fig_metric_summary_R.json (what panel B plots) recomputed from common_all.json and values_generative.json."""
    fig = load("fig_metric_summary_R.json")
    c = load("common_all.json")["datasets"]
    g = load("values_generative.json")["values"]
    mse_rows = [{r: v["R_self"][r]["mse"][0] for r in v["R_self"] if "mse" in v["R_self"][r]} for v in c.values()]
    fid_rows = [{r: (x["ratio_to_stranger"] if x["n"] / b["rungs"]["stranger"]["n"] >= 0.5 else None)
                 for r, x in b["rungs"].items()}   # set ratio, anchor-free; sets under half the stranger set left out
                for k, b in g.items() if k.startswith("fid|") and k != "fid|QTIM"]   # brain excluded (small repeat set)
    pers = {d: {r: v["R_self"][r]["persistence"][0] for r in v["R_self"] if "persistence" in v["R_self"][r]}
            for d, v in c.items()}
    pers = {d: r for d, r in pers.items() if r}
    for k, b in g.items():   # faces persistence exists only in the generative block (self-anchored)
        d = k.split("|")[1]
        if k.startswith("persistence|") and b["anchor"] == "self" and d not in pers:
            pers[d] = {r: x["R"] for r, x in b["rungs"].items()}
    pers_rows = list(pers.values())
    for metric, rows in (("mse", mse_rows), ("fid", fid_rows), ("persistence", pers_rows)):
        assert fig[metric]["n_datasets"] == len([r for r in rows if "repeat" in r or "other_side" in r]), metric
        for rung in ("ref", "dataset_mean", "shrink_ref", "noise"):
            want = _mean_R(rows, rung)
            got = fig[metric][rung]
            assert (want is None and got is None) or abs(got - want) < 1e-3, (metric, rung, got, want)


def test_summary_figure_paper_variant_plots_the_same_values():
    assert load("fig_metric_summary_paper_R.json") == load("fig_metric_summary_R.json")
    assert os.path.exists(os.path.join(OUT, "fig_metric_summary_paper.pdf"))


def test_summary_panel_a_equals_rank_table_gaps():
    """Panel A y = mean over datasets of (shrunk true person - true person) rank gap / (ladder length - 1); x the same
    with the stranger. Recomputed from ranks_all.json and ranks_generative.json."""
    a = load("fig_metric_summary_R.json")["panel_A"]
    ranks = {**load("ranks_all.json"), **load("ranks_generative.json")}
    g = load("values_generative.json")["values"]

    def ok(key, rung):   # set-level rule: sets under half the stranger set are left out; brain excluded
        if key.split("|")[0] not in ("fid", "fwd"):
            return True
        if key.endswith("|QTIM"):
            return False
        r = g[key]["rungs"]
        return r[rung]["n"] / r["stranger"]["n"] >= 0.5
    for metric, got in a.items():
        xs, ys = [], []
        for key, v in ranks.items():
            if key.split("|")[0] != metric:
                continue
            ref = "repeat" if "repeat" in v else ("other_side" if "other_side" in v else None)
            if ref is None or not ok(key, ref):
                continue
            n = max(v.values())
            if "stranger" in v:
                xs.append((v["stranger"] - v[ref]) / (n - 1))
            if "shrink_ref" in v and ok(key, "shrink_ref"):
                ys.append((v["shrink_ref"] - v[ref]) / (n - 1))
        assert abs(got["y_shrunk_gap"] - sum(ys) / len(ys)) < 1e-3 and got["n_y"] == len(ys), metric
        assert abs(got["x_stranger_gap"] - sum(xs) / len(xs)) < 1e-3 and got["n_x"] == len(xs), metric


def test_summary_small_sets_left_out_of_set_level_means():
    """QTIM's repeat set (138 windows vs 1326 strangers) inflates FID; it is listed as left out, never averaged."""
    rule = load("fig_metric_summary_R.json")["set_level_rule"]
    assert rule["min_set_fraction_of_stranger"] == 0.5
    assert "fid|QTIM:repeat" in rule["left_out"] and "fid|GRAPE:repeat" not in rule["left_out"]
    assert "QTIM" in rule["datasets_excluded"]
    fig = load("fig_metric_summary_R.json")
    assert "QTIM" not in fig["fid"]["datasets"] and "QTIM" not in fig["fwd"]["datasets"]
