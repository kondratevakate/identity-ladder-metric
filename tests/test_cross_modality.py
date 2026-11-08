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
               "dataset_card.json": "dataset_card.py", "ranks_generative.json": "generative_block.py",
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
        assert key.split("|")[1] not in status["not_run"], key


def test_tile_ranks_are_permutations():
    for key, rank in load("ranks_all.json").items():
        assert sorted(rank.values()) == list(range(1, len(rank) + 1)), key
