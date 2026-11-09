import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import face_vgg_vm as V  # noqa: E402

DESC = {"data": {"a": {"pool": [0, 3], "targets": [{"t": 5, "rep": 6, "sib": None}]},
                 "b": {"pool": [10, 12], "targets": [{"t": 13, "rep": 14, "sib": 2}]}}}


def test_source_rows_cover_pool_targets_retests_and_siblings():
    assert V.source_rows(DESC, "a") == [0, 1, 2, 5, 6]
    assert V.source_rows(DESC, "b") == [2, 10, 11, 13, 14]


def test_blank_other_rows_removes_every_row_of_the_other_source():
    X = np.arange(16, dtype=np.float32)[:, None].repeat(4, 1)
    keep = V.source_rows(DESC, "b")
    out = V.blank_other_rows(X, keep, np.nan)
    assert np.isfinite(out[keep]).all()
    assert np.isnan(np.delete(out, keep, axis=0)).all()
    M = V.blank_other_rows(np.ones((16, 2), bool), keep, False)
    assert M[keep].all() and not np.delete(M, keep, axis=0).any()


def _rows(vgg):
    return [{"src": "imdb", "target": t, "family": t, "patch": 0, "fill": "str00", "mse": 0.5, "lpips_vgg": v}
            for t, v in enumerate(vgg)]


def test_compare_accepts_last_bit_noise_and_rejects_real_differences():
    assert V.compare(_rows([0.1, np.nan]), _rows([0.1 + 1e-7, np.nan]))["passed"]
    assert not V.compare(_rows([0.1, np.nan]), _rows([0.1 + 1e-3, np.nan]))["passed"]
    assert not V.compare(_rows([0.1, np.nan]), _rows([0.1, 0.2]))["passed"]          # missing pattern differs
    assert not V.compare(_rows([0.1, 0.2]), _rows([0.1]))["passed"]                   # rows lost


def test_pack_round_trip_is_lossless(tmp_path):
    # 9 Nov: the default CSV float format changed MSE by 8.9e-16 and failed the laptop-vs-VM reference check
    import pickle
    parts = tmp_path / "parts"
    parts.mkdir()
    rows = [{"src": "imdb", "target": 1, "family": 1, "patch": 0, "fill": "str00", "mse": 0.1 + 0.2,
             "lpips_vgg": 1 / 3}, {"src": "imdb", "target": 2, "family": 2, "patch": 0, "fill": "mirror",
                                   "mse": np.pi / 7, "lpips_vgg": np.nan}]
    with open(parts / "imdb_p00_c00.pkl", "wb") as fh:
        pickle.dump((rows, 0.0), fh)
    V.pack(str(parts), str(tmp_path / "r.csv.gz"), str(tmp_path / "r.json"))
    import pandas as pd
    back = pd.read_csv(tmp_path / "r.csv.gz", float_precision="round_trip")
    assert (back.mse.values == np.array([r["mse"] for r in rows])).all()
    assert back.lpips_vgg[0] == 1 / 3 and np.isnan(back.lpips_vgg[1])
