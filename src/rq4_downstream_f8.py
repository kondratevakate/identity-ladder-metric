"""f = 1/8 versions of the cross-modality aggregates downstream of common_all.json (Amendment 10).

Runs the unchanged scripts in src/cross_modality (tests_uniform, rank_tiles, agreement, dataset_card) with their
inputs redirected: 2D row files -> the f = 1/8 rows of the RQ4 run, common_all/tests_all/ranks_all -> the *_f8_v1
versions; every output is written under a new *_f8_v1 name next to the originals, which stay untouched.
Brain rows (QTIM 16 mm cube, IXI) are already at f = 1/8 and are read unchanged.

Fail-closed: a 2D row file of the old window that is read without a redirect stops the run, and every redirect must
be used at least once by the scripts that need it (no silent fall-back to the old window).

    python src/rq4_downstream_f8.py
"""
import builtins
import io
import os
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CM = os.path.join(HERE, "src", "cross_modality")
OUT = os.path.join(HERE, "results", "cross_modality")
BIL = os.path.join(os.path.dirname(HERE), "identity-ladder-bilateral")
P = "C:/Projects/data/projects/identity-ladder-metric/"
RQ = P + "rq4-window-size-20261114-v1/"
TAG = "_f8_v1"

ROWS = {  # old-window 2D rows -> f = 1/8 rows (faces 10 px, iris 10 px, PAPILA 10 px, GRAPE 11 px)
    P + "face-ladder-metrics-20261106-v1/rows_v6f.csv.gz": RQ + "faces/rows_full_w10.csv.gz",
    P + "fundus-ladder-metrics-20261106-v1/rows_GRAPE_p8_full.csv.gz": RQ + "retina/rows_GRAPE_p8_w11_full.csv.gz",
    P + "fundus-ladder-metrics-20261106-v1/rows_PAPILA_p9_full.csv.gz": RQ + "retina/rows_PAPILA_p9_w10_full.csv.gz",
    P + "iris-ladder-metrics-20261107-v1/rows_UTIRIS_NIR_P2_full.csv.gz":
        P + "iris-ladder-metrics-20261107-v1/rows_UTIRIS_NIR_P2_full_w10.csv.gz",
    P + "iris-ladder-metrics-20261107-v1/rows_UTIRIS_VIS_P2_full.csv.gz": RQ + "iris/rows_UTIRIS_VIS_P2_w10_full.csv.gz",
    P + "iris-ladder-metrics-20261107-v1/rows_UPOL_P2_full.csv.gz": RQ + "iris/rows_UPOL_P2_w10_full.csv.gz",
}
OLD_2D = ("face-ladder-metrics-", "fundus-ladder-metrics-", "iris-ladder-metrics-")
NAMES = {"common_all.json": "common_f8_v1.json", "tests_all.json": "tests" + TAG + ".json",
         "ranks_all.json": "ranks" + TAG + ".json", "agreement.json": "agreement" + TAG + ".json",
         "dataset_card.json": "dataset_card" + TAG + ".json"}
used = set()


def norm(p):
    return os.path.normpath(str(p)).replace("\\", "/").lower()


ROWS_N = {norm(k): v for k, v in ROWS.items()}


def redirect(path, writing=False):
    n = norm(path)
    if n in ROWS_N:
        used.add(n)
        return ROWS_N[n]
    if n.endswith(".csv.gz") and any(s in n for s in OLD_2D) and n not in {norm(v) for v in ROWS.values()}:
        raise SystemExit(f"old-window 2D rows read without a redirect: {path}")
    d, b = os.path.split(str(path))
    if norm(d) == norm(OUT):
        if b in NAMES:
            return os.path.join(d, NAMES[b])
        stem, ext = os.path.splitext(b)
        if writing and not stem.endswith(TAG) and b not in NAMES.values():   # idempotent: savefig -> open nest
            return os.path.join(d, stem + TAG + ext)
    return path


_open, _read_csv, _savefig = builtins.open, pd.read_csv, plt.savefig
_fig_savefig = matplotlib.figure.Figure.savefig


def open_(file, mode="r", *a, **k):
    if isinstance(file, (str, os.PathLike)):
        file = redirect(file, writing=any(c in mode for c in "wax"))
    return _open(file, mode, *a, **k)


def read_csv_(path, *a, **k):
    return _read_csv(redirect(path) if isinstance(path, (str, os.PathLike)) else path, *a, **k)


def savefig_(fname, *a, **k):
    return _savefig(redirect(fname, writing=True) if isinstance(fname, (str, os.PathLike)) else fname, *a, **k)


def fig_savefig_(self, fname, *a, **k):
    return _fig_savefig(self, redirect(fname, writing=True) if isinstance(fname, (str, os.PathLike)) else fname,
                        *a, **k)


def run(script, *args):
    print(f"== {script} {' '.join(args)}", flush=True)
    argv = sys.argv
    sys.argv = [os.path.join(CM, script), *args]
    try:
        runpy.run_path(os.path.join(CM, script), run_name="__main__")
    finally:
        sys.argv = argv
        plt.close("all")


def main():
    builtins.open, pd.read_csv, plt.savefig = open_, read_csv_, savefig_
    matplotlib.figure.Figure.savefig = fig_savefig_
    try:
        run("tests_uniform.py", os.path.join(OUT, "tests_all.json"))
        run("rank_tiles.py", OUT)
        run("rank_tiles.py", OUT, "dataset")
        run("agreement.py", OUT)
        run("dataset_card.py", OUT, BIL)
    finally:
        builtins.open, pd.read_csv, plt.savefig = _open, _read_csv, _savefig
        matplotlib.figure.Figure.savefig = _fig_savefig
    missing = set(ROWS_N) - used
    if missing:
        raise SystemExit(f"redirects never used (inputs not at f = 1/8?): {sorted(missing)}")
    print("all 2D row inputs redirected to f = 1/8:", len(used), flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    main()
