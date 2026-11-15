"""RQ4 size curve and f = 1/8 summary input, iris (PROTOCOL_donor_ladder_general.md, Amendment 10; iris Amendment P4
extended to f = 1/16 and 1/4). Pre-QC.

Runs the iris metric panel (`iris_ladder_metrics`, frame P2, unchanged code, `lookalike16` and the oracle included)
at the window w = round_half_up(D / denom), D = 256 / pi, ring ceil(w / 2): 1/16 -> 5 px, 1/8 -> 10 px, 1/4 -> 20 px.
The panel module allows only 16 and 10 px (P4); this run widens the allowed set to the Amendment 10 sizes in this
process and in every worker, and changes nothing else. Rows go to this run's private folder; the per-set aggregate
(`summarise`, the panel's own) is written next to them as JSON (aggregates only).

Gate (smoke): at 16 px and 10 px the wrapper must reproduce the stored panel rows of the same targets exactly.

    python -I src/rq4_iris_size.py --set UTIRIS_VIS --denom 8 --workers 2 [--every 8] [--check]
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iris_ladder_metrics as IM  # noqa: E402
import iris_ladder_metrics_frame as IF  # noqa: E402
from rq4_checkpoint import blocked_rows  # noqa: E402

PRIVATE = os.path.join(os.path.dirname(IF.PRIVATE), "rq4-window-size-20261114-v1", "iris")
VERSION = "P2"
D_PX = IM.W / math.pi                                     # iris outer diameter in strip columns (P4)
DENOMS = (16, 8, 4)
ALLOWED = tuple(sorted({16} | {int(math.floor(D_PX / k + 0.5)) for k in DENOMS}))
TOL = 1e-9


def window_of(denom):
    return int(math.floor(D_PX / denom + 0.5))


def _init(state):
    IM.WINDOWS = ALLOWED
    IM._init(state)


def run_rows(st, targets, workers, part_dir):
    """As `iris_ladder_metrics.run_rows`, with the widened window set in the workers; block checkpoints in part_dir."""
    state = {k: v for k, v in st.items() if k in ("X", "M", "XF", "MF", "idx", "valid", "person", "side", "repeat",
                                                     "fellow", "pos", "side_units", "lam", "sigma", "name")}
    state["fundus_path"], state["brain_path"] = os.environ["FUNDUS_METRICS_SRC"], os.environ["BRAIN_METRICS_SRC"]
    state["window"] = IM.WIN
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(state,)) as ex:
        return blocked_rows(ex, IM.run_target, targets, part_dir, f"{st['name']} {IM.WIN} px", chunksize=2)


def check(r, stored_path):
    """Stored rows of the same targets against the new ones: same keys, missing pattern and values."""
    ref = pd.read_csv(stored_path)
    key = [c for c in ("unit", "patch", "fill") if c in r.columns]
    ref = ref.merge(r[["unit"]].drop_duplicates(), on="unit")
    m = ref.merge(r, on=key, how="outer", suffixes=("_ref", ""), indicator=True)
    out = {"stored": str(os.path.basename(stored_path)), "n_stored": int(len(ref)), "n_new": int(len(r)),
           "n_matched": int((m["_merge"] == "both").sum()), "max_abs_diff": {}}
    ok = out["n_stored"] == out["n_new"] == out["n_matched"] > 0
    for c in r.columns:
        if c in key or c + "_ref" not in m.columns or not np.issubdtype(r[c].dtype, np.number):
            continue
        x, y = m[c].to_numpy(float), m[c + "_ref"].to_numpy(float)
        same = bool((np.isnan(x) == np.isnan(y)).all())
        diff = float(np.nanmax(np.abs(x - y))) if np.isfinite(x - y).any() else 0.0
        out["max_abs_diff"][c] = diff if same else "missing pattern differs"
        ok = ok and same and diff <= TOL
    out["passed"] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=IF.SETS)
    ap.add_argument("--denom", type=int, required=True, choices=DENOMS + (0,), help="0 = the 16 px reference (check)")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--every", type=int, default=0, help="smoke: every N-th eye of each side")
    ap.add_argument("--check", action="store_true", help="compare with the stored panel rows of this size")
    ap.add_argument("--qc", action="store_true", help="private fill sheet on 12 evenly spaced eyes, no rows")
    a = ap.parse_args()
    if a.workers > 3:
        raise SystemExit("at most 3 workers (shared CPU)")
    w = 16 if a.denom == 0 else window_of(a.denom)
    IM.load_shared()
    IM.WINDOWS = ALLOWED
    IM.set_window(w)
    os.makedirs(PRIVATE, exist_ok=True)
    t0 = time.time()
    st = IM.build(a.set, VERSION)
    n = len(st["idx"])
    if a.qc:
        path = os.path.join(PRIVATE, f"qc_fills_{a.set}_{VERSION}_w{w}_private.png")
        IM.qc_sheet(st, list(np.linspace(0, n - 1, 12).round().astype(int)), path)
        print(path, flush=True)
        return
    targets = (sorted(np.concatenate([np.flatnonzero(st["side"] == s_)[::a.every] for s_ in ("R", "L")]).tolist())
               if a.every else list(range(n)))
    tag = f"{a.set}_{VERSION}_w{w}" + (f"_every{a.every}" if a.every else "_full")
    print(f"{a.set}: window {w} px (f = {w / D_PX:.3f}), ring {IM.RING}, positions {len(st['pos'])}, eyes {n}, "
          f"targets {len(targets)}", flush=True)
    r = run_rows(st, targets, a.workers, os.path.join(PRIVATE, f"parts_{tag}"))
    r.to_csv(os.path.join(PRIVATE, f"rows_{tag}.csv.gz"), index=False)
    info = {"set": a.set, "window_px": w, "ring_px": IM.RING, "f_exact": w / D_PX, "D_px": D_PX,
            "positions": len(st["pos"]), "targets": len(targets), "rows": int(len(r)),
            "minutes": round((time.time() - t0) / 60, 1), "workers": a.workers}
    if a.check:
        sfx = "" if w == 16 else f"_w{w}"
        info["check"] = check(r, os.path.join(IF.PRIVATE, f"rows_{a.set}_{VERSION}_full{sfx}.csv.gz"))
        print(json.dumps(info["check"]), flush=True)
    if not a.every:
        s = IM.summarise(st, r)
        info["panel"] = IM.FL.finite({"frame": VERSION, **s})
    with open(os.path.join(PRIVATE, f"summary_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1, allow_nan=False)
    print(json.dumps({k: v for k, v in info.items() if k not in ("panel", "check")}), flush=True)
    if a.check and not info["check"]["passed"]:
        raise SystemExit("the wrapper does not reproduce the stored panel rows")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
