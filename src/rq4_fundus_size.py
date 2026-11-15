"""RQ4 size curve, retina (GRAPE P8, PAPILA P9): fundus metric panel at windows in object units
(PROTOCOL_donor_ladder_general.md, Amendment 10). Pre-QC.

The panel code is `fundus_ladder_metrics.py` of the donor-ladder-bilateral worktree, loaded unchanged from the path in
FUNDUS_METRICS_SRC (no default; its directory goes first on sys.path so its own `donor_ladder_2d` is used). Object
extent L = sqrt(mean field-of-view area in the 128 px frame) over the dataset's units (mask `M` of
`donor_ladder_2d_facefmt.down2`); w = round_half_up(L / denom), ring = ceil(w / 2). The module's window is a module
constant and a default argument of `positions` and `ring_mask`; both are set explicitly here, in the main process and
in every worker, before anything runs. The Amendment 9.15 extras (LPIPS, Betti, set-level) are off: tier-1 rungs and
metrics only (the RQ4 curve uses MSE). `lookalike16` and the oracle are part of the panel.

Gate (smoke): at 16 px the wrapper must reproduce the stored panel rows of the same targets on every tier-1 column.

    python -I src/rq4_fundus_size.py --set PAPILA --denom 8 --workers 2 [--every 20] [--check]
    python -I src/rq4_fundus_size.py --geometry
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

_SRC = os.environ.get("FUNDUS_METRICS_SRC")
if not _SRC or not os.path.isfile(_SRC):
    raise SystemExit("FUNDUS_METRICS_SRC must point to fundus_ladder_metrics.py (no default)")
sys.path.insert(0, os.path.dirname(_SRC))
import fundus_ladder_metrics as FL  # noqa: E402
sys.path.append(os.path.dirname(os.path.abspath(__file__)))     # after the fundus dir: its modules win
from rq4_checkpoint import blocked_rows  # noqa: E402

PRIVATE = os.path.join(os.path.dirname(FL.PRIVATE), "rq4-window-size-20261114-v1", "retina")
ARMS = dict(FL.FF.SETS)
DENOMS = (16, 8, 4)
TOL = 1e-9


def extent(name):
    """L = sqrt(mean field-of-view area) in the 128 px frame over the dataset's units; geometry only."""
    _, img, fov, u, _, _, _, _ = FL.D.setup(name, ARMS[name])
    _, M = FL.FF.down2(img, fov)
    a = M[u.idx.to_numpy()].reshape(len(u), -1).sum(1)
    return {"L_px": float(np.sqrt(a.mean())), "n_units": int(len(u)),
            "equal_area_diameter_px": float(2 * np.sqrt(a.mean() / np.pi))}


def window_of(L, denom):
    w = 16 if denom == 0 else int(math.floor(L / denom + 0.5))
    return w, int(math.ceil(w / 2))


def set_window(w, rw):
    """Window and ring of the fundus module: constants and the defaults of positions / ring_mask."""
    FL.WIN, FL.RING = int(w), int(rw)
    FL.positions.__defaults__ = (FL.WIN, FL.RING)
    FL.ring_mask.__defaults__ = (FL.WIN, FL.RING)


def _init(state, w, rw):
    set_window(w, rw)
    FL._init(state)


def run_rows(name, targets, workers, w, rw, part_dir):
    from concurrent.futures import ProcessPoolExecutor
    st = FL.build(name, ARMS[name])
    n = len(st["idx"])
    targets = list(range(n)) if targets is None else list(targets)
    st["bm_path"] = os.environ.get("BRAIN_METRICS_SRC")
    st["models_path"] = None
    st["extra"] = False
    print(f"{name}: window {w} px, ring {rw}, positions {len(st['pos'])}, units {n}, targets {len(targets)}",
          flush=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(dict(st), w, rw)) as ex:
        r = blocked_rows(ex, FL.run_target, targets, part_dir, f"{name} {w} px", block=12)
    st.pop("bm_path", None)
    st.pop("models_path", None)
    return r, st


def check(r, stored_path):
    ref = pd.read_csv(stored_path)
    key = ["unit", "patch", "fill"]
    ref = ref.merge(r[["unit"]].drop_duplicates(), on="unit")
    cols = [c for c in r.columns if c in ref.columns and c not in key and np.issubdtype(r[c].dtype, np.number)]
    m = ref[key + cols].merge(r[key + cols], on=key, how="outer", suffixes=("_ref", ""), indicator=True)
    out = {"stored": os.path.basename(stored_path), "n_stored": int(len(ref)), "n_new": int(len(r)),
           "n_matched": int((m["_merge"] == "both").sum()), "max_abs_diff": {}}
    ok = out["n_stored"] == out["n_new"] == out["n_matched"] > 0
    for c in cols:
        x, y = m[c].to_numpy(float), m[c + "_ref"].to_numpy(float)
        same = bool((np.isnan(x) == np.isnan(y)).all())
        diff = float(np.nanmax(np.abs(x - y))) if np.isfinite(x - y).any() else 0.0
        out["max_abs_diff"][c] = diff if same else "missing pattern differs"
        ok = ok and same and diff <= TOL
    out["passed"] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=tuple(ARMS))
    ap.add_argument("--denom", type=int, choices=DENOMS + (0,), help="0 = the 16 px reference (check)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--every", type=int, default=0, help="smoke: every N-th unit of each side")
    ap.add_argument("--check", action="store_true", help="compare with the stored 16 px rows (denom 0)")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--qc", action="store_true", help="private fill sheet on 12 evenly spaced eyes, no rows")
    a = ap.parse_args()
    FL.setup_external()
    FL.load_bm()
    os.makedirs(PRIVATE, exist_ok=True)
    if a.geometry:
        geo = {}
        for name in ARMS:
            g = extent(name)
            g["f_of_16px"] = 16 / g["L_px"]
            g["sizes"] = {f"1/{k}": dict(zip(("window_px", "ring_px"), window_of(g["L_px"], k))) for k in DENOMS}
            geo[name] = g
        json.dump(geo, open(os.path.join(PRIVATE, "geometry.json"), "w"), indent=1)
        print(json.dumps(geo, indent=1))
        return
    if a.workers > 3:
        raise SystemExit("at most 3 workers (shared CPU)")
    geo = json.load(open(os.path.join(PRIVATE, "geometry.json")))
    w, rw = window_of(geo[a.set]["L_px"], a.denom)
    set_window(w, rw)
    if a.qc:
        st = FL.build(a.set, ARMS[a.set])
        st["extra"] = False
        n = len(st["idx"])
        path = os.path.join(PRIVATE, f"qc_fills_{a.set}_w{w}_private.png")
        FL.qc_sheet(a.set, st, list(np.linspace(0, n - 1, 12).round().astype(int)), path, {})
        print(path, flush=True)
        return
    targets = None
    if a.every:
        side = FL.D.setup(a.set, ARMS[a.set])[3].side.to_numpy()
        targets = sorted(np.concatenate([np.flatnonzero(side == s_)[::a.every] for s_ in ("R", "L")]).tolist())
    t0 = time.time()
    tag = f"{a.set}_{ARMS[a.set]}_w{w}" + (f"_every{a.every}" if a.every else "_full")
    r, st = run_rows(a.set, targets, a.workers, w, rw, os.path.join(PRIVATE, f"parts_{tag}"))
    r.to_csv(os.path.join(PRIVATE, f"rows_{tag}.csv.gz"), index=False)
    info = {"set": a.set, "window_px": w, "ring_px": rw, "L_px": geo[a.set]["L_px"], "f_exact": w / geo[a.set]["L_px"],
            "positions": len(st["pos"]), "targets": int(r.unit.nunique()), "rows": int(len(r)),
            "minutes": round((time.time() - t0) / 60, 1), "workers": a.workers, "extras": "off (tier-1 only)"}
    if a.check:
        info["check"] = check(r, os.path.join(FL.PRIVATE, f"rows_{a.set}_{ARMS[a.set]}_full.csv.gz"))
        print(json.dumps(info["check"]), flush=True)
    with open(os.path.join(PRIVATE, f"summary_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1, allow_nan=False)
    print(json.dumps({k: v for k, v in info.items() if k != "check"}), flush=True)
    if a.check and not info["check"]["passed"]:
        raise SystemExit("the wrapper does not reproduce the stored panel rows")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
