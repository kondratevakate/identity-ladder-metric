"""Block checkpoints for the RQ4 window-size wrappers (Amendment 10).

`blocked_rows` maps `fn` over targets in fixed blocks with an open executor; each finished block is written as a pickle
(exact float64 round trip) with an atomic rename, and a block whose file exists is loaded instead of recomputed. A
killed run therefore loses at most the blocks in flight. Block boundaries depend only on the target list and the block
size, so a resumed run must use the same targets and block size; the manifest file enforces that.
The pickles are read only from the run's own private folder, written by this function; never point part_dir at
files from elsewhere.
"""
import json
import os
import time

import pandas as pd

BLOCK = 24


def blocked_rows(ex, fn, targets, part_dir, label, block=BLOCK, chunksize=1):
    targets = [int(t) for t in targets]
    os.makedirs(part_dir, exist_ok=True)
    manifest = {"targets": targets, "block": int(block)}
    mpath = os.path.join(part_dir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            if json.load(fh) != manifest:
                raise SystemExit(f"{part_dir}: saved parts were made with other targets or block size")
    else:
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
    frames, t0, done, run_done = [], time.time(), 0, 0
    starts = range(0, len(targets), block)
    for b in starts:
        path = os.path.join(part_dir, f"b{b:05d}.pkl")
        chunk = targets[b:b + block]
        if os.path.exists(path):
            frames.append(pd.read_pickle(path))
            done += len(chunk)
            continue
        rows = []
        for r in ex.map(fn, chunk, chunksize=chunksize):
            rows.extend(r)
        df = pd.DataFrame(rows)
        tmp = path + ".tmp"
        df.to_pickle(tmp)
        os.replace(tmp, path)
        frames.append(df)
        done += len(chunk)
        run_done += len(chunk)
        el = time.time() - t0
        eta = el / run_done * (len(targets) - done)
        print(f"  {label}: {done}/{len(targets)} targets, {el / 60:.1f} min this run, ETA {eta / 60:.0f} min "
              f"({time.strftime('%H:%M', time.localtime(time.time() + eta))})", flush=True)
    if done and not run_done:
        print(f"  {label}: {done}/{len(targets)} targets, all loaded from saved parts", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
