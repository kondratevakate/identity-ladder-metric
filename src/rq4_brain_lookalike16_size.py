"""RQ4 size curve, brain (QTIM): mean of 16 look-alikes at 8 and 32 mm cubes (PROTOCOL_donor_ladder_general.md,
Amendment 10; geometry of brain Amendment 21; fill of brain Amendment 24 A). Pre-QC (affine registration only).

At each size the fill is the masked mean of the 16 donors nearest by the scaffold (shell) distance that
`donor_ladder_brain_metrics.run_position_at` uses for the k = 1 look-alike. The k = 1 look-alike is recomputed next to
it (`lookalike_k1_check`) and must equal the stored rows of the Amendment 21 size run on every metric; otherwise no
rows are kept. Region memmaps are built one position at a time into this run's private folder and deleted once the
position is saved. Rows stay private.

Usage: python src/rq4_brain_lookalike16_size.py --cube 8|32 --workers N [--smoke 3] [--positions 0,5]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as BL  # noqa: E402
import donor_ladder_brain_lookalike16 as L16  # noqa: E402
import donor_ladder_brain_metrics as M  # noqa: E402
import donor_ladder_brain_metrics_size as MS  # noqa: E402
import donor_ladder_brain_size as S  # noqa: E402

PRIVATE = os.path.join(B.PROJ, "rq4-window-size-20261114-v1", "brain")
K = L16.K
GB = 2 ** 30
RESERVE_GB = 2.0          # RAM kept free at all times (Kate, 15 Nov: no paging; the 14 Nov 32 mm run thrashed)
FIRST_CUBE_GB = 2.0       # minimum headroom for the single measurement cube (HANDOFF estimate ~1.3 GB per worker)


def run_position_geom(args):
    """lookalike16 and lookalike_k1_check rows for one position, both sides, at an explicit geometry.

    Same computation as `donor_ladder_brain_lookalike16.run_position`, with the shell, block-search half-window and
    region paths passed in `geom` (no module default)."""
    pi, t, tg_cols, pos, only, geom = args
    tg = pd.DataFrame(tg_cols)
    _, start, size = pos[pi]
    pv, pm = geom["paths"]
    vol, mask = np.load(pv), np.load(pm).astype(bool)
    drawn = BL.side_of(start, size, M.AFFINE_HOLDER["affine"])
    shell, W = geom["shell"], geom["W"]
    fam, sex, age = tg.family.to_numpy(), tg.sex.to_numpy(), tg.age.to_numpy()
    ix = tg["index"].to_numpy()
    row_of = {(p, s): k for k, (p, s) in enumerate(zip(t["person"], t["session"]))}
    win_sl = tuple(slice(W, W + s + 2 * shell) for s in size)
    cube_sl = tuple(slice(W + shell, W + shell + s) for s in size)
    sm = BL.shell_mask(size, shell)
    rows = []
    for side in BL.SIDES:
        fo = BL.own_frame(side, drawn)
        own = vol[:, fo]
        cube_all = own[(slice(None),) + cube_sl]
        mk_all = mask[:, fo][(slice(None),) + cube_sl]
        valid_all = mk_all.reshape(len(cube_all), -1).mean(1) >= BL.VALID
        X, MK, valid = cube_all[ix], mk_all[ix], valid_all[ix]
        win = own[(slice(None),) + win_sl][ix]
        wmask = mask[:, fo][(slice(None),) + win_sl][ix]
        C, CM = win[:, sm], wmask[:, sm]
        n_shell = int(sm.sum())
        rep_msd = {}
        for i in np.flatnonzero(valid):
            j2 = row_of.get((tg.person.iloc[i], "ses-02"))
            if j2 is not None and valid_all[j2]:
                both = MK[i] & mk_all[j2]
                if both.sum() >= 10:
                    rep_msd.setdefault(fam[i], []).append(float(np.mean((X[i][both] - cube_all[j2][both]) ** 2)))
        for i in np.flatnonzero(valid):
            if only is not None and int(i) not in only:
                continue
            others = np.flatnonzero((fam != fam[i]) & valid)
            pairs = [v for f_, vs in rep_msd.items() if f_ != fam[i] for v in vs]
            sigma = float(np.sqrt(np.mean(pairs))) if len(pairs) >= M.MIN_REP_PAIRS else None
            if others.size < 10:
                continue
            xi, ei = X[i], MK[i]
            mu_d, _ = M.masked_mean(X[others], MK[others])
            cpool = others[(sex[others] == sex[i]) & (np.abs(age[others] - age[i]) <= 2)]
            mu_c = M.masked_mean(X[cpool], MK[cpool])[0] if cpool.size else None
            d_s = M.masked_dist(C[others], CM[others], C[i], CM[i], n_shell)
            near = others[L16.nearest(d_s)]
            if near.size == 0:
                continue
            fills = [("lookalike_k1_check", X[near[0]], MK[near[0]], 1)]
            if near.size >= K:                    # fewer than 16 eligible donors: the rung is missing, not smaller
                m16, ok16 = M.masked_mean(X[near], MK[near])
                fills.append(("lookalike16", m16, ok16, int(near.size)))
            for name, f, ok, nd in fills:
                s = M.score(xi, f, ei & ok, mu_d, mu_c, sigma)
                if s is None:
                    continue
                rows.append({"target": int(i), "patch": pi, "side": side, "drawn": side == drawn, "fill": name,
                             "n_donors": nd, **s})
    return rows


def build_region(V, Mk, start, size, shell, W, a, pi):
    """As `donor_ladder_brain_metrics_size.build_region`, written into this run's private folder."""
    pad = shell + W
    lo = tuple(s0 - pad for s0 in start)
    rs = tuple(n + 2 * pad for n in size)
    os.makedirs(os.path.join(PRIVATE, "regions"), exist_ok=True)
    pv, pm = (os.path.join(PRIVATE, "regions", f"{k}{pi:02d}.npy") for k in ("vol", "mask"))
    vol = np.lib.format.open_memmap(pv + ".tmp.npy", "w+", np.float32, (len(V), 2) + rs)
    mask = np.lib.format.open_memmap(pm + ".tmp.npy", "w+", np.uint8, (len(V), 2) + rs)
    for k in range(len(V)):
        for fr, mirror in ((0, None), (1, a)):
            vol[k, fr] = S.get_box(V[k], lo, rs, mirror)
            mask[k, fr] = S.get_box(Mk[k], lo, rs, mirror)
    vol.flush()
    mask.flush()
    del vol, mask
    os.replace(pv + ".tmp.npy", pv)
    os.replace(pm + ".tmp.npy", pm)
    return pv, pm


def stored_rows(cube, sel, smoke):
    """Stored Amendment 21 rows at this size (full rows file, or the smoke parts)."""
    full = os.path.join(MS.PRIVATE, f"w{cube}_rows.csv.gz")
    if os.path.exists(full):
        return pd.read_csv(full)
    raise SystemExit(f"no stored Amendment 21 rows at {cube} mm: {full}")


def qc(position, n_targets):
    """Private sheet: central axial slice of the cube (shell drawn around it) for evenly spaced targets: target,
    k = 1 look-alike (shell distance), mean of 16, oracle (nearest on the hidden cube), at 8, 16 and 32 mm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from nilearn.datasets import load_mni152_brain_mask
    t = B.scans()
    pos16, affine, _ = B.positions(np.random.default_rng(M.SEED))
    M._init(affine)
    tmask = load_mni152_brain_mask(resolution=1).get_fdata() > 0
    V, Mk = S.cache_volumes(t, affine)
    mc = BL.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    tg = t[t.session == "ses-01"].reset_index()
    fam, ix = tg.family.to_numpy(), tg["index"].to_numpy()
    sel = np.unique(np.linspace(0, len(tg) - 1, n_targets).round().astype(int))
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#9ecae1")
    cols = ("target", "look-alike k=1", "mean of 16", "oracle")
    fig, ax = plt.subplots(len(sel), 3 * len(cols), figsize=(1.25 * 3 * len(cols), 1.35 * len(sel) + 0.6),
                           squeeze=False)
    for c, cube in enumerate((8, 16, 32)):
        g = MS.geometry(cube)
        pos, kept, _ = MS.sized_positions(pos16, tmask, cube, g["shell"])
        if position not in kept:
            raise SystemExit(f"position {position} is dropped at {cube} mm")
        paths = build_region(V, Mk, pos[position][1], pos[position][2], g["shell"], g["W"], mc, position)
        vol, mask = np.load(paths[0]), np.load(paths[1]).astype(bool)
        sh, W = g["shell"], g["W"]
        size = pos[position][2]
        drawn = BL.side_of(pos[position][1], size, affine)
        fo = BL.own_frame(drawn, drawn)
        win_sl = tuple(slice(W, W + s + 2 * sh) for s in size)
        cube_sl = tuple(slice(sh, sh + s) for s in size)
        win, wm = vol[:, fo][(slice(None),) + win_sl][ix], mask[:, fo][(slice(None),) + win_sl][ix]
        X, MK = win[(slice(None),) + cube_sl], wm[(slice(None),) + cube_sl]
        valid = MK.reshape(len(X), -1).mean(1) >= BL.VALID
        sm = BL.shell_mask(size, sh)
        C, CM = win[:, sm], wm[:, sm]
        Xf, XM = X.reshape(len(X), -1), MK.reshape(len(X), -1)
        z = sh + size[2] // 2
        for i, ti in enumerate(sel):
            others = np.flatnonzero((fam != fam[ti]) & valid)
            d_s = M.masked_dist(C[others], CM[others], C[ti], CM[ti], int(sm.sum()))
            d_o = M.masked_dist(Xf[others], XM[others], Xf[ti], XM[ti], Xf.shape[1])
            near = others[L16.nearest(d_s)]
            j_o = others[int(np.nanargmin(np.where(np.isfinite(d_o), d_o, np.nan)))]
            m16, ok16 = M.masked_mean(X[near], MK[near])
            panels = [(win[ti], wm[ti])]
            for f, ok in ((X[near[0]], MK[near[0]]), (m16, ok16), (X[j_o], MK[j_o])):
                w2 = win[ti].copy()
                k2 = wm[ti].copy()
                w2[cube_sl] = f
                k2[cube_sl] = ok & MK[ti]
                panels.append((w2, k2))
            for q, (img, ok) in enumerate(panels):
                a_ = ax[i, c * len(cols) + q]
                a_.imshow(np.where(ok[:, :, z], img[:, :, z], np.nan).T, cmap=cmap, vmin=-2.5, vmax=2.5,
                          origin="lower", interpolation="nearest")
                a_.add_patch(plt.Rectangle((sh - .5, sh - .5), size[0], size[1], fill=False, ec="#d33", lw=0.6))
                a_.set_xticks([])
                a_.set_yticks([])
                if i == 0:
                    a_.set_title(f"{cols[q]}\n{cube} mm (f {g['f']})", fontsize=6)
        for p_ in paths:
            os.remove(p_)
    fig.suptitle(f"PRIVATE QC QTIM, Amendment 10: random cube {position}, drawn side, central axial slice; red = "
                 "hidden cube, outside = shell context; light blue = outside the brain mask; targets evenly spaced",
                 fontsize=7)
    fig.tight_layout()
    out = os.path.join(PRIVATE, f"qc_lookalike16_sizes_p{position:02d}_private.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out, flush=True)


def main():
    import nibabel as nib
    from nilearn.datasets import load_mni152_brain_mask
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=int, choices=(8, 32), help="cube edge in mm (no default)")
    ap.add_argument("--qc", type=int, default=-1, help="private QC sheet of this position at 8, 16, 32 mm")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", type=int, default=0, help="score only the first N targets that have a repeat scan")
    ap.add_argument("--positions", default="", help="comma-separated position indices; default all kept")
    a = ap.parse_args()
    if a.qc >= 0:
        return qc(a.qc, 8)
    if a.cube is None:
        raise SystemExit("--cube is required")
    g = MS.geometry(a.cube)
    t0 = time.time()
    t = B.scans()
    pos16, affine, _ = B.positions(np.random.default_rng(M.SEED))
    tmask = load_mni152_brain_mask(resolution=1).get_fdata() > 0
    pos, kept, dropped = MS.sized_positions(pos16, tmask, g["cube"], g["shell"])
    sel = [int(p) for p in a.positions.split(",")] if a.positions else kept
    if set(sel) - set(kept):
        raise SystemExit(f"positions {sorted(set(sel) - set(kept))} are dropped at {a.cube} mm")
    tg = t[t.session == "ses-01"].reset_index()
    only = None
    if a.smoke:
        rep = set(t[t.session == "ses-02"].person)
        only = set(np.flatnonzero(tg.person.isin(rep).to_numpy())[:a.smoke].tolist())
    tag = f"w{a.cube}_" + (f"smoke{a.smoke}_" if a.smoke else "")
    parts = os.path.join(PRIVATE, "parts")
    os.makedirs(parts, exist_ok=True)
    part_path = lambda pi: os.path.join(parts, f"{tag}p{pi:02d}.csv.gz")  # noqa: E731
    todo = [pi for pi in sel if not os.path.exists(part_path(pi))]
    print(f"cube {a.cube} mm (f {g['f']}, shell {g['shell']}, search +-{g['W']}): kept {len(kept)}/32, dropped "
          f"{dropped}; done {len(sel) - len(todo)}/{len(sel)}, to run {len(todo)}", flush=True)
    V, Mk = S.cache_volumes(t, affine)
    mc = BL.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    tcols = t[["person", "session"]].to_dict("list")
    tgcols = tg[["index", "person", "family", "sex", "age"]].to_dict("list")
    me = psutil.Process()
    peak = {"gb": None}                                  # peak working set of one cube worker, measured on cube 1

    def workers_mem():
        out = {}
        for c in me.children():
            try:
                mi = c.memory_info()
                out[c.pid] = (mi.wset / GB, mi.peak_wset / GB)
            except psutil.Error:
                pass
        return out

    def may_submit(n_running):
        """Memory gate (Amendment 10 resume): never start a cube that would push the machine into paging."""
        free = psutil.virtual_memory().available / GB
        if peak["gb"] is None:                           # measurement round: one cube alone
            return n_running == 0 and free - RESERVE_GB >= FIRST_CUBE_GB, free
        mem = workers_mem()
        # growth still to come counts only busy workers: idle pool workers keep a small working set and must not
        # count as a future peak (15 Nov: counting them held the run at 1 to 2 workers with 6.7 GB available);
        # the busy ones are taken as the n_running largest working sets
        busy = sorted((ws for ws, _ in mem.values()), reverse=True)[:n_running]
        growth = sum(max(0.0, peak["gb"] - ws) for ws in busy)
        return n_running < a.workers and free - RESERVE_GB - growth >= peak["gb"], free

    with ProcessPoolExecutor(a.workers, initializer=M._init, initargs=(affine,)) as ex:
        running, queue, n_done, last_wait = {}, list(todo), 0, 0.0
        while queue or running:
            while queue:
                ok, free = may_submit(len(running))
                if not ok:
                    break
                pi = queue.pop(0)
                paths = build_region(V, Mk, pos[pi][1], pos[pi][2], g["shell"], g["W"], mc, pi)
                job = (pi, tcols, tgcols, pos, only, {"shell": g["shell"], "W": g["W"], "paths": paths})
                running[ex.submit(run_position_geom, job)] = (pi, paths)
                pk_txt = "not yet measured" if peak["gb"] is None else f"{peak['gb']:.2f} GB"
                print(f"start position {pi}: running {len(running)}, available RAM {free:.1f} GB, worker peak "
                      f"{pk_txt}", flush=True)
            if not running:
                if time.time() - last_wait > 600:
                    print(f"WAIT {time.strftime('%H:%M')}: available RAM {psutil.virtual_memory().available / GB:.1f}"
                          f" GB is below reserve {RESERVE_GB} GB + one worker; not starting a cube", flush=True)
                    last_wait = time.time()
                time.sleep(60)
                continue
            done, _ = wait(running, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                continue
            pk = max((p for _, p in workers_mem().values()), default=0.0)
            if peak["gb"] is None or pk > peak["gb"]:
                peak["gb"] = pk
                print(f"worker peak working set {pk:.2f} GB (measured)", flush=True)
            for fut in done:
                pi, paths = running.pop(fut)
                part = fut.result()                      # a worker error stops the run here, loudly
                pd.DataFrame(part).to_csv(part_path(pi) + ".tmp", index=False, compression="gzip")
                os.replace(part_path(pi) + ".tmp", part_path(pi))
                for p in paths:
                    os.remove(p)
                n_done += 1
                el = time.time() - t0
                eta = el / n_done * (len(todo) - n_done)
                print(f"position {pi} ({n_done}/{len(todo)}): rows {len(part)}, {el / 60:.1f} min, ETA {eta / 60:.0f} "
                      f"min ({time.strftime('%H:%M', time.localtime(time.time() + eta))})", flush=True)
    new = pd.concat([pd.read_csv(part_path(pi)) for pi in sel], ignore_index=True)
    rep = L16.check_rows(new, stored_rows(a.cube, sel, a.smoke), [m for m in M.METRICS if m in new.columns])
    n16 = new[new.fill == "lookalike16"]
    info = {"cube_mm": a.cube, "f": g["f"], "positions": len(sel), "smoke_targets": a.smoke, "rows": len(new),
            "k1_reproduces_stored": rep["ok"], "rows_check": rep["rows_check"], "rows_matched": rep["rows_matched"],
            "max_abs_diff_mse": rep["max_abs_diff"].get("mse"), "n_lookalike16_rows": int(len(n16)),
            "n_donors_min": int(n16.n_donors.min()) if len(n16) else None,
            "minutes": round((time.time() - t0) / 60, 1), "workers": a.workers,
            "worker_peak_gb": None if peak["gb"] is None else round(peak["gb"], 2), "reserve_gb": RESERVE_GB}
    with open(os.path.join(PRIVATE, f"{tag}gate.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info), flush=True)
    if not rep["ok"]:
        raise SystemExit("the nearest of the 16 does not reproduce the stored k = 1 look-alike rows")


if __name__ == "__main__":
    main()
