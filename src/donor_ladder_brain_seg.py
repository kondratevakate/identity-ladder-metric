"""Segmentation metrics on the deep and parietal boxes (PROTOCOL_donor_ladder_brain.md, Amendment 15). Pre-QC.

Core-rung fills are rebuilt exactly as in `donor_ladder_brain_metrics.run_position` (same donors, same seeds),
put back into the target's whole scan in intensity units, segmented, and scored against the segmentation of the
unaltered scan inside each box. Volumes and labels stay in the private folder; the JSON holds aggregates only.

Steps (each resumable):  build -> segment -> score.   `--timing` measures one volume per segmenter (D8).
"""
import argparse
import json
import os
import shutil
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as BL  # noqa: E402
import donor_ladder_brain_metrics as M  # noqa: E402

SEED = B.SEED
PRIVATE = os.path.join(B.PROJ, "donor-ladder-brain-seg-20261106-v1")
OUT_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_seg.json")
BOXES = ("deep", "parietal")
CORE = ("repeat", "mirror", "sibling", "stranger", "lookalike", "lookalike_oracle", "mean_k2", "dataset_mean", "noise")
BLUR_SIGMAS = (1, 2, 3, 4)
MIN_LABEL_VOX = 50
D8_LIMIT_S = 300


def box_positions(pos):
    return {name: pi for pi, (name, _, _) in enumerate(pos) if name in BOXES}


def eligible_targets(t, tg, pos):
    """Targets with a valid repeat in both boxes, one per family (first by participant id), sorted by id."""
    pis = box_positions(pos)
    ok = np.ones(len(tg), bool)
    row_of = {(p, s): k for k, (p, s) in enumerate(zip(t.person, t.session))}
    for pi in pis.values():
        _, start, size = pos[pi]
        mask = np.load(BL.region_paths(pi)[1], mmap_mode="r")
        cube_sl = tuple(slice(BL.W + BL.SHELL, BL.W + BL.SHELL + s) for s in size)
        valid = np.array([mask[k, 0][cube_sl].mean() >= BL.VALID for k in range(len(t))])
        for i, person in enumerate(tg.person):
            j2 = row_of.get((person, "ses-02"))
            ok[i] &= bool(valid[tg["index"].iloc[i]]) and j2 is not None and bool(valid[j2])
    c = tg[ok].sort_values("person").drop_duplicates("family")
    return list(c.index)


def pick(eligible, n):
    if n >= len(eligible):
        return list(eligible)
    return [eligible[int(round(k))] for k in np.linspace(0, len(eligible) - 1, n)]


def box_fills(i, pi, t, tg, pos):
    """Core fills of target i in box pi (drawn side, scan frame): {name: (fill, valid)} plus the target cube/mask
    and the dataset mean. Mirrors run_position line by line for the rungs it shares."""
    _, start, size = pos[pi]
    vol = np.load(BL.region_paths(pi)[0], mmap_mode="r")
    mask = np.load(BL.region_paths(pi)[1], mmap_mode="r")
    shell, W = BL.SHELL, BL.W
    fam, sex, age = tg.family.to_numpy(), tg.sex.to_numpy(), tg.age.to_numpy()
    ix = tg["index"].to_numpy()
    n = len(tg)
    row_of = {(p, s): k for k, (p, s) in enumerate(zip(t.person, t.session))}
    win_sl = tuple(slice(W, W + s + 2 * shell) for s in size)
    cube_sl = tuple(slice(W + shell, W + shell + s) for s in size)
    sm = BL.shell_mask(size, shell)
    fo = 0                                                    # drawn side, read from the scan itself
    own, other = vol[:, fo], vol[:, 1 - fo]
    cube_all = np.asarray(own[(slice(None),) + cube_sl])
    mk_all = np.asarray(mask[:, fo][(slice(None),) + cube_sl]).astype(bool)
    valid_all = mk_all.reshape(len(cube_all), -1).mean(1) >= BL.VALID
    X, MK, valid = cube_all[ix], mk_all[ix], valid_all[ix]
    Mc = np.asarray(other[(slice(None),) + cube_sl])[ix]
    MM = np.asarray(mask[:, 1 - fo][(slice(None),) + cube_sl]).astype(bool)[ix]
    mvalid = MM.reshape(n, -1).mean(1) >= BL.VALID
    win = np.asarray(own[(slice(None),) + win_sl])[ix]
    wmask = np.asarray(mask[:, fo][(slice(None),) + win_sl]).astype(bool)[ix]
    C, CM = win[:, sm], wmask[:, sm]
    Xf, XM = X.reshape(n, -1), MK.reshape(n, -1)
    n_shell, n_cube = int(sm.sum()), Xf.shape[1]
    others = np.flatnonzero((fam != fam[i]) & valid)
    xi, ei = X[i], MK[i]
    mu_d, mu_d_ok = M.masked_mean(X[others], MK[others])
    pop = np.random.default_rng([SEED, i, pi, B.DRAWN.index("population")]).permutation(others)
    f = {"stranger": (X[pop[0]], MK[pop[0]])}
    f["mean_k2"] = M.masked_mean(X[pop[:2]], MK[pop[:2]])
    f["dataset_mean"] = (mu_d, mu_d_ok)
    d_s = M.masked_dist(C[others], CM[others], C[i], CM[i], n_shell)
    d_o = M.masked_dist(Xf[others], XM[others], Xf[i], XM[i], n_cube)
    for name, d in (("lookalike", d_s), ("lookalike_oracle", d_o)):
        if np.isfinite(d).any():
            j = others[int(np.argmin(d))]
            f[name] = (X[j], MK[j])
    if mvalid[i]:
        f["mirror"] = (Mc[i], MM[i])
    rel = np.flatnonzero((fam == fam[i]) & (np.arange(n) != i) & valid)
    sib = [j for j in rel if age[j] != age[i]]
    if sib:
        f["sibling"] = (X[sib[0]], MK[sib[0]])
    j2 = row_of.get((tg.person.iloc[i], "ses-02"))
    if j2 is not None and valid_all[j2]:
        f["repeat"] = (cube_all[j2], mk_all[j2])
    side = BL.side_of(start, size, AFFINE[0])
    e_nz = ei & mu_d_ok
    f["noise"] = (M.noise_fill(xi, mu_d, e_nz, np.random.default_rng([SEED, i, pi, BL.SIDES.index(side), 99])), e_nz)
    return f, xi, ei, mu_d, mu_d_ok, side


AFFINE = [None]


def box_slices(pos, pi):
    _, start, size = pos[pi]
    return tuple(slice(a, a + s) for a, s in zip(start, size))


def target_image(t, row):
    f = t.path.iloc[row]
    img = nib.load(f.replace("_brain.nii.gz", ".nii.gz"))
    v = img.get_fdata(dtype=np.float32)
    m = nib.load(f).get_fdata(dtype=np.float32) > 0.5
    return img, v, m, float(v[m].mean()), float(v[m].std() + 1e-6)


def build(t, tg, pos, targets, rows_mse):
    """Write the volumes of every target: oracle, core fills, blur sweep. Checks mse against the metric rows."""
    pis = box_positions(pos)
    manifest = []
    for i in targets:
        tid = f"t{i:03d}"
        d = os.path.join(PRIVATE, tid, "in")
        done = os.path.join(PRIVATE, tid, "fills.json")
        if os.path.exists(done):
            manifest.append(json.load(open(done)))
            continue
        os.makedirs(d, exist_ok=True)
        img, v, m, mean, sd = target_image(t, tg["index"].iloc[i])
        vols = {"oracle": v.copy()}
        info = {"target": int(i), "family": str(tg.family.iloc[i]), "boxes": {}}
        blurred = {s: gaussian_filter(v, s) for s in BLUR_SIGMAS}
        for box, pi in pis.items():
            sl = box_slices(pos, pi)
            fills, xi, ei, mu_d, mu_ok, side = box_fills(i, pi, t, tg, pos)
            tm = m[sl]
            if not np.array_equal(tm, ei):
                raise RuntimeError(f"{tid} {box}: target mask in the region array differs from the scan mask")
            if not np.allclose(((v[sl] - mean) / sd)[tm], xi[tm], atol=1e-3):
                raise RuntimeError(f"{tid} {box}: target z differs from the region array")
            binfo = {"side": side, "fills": {}}
            for name in CORE:
                if name not in fills:
                    continue
                fz, ok = fills[name]
                e = ei & ok
                ref = rows_mse.get((int(i), int(pi), side, name))
                mse = float(np.mean((xi[e] - fz[e]) ** 2))
                if ref is None or not np.isclose(mse, ref, rtol=1e-4, atol=1e-6):
                    raise RuntimeError(f"{tid} {box} {name}: mse {mse} does not reproduce the metric rows ({ref})")
                patch = np.where(ok, fz, np.where(mu_ok, mu_d, xi))   # missing fill -> dataset mean, never scored
                vol = vols.setdefault(name, v.copy())
                vol[sl] = np.where(tm, patch * sd + mean, v[sl])
                binfo["fills"][name] = {"n_scored": int(e.sum()), "n_box_brain": int(tm.sum()),
                                        "n_missing_filled_with_mean": int((tm & ~ok & mu_ok).sum()),
                                        "n_kept_truth": int((tm & ~ok & ~mu_ok).sum()), "mse": mse}
                np.save(os.path.join(PRIVATE, tid, f"valid_{box}_{name}.npy"), e)
            for s, bv in blurred.items():
                vol = vols.setdefault(f"blur_{s}", v.copy())
                vol[sl] = np.where(tm, bv[sl], v[sl])
            np.save(os.path.join(PRIVATE, tid, f"valid_{box}_blur.npy"), ei)
            info["boxes"][box] = binfo
        for name, vol in vols.items():
            nib.save(nib.Nifti1Image(vol.astype(np.float32), img.affine), os.path.join(d, f"{name}.nii.gz"))
        info["volumes"] = sorted(vols)
        json.dump(info, open(done, "w"), indent=1)
        manifest.append(info)
        print(f"built {tid}: {len(vols)} volumes", flush=True)
    return manifest


def segment(tool, targets, threads, parallel):
    import real_cases as RC
    for i in targets:
        tid = f"t{i:03d}"
        base = os.path.join(PRIVATE, tid)
        names = json.load(open(os.path.join(base, "fills.json")))["volumes"]
        out_dir = os.path.join(base, f"seg_{tool}")
        todo = [n for n in names if not os.path.exists(os.path.join(base, f"label_{tool}_{n}.npy"))]
        if not todo:
            continue
        t0 = time.time()
        segs, _ = RC.run_segmenter(tool, os.path.join(base, "in"), out_dir, todo, threads=threads, parallel=parallel)
        failed = [k for k, p in segs.items() if p is None]
        if failed:   # D8: one retry
            retry, _ = RC.run_segmenter(tool, os.path.join(base, "in"), out_dir + "_retry", failed, threads=threads)
            segs.update({k: p for k, p in retry.items() if p is not None})
        ref = nib.load(os.path.join(base, "in", "oracle.nii.gz"))
        for k, p in segs.items():
            if p is None:
                continue
            np.save(os.path.join(base, f"label_{tool}_{k}.npy"), to_grid(p, ref))
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.rmtree(out_dir + "_retry", ignore_errors=True)
        print(f"segmented {tid} {tool}: {len(todo)} volumes, failed {sorted(k for k, p in segs.items() if p is None)}, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)


def to_grid(path, ref):
    from nibabel.processing import resample_from_to
    seg = nib.load(path)
    if seg.shape[:3] != ref.shape or not np.allclose(seg.affine, ref.affine, atol=1e-3):
        seg = resample_from_to(seg, ref, order=0)
    return np.asarray(seg.dataobj).astype(np.int16)


def label_scores(pred, oracle, s):
    p, o = pred[s], oracle[s]
    labels = [int(v) for v, c in zip(*np.unique(o, return_counts=True)) if c >= MIN_LABEL_VOX and v != 0]
    dice = [2 * np.sum((p == v) & (o == v)) / (np.sum(p == v) + np.sum(o == v)) for v in labels]
    return {"dice_mean": float(np.mean(dice)) if dice else np.nan, "accuracy": float(np.mean(p == o)),
            "n_labels": len(labels)}


def to_aseg(lab):
    """DKT cortical parcels (1000s left, 2000s right) collapsed to aseg cortex (3, 42); other labels unchanged."""
    out = lab.copy()
    out[(lab >= 1000) & (lab < 2000)] = 3
    out[(lab >= 2000) & (lab < 3000)] = 42
    return out


def label_both(pred, oracle, s):
    """aseg-level and DKT-level scores from one aparc.DKTatlas+aseg label map."""
    a = label_scores(to_aseg(pred), to_aseg(oracle), s)
    d = label_scores(pred, oracle, s)
    return {"dice_mean": a["dice_mean"], "accuracy": a["accuracy"], "n_labels": a["n_labels"],
            "dice_dkt": d["dice_mean"], "accuracy_dkt": d["accuracy"], "n_labels_dkt": d["n_labels"]}


def score(tool, targets, pos):
    pis = box_positions(pos)
    rows = []
    for i in targets:
        base = os.path.join(PRIVATE, f"t{i:03d}")
        info = json.load(open(os.path.join(base, "fills.json")))
        lab = lambda n: os.path.join(base, f"label_{tool}_{n}.npy")  # noqa: E731
        if not os.path.exists(lab("oracle")):
            continue
        oracle = np.load(lab("oracle"))
        for box, pi in pis.items():
            sl = box_slices(pos, pi)
            ob = oracle[sl]
            names = list(info["boxes"][box]["fills"]) + [f"blur_{s}" for s in BLUR_SIGMAS] + ["oracle"]
            for n in names:
                if not os.path.exists(lab(n)):
                    rows.append({"target": i, "family": info["family"], "box": box, "fill": n, "failed": True})
                    continue
                vfile = f"valid_{box}_{n}.npy" if n in CORE else f"valid_{box}_blur.npy"
                s = np.load(os.path.join(base, vfile))
                rows.append({"target": i, "family": info["family"], "box": box, "fill": n, "failed": False,
                             **label_both(np.load(lab(n))[sl], ob, s)})
    return pd.DataFrame(rows)


def boot(vals, reps=2000, seed=SEED):
    vals = np.asarray(vals, float)
    idx = np.random.default_rng(seed).integers(0, len(vals), (reps, len(vals)))
    b = vals[idx].mean(1)
    return {"mean": float(vals.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "n": int(len(vals))}


def summarise(r, tool, timing):
    out = {"status": "pre-QC: affine registration only; descriptive (few targets)", "protocol":
           "docs/PROTOCOL_donor_ladder_brain.md Amendment 15", "segmenter": tool, "timing": timing,
           "n_targets": int(r.target.nunique()), "failed": int(r.failed.sum()), "boxes": {}}
    ok = r[~r.failed]
    for box in BOXES:
        b = ok[ok.box == box]
        piv = {m: b.pivot(index="target", columns="fill", values=m)
               for m in ("dice_mean", "accuracy", "dice_dkt", "accuracy_dkt") if m in b}
        res = {"blur": {}, "R": {}, "raw": {}}
        for m, p in piv.items():
            res["raw"][m] = {f: boot(p[f].dropna()) for f in p.columns}
            res["blur"][m] = {s: boot((p["oracle"] - p[f"blur_{s}"]).dropna()) for s in BLUR_SIGMAS
                              if f"blur_{s}" in p}
            d = 1 - p                                                       # distance
            idx = np.random.default_rng(SEED).integers(0, len(d), (2000, len(d)))
            Rr = {}
            for f in CORE:
                if f not in d or d[f].isna().all():
                    continue
                sub = d[["repeat", "stranger", f]].dropna()
                a = sub.to_numpy()
                def R(x):
                    den = x[:, 1].mean() - x[:, 0].mean()
                    return (x[:, 2].mean() - x[:, 0].mean()) / den if den != 0 else np.nan
                bs = np.array([R(a[k % len(a)]) for k in idx[:, :len(a)]])
                Rr[f] = {"R": float(R(a)), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                         "n": int(len(a))}
            res["R"][m] = Rr
        drop3 = res["blur"]["dice_mean"].get(3)
        res["segmenter_sees_blur"] = bool(drop3 and drop3["ci95"][0] > 0)
        out["boxes"][box] = res
    return out


def timing(t, tg, pos):
    """D8: one unaltered target scan through each segmenter (timing only, not a result)."""
    import real_cases as RC
    d = os.path.join(PRIVATE, "timing")
    os.makedirs(os.path.join(d, "in"), exist_ok=True)
    img, v, _, _, _ = target_image(t, tg["index"].iloc[0])
    nib.save(nib.Nifti1Image(v, img.affine), os.path.join(d, "in", "timing.nii.gz"))
    out = {}
    for tool in ("fastsurfer", "synthseg"):
        res, secs = RC.run_segmenter(tool, os.path.join(d, "in"), os.path.join(d, tool), ["timing"], threads=TH[0])
        out[tool] = {"seconds": round(secs, 1), "ok": res["timing"] is not None, "image": RC.SEGMENTERS[tool],
                     "threads": TH[0]}
        print(tool, out[tool], flush=True)
        if tool == "fastsurfer" and out[tool]["ok"] and secs <= D8_LIMIT_S:
            break
    f = out["fastsurfer"]
    out["d8_choice"] = "fastsurfer" if f["ok"] and f["seconds"] <= D8_LIMIT_S else "synthseg"
    json.dump(out, open(os.path.join(PRIVATE, "timing.json"), "w"), indent=1)
    print(out, flush=True)


TH = [3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("timing", "list", "build", "segment", "score"))
    ap.add_argument("--n-targets", type=int, required=False)
    ap.add_argument("--tool", choices=("fastsurfer", "synthseg"))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=1)
    a = ap.parse_args()
    TH[0] = a.threads
    os.makedirs(PRIVATE, exist_ok=True)
    t = B.scans()
    pos, affine, _ = B.positions(np.random.default_rng(SEED))
    AFFINE[0] = affine
    tg = t[t.session == "ses-01"].reset_index()
    if a.step == "timing":
        return timing(t, tg, pos)
    el = eligible_targets(t, tg, pos)
    if a.step == "list":
        print(json.dumps({"eligible": len(el), "targets": el}))
        return
    if a.n_targets is None:
        raise SystemExit("--n-targets is required (Amendment 15: set from the measured time per volume)")
    targets = pick(el, a.n_targets)
    if a.step == "build":
        r = pd.read_csv(os.path.join(M.PRIVATE, "rows.csv.gz"), usecols=["target", "patch", "side", "fill", "mse"])
        pis = set(box_positions(pos).values())
        r = r[r.patch.isin(pis) & r.fill.isin(CORE) & r.target.isin(targets)]
        rows_mse = {(int(a_), int(b_), c_, d_): float(e_) for a_, b_, c_, d_, e_ in r.itertuples(index=False)}
        build(t, tg, pos, targets, rows_mse)
        return
    if a.tool is None:
        raise SystemExit("--tool is required (the D8 choice from timing.json)")
    if a.step == "segment":
        return segment(a.tool, targets, a.threads, a.parallel)
    r = score(a.tool, targets, pos)
    r.to_csv(os.path.join(PRIVATE, f"scores_{a.tool}.csv"), index=False)
    tim = json.load(open(os.path.join(PRIVATE, "timing.json"))) if os.path.exists(os.path.join(PRIVATE, "timing.json")) else None
    out = summarise(r, a.tool, tim)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(M.finite(out), fh, indent=1, allow_nan=False)
    print(json.dumps({b: {"sees_blur": v["segmenter_sees_blur"]} for b, v in out["boxes"].items()}))


if __name__ == "__main__":
    main()
