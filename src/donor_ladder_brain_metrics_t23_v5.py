"""Amendment 22 (PROTOCOL_donor_ladder_brain.md): LPIPS VGG and set-level FID / FWD per rung on the two frozen brain
boxes, a new version next to Amendment 20 (whose rows and JSON stay untouched). Pre-QC, affine only.

Steps (each refuses to run on an incomplete input):
  --workers N            score the boxes (resumable chunks: rows CSV + crops NPZ per chunk, private folder v5)
  --smoke K              the same on the first K targets with a repeat scan (own file tag)
  --set-level            Inception features and FID / FWD per rung with a family bootstrap (needs every chunk)
  --summary-only         check the v5 rows against the v4 rows, summarise LPIPS VGG, write the public JSON
Missing is NaN, never zero: a crop that leaves E is not a sample; a slice that leaves E does not enter LPIPS.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain_metrics as M  # noqa: E402
import donor_ladder_brain_metrics_t23 as T  # noqa: E402

PRIVATE = os.path.join(M.B.PROJ, "donor-ladder-brain-metrics-20261109-v5")
PRIVATE_V4 = T.PRIVATE
OUT_JSON = os.path.join(M.B.ROOT, "results", "donor_ladder_brain_metrics_t23_v5.json")
HOOK = "vgg_set"
METRICS = ("lpips_vgg",)
DESCRIPTIVE = ("lpips_vgg_slices",)
REPRODUCE_COLS = ("mse", "ssim", "lddt", "mse_allvox")    # tier-1 columns that must equal the Amendment 20 rows
SET_PX = 64                                                # crops are upsampled to SET_PX x SET_PX
HAAR_LEVEL = 2
BOOT = 100
VIEW_OF = {32: "deep/LR", 33: "parietal/LR"}


# ---------------------------------------------------------------------------------------------------- set metrics

def to_unit(c):
    """z crops -> [0, 1] (z / 4 clipped to [-1, 1], as LPIPS)."""
    return (np.clip(np.asarray(c, np.float64) / T.LPIPS_Z, -1, 1) + 1) / 2


def upsample(c, px):
    """Bilinear upsampling of a batch of square crops (n, S, S) to (n, px, px)."""
    import torch
    t = torch.from_numpy(np.asarray(c, np.float32))[:, None]
    return torch.nn.functional.interpolate(t, size=(px, px), mode="bilinear", align_corners=False)[:, 0].numpy()


def haar_packets(a, level):
    """Orthonormal 2D Haar wavelet packets (n, H, W) -> (n, 4**level, H / 2**level, W / 2**level), as the face panel."""
    out = a[:, None].astype(np.float64)
    for _ in range(level):
        p, q = out[..., 0::2, :], out[..., 1::2, :]
        lo, hi = (p + q) / np.sqrt(2), (p - q) / np.sqrt(2)
        bands = []
        for r in (lo, hi):
            u, w = r[..., 0::2], r[..., 1::2]
            bands += [(u + w) / np.sqrt(2), (u - w) / np.sqrt(2)]
        out = np.stack(bands, 2).reshape(out.shape[0], -1, *bands[0].shape[-2:])
    return out


def frechet(A, B):
    """Frechet distance between Gaussian fits (mean, unbiased covariance) of the rows of A and B. Exact in both
    regimes: through the d x d covariances when d <= n, through the n_a x n_b cross-Gram matrix when d > n (the
    nonzero eigenvalues of S_a S_b are the squared singular values of A_c B_c^T / sqrt((n_a - 1)(n_b - 1)))."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    if len(A) < 2 or len(B) < 2:
        raise ValueError("Frechet distance needs at least 2 samples per set")
    ma, mb = A.mean(0), B.mean(0)
    Ac, Bc = A - ma, B - mb
    na, nb = len(A) - 1, len(B) - 1
    tr = (Ac ** 2).sum() / na + (Bc ** 2).sum() / nb
    if A.shape[1] <= min(len(A), len(B)):
        Sa, Sb = Ac.T @ Ac / na, Bc.T @ Bc / nb
        w, V = np.linalg.eigh(Sa)
        R = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
        cross = np.sqrt(np.clip(np.linalg.eigvalsh(R @ Sb @ R), 0, None)).sum()
    else:
        cross = np.linalg.svd(Ac @ Bc.T, compute_uv=False).sum() / np.sqrt(na * nb)
    return float(((ma - mb) ** 2).sum() + tr - 2 * cross)


def fwd(Pa, Pb):
    """FWD from precomputed packets (n, K, h, w): Frechet distance per packet, unweighted mean."""
    return float(np.mean([frechet(Pa[:, k].reshape(len(Pa), -1), Pb[:, k].reshape(len(Pb), -1))
                          for k in range(Pa.shape[1])]))


def inception_features(U, threads):
    """Inception-v3 pool features (pytorch-fid weights from the project TORCH_HOME) of [0, 1] images (n, H, W)."""
    os.environ["TORCH_HOME"] = T.TORCH_HOME
    import torch
    from pytorch_fid.inception import InceptionV3
    torch.set_num_threads(threads)
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(U), 64):
            t = torch.from_numpy(U[b:b + 64].astype(np.float32))[:, None].repeat(1, 3, 1, 1)
            out.append(net(t)[0].reshape(len(t), -1).numpy())
    return np.concatenate(out).astype(np.float64)


def cached_features(U, threads, pi, tag):
    """Inception features of one box, cached in PRIVATE so a crash after a box does not redo hours of CPU.
    The cache is used only if it was built from exactly these samples (sha256 of U)."""
    import hashlib
    h = hashlib.sha256(np.ascontiguousarray(U, dtype=np.float32).tobytes()).hexdigest()
    path = os.path.join(PRIVATE, f"set_features_box{pi}{tag}.npz")
    if os.path.exists(path):
        with np.load(path) as z:                     # closed at once: Windows cannot replace an open file
            same, F = str(z["sha256"]) == h, z["F"]
        if same:
            return F
        print(f"box {pi}: feature cache built from other samples, recomputing", flush=True)
    F = inception_features(U, threads)
    np.savez(path + ".part.npz", F=F, sha256=np.array(h))
    os.replace(path + ".part.npz", path)
    return F


def set_entry(Fa, Fb, Pa, Pb, fam, seed):
    """fid, fwd and n_cells of two paired sample sets, with 95% intervals from BOOT family resamples."""
    e = {"n_cells": int(len(Fa)), "n_families": int(len(np.unique(fam))),
         "fid": frechet(Fa, Fb), "fwd": fwd(Pa, Pb)}
    fams = np.unique(fam)
    members = {f: np.flatnonzero(fam == f) for f in fams}
    rng = np.random.default_rng(seed)
    bf, bw = [], []
    for _ in range(BOOT):
        ix = np.concatenate([members[f] for f in rng.choice(fams, len(fams))])
        bf.append(frechet(Fa[ix], Fb[ix]))
        bw.append(fwd(Pa[ix], Pb[ix]))
    e["fid_ci95"] = [float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))]
    e["fwd_ci95"] = [float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))]
    return e


# ---------------------------------------------------------------------------------------------------- run

def split_rows(part):
    """Rows from run_position -> (scalar rows DataFrame with a crop index, crops dict). Array keys start with '_'."""
    scal, xc, fc, ok = [], [], [], []
    for row in part:
        s = {k: v for k, v in row.items() if not k.startswith("_")}
        if "_xc" in row:
            s["crop"] = len(xc)
            xc.append(row["_xc"])
            fc.append(row["_fc"])
            ok.append(row["_ok"])
        else:
            s["crop"] = -1
        scal.append(s)
    S = T.SET_SIDE
    crops = {"xc": np.array(xc, np.float32).reshape(-1, 3, S, S), "fc": np.array(fc, np.float32).reshape(-1, 3, S, S),
             "ok": np.array(ok, bool).reshape(-1, 3)}
    return pd.DataFrame(scal), crops


def chunk_paths(tag, pi, c):
    base = os.path.join(PRIVATE, "parts", f"{tag}p{pi:02d}_c{c:02d}")
    return base + ".csv.gz", base + ".npz"


def setup(smoke):
    import nibabel as nib
    t = M.B.scans()
    pos, affine, _ = M.B.positions(np.random.default_rng(M.SEED))
    assert [pos[p][0] for p in T.FROZEN] == ["deep", "parietal"], "frozen box positions moved"
    tg = t[t.session == "ses-01"].reset_index()
    only = None
    if smoke:
        rep = set(t[t.session == "ses-02"].person)
        only = set(np.flatnonzero(tg.person.isin(rep).to_numpy())[:smoke].tolist())
    targets = sorted(only) if only is not None else list(range(len(tg)))
    chunks = [targets[k:k + T.CHUNK] for k in range(0, len(targets), T.CHUNK)]
    mc = M.BL.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    return t, tg, pos, affine, chunks, mc


def run(workers, smoke):
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(os.path.join(PRIVATE, "parts"), exist_ok=True)
    t0 = time.time()
    t, tg, pos, affine, chunks, mc = setup(smoke)
    lam = M.rung_lambdas()
    tcols = t[["person", "session"]].to_dict("list")
    tgcols = tg[["index", "person", "family", "sex", "age"]].to_dict("list")
    tag = f"smoke{smoke}_" if smoke else ""
    jobs = [((pi, tcols, tgcols, pos, mc, lam, HOOK, set(ch)), pi, c) for pi in T.FROZEN for c, ch in enumerate(chunks)
            if not os.path.exists(chunk_paths(tag, pi, c)[1])]
    n_all = len(T.FROZEN) * len(chunks)
    print(f"chunks done {n_all - len(jobs)}/{n_all}, to run {len(jobs)}", flush=True)
    with ProcessPoolExecutor(workers, initializer=M._init, initargs=(affine,)) as ex:
        for k, ((_, pi, c), part) in enumerate(zip(jobs, ex.map(M.run_position, [j[0] for j in jobs]))):
            rows, crops = split_rows(part)
            pc, pn = chunk_paths(tag, pi, c)
            rows.to_csv(pc + ".tmp", index=False, compression="gzip")
            np.savez_compressed(pn + ".tmp.npz", **crops)
            os.replace(pc + ".tmp", pc)
            os.replace(pn + ".tmp.npz", pn)                  # the npz is written last: it marks the chunk done
            el = time.time() - t0
            eta = el / (k + 1) * (len(jobs) - k - 1)
            print(f"chunk {k + 1}/{len(jobs)} (box {pi}, chunk {c}): rows {len(rows)}, {el / 60:.0f} min, "
                  f"ETA {eta / 60:.0f} min ({time.strftime('%H:%M', time.localtime(time.time() + eta))})", flush=True)
    timing = {"minutes": round((time.time() - t0) / 60, 2), "workers": workers, "smoke_targets": smoke}
    with open(os.path.join(PRIVATE, f"{tag}timing.json"), "w", encoding="utf-8") as fh:
        json.dump(timing, fh, indent=1)
    print(json.dumps(timing), flush=True)


def load(smoke):
    """All v5 rows and crops; refuses when a chunk is missing."""
    _, tg, pos, _, chunks, _ = setup(smoke)
    tag = f"smoke{smoke}_" if smoke else ""
    rows, crops = [], {}
    for pi in T.FROZEN:
        xs, fs, oks, off = [], [], [], 0
        for c in range(len(chunks)):
            pc, pn = chunk_paths(tag, pi, c)
            if not os.path.exists(pn):
                raise SystemExit(f"chunk box {pi} {c} missing: run the scoring first")
            r = pd.read_csv(pc)
            z = np.load(pn)
            r.loc[r.crop >= 0, "crop"] += off
            off += len(z["ok"])
            xs.append(z["xc"])
            fs.append(z["fc"])
            oks.append(z["ok"])
            rows.append(r)
        crops[pi] = (np.concatenate(xs), np.concatenate(fs), np.concatenate(oks))
    return pd.concat(rows, ignore_index=True), crops, tg, pos


def reproduce(r, smoke):
    """Tier-1 columns of v5 against the Amendment 20 rows on the shared keys (same fills -> same numbers)."""
    import glob
    old = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(PRIVATE_V4, "parts", "p3[23]_c*.csv.gz")))],
                    ignore_index=True)
    key = ["target", "patch", "side", "fill"]
    m = r[key + list(REPRODUCE_COLS)].merge(old[key + list(REPRODUCE_COLS)], on=key, suffixes=("", "_v4"))
    out = {"rows_v5": int(len(r)), "rows_matched": int(len(m))}
    for col in REPRODUCE_COLS:
        a, b = m[col].to_numpy(), m[col + "_v4"].to_numpy()
        both = np.isfinite(a) & np.isfinite(b)
        out[col] = {"max_abs_diff": float(np.max(np.abs(a[both] - b[both]))) if both.any() else None,
                    "nan_pattern_equal": bool((np.isfinite(a) == np.isfinite(b)).all())}
    out["ok"] = bool(len(m) == len(r) and all(out[c]["nan_pattern_equal"] and (out[c]["max_abs_diff"] or 0) < 1e-5
                                               for c in REPRODUCE_COLS))
    if not smoke and len(m) != len(r):
        out["ok"] = False
    return out


def set_level(smoke, threads):
    r, crops, tg, _ = load(smoke)
    fam_of = tg.family.to_numpy()
    out = {}
    for pi in T.FROZEN:
        xc, fc, ok = crops[pi]
        rr = r[(r.patch == pi) & (r.crop >= 0) & ~r.fill.map(M.is_reference_stranger)]
        # one sample per valid cell: (row, axis); target samples deduplicated by (target, side, axis)
        cells = [(i, row.crop, ax, row.target, row.side, row.fill) for i, row in enumerate(rr.itertuples())
                 for ax in range(3) if ok[row.crop, ax]]
        tkey = {}
        for _, cr, ax, tgt, side, _ in cells:
            tkey.setdefault((tgt, side, ax), (cr, ax))
        tlist = sorted(tkey)
        timg = np.stack([xc[tkey[k][0], tkey[k][1]] for k in tlist])
        fimg = np.stack([fc[cr, ax] for _, cr, ax, *_ in cells])
        U = to_unit(upsample(np.concatenate([timg, fimg]), SET_PX))
        t0 = time.time()
        F = cached_features(U, threads, pi, f"_smoke{smoke}" if smoke else "")
        P = haar_packets(U, HAAR_LEVEL)
        print(f"box {pi}: {len(U)} samples, inception {time.time() - t0:.0f}s", flush=True)
        tix = {k: n for n, k in enumerate(tlist)}
        nt = len(tlist)
        view = {}
        for fill in sorted({c[5] for c in cells}):
            sel = [(n, c) for n, c in enumerate(cells) if c[5] == fill]
            ia = np.array([tix[(c[3], c[4], c[2])] for _, c in sel])
            ib = np.array([nt + n for n, _ in sel])
            fam = fam_of[np.array([c[3] for _, c in sel])]
            if len(np.unique(fam)) < 10:
                view[fill] = {"n_cells": len(sel), "status": "fewer than 10 families: not computed"}
                continue
            view[fill] = set_entry(F[ia], F[ib], P[ia], P[ib], fam, [M.SEED, pi, len(view)])
            print(pi, fill, {k: v for k, v in view[fill].items() if not k.endswith("ci95")}, flush=True)
        tfam = fam_of[np.array([k[0] for k in tlist])]
        uf = np.random.default_rng([M.SEED, pi, 3000]).permutation(np.unique(tfam))
        ha = np.isin(tfam, uf[:len(uf) // 2])
        a, b = np.flatnonzero(ha), np.flatnonzero(~ha)
        view["_target_split_half"] = {"n_cells": [int(len(a)), int(len(b))], "fid": frechet(F[a], F[b]),
                                      "fwd": fwd(P[a], P[b]), "note": "two family-disjoint halves of the target set"}
        out[VIEW_OF[pi]] = view
    tag = f"_smoke{smoke}" if smoke else ""
    with open(os.path.join(PRIVATE, f"set_level{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(M.finite(out), fh, indent=1, allow_nan=False)
    print("set level written", flush=True)
    return out


def summary(smoke):
    r, _, tg, pos = load(smoke)
    rep = reproduce(r, smoke)
    print(json.dumps(rep), flush=True)
    if not rep["ok"]:
        raise SystemExit("v5 rows do not reproduce the Amendment 20 rows: no summary written")
    if r.duplicated(["target", "patch", "side", "fill"]).any():
        raise SystemExit("duplicate rows: no summary written")
    views, n_hyp = T.summarise(r, tg, pos, METRICS, DESCRIPTIVE)
    tag = f"_smoke{smoke}" if smoke else ""
    sp = os.path.join(PRIVATE, f"set_level{tag}.json")
    sl = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else None
    out = {"status": "pre-QC: affine registration only, MRI registration gate not met (results/registration/status.md)",
           "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendment 22)", "version": "v5 (Amendment 20 = v4, kept)",
           "seed": M.SEED, "boxes": [pos[p][0] for p in T.FROZEN], "metrics": list(METRICS),
           "lower_is_better": list(METRICS), "descriptive": list(DESCRIPTIVE),
           "lpips_vgg": {"net": "vgg", "version": "0.1", "short_side_px": T.LPIPS_SIDE, "z_scale": T.LPIPS_Z,
                         "slices": "central orthogonal, fully inside E (as Amendment 20)"},
           "set_level_method": {"cell": "(target, side, slice axis)", "sample": f"central {T.SET_SIDE} x {T.SET_SIDE} "
                                f"square of the central slice, fully inside E, z/4 clipped to [0, 1], {SET_PX} px",
                                "fid": "Inception-v3 pool 2048 (pytorch-fid), exact Frechet",
                                "fwd": f"Haar packets level {HAAR_LEVEL}, Frechet per packet, mean",
                                "interval": f"{BOOT} family bootstrap resamples, 2.5-97.5 percentiles",
                                "floor": "_target_split_half", "lower_is_better": True},
           "missing_rule": "missing is NaN, never zero; counts per view under 'missing'",
           "n_hypotheses_holm_run": n_hyp, "reproduction_vs_v4": rep,
           "not_run": {"topology_distance": "needs equal-sized groups; own amendment"},
           "counts": {"n_targets": int(r.target.nunique()), "n_rows": int(len(r))},
           "views": views, "set_level": sl if sl is not None else "not run"}
    path = OUT_JSON if not smoke else os.path.join(PRIVATE, f"summary_smoke{smoke}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(M.finite(out), fh, indent=1, allow_nan=False)
    print(f"saved {path}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0, help="first N targets that have a repeat scan")
    ap.add_argument("--set-level", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.set_level:
        set_level(a.smoke, a.workers)
    elif a.summary_only:
        summary(a.smoke)
    else:
        run(a.workers, a.smoke)


if __name__ == "__main__":
    main()
