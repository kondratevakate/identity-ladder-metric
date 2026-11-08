"""Metrics of Amendment 9.15 (general protocol) on the fundus donor ladder: LPIPS, Betti matching, FID and FWD.

Used by `fundus_ladder_metrics.py`. Weights and libraries come from the shared folder given by METRIC_MODELS
(no default; `models-20261107`): TORCH_HOME = <folder>/torch, Betti matching 2D in <folder>/betti-matching-2d,
pytorchfwd / ptwt / PyWavelets in <folder>/pytorchfwd-lib (installed there with --no-deps).
Every score uses only the scored pixels E; pixels a network needs outside the target's field are set identically in
the reference and the composite and are never scored (missing is never zero in a score).
"""
import os
import sys

import numpy as np
from scipy import ndimage

LP_CTX = 64                 # LPIPS context (px) around the 16 px window
TOPO_PER_UNIT = 2           # topology subset S: positions per target unit (Betti matching, FID, LPIPS-VGG)
SET_SHARE_MAX = 0.05        # LPIPS sensitivity: cells with at most 5% set pixels in the context
FWD_LEVEL = 2               # Haar packets of 4 x 4 on 16 px windows
FID_BOOT = 200
Z_CLIP = 3.0
NETS = {}
MODELS = None
WEIGHTS = ("alexnet-owt-7be5be79.pth", "vgg16-397923af.pth", "pt_inception-2015-12-05-6726825d.pth")


def setup(path=None):
    """Point torch and the libraries at the shared folder. The folder is required (no silent default)."""
    global MODELS
    path = path or os.environ.get("METRIC_MODELS")
    if not path or not os.path.isdir(path):
        raise SystemExit("METRIC_MODELS must point to the shared models-20261107 folder")
    ck = os.path.join(path, "torch", "hub", "checkpoints")
    missing = [w for w in WEIGHTS if not os.path.isfile(os.path.join(ck, w))]
    if missing:                                   # never let torch download silently
        raise SystemExit(f"weights missing in {ck}: {missing}")
    os.environ["TORCH_HOME"] = os.path.join(path, "torch")
    for sub in ("betti-matching-2d", "pytorchfwd-lib"):
        p = os.path.join(path, sub)
        if p not in sys.path:
            sys.path.append(p)                    # appended: never shadows this checkout's modules
    MODELS = path
    return path


def load_nets():
    import torch
    import lpips
    from pytorch_fid.inception import InceptionV3
    torch.set_num_threads(1)
    if "alex" not in NETS:
        NETS["alex"] = lpips.LPIPS(net="alex", spatial=True, verbose=False).eval()
        NETS["vgg"] = lpips.LPIPS(net="vgg", spatial=True, verbose=False).eval()
        NETS["incep"] = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
        import BettiMatching
        NETS["bm"] = BettiMatching
    return NETS


# ---------------------------------------------------------------------------------------------- inputs

def to01(z):
    return (np.clip(z, -Z_CLIP, Z_CLIP) + Z_CLIP) / (2 * Z_CLIP)


def to11(z):
    return np.clip(z, -Z_CLIP, Z_CLIP) / Z_CLIP


def network_input(x, m):
    """The target with pixels outside m set to its masked Gaussian local mean (sd 4 px; where the weight is below
    1e-3, sd 16 px; else the mean over m). Returns (image, set-pixel mask). The set pixels are never scored."""
    out = np.where(m, x, np.nan).astype(np.float64)
    w = m.astype(np.float64)
    xv = np.where(m, x, 0.0).astype(np.float64)
    for sd in (4.0, 16.0):
        g = ndimage.gaussian_filter(w, sd, mode="constant", truncate=3.0)
        s = ndimage.gaussian_filter(xv, sd, mode="constant", truncate=3.0)
        ok = np.isnan(out) & (g >= 1e-3)
        out[ok] = s[ok] / g[ok]
    if np.isnan(out).any():
        if not m.any():
            raise ValueError("target has no valid pixel")
        out[np.isnan(out)] = float(x[m].mean())
    return out, ~m


def context_box(y, x, win, size, ctx=LP_CTX):
    """Slices of the ctx x ctx context centred on the window, shifted to stay inside the frame."""
    cy = min(max(y + win // 2 - ctx // 2, 0), size - ctx)
    cx = min(max(x + win // 2 - ctx // 2, 0), size - ctx)
    return slice(cy, cy + ctx), slice(cx, cx + ctx)


def topo_cells(ti, full_positions, seed):
    """Topology subset S of one target: TOPO_PER_UNIT positions drawn among those with a wholly valid window."""
    full_positions = list(full_positions)
    if not full_positions:
        return set()
    rng = np.random.default_rng([seed, int(ti), 11])
    k = min(TOPO_PER_UNIT, len(full_positions))
    return {int(p) for p in rng.choice(full_positions, size=k, replace=False)}


# ---------------------------------------------------------------------------------------------- per-row metrics

def lpips_batch(refs, comps, masks, net="alex", chunk=64):
    """Mean of the spatial LPIPS map over each mask. refs, comps: (n, H, W) z images; masks: (n, H, W) bool."""
    import torch
    L = NETS[net]
    out = np.full(len(refs), np.nan)
    for i in range(0, len(refs), chunk):
        a = torch.from_numpy(to11(np.asarray(refs[i:i + chunk])).astype(np.float32))[:, None].repeat(1, 3, 1, 1)
        b = torch.from_numpy(to11(np.asarray(comps[i:i + chunk])).astype(np.float32))[:, None].repeat(1, 3, 1, 1)
        with torch.no_grad():
            d = L(a, b).numpy()[:, 0]
        for j, mk in enumerate(masks[i:i + chunk]):
            if mk.any():
                out[i + j] = float(d[j][mk].mean())
    return out


def betti_parts(a, b):
    """(matched, unmatched) parts of the Betti-matching loss between two windows in z (mapped to [0, 1]); the two
    terms of the library's `loss` (dimensions 0 and 1), infinite deaths set as the library does."""
    import torch
    BMod = NETS["bm"]
    m = BMod.BettiMatching(torch.tensor(to01(a)), torch.tensor(to01(b)), relative=False, filtration="sublevel",
                           construction="V", comparison="union")

    def coarse(cp, interval):
        a_, b_ = cp.fine_to_coarse(interval)
        if b_ == np.inf:
            b_ = 1
        elif b_ == -np.inf:
            b_ = 0
        return float(a_), float(b_)
    matched = unmatched = 0.0
    for dim in (0, 1):
        for (i0, _, i1) in m.matched[dim]:
            a0, b0 = coarse(m.CP_0, i0)
            a1, b1 = coarse(m.CP_1, i1)
            matched += 2 * ((a0 - a1) ** 2 + (b0 - b1) ** 2)
        for cp, un in ((m.CP_0, m.unmatched_0[dim]), (m.CP_1, m.unmatched_1[dim])):
            for i in un:
                a_, b_ = coarse(cp, i)
                unmatched += (a_ - b_) ** 2
    return matched, unmatched


def inception_features(wins, chunk=32):
    """InceptionV3 pool features (2048) of 16 px z windows mapped to [0, 1], three equal channels (the package
    resizes to 299 px bilinearly and maps to [-1, 1])."""
    import torch
    feats = []
    for i in range(0, len(wins), chunk):
        t = torch.from_numpy(to01(np.asarray(wins[i:i + chunk])).astype(np.float32))[:, None].repeat(1, 3, 1, 1)
        with torch.no_grad():
            feats.append(NETS["incep"](t)[0][:, :, 0, 0].numpy())
    return np.concatenate(feats, 0) if feats else np.zeros((0, 2048), np.float32)


# ---------------------------------------------------------------------------------------------- set-level metrics

def frechet_pkg(mu1, s1, mu2, s2):
    from pytorch_fid.fid_score import calculate_frechet_distance
    return float(calculate_frechet_distance(mu1, s1, mu2, s2))


def fid_point(F1, F2):
    F1, F2 = np.asarray(F1, np.float64), np.asarray(F2, np.float64)
    return frechet_pkg(F1.mean(0), np.cov(F1, rowvar=False), F2.mean(0), np.cov(F2, rowvar=False))


def fid_nuclear(F1, F2):
    """Exact FID with Tr sqrt(C1 C2) = nuclear norm of A1^T A2 (A = centred features / sqrt(n - 1)); n x n work."""
    F1, F2 = np.asarray(F1, np.float64), np.asarray(F2, np.float64)
    m1, m2 = F1.mean(0), F2.mean(0)
    A1 = (F1 - m1) / np.sqrt(len(F1) - 1)
    A2 = (F2 - m2) / np.sqrt(len(F2) - 1)
    nuc = np.linalg.svd(A1 @ A2.T, compute_uv=False).sum()
    return float(((m1 - m2) ** 2).sum() + (A1 ** 2).sum() + (A2 ** 2).sum() - 2 * nuc)


def wavelet_packets(wins):
    """Haar packets of level FWD_LEVEL of 16 px z windows mapped to [0, 1], one grey channel: (P, n, features)."""
    import torch
    from pytorchfwd.freq_math import forward_wavelet_packet_transform
    t = torch.from_numpy(to01(np.asarray(wins)).astype(np.float64))[:, None]
    p = forward_wavelet_packet_transform(t, "Haar", FWD_LEVEL, False)        # (n, P, C, h, w)
    n, P = p.shape[:2]
    return p.permute(1, 0, 2, 3, 4).reshape(P, n, -1).numpy()


def fwd_from_packets(P1, P2):
    from pytorchfwd.freq_math import calculate_frechet_distance
    d = [calculate_frechet_distance(P1[k].mean(0), np.cov(P1[k], rowvar=False),
                                    P2[k].mean(0), np.cov(P2[k], rowvar=False)) for k in range(len(P1))]
    return float(np.mean(d))


def set_level(recs, persons_all, seed, reps=FID_BOOT):
    """FID (topology subset S) and FWD (all full-window cells) per rung against the target windows of the same
    cells, with the ratio to the stranger rung and a person bootstrap. recs: list of dicts with unit, person, patch,
    fill ('_target' for the target window), win (16 x 16 z), feat (2048 or None)."""
    import pandas as pd
    if not recs:
        return {"fid": {}, "fwd": {}}
    df = pd.DataFrame([{k: r[k] for k in ("unit", "person", "patch", "fill")} | {"i": i} for i, r in enumerate(recs)])
    tgt = df[df.fill == "_target"].set_index(["unit", "patch"])["i"]
    wins = np.stack([r["win"] for r in recs]).astype(np.float64)
    has_feat = np.array([r["feat"] is not None for r in recs])
    feat = np.zeros((len(recs), 2048), np.float64)
    if has_feat.any():
        feat[has_feat] = np.stack([r["feat"] for r in recs if r["feat"] is not None])
    rng = np.random.default_rng(seed)
    persons = np.array(sorted(persons_all))
    bidx = rng.integers(0, len(persons), size=(reps, len(persons)))
    out = {"fid": {}, "fwd": {}}
    pairs = {}
    for f, g in df[df.fill != "_target"].groupby("fill"):
        key = list(zip(g.unit, g.patch))
        ti = tgt.reindex(pd.MultiIndex.from_tuples(key, names=["unit", "patch"])).to_numpy()
        ok = ~pd.isna(ti)
        pairs[f] = (g.i.to_numpy()[ok], ti[ok].astype(int), g.person.to_numpy()[ok])
    for metric in ("fid", "fwd"):
        vals, boots = {}, {}
        for f, (fi, ti, pp) in pairs.items():
            sel = has_feat[fi] & has_feat[ti] if metric == "fid" else np.ones(len(fi), bool)
            fi, ti, pp = fi[sel], ti[sel], pp[sel]
            if len(fi) < 50:
                out[metric][f] = {"status": "not run (fewer than 50 windows)", "n_windows": int(len(fi))}
                continue
            if metric == "fid":
                vals[f] = fid_point(feat[ti], feat[fi])
                rows_of = {p: np.flatnonzero(pp == p) for p in np.unique(pp)}
                bs = []
                for b in bidx:
                    ix = np.concatenate([rows_of.get(persons[k], np.zeros(0, int)) for k in b])
                    bs.append(fid_nuclear(feat[ti[ix]], feat[fi[ix]]) if len(ix) > 2 else np.nan)
            else:
                Pt, Pf = wavelet_packets(wins[ti]), wavelet_packets(wins[fi])
                vals[f] = fwd_from_packets(Pt, Pf)
                rows_of = {p: np.flatnonzero(pp == p) for p in np.unique(pp)}
                bs = []
                for b in bidx:
                    ix = np.concatenate([rows_of.get(persons[k], np.zeros(0, int)) for k in b])
                    bs.append(fwd_from_packets(Pt[:, ix], Pf[:, ix]) if len(ix) > 2 else np.nan)
            boots[f] = np.array(bs)
            # Amendment 9.15 item 4a: a Frechet distance is >= 0; a negative value is numerical error of the matrix
            # square root, so it is set to 0 and the number of such values is reported.
            neg = {"point": int(vals[f] < 0), "bootstrap": int(np.sum(boots[f] < 0))}
            vals[f] = max(vals[f], 0.0)
            boots[f] = np.where(boots[f] < 0, 0.0, boots[f])
            out[metric][f] = {"negative_set_to_zero": neg, "value": vals[f], "ci95": [float(np.nanpercentile(boots[f], 2.5)),
                                                          float(np.nanpercentile(boots[f], 97.5))],
                              "n_windows": int(len(fi)), "n_persons": int(len(np.unique(pp)))}
        if "stranger" in vals and vals["stranger"] > 0:
            for f in vals:
                r = boots[f] / boots["stranger"]
                out[metric][f]["ratio_to_stranger"] = vals[f] / vals["stranger"]
                out[metric][f]["ratio_ci95"] = [float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))]
    return out
