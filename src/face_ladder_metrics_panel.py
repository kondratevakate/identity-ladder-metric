"""The rest of the 26-metric panel on the face ladder, frame v6f (docs/PROTOCOL_face_metrics.md, Amendment F2).

Adds to the frozen Amendment F1 run (same targets, positions, donors, fills, seeds and scored pixels E) the metrics
still missing on faces: MS-SSIM, LPIPS, amplitude vs matched mean (group A); intensity persistence, Betti matching
matched / unmatched, identity norm and PLDA on ArcFace (group B); plausibility (per row) and FID, FWD, Topology
Distance (group C, set-level). PFE and the latent metrics are blocked (reasons in BLOCKED).

Fills are rebuilt by `face_ladder_metrics.window_fills` (unchanged; its `ident` hook only reports the donor row) and
the rebuilt MSE of every row must equal the v1 rows before a new value is used. 2D functions shared with the fundus
and iris panels are imported unchanged by explicit path. Every path that changes behaviour is passed explicitly:
--version, FUNDUS_METRICS_SRC, BETTI_SRC and TORCH_HOME (no defaults). Missing is never zero: outside E nothing
enters a score, a network or a density (marginalised, or the cell is missing).

Rows, caches and QC images stay in the private folder; the public JSON holds aggregates only.
Usage:
  python src/face_ladder_metrics_panel.py prep  --version _v6f      raw ArcFace norms, PLDA and density models
  python src/face_ladder_metrics_panel.py rows  --version _v6f [--limit N]   per-row metrics (+ set-level patches)
  python src/face_ladder_metrics_panel.py set   --version _v6f [--limit N]   FID, FWD, Topology Distance
  python src/face_ladder_metrics_panel.py summary --version _v6f            merge into results/face_ladder_metrics.json
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

sys.path.insert(0, os.path.dirname(__file__))
import face_ladder_metrics as FM  # noqa: E402
import donor_ladder_brain_metrics as BM  # noqa: E402

V1 = FM.PRIVATE
PRIVATE = os.path.join(os.path.dirname(FM.PRIVATE), "face-ladder-metrics-20261108-v2")
SEED = FM.SEED
WIN = FM.WIN
LPIPS_SIDE, LPIPS_Z = 64, 4.0
BETTI_TARGETS = 100                 # evenly spaced targets per source (Amendment F2: 0.16 s per pair)
SET_CELLS, TD_GROUP = 1000, 500
PLDA_DIM = 128
EIG_FLOOR = 1e-6
NORM_GATE = 0.999
DENS_MIN = 100                      # complete pool windows per half and position for a density

NEW_ROW = ("ms_ssim", "lpips", "ampdev_dataset", "ampdev_cohort", "persistence", "betti_matched", "betti_unmatched",
           "id_norm", "id_plda", "plausibility")
LOWER = {"lpips", "ampdev_dataset", "ampdev_cohort", "persistence", "betti_matched", "betti_unmatched", "id_norm",
         "plausibility"}
SELF_ZERO = LOWER - {"plausibility"}
SELF_PER_ROW = {"id_plda": "id_plda_self", "plausibility": "plausibility_self"}   # d(self) differs per target / cell

BLOCK_PFE = ("needs a per-image variance from a trained PFE uncertainty head (Shi and Jain 2019, arXiv:1904.09658); "
             "none exists for the InsightFace w600k_r50 backbone; training one is a separate project")
BLOCK_LATENT = ("needs a face generator with an inversion encoder (e.g. StyleGAN2-FFHQ + encoder) for the 128 px grey "
                "masked frame, RGB re-rendering and a GPU (days on this CPU); LDDMM between two 16 px windows is a "
                "pairwise registration (fundus stop rule)")
# The 26 confirmatory metrics (fundus registry names) -> (group, column in rows v1 / v2 / set-level, blocked reason)
REGISTRY = {
    "mse": ("A fidelity", "mse", None), "rmse": ("A fidelity", "rmse", None), "psnr": ("A fidelity", "psnr", None),
    "mae": ("A fidelity", "mae", None), "ssim": ("A fidelity", "ssim", None),
    "ms_ssim": ("A fidelity", "ms_ssim", None), "lpips": ("A fidelity", "lpips", None),
    "acc_dataset": ("A anomaly", "acc_dataset", None), "acc_matched": ("A anomaly", "acc_cohort", None),
    "amplitude_dataset": ("A anomaly", "ampdev_dataset", None),
    "amplitude_matched": ("A anomaly", "ampdev_cohort", None),
    "lddt": ("B structure", "lddt", None), "intensity_persistence": ("B structure", "persistence", None),
    "betti_matching_matched": ("B structure", "betti_matched", None),
    "betti_matching_unmatched": ("B structure", "betti_unmatched", None),
    "id_angle": ("B identity", "arcface_cos", None), "id_norm": ("B identity", "id_norm", None),
    "id_pfe": ("B identity", None, BLOCK_PFE), "id_plda": ("B identity", "id_plda", None),
    "latent_pullback": ("B latent", None, BLOCK_LATENT), "latent_fisher_rao": ("B latent", None, BLOCK_LATENT),
    "latent_lddmm": ("B latent", None, BLOCK_LATENT),
    "plausibility": ("C reference-free", "plausibility", None), "fid": ("C reference-free", "set:fid", None),
    "fwd": ("C reference-free", "set:fwd", None), "topology_distance": ("C reference-free", "set:td", None),
}
assert len(REGISTRY) == 26
SET_SKIP = BM.is_reference_stranger            # str00-str15 are R anchors, not rungs, for the set-level metrics


def need_env(name):
    v = os.environ.get(name)
    if not v or not os.path.exists(v):
        raise SystemExit(f"{name} must point to an existing path (no default)")
    return v


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_fundus(path):
    """The fundus panel module by explicit path (read only); only its 2D ms_ssim and persistence_distance are used."""
    spec = importlib.util.spec_from_file_location("fundus_ladder_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


# ---------------------------------------------------------------------------------------------------- metrics

def to_unit(z):
    return np.clip((np.asarray(z, np.float64) / LPIPS_Z + 1) / 2, 0, 1)


def betti_parts(B, x, f, e):
    """Betti matching (library unchanged) split into its matched and unmatched loss parts, dimensions 0 and 1.
    Pixels outside e are 1 in both images (identical, enter last)."""
    a, b = np.where(e, to_unit(x), 1.0), np.where(e, to_unit(f), 1.0)
    bm = B.BettiMatching(a, b, relative=False, filtration="sublevel", comparison="union")

    def cap(iv, cp):
        s, t = cp.fine_to_coarse(iv)
        return s, (1.0 if t == np.inf else 0.0 if t == -np.inf else t)
    matched = unmatched = 0.0
    for dim in (0, 1):
        for i0, _, i1 in bm.matched[dim]:
            (a0, b0), (a1, b1) = cap(i0, bm.CP_0), cap(i1, bm.CP_1)
            matched += 2 * ((a0 - a1) ** 2 + (b0 - b1) ** 2)
        for iv in bm.unmatched_0[dim]:
            s, t = cap(iv, bm.CP_0)
            unmatched += (s - t) ** 2
        for iv in bm.unmatched_1[dim]:
            s, t = cap(iv, bm.CP_1)
            unmatched += (s - t) ** 2
    return float(matched), float(unmatched), bm


def ampdev(x, f, mu, e):
    xe, fe, me = x[e].astype(np.float64), f[e].astype(np.float64), mu[e].astype(np.float64)
    nx = np.linalg.norm(xe - me)
    return abs(1.0 - float(np.linalg.norm(fe - me) / nx)) if nx > 0 else np.nan


def gauss_nll(v, ok, mean, cov, cache=None, key=None):
    """Negative log-likelihood per pixel of the marginal Gaussian over the pixels in ok (flattened window).
    cache/key: optional dict and model key; the Cholesky factor of the marginal covariance is reused per pixel set."""
    s = np.flatnonzero(ok.ravel())
    if s.size < 10:
        return np.nan
    r = v.ravel()[s].astype(np.float64) - mean[s]
    ck = None if cache is None else (key, s.tobytes())
    L = cache.get(ck) if ck is not None else None
    if L is None:
        L = np.linalg.cholesky(cov[np.ix_(s, s)])
        if ck is not None:
            cache[ck] = L
    y = solve_triangular(L, r, lower=True)
    return float((0.5 * y @ y + np.log(np.diag(L)).sum()) / s.size + 0.5 * np.log(2 * np.pi))


def haar_packets(a, level=2):
    """Orthonormal 2D Haar wavelet packets of a batch (n, H, W): (n, 4**level, H / 2**level, W / 2**level), natural
    order of the four sub-bands at each split (LL, LH, HL, HH)."""
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


def frechet(m1, s1, m2, s2):
    from scipy import linalg
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    if np.iscomplexobj(covmean):
        if np.abs(covmean.imag).max() > 1e-3:
            raise ValueError("imaginary component in the Frechet square root")
        covmean = covmean.real
    d = m1 - m2
    return float(d @ d + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fwd(A, B_, level=2):
    """FWD between two sets of windows in [0, 1]: Frechet distance per Haar packet, unweighted mean."""
    pa, pb = haar_packets(A, level), haar_packets(B_, level)
    out = []
    for k in range(pa.shape[1]):
        xa, xb = pa[:, k].reshape(len(A), -1), pb[:, k].reshape(len(B_), -1)
        out.append(frechet(xa.mean(0), np.cov(xa, rowvar=False), xb.mean(0), np.cov(xb, rowvar=False)))
    return float(np.mean(out))


def mst_deaths(F_):
    """H0 Vietoris-Rips death times of a point cloud = minimum-spanning-tree edge lengths (Euclidean), sorted."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import cdist
    D = cdist(F_, F_)
    D[D == 0] = 1e-12                                   # coincident points stay connected (zero-length edge)
    np.fill_diagonal(D, 0)
    return np.sort(minimum_spanning_tree(D).data)


def topology_distance(Fr, Fg):
    """Horak et al. 2021: L2 distance of the sorted H0 longevity vectors; the essential bars cancel (inf - inf = 0)."""
    n = min(len(Fr), len(Fg))
    return float(np.linalg.norm(mst_deaths(Fr[:n]) - mst_deaths(Fg[:n])))


# ---------------------------------------------------------------------------------------------------- PLDA

def plda_fit(Z, person):
    """Two-covariance model by moments on centred data: W pooled within-person covariance (people with >= 2
    images), B covariance of person means minus mean(W / n_i); eigenvalues floored."""
    groups = pd.Series(np.arange(len(Z))).groupby(person).apply(list)
    multi = [g for g in groups if len(g) >= 2]
    if len(multi) < 20:
        raise ValueError(f"only {len(multi)} pool people with >= 2 images; PLDA not fitted")
    d = Z.shape[1]
    Sw, nw = np.zeros((d, d)), 0
    for g in multi:
        r = Z[g] - Z[g].mean(0)
        Sw += r.T @ r
        nw += len(g) - 1
    W = Sw / nw
    means = np.stack([Z[g].mean(0) for g in groups])
    ns = np.array([len(g) for g in groups])
    B = np.cov(means, rowvar=False) - W * np.mean(1.0 / ns)

    def floor(S):
        w, V = np.linalg.eigh((S + S.T) / 2)
        return (V * np.maximum(w, EIG_FLOOR)) @ V.T
    return floor(B), floor(W), len(multi), len(groups)


def plda_scorer(B, W):
    T = B + W
    S2 = np.block([[T, B], [B, T]])
    iS2, iT = np.linalg.inv(S2), np.linalg.inv(T)
    c = -0.5 * np.linalg.slogdet(S2)[1] + np.linalg.slogdet(T)[1]
    return {"iS2": iS2, "iT": iT, "c": c}


def plda_llr(P, a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    ab = np.concatenate([a, b])
    return float(-0.5 * ab @ P["iS2"] @ ab + 0.5 * a @ P["iT"] @ a + 0.5 * b @ P["iT"] @ b + P["c"])


# ---------------------------------------------------------------------------------------------------- prep

def prep(version):
    """Raw ArcFace norms of the cached images (gate: normalised rerun = stored embedding), the PLDA model per source
    (pool people only) and the per-position window densities (two pool halves)."""
    F = FM.load_face(version)
    snap = FM.snapshot(version)
    cache = os.path.join(V1, f"cache{version}")
    os.makedirs(PRIVATE, exist_ok=True)
    out = os.path.join(PRIVATE, f"prep{version}")
    os.makedirs(out, exist_ok=True)
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    E = np.load(os.path.join(cache, "emb.npy"))
    ident = np.load(os.path.join(F.OUT, snap["identity"][0]))
    emb_all, excluded = ident["emb"].astype(np.float32), ident["excluded"]
    key = {}
    for j in range(len(emb_all)):
        if np.isfinite(emb_all[j]).all():
            key.setdefault(emb_all[j].tobytes(), j)
    rows = np.array([key.get(E[i].tobytes(), -1) if np.isfinite(E[i]).all() else -1 for i in range(len(E))])
    print(f"cache rows matched to image rows: {(rows >= 0).sum()}/{len(rows)}", flush=True)
    imgs = json.load(open(os.path.join(F.OUT, snap["images"]), encoding="utf-8"))
    gate_path = os.path.join(out, "norm_gate.json")
    if os.path.exists(gate_path) and os.path.exists(os.path.join(out, "raw_norm.npy")):
        info = {"norm_gate": json.load(open(gate_path)), "plda": {}, "density": {}}
        print("raw norms reused", info["norm_gate"], flush=True)
    else:
        info = {"norm_gate": raw_norms(F, snap, imgs, rows, emb_all, out), "plda": {}, "density": {}}
    # PLDA per source on pool people (all analysable images of theirs, any pose), PCA to PLDA_DIM
    Z_cache = np.full((len(E), PLDA_DIM), np.nan)
    for src, d in desc.items():
        pool_people = sorted({im["person"] for im in imgs if im["src"] == src and im["role"] == "pool"})
        pset = set(pool_people)
        jj = [j for j, im in enumerate(imgs) if im["src"] == src and im["person"] in pset
              and j < len(emb_all) and np.isfinite(emb_all[j]).all() and not excluded[j]]
        Xp = emb_all[jj].astype(np.float64)
        person = [imgs[j]["person"] for j in jj]
        mu = Xp.mean(0)
        _, _, Vt = np.linalg.svd(Xp - mu, full_matrices=False)
        Pm = Vt[:PLDA_DIM].T
        Zp = (Xp - mu) @ Pm
        B, W, n_multi, n_people = plda_fit(Zp, person)
        np.savez(os.path.join(out, f"plda_{src}.npz"), B=B, W=W)
        rows_src = sorted({t[k] for t in d["targets"] for k in ("t", "rep", "sib") if t.get(k) is not None}
                          | set(range(*d["pool"])))
        Z_cache[rows_src] = (E[rows_src].astype(np.float64) - mu) @ Pm
        info["plda"][src] = {"n_images": len(jj), "n_people": n_people, "n_people_multi": n_multi, "dim": PLDA_DIM}
        print(src, info["plda"][src], flush=True)
    np.save(os.path.join(out, "plda_z.npy"), Z_cache)
    # window densities per (source, position), two seeded pool halves
    X = np.load(os.path.join(cache, "X.npy"), mmap_mode="r")
    M = np.load(os.path.join(cache, "M.npy"), mmap_mode="r")
    from sklearn.covariance import LedoitWolf
    for src, d in desc.items():
        valid = np.load(os.path.join(cache, f"{src}_valid.npy"))
        p0, p1 = d["pool"]
        half = pool_half(src, p1 - p0)
        pos = FM.positions(valid)
        means, covs, ns = np.zeros((2, len(pos), WIN * WIN)), np.zeros((2, len(pos), WIN * WIN, WIN * WIN)), []
        Xp, Mp = np.asarray(X[p0:p1], np.float64), np.asarray(M[p0:p1])
        for pk, (y0, x0) in enumerate(pos):
            sl = (slice(y0, y0 + WIN), slice(x0, x0 + WIN))
            vw = valid[sl].ravel()                      # E never leaves `valid`; outside it the model is unused
            full = Mp[:, sl[0], sl[1]].reshape(p1 - p0, -1)[:, vw].all(1)
            nn = []
            for h in (0, 1):
                A = Xp[full & (half == h)][:, sl[0], sl[1]].reshape(-1, WIN * WIN)[:, vw]
                if len(A) < DENS_MIN:
                    raise SystemExit(f"{src} position {pk}: {len(A)} complete pool windows in half {h}")
                lw = LedoitWolf().fit(A)
                means[h, pk, vw] = lw.location_
                covs[h, pk] = np.eye(WIN * WIN)
                covs[h, pk][np.ix_(vw, vw)] = lw.covariance_
                nn.append(int(len(A)))
            ns.append(nn)
        np.savez(os.path.join(out, f"density_{src}.npz"), mean=means, cov=covs)
        info["density"][src] = {"n_pool_windows_per_half_min": int(np.min(ns)), "n_positions": len(pos)}
        print(src, info["density"][src], flush=True)
    json.dump(info, open(os.path.join(out, "prep.json"), "w"), indent=1)


def raw_norms(F, snap, imgs, rows, emb_all, out):
    """Raw ArcFace norms of the cached images; gate: the normalised rerun equals the stored embedding."""
    import cv2
    import onnxruntime as ort
    from PIL import Image
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 3, 1
    sess = ort.InferenceSession(F.ARCFACE, sess_options=so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    want = np.unique(rows[rows >= 0])
    L = FM.npz_rows(os.path.join(F.OUT, snap["landmarks"]), "L", want).astype(np.float64)
    norm_of, cos_gate = {}, []
    t0 = time.time()
    for b in range(0, len(want), 32):
        part = want[b:b + 32]
        xs = []
        for k, j in enumerate(part):
            d = imgs[j]
            path = os.path.join(F.CELEBA, "raw", "img_align_celeba", d["file"]) if d["src"] == "celeba" \
                else os.path.join(F.IMDB, *d["file"].split("/"))
            rgb = np.asarray(Image.open(path).convert("RGB"), np.float32)
            crop = cv2.warpAffine(rgb, F.similarity(F.five_points(L[b + k]), F.ARCFACE_DST), (112, 112),
                                  flags=cv2.INTER_LINEAR, borderValue=0)
            xs.append(((crop - 127.5) / 127.5).transpose(2, 0, 1))
        e = sess.run(None, {name: np.stack(xs).astype(np.float32)})[0].astype(np.float64)
        for k, j in enumerate(part):
            n = float(np.linalg.norm(e[k]))
            norm_of[int(j)] = n
            cos_gate.append(float(e[k] @ emb_all[j] / (n * np.linalg.norm(emb_all[j]))))
        if b % 640 == 0:
            print(f"arcface {b + len(part)}/{len(want)} {time.time() - t0:.0f}s", flush=True)
    cos_gate = np.array(cos_gate)
    gate = {"n": int(cos_gate.size), "min_cos": float(cos_gate.min()), "median_cos": float(np.median(cos_gate))}
    print("norm gate", gate, flush=True)
    if gate["min_cos"] < NORM_GATE:
        raise SystemExit("rerun ArcFace does not reproduce the stored embeddings; no identity norm")
    raw_norm = np.array([norm_of.get(int(j), np.nan) if j >= 0 else np.nan for j in rows])
    np.save(os.path.join(out, "raw_norm.npy"), raw_norm)
    json.dump(gate, open(os.path.join(out, "norm_gate.json"), "w"))
    return gate


def pool_half(src, n):
    return (np.random.default_rng([SEED, FM.SRC_CODE[src], 5]).permutation(n) >= n // 2).astype(int)


# ---------------------------------------------------------------------------------------------------- rows

W2 = {}


def _init(state):
    from threadpoolctl import threadpool_limits
    W2["blas"] = threadpool_limits(1)                  # one BLAS thread per worker: 3 workers x 8 threads thrash
    FM._init(state["cache"])
    W2["FL"] = load_fundus(state["fundus"])
    sys.path.insert(0, state["betti"])
    import BettiMatching
    W2["B"] = BettiMatching
    os.environ["TORCH_HOME"] = state["torch_home"]
    import lpips
    import torch
    torch.set_num_threads(1)
    W2["lpips"], W2["torch"] = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval(), torch
    prepd = state["prep"]
    W2["norm"] = np.load(os.path.join(prepd, "raw_norm.npy"))
    W2["z"] = np.load(os.path.join(prepd, "plda_z.npy"))
    for src in ("celeba", "imdb"):
        p = np.load(os.path.join(prepd, f"plda_{src}.npz"))
        W2[f"plda_{src}"] = plda_scorer(p["B"], p["W"])
        dz = np.load(os.path.join(prepd, f"density_{src}.npz"))
        W2[f"dens_{src}"] = (dz["mean"], dz["cov"])


def lpips_batch(pairs):
    if not pairs:
        return []
    torch = W2["torch"]
    a = np.stack([p for p, _ in pairs] + [q for _, q in pairs]).astype(np.float32)
    t = torch.from_numpy(np.clip(a / LPIPS_Z, -1, 1))[:, None]
    t = torch.nn.functional.interpolate(t, size=LPIPS_SIDE, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    n = len(pairs)
    with torch.no_grad():
        return W2["lpips"](t[:n], t[n:]).reshape(-1).numpy().astype(float).tolist()


def score_position(args):
    """New metrics for one (source, window position) over the given targets; returns rows, set-level patches and the
    reproduction residual of MSE against v1."""
    src, pk, (y0, x0), targets, lam, pool, n_sub, shift_px, betti_set, set_cells, v1_mse, cell_cap = args
    FL, B = W2["FL"], W2["B"]
    sl = (slice(y0, y0 + WIN), slice(x0, x0 + WIN))
    valid = np.asarray(FM.WORK[f"{src}_valid"])
    ring = FM.ring_mask(y0, x0) & valid
    P = FM.pool_window(src, {"pool": pool}, sl, ring)
    P["shift_px"] = shift_px
    mu_full = np.asarray(FM.WORK[f"{src}_mean"])
    dmean, dcov = W2[f"dens_{src}"]
    half = pool_half(src, pool[1] - pool[0])
    plda, Z, NRM = W2[f"plda_{src}"], W2["z"], W2["norm"]
    rows, patches, lp_pairs, lp_idx, worst = [], {}, [], [], 0.0
    for ti, d in targets:
        if (np.asarray(FM.WORK["M"][d["t"]])[sl] & valid[sl]).mean() < FM.MIN_IN_MASK:
            continue
        donor, chol = {}, {}                           # Cholesky cache per target (pixel sets depend on its mask)
        xw, tv, fills = FM.window_fills(src, ti, d, pk, sl, ring, lam, P, n_sub, ident=lambda t, g: g)
        t = d["t"]
        xt_full = np.asarray(FM.WORK["X"][t], np.float32)
        mx_full = np.asarray(FM.WORK["M"][t]) & valid
        mu_c = np.asarray(FM.WORK[f"{src}_cells_m"][d["cell"]][sl]) if d["cell"] is not None else None
        in_cells = (ti, pk) in set_cells
        if in_cells and tv.all():
            patches[("target", ti)] = xw.astype(np.float16)
        for name, (f, ok, g) in fills.items():
            e = tv & ok
            if e.sum() < 10:
                continue
            f = np.asarray(f, np.float32)
            mse = float(np.mean((f[e].astype(np.float64) - xw[e]) ** 2))
            ref = v1_mse.get((ti, name))
            if ref is None or not np.isclose(mse, ref, rtol=0, atol=1e-5):
                worst = np.inf if ref is None else max(worst, abs(mse - ref))
            ff, ef = np.zeros_like(xt_full), np.zeros(xt_full.shape, bool)
            ff[sl], ef[sl] = f, e
            row = {"src": src, "target": ti, "family": d["family"], "patch": pk, "fill": name, "mse": mse,
                   "ms_ssim": FL.ms_ssim(xt_full, mx_full, ff, ef, sl),
                   "lpips": np.nan, "ampdev_dataset": ampdev(xw, f, mu_full[sl], e),
                   "ampdev_cohort": ampdev(xw, f, mu_c, e) if mu_c is not None else np.nan,
                   "persistence": FL.persistence_distance(xw, f, e),
                   "betti_matched": np.nan, "betti_unmatched": np.nan,
                   "id_norm": np.nan, "id_plda": np.nan, "id_plda_self": plda_llr(plda, Z[t], Z[t])}
            if ti in betti_set:
                row["betti_matched"], row["betti_unmatched"], _ = betti_parts(B, xw, f, e)
            is_img = isinstance(g, (int, np.integer))
            if is_img:
                row["id_norm"] = abs(NRM[g] - NRM[t]) if np.isfinite(NRM[g]) and np.isfinite(NRM[t]) else np.nan
                row["id_plda"] = plda_llr(plda, Z[g], Z[t])
            # plausibility: a pool donor by the model of the other half, everything else by the mean of both
            hs = [1 - half[g - pool[0]]] if is_img and pool[0] <= g < pool[1] else [0, 1]
            row["plausibility"] = float(np.mean([gauss_nll(f, e, dmean[h, pk], dcov[h, pk], chol, h) for h in hs]))
            row["plausibility_self"] = float(np.mean([gauss_nll(xw, e, dmean[h, pk], dcov[h, pk], chol, h)
                                                      for h in (0, 1)]))
            if e.all():
                lp_pairs.append((xw, f))
                lp_idx.append(len(rows))
                if in_cells and not SET_SKIP(name):
                    patches[(name, ti)] = f.astype(np.float16)
            rows.append(row)
        if len(lp_pairs) >= cell_cap:
            for k, v in zip(lp_idx, lpips_batch(lp_pairs)):
                rows[k]["lpips"] = v
            lp_pairs, lp_idx = [], []
    for k, v in zip(lp_idx, lpips_batch(lp_pairs)):
        rows[k]["lpips"] = v
    return rows, {(src, pk, n, ti): p for (n, ti), p in patches.items()}, worst


CHUNK = 50                                             # targets per saved job: a crash loses one chunk only


def target_chunks(sel, size):
    """Consecutive chunks of at most `size` targets covering `sel` once, in order."""
    sel = [int(t) for t in sel]
    return [sel[k:k + size] for k in range(0, len(sel), size)]


def run_rows(version, limit, workers):
    state = {"cache": os.path.join(V1, f"cache{version}"), "fundus": need_env("FUNDUS_METRICS_SRC"),
             "betti": need_env("BETTI_SRC"), "torch_home": need_env("TORCH_HOME"),
             "prep": os.path.join(PRIVATE, f"prep{version}")}
    if workers > 3:
        raise SystemExit("at most 3 workers (shared CPU)")
    desc = json.load(open(os.path.join(state["cache"], "desc.json")))["data"]
    lam = FM.lambdas(version)
    tag = version + (f"_n{limit}" if limit else "")
    v1 = pd.read_csv(os.path.join(V1, f"rows{version}.csv.gz"), usecols=["src", "target", "patch", "fill", "mse"])
    jobs = []
    meta = {"version": version, "sources": {}, "sha256": {"fundus_metrics_source": sha(state["fundus"]),
                                                          "betti_matching": sha(os.path.join(state["betti"], "BettiMatching.py")),
                                                          "face_ladder_metrics": sha(FM.__file__)}}
    names = []
    for src, d in desc.items():
        valid = np.load(os.path.join(state["cache"], f"{src}_valid.npy"))
        pos = FM.positions(valid)
        T = d["targets"]
        sel = np.unique(np.linspace(0, len(T) - 1, limit).round().astype(int)) if limit else np.arange(len(T))
        bsel = set(np.unique(np.linspace(0, len(T) - 1, BETTI_TARGETS).round().astype(int)).tolist())
        if limit:
            bsel = set(sel.tolist())                        # smoke: Betti on every smoke target
        rng = np.random.default_rng([SEED, FM.SRC_CODE[src], 1000])
        allc = [(int(ti), pk) for ti in sel for pk in range(len(pos))]
        cells = set(map(tuple, np.array(allc)[rng.choice(len(allc), min(SET_CELLS, len(allc)), replace=False)].tolist()))
        n_sub = min(FM.POOL_SUB, d["n_pool"])
        vs = v1[(v1.src == src) & v1.target.isin(sel)]
        for pk, yx in enumerate(pos):
            vp = vs[vs.patch == pk]
            vm = {(int(a), b): float(c) for a, b, c in zip(vp.target, vp.fill, vp.mse)}
            for c, ch in enumerate(target_chunks(sel, CHUNK)):
                vmc = {k: v for k, v in vm.items() if k[0] in set(ch)}
                jobs.append((src, pk, tuple(yx), [(int(ti), T[ti]) for ti in ch], lam[src], tuple(d["pool"]), n_sub,
                             d["shift_px"], bsel, cells, vmc, 512))
                names.append(f"{src}_p{pk:02d}_c{c:02d}")
        meta["sources"][src] = {"n_targets_scored": int(len(sel)), "n_betti_targets": len(bsel & set(sel.tolist())),
                                "n_set_cells": len(cells), "n_positions": len(pos)}
    print(json.dumps(meta["sources"]), flush=True)
    t0 = time.time()
    # resumable: every (source, position) job is saved on its own, so a crash loses one position only
    parts = os.path.join(PRIVATE, f"parts{tag}")
    os.makedirs(parts, exist_ok=True)
    # pickles written by this function into the private folder only (never untrusted input)
    part = lambda k: os.path.join(parts, f"{names[k]}.pkl")  # noqa: E731
    todo = [k for k in range(len(jobs)) if not os.path.exists(part(k))]
    print(f"jobs done {len(jobs) - len(todo)}/{len(jobs)}, to run {len(todo)}", flush=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(state,)) as ex:
        for n, (k, res) in enumerate(zip(todo, ex.map(score_position, [jobs[k] for k in todo]))):
            with open(part(k) + ".tmp", "wb") as fh:
                pickle.dump(res, fh)
            os.replace(part(k) + ".tmp", part(k))
            el = time.time() - t0
            eta = el / (n + 1) * (len(todo) - n - 1)
            print(f"job {n + 1}/{len(todo)} ({names[k]}): rows {len(res[0])} repro {res[2]:.2e}, "
                  f"{el / 60:.0f} min, ETA {eta / 60:.0f} min ({time.strftime('%H:%M', time.localtime(time.time() + eta))})",
                  flush=True)
    rows, patches, worst = [], {}, 0.0
    for k in range(len(jobs)):
        with open(part(k), "rb") as fh:
            r, p, w = pickle.load(fh)
        rows += r
        patches.update(p)
        worst = max(worst, w)
    n_v1 = int(sum(len(j[10]) for j in jobs))
    meta["reproduction"] = {"max_abs_mse_diff": worst, "n_rows": len(rows), "n_rows_v1": n_v1,
                            "passed": bool(worst <= 1e-5 and len(rows) == n_v1)}
    print(json.dumps(meta["reproduction"]), flush=True)
    if not meta["reproduction"]["passed"]:
        raise SystemExit("rebuilt fills do not reproduce the v1 rows; nothing written")
    os.makedirs(PRIVATE, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(PRIVATE, f"rows{tag}.csv.gz"), index=False)
    keys = sorted(patches)
    np.savez_compressed(os.path.join(PRIVATE, f"set_patches{tag}.npz"), patches=np.stack([patches[k] for k in keys]),
                        keys=np.array([f"{s}|{pk}|{n}|{ti}" for s, pk, n, ti in keys]))
    meta["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(meta, open(os.path.join(PRIVATE, f"meta{tag}.json"), "w"), indent=1)
    print("rows written", meta["minutes"], "min", flush=True)


# ---------------------------------------------------------------------------------------------------- set level

def inception_features(P):
    os.environ["TORCH_HOME"] = need_env("TORCH_HOME")
    import torch
    from pytorch_fid.inception import InceptionV3
    torch.set_num_threads(3)
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(P), 64):
            t = torch.from_numpy(to_unit(P[b:b + 64]).astype(np.float32))[:, None].repeat(1, 3, 1, 1)
            out.append(net(t)[0].reshape(len(t), -1).numpy())
    return np.concatenate(out).astype(np.float64)


def run_set(version, limit):
    tag = version + (f"_n{limit}" if limit else "")
    z = np.load(os.path.join(PRIVATE, f"set_patches{tag}.npz"))
    P, keys = z["patches"].astype(np.float64), [k.split("|") for k in z["keys"]]
    t0 = time.time()
    feats = inception_features(P)
    print(f"inception {len(P)} patches {time.time() - t0:.0f}s", flush=True)
    idx = {}
    for i, (s, pk, n, ti) in enumerate(keys):
        idx.setdefault((s, n), {})[(int(ti), int(pk))] = i
    out = {}
    for src in sorted({s for s, _ in idx}):
        tgt = idx[(src, "target")]
        out[src] = {}
        for (s, n), cells in sorted(idx.items()):
            if s != src:
                continue
            common = sorted(set(cells) & set(tgt))
            if len(common) < 2 * TD_GROUP // 5:
                out[src][n] = {"n_cells": len(common), "status": "too few complete windows"}
                continue
            ia, ib = [tgt[c] for c in common], [cells[c] for c in common]
            fa, fb = feats[ia], feats[ib]
            e = {"n_cells": len(common)}
            e["fid"] = frechet(fa.mean(0), np.cov(fa, rowvar=False), fb.mean(0), np.cov(fb, rowvar=False))
            e["fwd"] = fwd(to_unit(P[ia]), to_unit(P[ib]))
            order = np.random.default_rng([SEED, FM.SRC_CODE[src], 2000]).permutation(len(common))
            groups = [order[k:k + TD_GROUP] for k in range(0, len(order) - TD_GROUP + 1, TD_GROUP)] or [order]
            e["td"] = float(np.mean([topology_distance(fa[g], fb[g]) for g in groups]))
            e["td_groups"] = len(groups)
            out[src][n] = e
            print(src, n, e, flush=True)
        # floor: two disjoint halves of the target set
        tk = sorted(tgt)
        h = np.random.default_rng([SEED, FM.SRC_CODE[src], 3000]).permutation(len(tk))
        a, b = [tgt[tk[i]] for i in h[:len(tk) // 2]], [tgt[tk[i]] for i in h[len(tk) // 2:]]
        out[src]["_target_split_half"] = {
            "n_cells": len(a), "fid": frechet(feats[a].mean(0), np.cov(feats[a], rowvar=False),
                                              feats[b].mean(0), np.cov(feats[b], rowvar=False)),
            "fwd": fwd(to_unit(P[a]), to_unit(P[b])), "td": topology_distance(feats[a], feats[b])}
    json.dump(BM.finite(out), open(os.path.join(PRIVATE, f"set_level{tag}.json"), "w"), indent=1)
    print("set level written", round((time.time() - t0) / 60, 1), "min")


# ---------------------------------------------------------------------------------------------------- summary

def with_self_panel(r, metrics):
    """FM.with_self with the per-row self values of PLDA and plausibility (d(self) differs per target / cell)."""
    out = r.copy()
    out.loc[out.fill == "repeat", "fill"] = "retest"
    ref = r[r.fill == "str00"].copy()
    ref["fill"] = "repeat"
    for m in metrics:
        if m in SELF_PER_ROW:
            val = ref[SELF_PER_ROW[m]]
        else:
            val = 0.0 if m in SELF_ZERO else 1.0
        ref[m] = np.where(ref[m].notna(), val, np.nan)
    return pd.concat([out, ref], ignore_index=True)


def summarise(version, limit):
    tag = version + (f"_n{limit}" if limit else "")
    r = pd.read_csv(os.path.join(PRIVATE, f"rows{tag}.csv.gz"))
    meta = json.load(open(os.path.join(PRIVATE, f"meta{tag}.json")))
    BM.LOWER_BETTER.update(LOWER)
    FM.with_self = with_self_panel
    views = {}
    for src in ("celeba", "imdb"):
        rs = r[r.src == src]
        metrics = [m for m in NEW_ROW if rs[m].notna().any()]
        v = FM.summarise(rs, metrics)
        v["coverage"] = {m: {f: int(g[m].notna().sum()) for f, g in rs.groupby("fill")} for m in metrics}
        views[src] = v
    set_path = os.path.join(PRIVATE, f"set_level{tag}.json")
    set_level = json.load(open(set_path)) if os.path.exists(set_path) else None
    reg = {}
    for k, (g, col, why) in REGISTRY.items():
        if why:
            reg[k] = {"group": g, "status": "blocked", "reason": why}
        elif col.startswith("set:"):
            reg[k] = {"group": g, "status": "computed" if set_level else "not run", "column": col}
        else:
            reg[k] = {"group": g, "status": "computed", "column": col}
    reg["betti_matching_matched"]["subset"] = reg["betti_matching_unmatched"]["subset"] = \
        f"{BETTI_TARGETS} evenly spaced targets per source (Amendment F2)"
    reg["acc_matched"]["note"] = reg["amplitude_matched"]["note"] = "targets with a sex-age cell of >= 10 people"
    reg["plausibility"]["note"] = "reference-free: lower = more plausible, not closer; R_self descriptive"
    for k in ("fid", "fwd", "topology_distance"):
        reg[k]["note"] = "set-level per rung; window patches; no R"
    return views, set_level, reg, meta


def write_json(version, limit):
    views, set_level, reg, meta = summarise(version, limit)
    panel = {"amendment": "F2 (docs/PROTOCOL_face_metrics.md)", "rows": "face-ladder-metrics-20261108-v2",
             "reproduction_of_v1_rows": meta["reproduction"], "sha256": meta["sha256"],
             "counts": meta["sources"], "lower_is_better": sorted(LOWER),
             "views": views, "set_level": set_level}
    prep = json.load(open(os.path.join(PRIVATE, f"prep{version}", "prep.json")))
    panel["prep"] = prep
    if limit:
        path = os.path.join(PRIVATE, f"summary_panel{version}_n{limit}.json")
        json.dump(BM.finite({"panel_f2": panel, "metrics_confirmatory": reg}), open(path, "w"), indent=1,
                  allow_nan=False)
    else:
        path = FM.OUT_JSON
        out = json.load(open(path, encoding="utf-8"))
        if out.get("frame") != version:
            raise SystemExit(f"{path} holds frame {out.get('frame')}, not {version}")
        out["metrics_confirmatory"] = reg
        out["panel_f2"] = panel
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(BM.finite(out), fh, indent=1, allow_nan=False)
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prep", "rows", "set", "summary"))
    ap.add_argument("--version", required=True, help="face frame version, e.g. _v6f (no default)")
    ap.add_argument("--limit", type=int, default=0, help="evenly spaced targets per source (smoke)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    FM.snapshot(a.version)
    if a.stage == "prep":
        prep(a.version)
    elif a.stage == "rows":
        run_rows(a.version, a.limit, a.workers)
    elif a.stage == "set":
        run_set(a.version, a.limit)
    else:
        write_json(a.version, a.limit)


if __name__ == "__main__":
    main()
