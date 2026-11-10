"""Donor ladder for face photographs (docs/PROTOCOL_donor_ladder_face.md): landmarks, canonical frame, rung fills,
local-error ratio maps and the map figure. Runs in `.venv-face` (mediapipe). Private rows go to the data folder;
only aggregates are printed or written to the repository. Never prints names, file names or identity ids.

Stages: `select` (image lists), `landmarks` (MediaPipe), `frame` (alignment, masks, QC), `maps`, `figure`.
"""
import argparse
import json
import math
import os
import re

import numpy as np

DATA = os.environ.get("DATA_ROOT", r"C:\Projects\data")
CELEBA = os.path.join(DATA, "datasets", "biometrics", "celeba", "version-unverified")
IMDB = os.path.join(DATA, "datasets", "biometrics", "imdb-wiki", "version-unverified", "raw", "imdb_crop")
KIN = os.path.join(DATA, "projects", "identity-ladder-metric", "imdb-wiki-kinship-2026-11-04", "pairs.json")
MODEL = os.path.join(DATA, "projects", "identity-ladder-metric", "models", "mediapipe-face-landmarker-v2",
                     "face_landmarker.task")
OUT = os.path.join(DATA, "projects", "identity-ladder-metric", "face-ladder-20261105-v1")
PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 2026

SIZE = 128
EYE_L, EYE_R, MOUTH = (44.0, 52.0), (84.0, 52.0), (64.0, 92.0)  # canonical points (x, y) in the frame
IRIS_A, IRIS_B, LIP_UP, LIP_LO, NOSE = 468, 473, 13, 14, 1
POSE_MAX = 10.0  # degrees, |yaw| and |pitch|
WIN, GRID, K_LOOK, N_STRANGER = 16, 8, 16, 8
SUF = None  # analysed version: "" = v1 frame (Amendment 1), "_v2" ... "_v7"; must be set explicitly (CLI or caller)


def _version():
    """The analysed version. Version-dependent behaviour (error window, masks, rules) must never fall back silently to
    the oldest version: a script that imports this module without setting SUF fails here (6 Nov 2026: QC scripts drew
    v6 error maps with the v1 box window because SUF was left at its default)."""
    if SUF is None:
        raise RuntimeError("donor_ladder_face.SUF is not set; set it to the analysed version, e.g. f.SUF = '_v6'")
    return SUF
DUP_R = 0.9


# ---------------------------------------------------------------- geometry (pure functions, tested)

def similarity(src, dst):
    """Least-squares similarity (scale, rotation, translation) mapping src (n x 2) to dst; returns a 2 x 3 matrix."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    a = (s * d).sum() / (s ** 2).sum()
    b = (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum() / (s ** 2).sum()
    R = np.array([[a, -b], [b, a]])
    return np.hstack([R, (md - R @ ms)[:, None]])


def apply(M, pts):
    pts = np.asarray(pts, float)
    return pts @ M[:, :2].T + M[:, 2]


def mirror(img):
    """Flip about the frame midline x = (SIZE - 1) / 2, the facial midline by construction."""
    return img[..., ::-1]


def box(x, w=WIN):
    """Mean over a w x w window (same size, zero padding outside), via cumulative sums."""
    p = w // 2
    c = np.pad(x, ((p + 1, w - p - 1), (p + 1, w - p - 1))).cumsum(0).cumsum(1)
    return (c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]) / (w * w)


def local_error(a, b, mask):
    """Windowed squared difference inside the joint mask (mask-weighted mean in each window): a 16 px box up to
    v2, a Gaussian of the same standard deviation from v3 on (Amendment 4.7)."""
    m = mask.astype(float)
    if _version() >= "_v3":
        from scipy.ndimage import gaussian_filter
        sd = WIN / np.sqrt(12)
        num, den = gaussian_filter(((a - b) ** 2) * m, sd, mode="constant"), gaussian_filter(m, sd, mode="constant")
    else:
        num, den = box(((a - b) ** 2) * m), box(m)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), den


def pose_from_matrix(T):
    """Yaw, pitch, roll in degrees from MediaPipe's 4 x 4 canonical-to-observed transformation."""
    R = np.asarray(T)[:3, :3]
    yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])))
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


# ---------------------------------------------------------------- stage: select

def select():
    """Image lists for the landmark pass (private). CelebA: all test images plus 3 images of 2,500 train identities.
    IMDb: up to 20 single-face crops of every kinship identity plus 3 crops of 3,000 identities outside kinship."""
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=False)
    dl = os.path.join(CELEBA, "downloads")
    part = dict(l.split() for l in open(os.path.join(dl, "list_eval_partition.txt")))
    ident = dict(l.split() for l in open(os.path.join(dl, "identity_CelebA.txt")))
    attr_lines = open(os.path.join(dl, "list_attr_celeba.txt")).read().split("\n")
    names = attr_lines[1].split()
    attr = {l.split()[0]: l.split()[1:] for l in attr_lines[2:] if l.strip()}
    im_male, im_young = names.index("Male"), names.index("Young")
    test = sorted(k for k, v in part.items() if v == "2")
    by_id = {}
    for k, v in part.items():
        if v == "0":
            by_id.setdefault(ident[k], []).append(k)
    ids = sorted(i for i, ks in by_id.items() if len(ks) >= 3)
    pool_ids = rng.choice(ids, 2500, replace=False)
    pool = sorted(k for i in pool_ids for k in rng.choice(sorted(by_id[i]), 3, replace=False))
    ce = [{"src": "celeba", "role": "target" if k in set(test) else "pool", "file": k, "person": ident[k],
           "male": attr[k][im_male] == "1", "young": attr[k][im_young] == "1"} for k in test + pool]

    import scipy.io as sio
    r = sio.loadmat(os.path.join(IMDB, "imdb.mat"), squeeze_me=True)["imdb"]
    paths, fs, sf = r["full_path"].item(), r["face_score"].item(), r["second_face_score"].item()
    dob, taken, gender = r["dob"].item(), r["photo_taken"].item(), r["gender"].item()
    good = np.isfinite(fs) & ~np.isfinite(sf)
    nm = np.array([re.search(r"nm\d{7}", p).group(0) for p in paths])
    pairs = json.load(open(KIN, encoding="utf-8"))["pairs"]
    kin = {x for p in pairs for x in (p["a"], p["b"])}
    idx = {}
    for i in np.flatnonzero(good):
        idx.setdefault(nm[i], []).append(i)
    rows = []
    for person in sorted(kin & set(idx)):
        take = idx[person] if len(idx[person]) <= 20 else rng.choice(idx[person], 20, replace=False)
        rows += [(person, int(i), "target") for i in sorted(take)]
    others = sorted(p for p in idx if p not in kin and len(idx[p]) >= 3)
    for person in rng.choice(others, 3000, replace=False):
        rows += [(person, int(i), "pool") for i in sorted(rng.choice(idx[person], 3, replace=False))]
    def age(i):  # Matlab datenum to year; photo year minus birth year
        y = 1 + (dob[i] - 367) / 365.2425
        return float(int(taken[i]) - math.floor(y)) if dob[i] > 0 else float("nan")  # taken is unsigned
    im = [{"src": "imdb", "role": role, "file": paths[i], "person": person,
           "male": bool(gender[i] == 1) if np.isfinite(gender[i]) else None, "age": age(i)} for person, i, role in rows]
    with open(os.path.join(OUT, "images.json"), "x", encoding="utf-8") as f:
        json.dump(ce + im, f)
    print({"celeba_target_images": len(test), "celeba_pool_images": len(pool), "imdb_target_images":
           sum(r[2] == "target" for r in rows), "imdb_pool_images": sum(r[2] == "pool" for r in rows)})


# ---------------------------------------------------------------- stage: landmarks

def _worker(chunk):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from PIL import Image
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.IMAGE, num_faces=2,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    out = []
    for j, path in chunk:
        try:
            a = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=a))
        except Exception as e:  # noqa: BLE001 - recorded per image, counted in the summary
            out.append((j, "error:" + type(e).__name__, None, None, None))
            continue
        if len(r.face_landmarks) != 1:
            out.append((j, f"faces:{len(r.face_landmarks)}", None, None, None))
            continue
        h, w = a.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]], np.float32)
        bs = {c.category_name: c.score for c in r.face_blendshapes[0]}
        expr = (0.5 * (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)), bs.get("jawOpen", 0))
        out.append((j, "ok", pts, pose_from_matrix(r.facial_transformation_matrixes[0]), expr))
    return out


def landmarks(workers):
    from concurrent.futures import ProcessPoolExecutor
    imgs = json.load(open(os.path.join(OUT, "images.json"), encoding="utf-8"))
    full = [os.path.join(CELEBA, "raw", "img_align_celeba", d["file"]) if d["src"] == "celeba"
            else os.path.join(IMDB, *d["file"].split("/")) for d in imgs]
    jobs = list(enumerate(full))
    chunks = [jobs[i::workers * 8] for i in range(workers * 8)]
    n = len(imgs)
    L = np.full((n, 478, 2), np.nan, np.float32)
    P, E = np.full((n, 3), np.nan, np.float32), np.full((n, 2), np.nan, np.float32)
    status = [""] * n
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_worker, chunks):
            for j, st, pts, pose, expr in res:
                status[j] = st
                if st == "ok":
                    L[j], P[j], E[j] = pts, pose, expr
    np.savez_compressed(os.path.join(OUT, "landmarks.npz"), L=L, P=P, E=E, status=np.array(status))
    from collections import Counter
    c = Counter((d["src"], d["role"], s.split(":")[0] if s != "ok" else "ok") for d, s in zip(imgs, status))
    print(json.dumps({" ".join(k): v for k, v in sorted(c.items())}, indent=1))


# ---------------------------------------------------------------- stage: frame

def frame():
    """Align every usable image (one face, |yaw| and |pitch| < POSE_MAX) to the canonical frame; z-scored grey level
    inside the warped face-oval mask; quantitative QC residual; private contact sheet for a human check."""
    import cv2
    from mediapipe.tasks.python import vision
    from PIL import Image
    oval = sorted({c.start for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL})
    imgs = json.load(open(os.path.join(OUT, "images.json"), encoding="utf-8"))
    z = np.load(os.path.join(OUT, "landmarks.npz"))
    L, P, status = z["L"], z["P"], z["status"]
    ok = (status == "ok") & (np.abs(P[:, 0]) < POSE_MAX) & (np.abs(P[:, 1]) < POSE_MAX)
    idx = np.flatnonzero(ok)
    dst = np.array([EYE_L, EYE_R, MOUTH])
    X = np.zeros((len(idx), SIZE, SIZE), np.float16)
    Mk = np.zeros((len(idx), SIZE, SIZE), bool)
    res_pts = np.zeros((len(idx), len(oval) + 3, 2), np.float32)
    for n, j in enumerate(idx):
        d = imgs[j]
        path = os.path.join(CELEBA, "raw", "img_align_celeba", d["file"]) if d["src"] == "celeba" \
            else os.path.join(IMDB, *d["file"].split("/"))
        rgb = np.asarray(Image.open(path).convert("RGB"), np.float32)
        y = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        eyes = sorted([L[j, IRIS_A], L[j, IRIS_B]], key=lambda p: p[0])
        src = np.array([eyes[0], eyes[1], 0.5 * (L[j, LIP_UP] + L[j, LIP_LO])])
        M = similarity(src, dst)
        g = cv2.warpAffine(y, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
        inb = cv2.warpAffine(np.ones_like(y), M, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0) > 0
        poly = apply(M, L[j, oval]).astype(np.int32)
        m = np.zeros((SIZE, SIZE), np.uint8)
        cv2.fillPoly(m, [cv2.convexHull(poly)], 1)
        m = (m > 0) & inb
        v = g[m]
        X[n] = np.where(m, (g - v.mean()) / max(v.std(), 1e-6), 0)
        Mk[n] = m
        res_pts[n] = apply(M, L[j, [NOSE, 61, 291] + oval])
    pool = np.array([imgs[j]["role"] == "pool" for j in idx])
    med = np.median(res_pts[pool], 0)
    resid = np.linalg.norm(res_pts - med, axis=2).mean(1) / (EYE_R[0] - EYE_L[0])
    cut = float(np.percentile(resid[pool], 99))
    keep = resid <= cut
    np.savez_compressed(os.path.join(OUT, "frame.npz"), idx=idx[keep], X=X[keep], M=Mk[keep], resid=resid[keep])
    rng = np.random.default_rng(SEED)
    pick = rng.choice(np.flatnonzero(keep), 32, replace=False)
    tiles = [np.hstack([X[i].astype(float), mirror(X[i].astype(float))]) for i in pick]
    sheet = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(8)])
    sheet = np.clip((sheet + 3) / 6 * 255, 0, 255).astype(np.uint8)
    Image.fromarray(sheet).save(os.path.join(OUT, "qc_contact_sheet_aligned_and_mirrored.png"))
    from collections import Counter
    c = Counter((imgs[j]["src"], imgs[j]["role"]) for j in idx[keep])
    print(json.dumps({"landmark_ok": int((status == "ok").sum()), "pose_ok": int(ok.sum()),
                      "qc_residual_cut_iod": round(cut, 3), "excluded_by_qc": int((~keep).sum()),
                      "kept": {" ".join(k): v for k, v in sorted(c.items())},
                      "qc_residual_median_iod": round(float(np.median(resid[keep])), 3)}, indent=1))


# ---------------------------------------------------------------- stage: maps

def _corr(a, b, m):
    a, b = a[m].astype(np.float64), b[m].astype(np.float64)
    if a.size < 10 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def masked_mean(X, M):
    s, n = (X * M).sum(0), M.sum(0)
    return np.where(n > 0, s / np.maximum(n, 1), 0), n


def ratio_map(errs, dens, v_errs):
    """Sum over targets of a rung's local error / sum of the dataset-mean local error, pixelwise; NaN-safe."""
    e, v = np.nansum(errs, 0), np.nansum(v_errs, 0)
    return np.where(v > 0, e / np.where(v > 0, v, 1), np.nan)


def _pose_expr_dist(a, b):
    return np.sqrt(((a[:2] - b[:2]) / 10.0) ** 2).sum() + np.abs(a[2:] - b[2:]).sum()


LOOK_STEP, LOOK_RING = 4, 8


def lookalike_fill(Xt, Xp, valid, ring=LOOK_RING, step=LOOK_STEP, k=K_LOOK, Mt=None, Mp=None, min_have=8):
    """Full fill images: for every step x step block whose pixels are valid, the k pool images nearest to the target
    on the ring (width `ring`) around the 16 px window centred on the block; the window takes their mean, and
    overlapping windows are tent-blended (Amendment 3). Ring pixels outside `valid` are ignored. Blocks that cannot be
    filled stay NaN.

    With masks (Amendment 7 erratum): the ring distance is the mean squared difference over pixels inside both the
    target's and the candidate's mask (candidates sharing < half of the target's ring pixels are ineligible), and each
    pixel of the window is the mean over the neighbours whose mask contains it (unfilled if < `min_have`)."""
    half = WIN // 2
    masked = Mt is not None and Mp is not None
    acc = np.zeros(Xt.shape, np.float32)
    wsum = np.zeros(Xt.shape if masked else (SIZE, SIZE), np.float32)
    tent1 = 1 - np.abs(np.arange(WIN) - (WIN - 1) / 2) / (WIN / 2)
    tent = np.outer(tent1, tent1).astype(np.float32)  # Amendment 3: overlapping windows, tent-weighted
    r = half + ring
    yy, xx = np.mgrid[-r:r, -r:r]
    in_win = (np.abs(yy + 0.5) < half) & (np.abs(xx + 0.5) < half)
    if masked:
        Mt, Mp = Mt.astype(np.float32), Mp.astype(np.float32)
    for y0 in range(0, SIZE, step):
        for x0 in range(0, SIZE, step):
            if not valid[y0:y0 + step, x0:x0 + step].all():
                continue
            cy, cx = y0 + step // 2, x0 + step // 2
            ys, xs = yy + cy, xx + cx
            ok = (ys >= 0) & (ys < SIZE) & (xs >= 0) & (xs < SIZE)
            ok[ok] = valid[ys[ok], xs[ok]]
            sel = ok & ~in_win
            if sel.sum() < 16:
                continue
            Rt = Xt[:, ys[sel], xs[sel]]
            Rp = Xp[:, ys[sel], xs[sel]]
            wy0, wx0 = cy - half, cx - half
            sy = slice(max(wy0, 0), min(wy0 + WIN, SIZE))
            sx = slice(max(wx0, 0), min(wx0 + WIN, SIZE))
            ty = slice(sy.start - wy0, sy.stop - wy0)
            tx = slice(sx.start - wx0, sx.stop - wx0)
            w = tent[ty, tx] * valid[sy, sx]
            if not masked:
                d = (Rt ** 2).sum(1)[:, None] + (Rp ** 2).sum(1)[None] - 2 * Rt @ Rp.T
                nn = np.argpartition(d, k, axis=1)[:, :k]
                acc[:, sy, sx] += Xp[:, sy, sx][nn].mean(1) * w
                wsum[sy, sx] += w
                continue
            Wt, Wp = Mt[:, ys[sel], xs[sel]], Mp[:, ys[sel], xs[sel]]
            Rt, Rp = Rt * Wt, Rp * Wp  # stored zeros outside masks, made explicit
            cnt = Wt @ Wp.T
            num = (Rt ** 2) @ Wp.T + Wt @ (Rp ** 2).T - 2 * Rt @ Rp.T
            d = np.where(cnt >= 0.5 * Wt.sum(1, keepdims=True), num / np.maximum(cnt, 1), np.inf)
            nn = np.argpartition(d, k, axis=1)[:, :k]
            good = np.isfinite(np.take_along_axis(d, nn, 1)).astype(np.float32)  # (T, k)
            Pm = Mp[:, sy, sx][nn] * good[:, :, None, None]
            have = Pm.sum(1)
            val = (Xp[:, sy, sx][nn] * Pm).sum(1) / np.maximum(have, 1)
            wk = w[None] * (have >= min_have)
            acc[:, sy, sx] += val * wk
            wsum[:, sy, sx] += wk
    covered = (wsum > 1e-6) & valid
    return np.where(covered, acc / np.where(wsum > 1e-6, wsum, 1), np.nan).astype(np.float32)


def _families(src_rows, pairs):
    """Connected components over full-sibling pairs (IMDb), else one family per person."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    return {p: find(p) for p in src_rows}


def frame_suf():
    """Amendment 8.4: '_v7_p15' analyses frame '_v7' with a 15 deg pose filter."""
    return _version().split("_p")[0]


def map_pose_max():
    return int(_version().split("_p")[1]) if "_p" in _version() else POSE_MAX


def maps():
    imgs = json.load(open(os.path.join(OUT, "images.json"), encoding="utf-8"))
    lz = np.load(os.path.join(OUT, "landmarks.npz"))
    fz = np.load(os.path.join(OUT, f"frame{frame_suf()}.npz"))
    idx, X, M = fz["idx"], fz["X"], fz["M"]
    PE = np.hstack([lz["P"][idx, :2], lz["E"][idx]])
    meta = [imgs[j] for j in idx]
    pose_ok = (np.abs(PE[:, 0]) < map_pose_max()) & (np.abs(PE[:, 1]) < map_pose_max())  # Amendment 8.4
    attr = {}
    if SUF >= "_v7":  # CelebA published attributes for the facial-hair stratum (Amendment 7d)
        al = open(os.path.join(CELEBA, "downloads", "list_attr_celeba.txt")).read().split("\n")
        an = al[1].split()
        attr = {l.split()[0]: l.split()[an.index("No_Beard") + 1] == "1" for l in al[2:] if l.strip()}
    kin = json.load(open(KIN, encoding="utf-8"))["pairs"]
    lab = json.load(open(os.path.join(os.path.dirname(KIN), "qualifier_labels.json"), encoding="utf-8"))
    full_sib = [(p["a"], p["b"]) for p in kin if p["relation"] == "sibling"
                and not any("half" in lab.get(q, "") or "step" in lab.get(q, "") for q in p["kinship_qualifiers"])]
    rng = np.random.default_rng(SEED)
    result, store = {}, {}
    for src in ("celeba", "imdb"):
        rows = [i for i, d in enumerate(meta) if d["src"] == src and pose_ok[i]]
        by_person = {}
        for i in rows:
            by_person.setdefault(meta[i]["person"], []).append(i)
        best = {p: min(ii, key=lambda i: (PE[i, :2] ** 2).sum()) for p, ii in by_person.items()}
        pool_p = sorted(p for p, ii in by_person.items() if meta[ii[0]]["role"] == "pool")
        sibs = {}
        for a, b in full_sib:
            if a in by_person and b in by_person:
                sibs.setdefault(a, set()).add(b)
                sibs.setdefault(b, set()).add(a)
        kin_people = {x for q in kin for x in (q["a"], q["b"])}
        targets = sorted(p for p, ii in by_person.items() if meta[ii[0]]["role"] == "target" and len(ii) >= 2
                         and (src == "celeba" or (p in kin_people if SUF >= "_v3" else p in sibs)))
        fam_pairs = full_sib + ([(q["a"], q["b"]) for q in kin if q["relation"] == "parent-child"]
                                if SUF >= "_v3" else [])
        fam = _families(targets + pool_p, fam_pairs if src == "imdb" else [])
        fam_ids = sorted(set(fam.values()))
        half_of = dict(zip(fam_ids, rng.permutation(len(fam_ids)) % 2))
        res_src, maps_src = {}, {}
        for h in ("all", 0, 1):
            T = [p for p in targets if h == "all" or half_of[fam[p]] == h]
            PP = [p for p in pool_p if h == "all" or half_of[fam[p]] == h]
            pi = np.array([best[p] for p in PP])
            Xp, Mp = X[pi].astype(np.float32), M[pi]
            mean, cov = masked_mean(Xp, Mp)
            valid = cov >= 0.9 * len(pi)
            if SUF >= "_v6":  # Amendment 7: anatomical coverage >= 90%, unmasked coverage >= 50%
                valid = (fz["A"][pi].mean(0) >= 0.9) & (cov >= 0.5 * len(pi))
            def cell(i):
                d = meta[i]
                return (d["male"], d["young"]) if src == "celeba" else (d["male"], int(d["age"] // 5))
            cells = {}
            for k, i in zip(PP, pi):
                cells.setdefault(cell(i), []).append(i)
            cell_mean = {c: masked_mean(X[ii].astype(np.float32), M[ii])[0] for c, ii in cells.items() if len(ii) >= 10}
            cols = {c: [] for c in ("repeat", "repeat_random", "repeat_age_far", "mirror", "sibling", "stranger",
                                    "lookalike", "lookalike_ring4", "sexage", "mean")}
            stratum = {}
            tot = {c: [] for c in cols}
            Xall = X[[best[p] for p in T]].astype(np.float32)
            mk = {}
            if SUF >= "_v5":  # Amendment 7 erratum: masks enter the look-alike search and mean
                mk = {"Mt": M[[best[p] for p in T]], "Mp": Mp}
            LF = lookalike_fill(Xall, Xp, valid, **mk)
            LF4 = lookalike_fill(Xall, Xp, valid, ring=4, **mk) if h == "all" else None
            strip = 2.5 if SUF >= "_v4" else WIN // 2  # Amendment 5.4 (v4) / Amendment 1 (earlier)
            midline = np.abs(np.arange(SIZE) - (SIZE - 1) / 2) >= strip
            for ti, p in enumerate(T):
                t = best[p]
                xt, mt = X[t].astype(np.float32), M[t]
                others = [i for i in by_person[p] if i != t and _corr(X[i], X[t], M[i] & M[t]) <= DUP_R]
                if not others:
                    continue
                rep = min(others, key=lambda i: _pose_expr_dist(PE[i], PE[t]))
                rep_rand = others[rng.integers(len(others))]
                neutral = PE[t, 2] < SMILE_MAX and PE[t, 3] < JAW_MAX
                stratum[p] = ["neutral" if neutral else "expressive"]
                if SUF >= "_v7" and src == "imdb":  # Amendment 8.2: age explicit
                    gap = abs(meta[rep]["age"] - meta[t]["age"])
                    if np.isfinite(gap):
                        stratum[p].append("agegap_0_4" if gap < 5 else "agegap_5_14" if gap < 15 else "agegap_15plus")
                    aged = [i for i in others if np.isfinite(meta[i]["age"]) and np.isfinite(meta[t]["age"])]
                    rep_far = max(aged, key=lambda i: abs(meta[i]["age"] - meta[t]["age"])) if aged else None
                else:
                    rep_far = None
                if SUF >= "_v7" and src == "celeba" and meta[t]["male"]:  # Amendment 7d: published attribute only
                    stratum[p].append("men_no_facial_hair" if attr.get(meta[t]["file"], True) else "men_facial_hair")
                fills = {"repeat": [(X[rep], M[rep])], "repeat_random": [(X[rep_rand], M[rep_rand])], "mirror": [(mirror(xt), mirror(mt) & midline[None, :])],
                         "mean": [(mean, valid)],
                         **({"repeat_age_far": [(X[rep_far], M[rep_far])]} if rep_far is not None else {}),
                         "lookalike": [(np.nan_to_num(LF[ti]), np.isfinite(LF[ti]))]}
                if LF4 is not None:
                    fills["lookalike_ring4"] = [(np.nan_to_num(LF4[ti]), np.isfinite(LF4[ti]))]
                if src == "imdb" and p in sibs:
                    fills["sibling"] = [(X[s], M[s]) for s in
                                        (min(by_person[q], key=lambda i: _pose_expr_dist(PE[i], PE[t])) for q in sibs[p])]
                c = cell(t)
                if c in cell_mean:
                    fills["sexage"] = [(cell_mean[c], valid)]
                fills["stranger"] = [(X[i], M[i]) for i in rng.choice(pi, N_STRANGER, replace=False)]
                vt = mt & valid
                for col, fl in fills.items():
                    es, nums, dens = [], [], []
                    for f, mf in fl:
                        jm = vt & mf.astype(bool)
                        e, cover = local_error(xt, np.asarray(f, np.float32), jm)
                        es.append(np.where(valid & (cover >= 0.5), e, np.nan))
                        nums.append(((xt - f) ** 2 * jm).sum())
                        dens.append(((xt - mean) ** 2 * jm).sum())  # same pixels for the dataset-mean reference
                    cols[col].append(np.nanmean(es, 0))
                    tot[col].append((p, (float(np.mean(nums)), float(np.mean(dens)))))
                for col in cols:
                    if col not in fills:  # keep targets aligned across columns
                        cols[col].append(np.full((SIZE, SIZE), np.nan))
            vmap = np.array(cols["mean"])
            rm = {c: ratio_map(np.array(v), None, np.where(np.isfinite(np.array(v)), vmap[:len(v)], np.nan))
                  for c, v in cols.items() if sum(np.isfinite(m).any() for m in v) >= 10}
            maps_src[str(h)] = {c: m for c, m in rm.items()}
            maps_src[str(h)]["V"] = np.nanmean(vmap, 0)
            maps_src[str(h)]["meanface"] = mean
            maps_src[str(h)]["valid"] = valid
            if h == "all":
                for c, v in tot.items():
                    if len(v) < 10:
                        continue
                    num = {k: a for k, (a, _) in v}
                    den = {k: b for k, (_, b) in v}
                    keys = list(num)
                    fams = sorted({fam[k] for k in keys})
                    byf = {f: [k for k in keys if fam[k] == f] for f in fams}
                    nf = np.array([sum(num[k] for k in byf[f]) for f in fams])
                    df = np.array([sum(den[k] for k in byf[f]) for f in fams])
                    bs = [nf[s].sum() / df[s].sum() for s in (rng.integers(0, len(fams), len(fams)) for _ in range(2000))]
                    res_src[c] = {"ratio": round(float(nf.sum() / df.sum()), 3),
                                  "ci95": [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])],
                                  "targets": len(keys), "families": len(fams)}
                    for st in sorted({x for v in stratum.values() for x in v}):  # Amendment 6 / 8.2 / 7d strata
                        ks = [k for k in keys if st in stratum.get(k, ())]
                        if len(ks) >= 10:
                            res_src[c][f"ratio_{st}"] = round(sum(num[k] for k in ks) / sum(den[k] for k in ks), 3)
                            res_src[c][f"targets_{st}"] = len(ks)
                res_src["pool_people"] = len(PP)
        for c in list(maps_src["all"]):
            if c in ("V", "meanface", "valid"):
                continue
            a, b = maps_src["0"].get(c), maps_src["1"].get(c)
            if a is not None and b is not None:
                ok = np.isfinite(a) & np.isfinite(b) & maps_src["all"]["valid"]
                if c in res_src and ok.sum() > 10:
                    res_src[c]["split_half_r"] = round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 2)
        result[src], store[src] = res_src, maps_src["all"]
    np.savez_compressed(os.path.join(OUT, f"maps{SUF}.npz"),
                        **{f"{s}_{k}": v for s, m in store.items() for k, v in m.items()})
    meta_out = {"protocol": "docs/PROTOCOL_donor_ladder_face.md", "status": qc_status(),
                "scale": "1 = dataset mean, 2 = stranger (squared error, z-scored grey level, 128 px frame)",
                "window_px": WIN, "pose_max_deg": map_pose_max(), "frame": frame_suf(), "results": result}
    with open(os.path.join(PROJECT, "results", f"donor_ladder_face{SUF}.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=1)
    print(json.dumps(result, indent=1))


# ---------------------------------------------------------------- stage: figure

def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(OUT, f"maps{SUF}.npz"))
    res = json.load(open(os.path.join(PROJECT, "results", f"donor_ladder_face{SUF}.json"), encoding="utf-8"))["results"]
    cols = [("meanface", "Mean face"), ("repeat", "Another photo"), ("mirror", "Mirror side"),
            ("sibling", "Sibling"), ("stranger", "Stranger"), ("lookalike", "16 look-alikes*"),
            ("sexage", "Same sex and age"), ("V", "Dataset mean")]
    rows = [("celeba", "CelebA"), ("imdb", "IMDb-WIKI")]
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.1 * len(cols), 2.6 * len(rows) + 0.9))
    for r, (src, rname) in enumerate(rows):
        face = z[f"{src}_meanface"]
        valid = z[f"{src}_valid"]
        for c, (key, title) in enumerate(cols):
            a = ax[r, c]
            a.axis("off")
            if r == 0:
                a.set_title(title, fontsize=11)
            a.imshow(np.where(valid, face, np.nan), cmap="gray", vmin=-2.5, vmax=2.5)
            name = f"{src}_{key}"
            if key == "meanface":
                a.text(0.5, -0.08, f"{rname}\n(pool mean, no person)", ha="center", va="top",
                       transform=a.transAxes, fontsize=8.5)
                continue
            if name not in z.files:
                a.text(0.5, 0.5, "not available", ha="center", va="center", transform=a.transAxes, fontsize=9)
                continue
            m = np.where(valid, z[name], np.nan)
            if key == "V":
                a.imshow(m, cmap="viridis")
                a.text(0.5, -0.08, "MSE = 1 x mean\n(shown: V, z units)", ha="center", va="top",
                       transform=a.transAxes, fontsize=8.5)
                continue
            s = res[src].get(key, {})
            if SUF in ("_v3", "_v4") and s.get("split_half_r", 0) < 0.5:  # Amendment 4.4 (withdrawn in v5)
                a.text(0.5, 0.5, "map not\nreproducible", ha="center", va="center", transform=a.transAxes,
                       fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.8))
            else:
                a.imshow(m, cmap="RdBu_r", vmin=0, vmax=2, alpha=0.9)
            r_txt = f"split-half r {s['split_half_r']:.2f}" if "split_half_r" in s else ""
            ci = s.get("ci95")
            ci_txt = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
            a.text(0.5, -0.08, f"MSE = {s.get('ratio', float('nan')):.2f} x mean{ci_txt}\n{r_txt}\nn = {s.get('targets', 0)}",
                   ha="center", va="top", transform=a.transAxes, fontsize=8.5)
    fig.suptitle("Faces, every location: pixel error against the dataset mean"
                 + ("" if frame_suf() in QC_APPROVED else " (pre-QC)"), fontsize=13)
    fig.text(0.5, 0.01, "One scale: 1 = dataset mean (white), 2 = stranger; blue beats the mean. "
             "* neighbours chosen on a 4 px grid by the 8 px ring around the 16 px window; mirror side excludes the midline. 128 px frame, 16 px window, |yaw|, |pitch| < 10 deg.",
             ha="center", fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PROJECT, "results", "figures", f"fig_face_ladder_maps{SUF}.{ext}"), dpi=200)
    print("figure written")


# ---------------------------------------------------------------- stage: prep (v2, Amendment 2)

SEGMENTER = os.path.join(DATA, "projects", "identity-ladder-metric", "models", "mediapipe-selfie-multiclass",
                         "selfie_multiclass_256x256.tflite")
HAIR, BODY_SKIN, FACE_SKIN, ACCESSORY = 1, 2, 3, 5
CAL, LENS = None, None  # Amendment 7 thresholds and lens mask, set in worker processes for v6


def _dilate(m, r):
    import cv2
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0
OCCL_MAX = {HAIR: 0.05, BODY_SKIN: 0.02, ACCESSORY: 0.01}
SMILE_MAX, JAW_MAX, ILLUM_SIGMA = 0.5, 0.3, 16.0


def affine_fit(src, dst):
    """Least-squares 2D affine (2 x 3) mapping src points to dst."""
    A = np.hstack([np.asarray(src, float), np.ones((len(src), 1))])
    return np.linalg.lstsq(A, np.asarray(dst, float), rcond=None)[0].T


def symmetric_template(S):
    """Average a frame-space shape with its mirror image; correspondence by nearest mirrored point."""
    Sm = np.c_[(SIZE - 1) - S[:, 0], S[:, 1]]
    j = np.argmin(((S[:, None] - Sm[None]) ** 2).sum(-1), axis=1)
    return 0.5 * (S + Sm[j])


def illumination_correct(y, m, sigma=ILLUM_SIGMA):
    """Divide by the mask-normalized Gaussian-smoothed grey level (the N4 analogue)."""
    from scipy.ndimage import gaussian_filter
    num, den = gaussian_filter(y * m, sigma), gaussian_filter(m.astype(float), sigma)
    field = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
    return np.where(m & (field > 0), y / np.where(field > 0, field, 1), 0)


# ---------------------------------------------------------------- Amendment 7: pixel masks (v6)

EYE_BAND = (36, 68)
LENS_A, LENS_B, BRIDGE_HALF, ACC_DILATE = 17, 12, 3, 3
L_WEIGHT, MEDIAN_FILT, GREY_CHROMA = 0.5, 3, 3.0
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
EYE_L_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_R_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
BROW_L_IDX = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
BROW_R_IDX = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
NOSE_BASE = 2
NOSTRIL_IDX = [1, 2, 4, 19, 64, 94, 97, 98, 99, 129, 168, 219, 240, 294, 326, 327, 328, 358, 439, 460]  # Amendment 7b: nose tip and alae


def _poly(pts, dilate=0):
    import cv2
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(m, [cv2.convexHull(np.round(pts).astype(np.int32))], 1)
    if dilate:
        m = cv2.dilate(m, np.ones((2 * dilate + 1,) * 2, np.uint8))
    return m > 0


def lens_mask(tmpl):
    """Template lens ellipses and bridge (Amendment 7.4), symmetric about the frame midline by construction."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    eyes = sorted([tmpl[IRIS_A], tmpl[IRIS_B]], key=lambda q: q[0])
    cy = 0.5 * (eyes[0][1] + eyes[1][1])
    dx = 0.5 * (eyes[1][0] - eyes[0][0])
    m = np.zeros((SIZE, SIZE), bool)
    for ex in ((SIZE - 1) / 2 - dx, (SIZE - 1) / 2 + dx):
        m |= ((xx - ex) / LENS_A) ** 2 + ((yy - cy) / LENS_B) ** 2 <= 1
    m |= (np.abs(yy - cy) <= BRIDGE_HALF) & (np.abs(xx - (SIZE - 1) / 2) <= dx)
    return m


def colour_distance(lab, ref):
    """CIELAB distance to the skin reference with lightness weighted L_WEIGHT (Amendment 7.5)."""
    d = lab - np.asarray(ref, np.float32)[:, None, None]
    return np.sqrt((L_WEIGHT * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2)


def colour_flags(dist, eligible, thr):
    from scipy.ndimage import median_filter
    return median_filter((dist > thr) & eligible, size=MEDIAN_FILT) & eligible


def _v6_features(rgb, cat, L, A, tmpl, oval):
    """Frame-space quantities for Amendment 7, shared by calibration and prep."""
    import cv2
    W = lambda pts: apply(A, pts)
    hull = np.zeros(cat.shape, np.uint8)
    cv2.fillPoly(hull, [cv2.convexHull(L[oval].astype(np.int32))], 1)
    hw = cv2.warpAffine(hull, A, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    inb = cv2.warpAffine(np.ones(cat.shape, np.float32), A, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    cw = cv2.warpAffine(cat, A, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0)
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    labw = np.stack([cv2.warpAffine(lab[..., c], A, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
                     for c in range(3)])
    anat = hw & inb
    skin = (cw == FACE_SKIN) & anat
    acc = (cw == ACCESSORY) & anat
    lips = _poly(W(L[LIPS_OUTER]))
    exempt = lips | _poly(W(L[EYE_L_IDX]), 2) | _poly(W(L[EYE_R_IDX]), 2) | _poly(W(L[BROW_L_IDX]), 2) \
        | _poly(W(L[BROW_R_IDX]), 2) | _poly(W(L[NOSTRIL_IDX]), 2)
    nose_y = W(L[[NOSE_BASE]])[0, 1]
    rows = np.arange(SIZE)[:, None]
    ref_zone = skin & ~exempt & (rows > EYE_BAND[1]) & (rows < nose_y)
    if ref_zone.sum() < 30:  # fall back to all non-exempt skin
        ref_zone = skin & ~exempt
    ref = np.median(labw[:, ref_zone], axis=1) if ref_zone.any() else np.array([50, 0, 0], np.float32)
    grey = float(np.hypot(ref[1], ref[2])) < GREY_CHROMA and float(np.median(np.hypot(labw[1][skin], labw[2][skin]))) < GREY_CHROMA if skin.any() else False
    if grey:
        labw = labw.copy(); labw[1:] = ref[1:, None, None]
    dist = colour_distance(labw, ref)
    band = np.zeros((SIZE, SIZE), bool); band[EYE_BAND[0]:EYE_BAND[1]] = True
    eye_acc = float((acc & band).sum() / max((anat & band).sum(), 1))
    lip_px = lips & anat
    lip_contrast = float(np.median(labw[1][lip_px]) - ref[1]) if lip_px.sum() >= 10 else float("nan")
    chin = anat & ~lips & (rows > nose_y)
    el = skin & ~exempt  # Amendment 7b: bilateral statistic, the pixel and its mirror partner must both deviate
    sym = np.where(el & el[:, ::-1], np.minimum(dist, dist[:, ::-1]), 0).astype(np.float32)
    return dict(anat=anat, skin=skin, acc=acc, lips=lips, exempt=exempt, dist=dist, sym=sym, eye_acc=eye_acc,
                lip_contrast=lip_contrast, grey=grey, chin=chin)


def zscore_in_mask(g, m):
    v = g[m]
    return np.where(m, (g - v.mean()) / max(v.std(), 1e-6), 0) if v.size > 50 else np.zeros_like(g)


BEARD_ROWS, BEARD_COLS = (76, 124), (24, 104)


def beard_features(x, m):
    """Amendment 7c: grey lower-face block means (2 x 2) and Sobel gradient magnitude in 8 x 8 blocks, one image."""
    from scipy.ndimage import sobel
    c = (x * m)[BEARD_ROWS[0]:BEARD_ROWS[1], BEARD_COLS[0]:BEARD_COLS[1]].astype(np.float64)
    h, w = c.shape
    gm = np.hypot(sobel(c, axis=0), sobel(c, axis=1))
    px = c.reshape(h // 2, 2, w // 2, 2).mean((1, 3)).ravel()
    ge = gm.reshape(h // 8, 8, w // 8, 8).mean((1, 3)).ravel()
    return np.concatenate([px, ge])


def beard_score(x, m, B):
    z = (beard_features(x, m) - np.asarray(B["mu"])) / np.asarray(B["sd"])
    return float(z @ np.asarray(B["w"][:-1]) + B["w"][-1])


def _logit(A, y, lam=30.0, it=25):
    w = np.zeros(A.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-A @ w))
        reg = lam * np.r_[np.ones(len(w) - 1), 0]
        w -= np.linalg.solve((A * (p * (1 - p))[:, None]).T @ A + np.diag(reg + 1e-9), A.T @ (p - y) + reg * w)
    return w


def auc(s, y):
    o = np.argsort(s)
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_beard():
    """Amendment 7c: facial-hair detector on CelebA men (v5 frame table), CV by person, Youden threshold."""
    imgs = json.load(open(os.path.join(OUT, "images.json"), encoding="utf-8"))
    z = np.load(os.path.join(OUT, "frame_v5.npz"))
    idx, X, M = z["idx"], z["X"], z["M"]
    al = open(os.path.join(CELEBA, "downloads", "list_attr_celeba.txt")).read().split("\n")
    an = al[1].split()
    attr = {l.split()[0]: l.split()[1:] for l in al[2:] if l.strip()}
    K = [k for k, j in enumerate(idx) if imgs[j]["src"] == "celeba" and attr[imgs[j]["file"]][an.index("Male")] == "1"]
    Fe = np.array([beard_features(X[k].astype(np.float32), M[k]) for k in K])
    mu, sd = Fe.mean(0), Fe.std(0) + 1e-6
    A = np.hstack([(Fe - mu) / sd, np.ones((len(K), 1))])
    lab = lambda k, a: attr[imgs[idx[k]]["file"]][an.index(a)] == "1"
    y = np.array([not lab(k, "No_Beard") for k in K])
    person = np.array([imgs[idx[k]]["person"] for k in K])
    ups = np.unique(person)
    fold = dict(zip(ups, np.random.default_rng(SEED).integers(0, 5, len(ups))))
    Fo = np.array([fold[p] for p in person])
    s = np.zeros(len(K))
    for f_ in range(5):
        w = _logit(A[Fo != f_], y[Fo != f_])
        s[Fo == f_] = A[Fo == f_] @ w
    J = [((s > t)[y].mean() - (s > t)[~y].mean(), t) for t in np.quantile(s, np.linspace(0.01, 0.99, 197))]
    jbest, thr = max(J)
    w = _logit(A, y)
    rep = {"n_men": len(K), "prevalence": round(float(y.mean()), 3), "cv_auc_no_beard": round(auc(s, y), 3),
           "cv_auc_mustache": round(auc(s, np.array([lab(k, "Mustache") for k in K])), 3),
           "cv_auc_goatee": round(auc(s, np.array([lab(k, "Goatee") for k in K])), 3),
           "cv_auc_5_o_clock_shadow": round(auc(s, np.array([lab(k, "5_o_Clock_Shadow") for k in K])), 3),
           "youden_J": round(float(jbest), 3), "cv_sensitivity": round(float((s > thr)[y].mean()), 3),
           "cv_false_positive_rate": round(float((s > thr)[~y].mean()), 3)}
    with open(os.path.join(OUT, "beard_v6c.json"), "w", encoding="utf-8") as f:
        json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "threshold": float(thr), "report": rep}, f)
    print(json.dumps(rep, indent=1))


def _prep_worker(args):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from PIL import Image
    global CAL, LENS
    chunk, tmpl, oval = args[:3]
    if len(args) > 3:
        CAL = args[3]
        LENS = lens_mask(tmpl)
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=SEGMENTER), running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True))
    out = []
    for j, path, L in chunk:
        rgb = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))
        cat = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).category_mask.numpy_view().copy()
        hull = np.zeros(cat.shape, np.uint8)
        cv2.fillPoly(hull, [cv2.convexHull(L[oval].astype(np.int32))], 1)
        h = hull > 0
        frac = {c: float((cat[h] == c).mean()) if h.any() else 1.0 for c in OCCL_MAX}
        A = affine_fit(L, tmpl)
        y = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        g = cv2.warpAffine(y, A, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
        cw = cv2.warpAffine(cat, A, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0)
        hw = cv2.warpAffine(hull, A, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0) > 0
        inb = cv2.warpAffine(np.ones_like(y), A, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0) > 0
        m = (cw == FACE_SKIN) & hw & inb
        if CAL is not None:  # Amendment 7 (v6)
            F = _v6_features(rgb, cat, L, A, tmpl, oval)
            glasses = F["eye_acc"] > CAL["glasses_eye_acc"]
            mg = (LENS & F["anat"]) | _dilate(F["acc"], ACC_DILATE) if glasses else np.zeros_like(m)
            if CAL.get("glasses_only"):  # Amendment 7d: only glasses are masked beyond the segmenter
                mc = np.zeros_like(m)
            elif CAL.get("symmetric"):  # Amendment 7b
                mc = colour_flags(F["sym"], F["skin"] & ~F["exempt"], CAL["colour_dist_sym"])
            else:
                mc = colour_flags(F["dist"], F["skin"] & ~F["exempt"], CAL["colour_dist"])
            ml = F["lips"] & F["skin"] if (not CAL.get("symmetric") and not CAL.get("glasses_only")  # Amendment 7b: lipstick rule withdrawn
                                           and np.isfinite(F["lip_contrast"]) and F["lip_contrast"] > CAL["lip_contrast"]) \
                else np.zeros_like(m)
            mb = np.zeros_like(m)
            if CAL.get("beard"):  # Amendment 7c: colour rule withdrawn; image-level facial-hair detector + zone
                mc = np.zeros_like(m)
                m1 = F["skin"] & ~mg
                score = beard_score(zscore_in_mask(illumination_correct(g, m1), m1), m1, CAL["beard"])
                frac.update(beard=bool(score > CAL["beard"]["threshold"]), beard_score=float(score))
                if frac["beard"]:
                    mb = F["chin"] & ~F["lips"]
            m = F["skin"] & ~mg & ~mc & ~ml & ~mb
            chin_n = max(int(F["chin"].sum()), 1)
            frac.update(anat=F["anat"], glasses=bool(glasses), grey=F["grey"], px_glasses=int((mg & F["skin"]).sum()),
                        px_colour=int(mc.sum()), px_lips=int(ml.sum()), chin_colour=float((mc & F["chin"]).sum() / chin_n),
                        removed=float(1 - m.sum() / max(int(F["skin"].sum()), 1)))
        x = zscore_in_mask(illumination_correct(g, m), m)
        out.append((j, x.astype(np.float16), m, frac))
    return out


N_ROWS_V6 = 66713  # rows of images.json before Amendment 8.3 added pool images (template uses the same rows as v6)


QC_APPROVED = {"_v6f": "visual QC approved by Kate, 6 Nov 2026 (Amendment 10)"}  # explicit, per version


def qc_status():
    return QC_APPROVED.get(frame_suf(), "pre-QC (no human visual check yet)")


def mask_rules(suf):
    """Pixel-mask rules of a version (Amendments 7 to 7d, 10), as an explicit table rather than string order:
    '_v6f' sorts after '_v6c' but must not inherit the withdrawn facial-hair detector or colour rule."""
    if suf < "_v6":
        return None
    if suf == "_v6c":
        return {"symmetric": True, "beard": True, "glasses_only": False}
    if suf == "_v6f" or suf >= "_v7":
        return {"symmetric": False, "beard": False, "glasses_only": True}
    return {"symmetric": suf >= "_v6b", "beard": False, "glasses_only": False}


def landmark_residual(L, tmpl):
    """Amendment 8.1: RMS distance of all 478 landmarks to the template after the least-squares affine, in
    inter-ocular distances of the template."""
    A = affine_fit(L, tmpl)
    iod = float(np.linalg.norm(tmpl[IRIS_A] - tmpl[IRIS_B]))
    return float(np.sqrt(((apply(A, L) - tmpl) ** 2).sum(1).mean()) / iod)


def prep(workers):
    from concurrent.futures import ProcessPoolExecutor
    from mediapipe.tasks.python import vision
    oval = sorted({c.start for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL})
    imgs = json.load(open(os.path.join(OUT, "images.json"), encoding="utf-8"))
    z = np.load(os.path.join(OUT, "landmarks.npz"))
    L, P, E, status = z["L"], z["P"], z["E"], z["status"]
    pm = prep_pose_max()
    base = (status == "ok") & (np.abs(P[:, 0]) < pm) & (np.abs(P[:, 1]) < pm)
    # versions before Amendment 8.3 use only the rows that existed then; later rows get no reason (not in the table)
    rows_in = np.arange(len(status)) < N_ROWS_V6 if SUF < "_v7" else np.ones(len(status), bool)
    base &= rows_in
    frontal = (status == "ok") & (np.abs(P[:, 0]) < POSE_MAX) & (np.abs(P[:, 1]) < POSE_MAX)
    reasons = {}
    for j in np.flatnonzero((status != "ok") & rows_in):
        reasons[int(j)] = "landmarks"
    for j in np.flatnonzero((status == "ok") & rows_in & ~base):
        reasons[int(j)] = "pose"
    # template: pool mean shape after the v1 similarity alignment, made symmetric
    dst = np.array([EYE_L, EYE_R, MOUTH])
    shapes = []
    for j in np.flatnonzero(frontal & np.array([k < len(imgs) and imgs[k]["role"] == "pool" for k in range(len(status))])
                            & (np.arange(len(status)) < N_ROWS_V6)):  # same template rows in every version
        eyes = sorted([L[j, IRIS_A], L[j, IRIS_B]], key=lambda q: q[0])
        shapes.append(apply(similarity(np.array([eyes[0], eyes[1], 0.5 * (L[j, LIP_UP] + L[j, LIP_LO])]), dst), L[j]))
    tmpl = symmetric_template(np.mean(shapes, 0))
    dl = os.path.join(CELEBA, "downloads")
    al = open(os.path.join(dl, "list_attr_celeba.txt")).read().split("\n")
    an = al[1].split()
    attr = {l.split()[0]: l.split()[1:] for l in al[2:] if l.strip()}
    id_bad = np.load(os.path.join(OUT, identity_file()))["excluded"] if SUF >= "_v4" else None
    cel_occ = {k: any(v[an.index(a)] == "1" for a in ("Eyeglasses", "Wearing_Hat", "Bangs")) for k, v in attr.items()}
    todo = []
    for j in np.flatnonzero(base):
        d = imgs[j]
        minimal = SUF >= "_v5"  # Amendment 6: only validity exclusions
        if not minimal and d["src"] == "celeba" and cel_occ[d["file"]]:
            reasons[int(j)] = "occlusion_attribute"
        elif not minimal and (E[j, 0] >= SMILE_MAX or E[j, 1] >= JAW_MAX):
            reasons[int(j)] = "expression"
        elif SUF >= "_v4" and id_bad[j]:  # Amendment 5
            reasons[int(j)] = "identity_inconsistent"
        else:
            path = os.path.join(CELEBA, "raw", "img_align_celeba", d["file"]) if d["src"] == "celeba" \
                else os.path.join(IMDB, *d["file"].split("/"))
            todo.append((int(j), path, L[j]))
    # segmentation also on the attribute-excluded CelebA images, for the agreement check only
    check = [(int(j), os.path.join(CELEBA, "raw", "img_align_celeba", imgs[j]["file"]), L[j])
             for j in np.flatnonzero(base) if imgs[j]["src"] == "celeba" and reasons.get(int(j)) == "occlusion_attribute"]
    jobs = todo + check
    cal = json.load(open(os.path.join(OUT, "calib_v6.json"), encoding="utf-8")) if SUF >= "_v6" else None
    if cal:
        rules = mask_rules(SUF)
        cal["symmetric"] = rules["symmetric"]  # Amendment 7b
        if rules["beard"]:  # Amendment 7c
            cal["beard"] = json.load(open(os.path.join(OUT, "beard_v6c.json"), encoding="utf-8"))
        cal["glasses_only"] = rules["glasses_only"]  # Amendment 7d / 10
    chunks = [(jobs[i::workers * 8], tmpl, oval) + ((cal,) if cal else ()) for i in range(workers * 8)]
    got = {}
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_prep_worker, chunks):
            for j, x, m, frac in res:
                got[j] = (x, m, frac)
    occl = lambda fr: any(fr[c] > t for c, t in OCCL_MAX.items())
    kept = []
    for j, _, _ in todo:
        x, m, fr = got[j]
        if SUF < "_v5" and occl(fr):
            reasons[j] = "occlusion_segmenter"
        elif m.sum() < 2000:  # less than ~12% of the frame is face skin
            reasons[j] = "small_mask"
        else:
            kept.append(j)
    # duplicates across people
    Xk = np.array([got[j][0] for j in kept], np.float32)
    Mk = np.array([got[j][1] for j in kept])
    com = Mk.mean(0) > 0.9
    V = Xk[:, com]
    V = (V - V.mean(1, keepdims=True)) / np.maximum(V.std(1, keepdims=True), 1e-6)
    person = np.array([imgs[j]["src"] + ":" + imgs[j]["person"] for j in kept])
    dup = np.zeros(len(kept), bool)
    for a in range(0, len(kept), 2000):
        C = V[a:a + 2000] @ V.T / V.shape[1]
        r, c = np.nonzero(C > DUP_R)
        sel = person[a + r] != person[c]
        dup[a + r[sel]] = True
        dup[c[sel]] = True
    for n, j in enumerate(kept):
        if dup[n]:
            reasons[j] = "duplicate_other_person"
    keep = ~dup
    kj = np.array(kept)[keep]
    X, M = Xk[keep], Mk[keep]
    pool = np.array([imgs[j]["role"] == "pool" for j in kj])
    mean, _ = masked_mean(X[pool], M[pool])
    tc = np.array([_corr(X[i], mean, M[i]) for i in range(len(kj))])
    med, mad = np.median(tc), np.median(np.abs(tc - np.median(tc)))
    flag = tc < med - 3 * 1.4826 * mad
    lfit = np.array([landmark_residual(L[int(j)], tmpl) for j in kj])
    if SUF >= "_v7":  # Amendment 8.1: fit failure by landmark geometry; template correlation kept as a flag only
        cut = float(np.percentile(lfit[pool], 99))
        bad = lfit > cut
        for j in kj[bad]:
            reasons[int(j)] = "landmark_fit"
        kj, X, M, tc, flag, lfit = kj[~bad], X[~bad], M[~bad], tc[~bad], flag[~bad], lfit[~bad]
        extra_keep = ~bad
    elif SUF >= "_v3":  # Amendment 4.6 / 6: template-correlation flags are excluded
        for j in kj[flag]:
            reasons[int(j)] = "template_correlation"
        extra_keep = ~flag
        kj, X, M, tc, flag, lfit = kj[~flag], X[~flag], M[~flag], tc[~flag], flag[~flag], lfit[~flag]
    else:
        extra_keep = np.ones(len(kj), bool)
    extra = {}
    if cal:
        extra["A"] = np.array([got[int(j)][2]["anat"] for j in kj])
        keys = ("glasses", "grey", "px_glasses", "px_colour", "px_lips", "chin_colour", "removed")             + (("beard", "beard_score") if cal.get("beard") else ())
        for k in keys:
            extra["mask_" + k] = np.array([got[int(j)][2][k] for j in kj])
    np.savez_compressed(os.path.join(OUT, f"frame{SUF}.npz"), idx=kj, X=X.astype(np.float16), M=M, flag=flag,
                        template_corr=tc, template_shape=tmpl, landmark_resid=lfit, **extra)
    if cal:
        mask_report(kj, extra, imgs, attr, an)
    with open(os.path.join(OUT, f"prep{SUF}_reasons.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in reasons.items()}, f)
    from collections import Counter
    cnt = Counter((imgs[int(j)]["src"], imgs[int(j)]["role"], r) for j, r in reasons.items())
    kc = Counter((imgs[j]["src"], imgs[j]["role"]) for j in kj)
    agree = {}
    for a_name, cls in (("Eyeglasses", ACCESSORY), ("Bangs", HAIR), ("Wearing_Hat", HAIR)):
        ys, ps = [], []
        for j, _, _ in jobs:
            if imgs[j]["src"] != "celeba":
                continue
            ys.append(attr[imgs[j]["file"]][an.index(a_name)] == "1")
            ps.append(got[j][2][cls] > OCCL_MAX[cls])
        ys, ps = np.array(ys), np.array(ps)
        agree[a_name] = {"sensitivity": round(float(ps[ys].mean()), 2) if ys.any() else None,
                         "false_positive_rate": round(float(ps[~ys].mean()), 2), "n_attr": int(ys.sum())}
    print(json.dumps({"excluded": {" ".join(k): v for k, v in sorted(cnt.items())},
                      "kept": {" ".join(k): v for k, v in sorted(kc.items())},
                      "qc_flagged_template_corr": int(flag.sum()),
                      "segmenter_vs_celeba_attributes": agree}, indent=1))


def mask_report(kj, ex, imgs, attr, an):
    """Amendment 7 counts per rule, source and sex, and agreement with CelebA attributes (report only)."""
    out = {}
    for src in ("celeba", "imdb"):
        rows = [n for n, j in enumerate(kj) if imgs[int(j)]["src"] == src]
        for sex in (True, False):
            rr = [n for n in rows if imgs[int(kj[n])].get("male") == sex]
            if not rr:
                continue
            key = f"{src} {'male' if sex else 'female'}"
            out[key] = {"images": len(rr), "glasses": round(float(ex["mask_glasses"][rr].mean()), 3),
                        "greyscale": int(ex["mask_grey"][rr].sum()),
                        "lips_removed": round(float((ex["mask_px_lips"][rr] > 0).mean()), 3),
                        "facial_hair_gt10pct_chin": round(float((ex["mask_chin_colour"][rr] > 0.1).mean()), 3),
                        "median_fraction_of_face_removed": round(float(np.median(ex["mask_removed"][rr])), 3)}
            if "mask_beard" in ex:
                out[key]["facial_hair_detector"] = round(float(ex["mask_beard"][rr].mean()), 3)
    agree = {}
    cel = [n for n, j in enumerate(kj) if imgs[int(j)]["src"] == "celeba"]
    A = lambda n, a: attr[imgs[int(kj[n])]["file"]][an.index(a)] == "1"
    men = [n for n in cel if A(n, "Male")]
    women = [n for n in cel if not A(n, "Male")]
    def rate(sub, pred):
        return round(float(np.mean([pred(n) for n in sub])), 3) if sub else None
    hair = lambda n: ex["mask_chin_colour"][n] > 0.1
    agree["facial_hair (men): No_Beard=0 vs 1"] = [rate([n for n in men if not A(n, "No_Beard")], hair),
                                                  rate([n for n in men if A(n, "No_Beard")], hair)]
    lip = lambda n: ex["mask_px_lips"][n] > 0
    agree["lips (women): Wearing_Lipstick=1 vs 0"] = [rate([n for n in women if A(n, "Wearing_Lipstick")], lip),
                                                      rate([n for n in women if not A(n, "Wearing_Lipstick")], lip)]
    col = lambda n: ex["mask_px_colour"][n] > 200
    agree["colour>200px (women): Heavy_Makeup=1 vs 0"] = [rate([n for n in women if A(n, "Heavy_Makeup")], col),
                                                          rate([n for n in women if not A(n, "Heavy_Makeup")], col)]
    g = lambda n: bool(ex["mask_glasses"][n])
    agree["glasses: Eyeglasses=1 vs 0"] = [rate([n for n in cel if A(n, "Eyeglasses")], g),
                                           rate([n for n in cel if not A(n, "Eyeglasses")], g)]
    if "mask_beard" in ex:
        bd = lambda n: bool(ex["mask_beard"][n])
        agree["facial-hair detector (men): No_Beard=0 vs 1"] = [rate([n for n in men if not A(n, "No_Beard")], bd),
                                                               rate([n for n in men if A(n, "No_Beard")], bd)]
    rep = {"by_group": out, "attribute_agreement_[pos,neg]": agree}
    with open(os.path.join(PROJECT, "results", f"face_masks{SUF}.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


def _calib_worker(args):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from PIL import Image
    chunk, tmpl, oval = args
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=SEGMENTER), running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True))
    rng = np.random.default_rng(len(chunk))
    out = []
    for j, path, L in chunk:
        rgb = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))
        cat = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).category_mask.numpy_view().copy()
        F = _v6_features(rgb, cat, L, affine_fit(L, tmpl), tmpl, oval)
        el = F["skin"] & ~F["exempt"]
        d = F["dist"][el]
        sel = rng.choice(d.size, min(200, d.size), replace=False) if d.size else np.zeros(0, int)
        out.append((j, F["eye_acc"], F["lip_contrast"], F["grey"], d[sel].astype(np.float32),
                    F["sym"][el][sel].astype(np.float32)))
    return out


def calibrate(workers, n_celeba=4000, n_imdb=1500):
    """Amendment 7 thresholds from CelebA attributes and male reference images; no ladder metric is used."""
    from concurrent.futures import ProcessPoolExecutor
    from mediapipe.tasks.python import vision
    oval = sorted({c.start for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL})
    imgs = json.load(open(os.path.join(OUT, "images.json"), encoding="utf-8"))
    z = np.load(os.path.join(OUT, "landmarks.npz"))
    L, P, status = z["L"], z["P"], z["status"]
    base = (status == "ok") & (np.abs(P[:, 0]) < POSE_MAX) & (np.abs(P[:, 1]) < POSE_MAX)
    tmpl = np.load(os.path.join(OUT, "frame_v5.npz"))["template_shape"]
    dl = os.path.join(CELEBA, "downloads")
    al = open(os.path.join(dl, "list_attr_celeba.txt")).read().split("\n")
    an = al[1].split()
    attr = {l.split()[0]: l.split()[1:] for l in al[2:] if l.strip()}
    rng = np.random.default_rng(SEED)
    cel = [j for j in np.flatnonzero(base) if imgs[j]["src"] == "celeba"]
    imd = [j for j in np.flatnonzero(base) if imgs[j]["src"] == "imdb" and imgs[j].get("male") is True]
    pick = list(rng.choice(cel, n_celeba, replace=False)) + list(rng.choice(imd, min(n_imdb, len(imd)), replace=False))
    jobs = []
    for j in pick:
        d = imgs[j]
        path = os.path.join(CELEBA, "raw", "img_align_celeba", d["file"]) if d["src"] == "celeba" \
            else os.path.join(IMDB, *d["file"].split("/"))
        jobs.append((int(j), path, L[j]))
    res = []
    with ProcessPoolExecutor(workers) as ex:
        for n, r in enumerate(ex.map(_calib_worker, [(jobs[i::workers * 8], tmpl, oval) for i in range(workers * 8)])):
            res += r
            print(f"calibration chunk {n + 1}/{workers * 8}", flush=True)
    A = lambda j, a: imgs[j]["src"] == "celeba" and attr[imgs[j]["file"]][an.index(a)] == "1"
    # glasses: Youden's J on CelebA
    ce = [(e, A(j, "Eyeglasses")) for j, e, *_ in res if imgs[j]["src"] == "celeba"]
    e = np.array([x for x, _ in ce]); y = np.array([t for _, t in ce])
    cands = np.unique(np.r_[0, e])
    J = [((e > c)[y].mean() - (e > c)[~y].mean(), c) for c in cands]
    jbest, cbest = max(J)
    # colour: 99th percentile of pixel distances in clean CelebA references
    cl = [(d, sy) for j, _, _, _, d, sy in res if imgs[j]["src"] == "celeba" and attr[imgs[j]["file"]][an.index("No_Beard")] == "1"
          and not any(A(j, a) for a in ("Heavy_Makeup", "Wearing_Lipstick", "Eyeglasses"))]
    clean = [d for d, _ in cl]
    colour = float(np.percentile(np.concatenate(clean), 99))
    colour_sym = float(np.percentile(np.concatenate([sy for _, sy in cl]), 99))  # Amendment 7b
    # lips: 95th percentile among men
    men = [lc for j, _, lc, grey, *_ in res if not grey and np.isfinite(lc)
           and (A(j, "Male") or (imgs[j]["src"] == "imdb" and imgs[j].get("male") is True))]
    lip = float(np.percentile(men, 95))
    cal = {"glasses_eye_acc": float(cbest), "colour_dist": colour, "colour_dist_sym": colour_sym, "lip_contrast": lip,
           "report": {"glasses_youden_J": round(float(jbest), 3),
                      "glasses_sensitivity": round(float((e > cbest)[y].mean()), 3),
                      "glasses_false_positive_rate": round(float((e > cbest)[~y].mean()), 3),
                      "n_celeba": int(len(ce)), "n_eyeglasses": int(y.sum()), "n_clean_reference_images": len(clean),
                      "n_men_lip_reference": len(men), "n_imdb_men": int(sum(imgs[j]["src"] == "imdb" for j, *_ in res))}}
    with open(os.path.join(OUT, "calib_v6.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=1)
    print(json.dumps(cal, indent=1))


# ---------------------------------------------------------------- stage: flow (selection diagram)

FLOW_STEPS = [("landmarks", "no single face found (MediaPipe Face Landmarker)"),
              ("pose", "|yaw| or |pitch| >= 10 deg"),
              ("identity_inconsistent", "identity inconsistent (ArcFace cosine to own centroid < 0.3)"),
              ("identity_unresolved", "identity unresolved (two images, cosine < 0.3)"),
              ("small_mask", "face-skin mask < 2000 px"),
              ("duplicate_other_person", "duplicate of another person's image (r > 0.9)"),
              ("template_correlation", "template fit failure (correlation < median - 3 MAD)")]


def flow():
    """Selection flow (images and people per step, per dataset and role) from the same files the analysis reads.
    Writes results/face_flow{SUF}.json and results/figures/fig_face_flow{SUF}.png/pdf; aggregates only."""
    import scipy.io as sio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imgs = json.load(open(os.path.join(OUT, "images.json"), encoding="utf-8"))
    reasons = {int(k): v for k, v in json.load(open(os.path.join(OUT, f"prep{SUF}_reasons.json"), encoding="utf-8")).items()}
    res = json.load(open(os.path.join(PROJECT, "results", f"donor_ladder_face{SUF}.json"), encoding="utf-8"))["results"]
    dl = os.path.join(CELEBA, "downloads")
    ident = dict(l.split() for l in open(os.path.join(dl, "identity_CelebA.txt")))
    part = dict(l.split() for l in open(os.path.join(dl, "list_eval_partition.txt")))
    r = sio.loadmat(os.path.join(IMDB, "imdb.mat"), squeeze_me=True)["imdb"]
    paths, fs, sf = r["full_path"].item(), r["face_score"].item(), r["second_face_score"].item()
    nm = np.array([re.search(r"nm\d{7}", q).group(0) for q in paths])
    single = np.isfinite(fs) & ~np.isfinite(sf)
    kin = json.load(open(KIN, encoding="utf-8"))["pairs"]
    from collections import Counter
    source = {
        "celeba": {"images": len(ident), "people": len(set(ident.values())),
                   "test_partition_images": sum(v == "2" for v in part.values())},
        "imdb": {"crops": int(len(paths)), "single_face_crops": int(single.sum()), "people_single_face": len(set(nm[single])),
                 "kinship_pairs_wikidata": dict(Counter(q["relation"] for q in kin))}}

    def count(rows):
        out = {}
        for role in ("pool", "target"):
            rr = [j for j in rows if imgs[j]["role"] == role]
            out[role] = {"images": len(rr), "people": len({imgs[j]["person"] for j in rr})}
        return out

    flow_src = {}
    for src in ("celeba", "imdb"):
        rows = [j for j, d in enumerate(imgs) if d["src"] == src]
        steps = [{"step": "sampled", **count(rows)}]
        alive = set(rows)
        for key, label in FLOW_STEPS:
            gone = {j for j in alive if reasons.get(j) == key}
            alive -= gone
            if gone or key in ("landmarks", "pose"):
                steps.append({"step": key, "label": label, "excluded": count(sorted(gone)), "remaining": count(sorted(alive))})
        other = {j for j in alive if j in reasons}
        if other:
            steps.append({"step": "other", "excluded": count(sorted(other)), "reasons": dict(Counter(reasons[j] for j in other))})
            alive -= other
        cols = res[src]
        final = {"targets_analysed": cols["mean"]["targets"], "pool_people": cols.get("pool_people"),
                 "targets_with_sibling": cols.get("sibling", {}).get("targets")}
        if "ratio_neutral" in cols["mean"]:
            final.update(neutral=cols["mean"].get("targets_neutral"), expressive=cols["mean"].get("targets_expressive"))
        usable = count(sorted(alive))
        final["target_people_without_usable_repeat"] = usable["target"]["people"] - final["targets_analysed"]
        flow_src[src] = {"steps": steps, "usable": usable, "analysis": final}
    masks = None
    mp_ = os.path.join(PROJECT, "results", f"face_masks{SUF}.json")
    if os.path.exists(mp_):
        masks = json.load(open(mp_, encoding="utf-8"))["by_group"]
    out = {"protocol": "docs/PROTOCOL_donor_ladder_face.md", "version": SUF.strip("_") or "v1", "source": source,
           "flow": flow_src, "pixel_masks": masks}
    with open(os.path.join(PROJECT, "results", f"face_flow{SUF}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    # ---- figure
    fmt = lambda c: (f"pool {c['pool']['images']:,} img / {c['pool']['people']:,} people\n"
                     f"targets {c['target']['images']:,} img / {c['target']['people']:,} people")
    fig, ax = plt.subplots(1, 2, figsize=(15, 13))
    head = {"celeba": (f"CelebA (aligned crops)\n{source['celeba']['images']:,} images, {source['celeba']['people']:,} people",
                       "targets: every test-partition image (identities disjoint from train)\n"
                       "pool: 2,500 train identities with >= 3 images, 3 random images each (seed 2026)"),
            "imdb": (f"IMDb-WIKI crops\n{source['imdb']['crops']:,} crops, {source['imdb']['single_face_crops']:,} single-face "
                     f"({source['imdb']['people_single_face']:,} people)",
                     "targets: every single-face crop of every person in a Wikidata kinship pair\n"
                     + ", ".join(f"{v} {k} pairs" for k, v in source['imdb']['kinship_pairs_wikidata'].items())
                     + "\npool: 3,000 people outside kinship with >= 3 crops, 3 random crops each (seed 2026)")}
    box = dict(boxstyle="round,pad=0.5", fc="#f3f2ee", ec="#52514e")
    xbox = dict(boxstyle="round,pad=0.4", fc="#fbeaea", ec="#a33")
    for a, src in zip(ax, ("celeba", "imdb")):
        a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        F = flow_src[src]
        items = [(head[src][0] + "\n" + head[src][1], None)]
        items.append(("Sampled\n" + fmt(F["steps"][0]), None))
        for st in F["steps"][1:]:
            if st["step"] == "other":
                continue
            items.append((fmt(st["remaining"]), st["label"] + "\n" + fmt(st["excluded"])))
        A = F["analysis"]
        last = (f"Analysis set\n{A['targets_analysed']:,} targets (one most-frontal image per person with a repeat)\n"
                f"pool mean over {A['pool_people']:,} people (most-frontal image each)")
        if A.get("targets_with_sibling"):
            last += f"\n{A['targets_with_sibling']:,} targets with a full sibling"
        if A.get("neutral") is not None:
            last += f"\nexpression strata: {A['neutral']:,} neutral, {A['expressive']:,} expressive"
        lost = F["usable"]["target"]["people"] - A["targets_analysed"]
        items.append((last, "no usable repeat\n(fewer than 2 usable images,\nor only near-duplicates r > 0.9)"
                      + f"\ntargets {lost:,} people"))
        n = len(items)
        ys = np.linspace(0.95, 0.04, n)
        for k, ((txt, exc), y) in enumerate(zip(items, ys)):
            a.text(0.3, y, txt, ha="center", va="center", fontsize=8.2, bbox=box)
            if k:
                a.annotate("", xy=(0.3, y + 0.028), xytext=(0.3, ys[k - 1] - 0.03),
                           arrowprops=dict(arrowstyle="->", color="#52514e"))
            if exc:
                yy = 0.5 * (y + ys[k - 1])
                a.text(0.78, yy, "excluded: " + exc, ha="center", va="center", fontsize=7.4, bbox=xbox)
                a.annotate("", xy=(0.58, yy), xytext=(0.3, yy), arrowprops=dict(arrowstyle="->", color="#a33"))
        a.set_title("CelebA" if src == "celeba" else "IMDb-WIKI", fontsize=13)
    note = ("Pixel masks (not exclusions): face-skin class of MediaPipe Selfie Multiclass inside the face-oval hull and the photograph; "
            "illumination field division; 478-landmark affine to a symmetric template; z-score in the mask.")
    if SUF >= "_v6":
        note += ("\nAmendment 7: glasses (template lens ellipses when triggered), facial hair, theatrical makeup and bright "
                 "lipstick (CIELAB colour rules) removed pixel-wise; thresholds calibrated on CelebA attributes, see face_masks.")
    fig.text(0.5, 0.01, note, ha="center", fontsize=8.5, color="#52514e")
    fig.suptitle(f"Face ladder {out['version']}: selection flow (images / people per step)", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PROJECT, "results", "figures", f"fig_face_flow{SUF}.{ext}"), dpi=200)
    print(json.dumps({k: v["analysis"] for k, v in flow_src.items()}))


# ---------------------------------------------------------------- stage: extend (Amendment 4.1)

def extend_pool(workers, cap=10):
    """Amendment 8.3: every pool person gets up to `cap` images (all if fewer; same identities, seed 2026); landmarks
    run on the new rows only. images.json and landmarks.npz are backed up as *_v6.*; existing row indices stay."""
    import shutil
    import scipy.io as sio
    from concurrent.futures import ProcessPoolExecutor
    ip, lp = os.path.join(OUT, "images.json"), os.path.join(OUT, "landmarks.npz")
    for a, b in ((ip, os.path.join(OUT, "images_v6.json")), (lp, os.path.join(OUT, "landmarks_v6.npz"))):
        if not os.path.exists(b):
            shutil.copyfile(a, b)
    imgs = json.load(open(ip, encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    have = {(d["src"], d["file"]) for d in imgs}
    pool = {}
    for d in imgs:
        if d["role"] == "pool":
            pool.setdefault((d["src"], d["person"]), []).append(d)
    new = []
    # CelebA: train-partition images of each pool identity
    dl = os.path.join(CELEBA, "downloads")
    part = dict(l.split() for l in open(os.path.join(dl, "list_eval_partition.txt")))
    ident = dict(l.split() for l in open(os.path.join(dl, "identity_CelebA.txt")))
    by_id = {}
    for k, v in part.items():
        if v == "0":
            by_id.setdefault(ident[k], []).append(k)
    for (src, person), ds in sorted(pool.items()):
        if src != "celeba":
            continue
        cand = sorted(k for k in by_id.get(person, []) if ("celeba", k) not in have)
        take = rng.choice(cand, min(len(cand), max(cap - len(ds), 0)), replace=False) if cand else []
        for k in sorted(take):
            new.append({**{x: ds[0][x] for x in ("src", "role", "person")}, "file": str(k),
                        "male": ds[0]["male"], "young": ds[0]["young"]})
    # IMDb-WIKI: single-face crops of each pool person
    r = sio.loadmat(os.path.join(IMDB, "imdb.mat"), squeeze_me=True)["imdb"]
    paths, fs, sf = r["full_path"].item(), r["face_score"].item(), r["second_face_score"].item()
    dob, taken, gender = r["dob"].item(), r["photo_taken"].item(), r["gender"].item()
    crops = {}
    for i in np.flatnonzero(np.isfinite(fs) & ~np.isfinite(sf)):
        crops.setdefault(re.search(r"nm\d{7}", paths[i]).group(0), []).append(i)
    for (src, person), ds in sorted(pool.items()):
        if src != "imdb":
            continue
        cand = sorted(i for i in crops.get(person, []) if ("imdb", paths[i]) not in have)
        take = rng.choice(cand, min(len(cand), max(cap - len(ds), 0)), replace=False) if cand else []
        for i in sorted(int(i) for i in take):
            y = 1 + (dob[i] - 367) / 365.2425
            new.append({"src": "imdb", "role": "pool", "file": paths[i], "person": person,
                        "male": bool(gender[i] == 1) if np.isfinite(gender[i]) else None,
                        "age": float(int(taken[i]) - math.floor(y)) if dob[i] > 0 else float("nan")})
    n0 = len(imgs)
    path = lambda d: os.path.join(CELEBA, "raw", "img_align_celeba", d["file"]) if d["src"] == "celeba"         else os.path.join(IMDB, *d["file"].split("/"))
    jobs = [(n0 + k, path(d)) for k, d in enumerate(new)]
    z = np.load(lp)
    L = np.concatenate([z["L"], np.full((len(new), 478, 2), np.nan, np.float32)])
    P = np.concatenate([z["P"], np.full((len(new), 3), np.nan, np.float32)])
    E = np.concatenate([z["E"], np.full((len(new), 2), np.nan, np.float32)])
    status = list(z["status"]) + [""] * len(new)
    chunks = [jobs[i::workers * 8] for i in range(workers * 8)]
    with ProcessPoolExecutor(workers) as ex:
        for n, res in enumerate(ex.map(_worker, chunks)):
            for j, st, pts, pose, expr in res:
                status[j] = st
                if st == "ok":
                    L[j], P[j], E[j] = pts, pose, expr
            print(f"landmarks chunk {n + 1}/{len(chunks)}", flush=True)
    np.savez_compressed(lp, L=L, P=P, E=E, status=np.array(status))
    with open(ip, "w", encoding="utf-8") as f:
        json.dump(imgs + new, f)
    st = np.array(status[n0:])
    from collections import Counter
    print({"added_images": dict(Counter(d["src"] for d in new)), "one_face": int((st == "ok").sum()),
           "frontal_20deg": int(((st == "ok") & (np.abs(P[n0:, 0]) < 20) & (np.abs(P[n0:, 1]) < 20)).sum())})


def extend(workers):
    """Add every remaining single-face crop of every kinship person (IMDb-WIKI) and run landmarks on the new rows
    only. The previous images.json and landmarks.npz are kept as *_v2.*; existing row indices do not change."""
    import shutil
    import scipy.io as sio
    from concurrent.futures import ProcessPoolExecutor
    ip, lp = os.path.join(OUT, "images.json"), os.path.join(OUT, "landmarks.npz")
    for a, b in ((ip, os.path.join(OUT, "images_v2.json")), (lp, os.path.join(OUT, "landmarks_v2.npz"))):
        if not os.path.exists(b):
            shutil.copyfile(a, b)
    imgs = json.load(open(ip, encoding="utf-8"))
    have = {d["file"] for d in imgs if d["src"] == "imdb"}
    r = sio.loadmat(os.path.join(IMDB, "imdb.mat"), squeeze_me=True)["imdb"]
    paths, fs, sf = r["full_path"].item(), r["face_score"].item(), r["second_face_score"].item()
    dob, taken, gender = r["dob"].item(), r["photo_taken"].item(), r["gender"].item()
    kin = {x for q in json.load(open(KIN, encoding="utf-8"))["pairs"] for x in (q["a"], q["b"])}
    new = []
    for i in np.flatnonzero(np.isfinite(fs) & ~np.isfinite(sf)):
        person = re.search(r"nm\d{7}", paths[i]).group(0)
        if person in kin and paths[i] not in have:
            y = 1 + (dob[i] - 367) / 365.2425
            new.append({"src": "imdb", "role": "target", "file": paths[i], "person": person,
                        "male": bool(gender[i] == 1) if np.isfinite(gender[i]) else None,
                        "age": float(int(taken[i]) - math.floor(y)) if dob[i] > 0 else float("nan")})
    n0 = len(imgs)
    jobs = [(n0 + k, os.path.join(IMDB, *d["file"].split("/"))) for k, d in enumerate(new)]
    z = np.load(lp)
    L = np.concatenate([z["L"], np.full((len(new), 478, 2), np.nan, np.float32)])
    P = np.concatenate([z["P"], np.full((len(new), 3), np.nan, np.float32)])
    E = np.concatenate([z["E"], np.full((len(new), 2), np.nan, np.float32)])
    status = list(z["status"]) + [""] * len(new)
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_worker, [jobs[i::workers * 8] for i in range(workers * 8)]):
            for j, st, pts, pose, expr in res:
                status[j] = st
                if st == "ok":
                    L[j], P[j], E[j] = pts, pose, expr
    np.savez_compressed(lp, L=L, P=P, E=E, status=np.array(status))
    with open(ip, "w", encoding="utf-8") as f:
        json.dump(imgs + new, f)
    st = np.array(status[n0:])
    print({"added_images": len(new), "one_face": int((st == "ok").sum()),
           "frontal": int(((st == "ok") & (np.abs(P[n0:, 0]) < POSE_MAX) & (np.abs(P[n0:, 1]) < POSE_MAX)).sum())})


# ---------------------------------------------------------------- stage: identity (Amendment 5)

ARCFACE = os.path.join(DATA, "projects", "identity-ladder-metric", "models", "insightface-buffalo-l", "extracted",
                       "w600k_r50.onnx")
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]])  # insightface/utils/face_align.py
ID_COS = 0.3


def prep_pose_max():
    """Amendment 8.4: from v7 the image set is prepared up to 20 deg; maps filter at 10, 15 or 20 deg."""
    return 20 if _version() >= "_v7" else POSE_MAX


def identity_file():
    return "identity_v7.npz" if _version() >= "_v7" else "identity.npz"


def five_points(L):
    """Left eye, right eye (iris centres, by image x), nose tip, left and right mouth corner (by image x)."""
    eyes = sorted([L[IRIS_A], L[IRIS_B]], key=lambda q: q[0])
    mouth = sorted([L[61], L[291]], key=lambda q: q[0])
    return np.array([eyes[0], eyes[1], L[NOSE], mouth[0], mouth[1]])


def person_outliers(E, cos=ID_COS):
    """Indices (into E, rows = one person's L2-normalized embeddings) excluded by the Amendment 5 rule."""
    if len(E) == 1:
        return []
    if len(E) == 2:
        return [0, 1] if float(E[0] @ E[1]) < cos else []
    c = E.mean(0)
    c /= np.linalg.norm(c)
    keep = E @ c >= cos
    if keep.any():
        c = E[keep].mean(0)
        c /= np.linalg.norm(c)
    return list(np.flatnonzero(E @ c < cos))


def _id_worker(chunk, threads=2, batch=32):
    import cv2
    import onnxruntime as ort
    from PIL import Image
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1  # one process must not take every core
    sess = ort.InferenceSession(ARCFACE, sess_options=so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for b in range(0, len(chunk), batch):
        part = chunk[b:b + batch]
        xs = []
        for j, path, pts in part:
            rgb = np.asarray(Image.open(path).convert("RGB"), np.float32)
            crop = cv2.warpAffine(rgb, similarity(pts, ARCFACE_DST), (112, 112), flags=cv2.INTER_LINEAR, borderValue=0)
            xs.append(((crop - 127.5) / 127.5).transpose(2, 0, 1))
        e = sess.run(None, {name: np.stack(xs).astype(np.float32)})[0]
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        out += [(j, e[k]) for k, (j, _, _) in enumerate(part)]
    return out


def identity(workers):
    from concurrent.futures import ProcessPoolExecutor
    imgs = json.load(open(os.path.join(OUT, "images.json"), encoding="utf-8"))
    z = np.load(os.path.join(OUT, "landmarks.npz"))
    L, P, status = z["L"], z["P"], z["status"]
    pmax = prep_pose_max()
    ok = np.flatnonzero((status == "ok") & (np.abs(P[:, 0]) < pmax) & (np.abs(P[:, 1]) < pmax))  # analysable images
    emb = np.full((len(imgs), 512), np.nan, np.float32)
    old = os.path.join(OUT, "identity.npz")
    if _version() >= "_v7" and os.path.exists(old):  # Amendment 8: reuse v4-v6 embeddings, embed only new rows
        e0 = np.load(old)["emb"].astype(np.float32)
        emb[:len(e0)] = e0
    jobs = []
    for j in ok:
        if np.isfinite(emb[j]).all():
            continue
        d = imgs[j]
        path = os.path.join(CELEBA, "raw", "img_align_celeba", d["file"]) if d["src"] == "celeba" \
            else os.path.join(IMDB, *d["file"].split("/"))
        jobs.append((int(j), path, five_points(L[j])))
    chunks = [jobs[i::workers * 4] for i in range(workers * 4)]
    with ProcessPoolExecutor(workers) as ex:
        for n, res in enumerate(ex.map(_id_worker, chunks)):
            for j, e in res:
                emb[j] = e
            print(f"identity chunk {n + 1}/{len(chunks)}", flush=True)
    by = {}
    for j in ok:
        by.setdefault((imgs[j]["src"], imgs[j]["person"]), []).append(int(j))
    bad = np.zeros(len(imgs), bool)
    two = 0
    for (src, person), jj in by.items():
        out = person_outliers(emb[jj])
        two += len(jj) == 2 and len(out) == 2
        for k in out:
            bad[jj[k]] = True
    np.savez_compressed(os.path.join(OUT, identity_file()), emb=emb.astype(np.float16), excluded=bad)
    from collections import Counter
    c = Counter((imgs[j]["src"], imgs[j]["role"]) for j in np.flatnonzero(bad))
    t = Counter((imgs[j]["src"], imgs[j]["role"]) for j in ok)
    print(json.dumps({"embedded": int(len(ok)), "excluded": {" ".join(k): f"{v} of {t[k]}" for k, v in sorted(c.items())},
                      "two_image_people_both_excluded": int(two)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "landmarks", "extend", "identity", "calibrate", "train_beard", "extend_pool", "flow", "frame", "prep", "maps", "figure"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--v2", action="store_true", help="use the Amendment 2 frame (frame_v2.npz) and _v2 outputs")
    ap.add_argument("--suffix", default="", help="output version suffix, e.g. _v3 (Amendment 4)")
    a = ap.parse_args()
    SUF = a.suffix or ("_v2" if a.v2 else "")
    {"select": select, "landmarks": lambda: landmarks(a.workers), "extend": lambda: extend(a.workers), "identity": lambda: identity(a.workers), "calibrate": lambda: calibrate(a.workers), "train_beard": train_beard, "extend_pool": lambda: extend_pool(a.workers), "flow": flow, "frame": frame, "prep": lambda: prep(a.workers),
     "maps": maps, "figure": figure}[a.stage]()
