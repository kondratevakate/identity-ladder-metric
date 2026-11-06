"""Versioned fundus preprocessing (docs/PROTOCOL_donor_ladder_general.md, Amendment 9.2). Version P1.

P1 keeps the P0 disc detector, estimates the disc radius, removes the disc before a ~100 px median background,
looks for the fovea on the background-divided green channel, and accepts it only as a strict minimum away from
the band edge at 1.8-3.5 disc diameters from the disc; otherwise the image is a counted frame failure. The frame
is the Amendment 2b similarity transform; the analysis image is green / background, 256 px, z-scored in the field
of view. P0 (ladder2d_register.fundus_landmarks + z-score) is left untouched for reproduction.
"""
import numpy as np
from scipy import ndimage

VERSION = "P1"
DISC_R, FOVEA_R = np.array([330.0, 256.0]), np.array([181.0, 256.0])   # Amendment 2b targets (512 grid)
G = 256


def field_of_view(rgb):
    return rgb[..., 0] > 0.1


def disc_centre(rgb, fov):
    """P0 disc: brightest point of the smoothed red + green inside the field of view eroded by 25 px."""
    inner = ndimage.binary_erosion(fov, iterations=25)
    bright = np.where(inner, ndimage.gaussian_filter(rgb[..., 0] + rgb[..., 1], 12), -np.inf)
    dy, dx = np.unravel_index(np.argmax(bright), bright.shape)
    return np.array([dx, dy], float), inner


def disc_radius(rgb, fov, disc, lo=20, hi=80):
    """Radius where the smoothed brightness profile around the disc falls below halfway between the peak and the
    median of the 100-150 px ring; clamped to [lo, hi] px on the 512 grid."""
    b = ndimage.gaussian_filter(rgb[..., 0] + rgb[..., 1], 3)
    yy, xx = np.mgrid[:b.shape[0], :b.shape[1]]
    d = np.hypot(xx - disc[0], yy - disc[1])
    ring = fov & (d > 100) & (d < 150)
    if not ring.any():
        return float(lo)
    peak, base = b[int(disc[1]), int(disc[0])], np.median(b[ring])
    half = (peak + base) / 2
    for r in range(lo, hi + 1):
        shell = fov & (d >= r) & (d < r + 1)
        if shell.any() and np.mean(b[shell]) < half:
            return float(r)
    return float(hi)


def background(g, fov, disc=None, radius=None, factor=4, size=25):
    """Median background over size * factor px at 1 / factor resolution; the disc circle (1.5 radii) and the
    outside of the field of view are filled from the nearest valid pixel first."""
    n = g.shape[0] // factor
    use = fov.copy()
    if disc is not None:
        yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
        use &= np.hypot(xx - disc[0], yy - disc[1]) > 1.5 * radius
    gs = g[:n * factor, :n * factor].reshape(n, factor, n, factor)
    us = use[:n * factor, :n * factor].reshape(n, factor, n, factor)
    cnt = us.sum(axis=(1, 3))
    small = np.where(cnt > 0, (gs * us).sum(axis=(1, 3)) / np.maximum(cnt, 1), 0)
    if not (cnt > 0).any():
        return np.ones_like(g)
    _, (iy, ix) = ndimage.distance_transform_edt(~(cnt > 0), return_indices=True)
    bg = ndimage.median_filter(small[iy, ix], size=size, mode="nearest")
    return np.maximum(ndimage.zoom(bg, g.shape[0] / n, order=1)[:g.shape[0], :g.shape[1]], 1e-3)


def landmarks(rgb):
    """P1 landmarks: dict(disc, radius, fovea or None, status)."""
    fov = field_of_view(rgb)
    if fov.sum() < 1000:
        return {"status": "no_fov", "disc": None, "radius": None, "fovea": None}
    disc, inner = disc_centre(rgb, fov)
    rd = disc_radius(rgb, fov, disc)
    g = rgb[..., 1].astype(np.float32)
    flat = np.where(fov, g / background(g, fov, disc, rd), np.inf)
    r = np.sqrt(fov.sum() / np.pi)
    yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
    d = np.hypot(xx - disc[0], yy - disc[1])
    band = (d > 0.35 * r) & (d < 0.75 * r) & (np.abs(yy - disc[1]) < 0.25 * r) & inner
    out = {"disc": disc, "radius": rd, "fovea": None}
    if not band.any():
        return {**out, "status": "no_band"}
    sm = np.where(band, ndimage.gaussian_filter(np.where(fov, flat, 1.0), 8), np.inf)
    fy, fx = np.unravel_index(np.argmin(sm), sm.shape)
    edge = ndimage.distance_transform_edt(band)[fy, fx]
    dist = float(np.hypot(fx - disc[0], fy - disc[1])) / (2 * rd)
    if edge <= 5:
        return {**out, "status": "fovea_at_band_edge"}
    if not 1.8 <= dist <= 3.5:
        return {**out, "status": "fovea_distance"}
    return {**out, "fovea": np.array([fx, fy], float), "status": "ok"}


def canonical(rgb, lm):
    """Amendment 2b similarity frame from the given landmarks, then disc-right (left eyes reflected)."""
    from skimage.transform import SimilarityTransform, warp
    disc, fovea = lm["disc"], lm["fovea"]
    targets = (DISC_R, FOVEA_R) if disc[0] > fovea[0] else (FOVEA_R, DISC_R)
    t = SimilarityTransform()
    t.estimate(np.array(targets), np.array([disc, fovea]))
    c = warp(rgb, t, order=1, mode="constant", cval=0.0).astype(np.float32)
    return c if disc[0] > fovea[0] else c[:, ::-1]


def analysis_image(rgb, lm):
    """P1 analysis image on the 256 grid: canonical green / background (disc removed), z-scored in the field of
    view; returns (image, fov) or None when the frame failed."""
    if lm["status"] != "ok":
        return None
    c = canonical(rgb, lm)
    fov512 = field_of_view(c)
    scale = np.linalg.norm(DISC_R - FOVEA_R) / np.linalg.norm(lm["fovea"] - lm["disc"])
    g = c[..., 1].astype(np.float32)
    flat = np.where(fov512, g / background(g, fov512, DISC_R, lm["radius"] * scale), 0)
    img = flat.reshape(G, 2, G, 2).mean(axis=(1, 3))
    fov = fov512.reshape(G, 2, G, 2).all(axis=(1, 3))
    v = img[fov]
    return np.where(fov, (img - v.mean()) / (v.std() + 1e-6), 0.0).astype(np.float32), fov


# ---------- P2 (Amendment 9.2): laterality from metadata, temporal fovea annulus, frame per dataset ----------

DISC_RADIUS_P2 = 30.0          # disc radius in the disc-only frame (512 grid)
FRAMES = {"PAPILA": "disc", "GRAPE": "disc_fovea"}


def oriented(rgb, side):
    """Reflect left eyes so the disc is on the right (the codebase's 'R' means disc right)."""
    return rgb[:, ::-1] if side == "L" else rgb


def broad_background(g, fov, disc, radius, sigma=60):
    """Gaussian background over the field of view with the disc circle and the outside filled from the nearest
    valid pixel (normalized convolution would bleed the disc in)."""
    yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
    use = fov & (np.hypot(xx - disc[0], yy - disc[1]) > 1.5 * radius)
    _, (iy, ix) = ndimage.distance_transform_edt(~use, return_indices=True)
    return np.maximum(ndimage.gaussian_filter(g[iy, ix], sigma), 1e-3)


def landmarks_p2(rgb, side):
    """P2 landmarks on the oriented image (disc right): dict(disc, radius, fovea or None, status)."""
    rgb = oriented(rgb, side)
    fov = field_of_view(rgb)
    if fov.sum() < 1000:
        return {"status": "no_fov", "disc": None, "radius": None, "fovea": None}
    disc, inner = disc_centre(rgb, fov)
    rd = disc_radius(rgb, fov, disc)
    out = {"disc": disc, "radius": rd, "fovea": None}
    g = rgb[..., 1].astype(np.float32)
    flat = g / broad_background(g, fov, disc, rd)
    yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
    d = np.hypot(xx - disc[0], yy - disc[1]) / (2 * rd)
    ann = (d >= 1.8) & (d <= 3.5) & (xx < disc[0]) & (np.abs(yy - disc[1]) < 2 * rd) & inner
    if not ann.any():
        return {**out, "status": "no_annulus"}
    sm = np.where(ann, ndimage.gaussian_filter(np.where(fov, flat, 1.0), 8), np.inf)
    fy, fx = np.unravel_index(np.argmin(sm), sm.shape)
    if ndimage.distance_transform_edt(ann)[fy, fx] <= 3:
        return {**out, "status": "fovea_at_annulus_edge"}
    return {**out, "fovea": np.array([fx, fy], float), "status": "ok"}


def frame_ok(lm, frame):
    return lm["disc"] is not None and (frame == "disc" or lm["status"] == "ok")


def canonical_p2(rgb, lm, side, frame):
    """Oriented image warped to the dataset's frame: disc-fovea similarity, or disc-only (translation + scale)."""
    from skimage.transform import SimilarityTransform, warp
    rgb = oriented(rgb, side)
    t = SimilarityTransform()
    if frame == "disc_fovea":
        t.estimate(np.array([DISC_R, FOVEA_R]), np.array([lm["disc"], lm["fovea"]]))
        scale = np.linalg.norm(DISC_R - FOVEA_R) / np.linalg.norm(lm["fovea"] - lm["disc"])
    else:
        scale = DISC_RADIUS_P2 / lm["radius"]
        t = SimilarityTransform(scale=1 / scale, translation=lm["disc"] - DISC_R / scale)
    return warp(rgb, t, order=1, mode="constant", cval=0.0).astype(np.float32), scale


def analysis_image_p2(rgb, lm, side, frame):
    """P2 analysis image (256 grid): frame green / background (~100 px, disc removed), z-scored in the field of
    view; (image, fov) or None on a frame failure."""
    if not frame_ok(lm, frame):
        return None
    c, scale = canonical_p2(rgb, lm, side, frame)
    fov512 = field_of_view(c)
    g = c[..., 1].astype(np.float32)
    flat = np.where(fov512, g / background(g, fov512, DISC_R, lm["radius"] * scale), 0)
    img = flat.reshape(G, 2, G, 2).mean(axis=(1, 3))
    fov = fov512.reshape(G, 2, G, 2).all(axis=(1, 3))
    v = img[fov]
    return np.where(fov, (img - v.mean()) / (v.std() + 1e-6), 0.0).astype(np.float32), fov


# ---------- P3 (Amendment 9.2): flattened disc detection, FOV-relative fovea search, no radius-based scale ----------

def fill_outside(x, valid):
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return x[iy, ix]


def landmarks_p3(rgb, side):
    """P3 landmarks on the oriented image (disc right)."""
    rgb = oriented(rgb, side)
    fov = field_of_view(rgb)
    out = {"disc": None, "radius": None, "fovea": None}
    if fov.sum() < 1000:
        return {**out, "status": "no_fov"}
    r = np.sqrt(fov.sum() / np.pi)
    inner = ndimage.binary_erosion(fov, iterations=25)
    rg = (rgb[..., 0] + rgb[..., 1]).astype(np.float32)
    ratio = ndimage.gaussian_filter(rg, 12) / np.maximum(ndimage.gaussian_filter(fill_outside(rg, fov), 60), 1e-3)
    score = np.where(inner, ratio, -np.inf)
    dy, dx = np.unravel_index(np.argmax(score), score.shape)
    if score[dy, dx] < 1.3:
        return {**out, "status": "disc_low_contrast"}
    disc = np.array([dx, dy], float)
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    d = np.hypot(xx - dx, yy - dy)
    half = (score[dy, dx] + 1.0) / 2
    rd = 0.20 * r
    for rr in range(int(0.08 * r), int(0.20 * r) + 1):
        shell = fov & (d >= rr) & (d < rr + 1)
        if shell.any() and np.mean(ratio[shell]) < half:
            rd = float(rr)
            break
    out = {**out, "disc": disc, "radius": max(rd, 0.08 * r)}
    g = rgb[..., 1].astype(np.float32)
    use = fov & (d > 2 * out["radius"])
    flat = g / np.maximum(ndimage.gaussian_filter(fill_outside(g, use), 60), 1e-3)
    region = (d >= 0.40 * r) & (d <= 0.95 * r) & (xx < dx) & (np.abs(yy - dy) < 0.30 * r) & inner
    if not region.any():
        return {**out, "status": "no_region"}
    sm = ndimage.gaussian_filter(np.where(fov, flat, 1.0), 8)
    cand = np.where(region, sm, np.inf)
    fy, fx = np.unravel_index(np.argmin(cand), cand.shape)
    if ndimage.distance_transform_edt(region)[fy, fx] <= 3:
        return {**out, "status": "fovea_at_region_edge"}
    if sm[fy, fx] > 0.9 * np.median(sm[region]):
        return {**out, "status": "fovea_not_dark"}
    return {**out, "fovea": np.array([fx, fy], float), "status": "ok"}


def canonical_p3(rgb, lm, side, frame):
    """GRAPE: disc-fovea similarity; PAPILA: translation of the disc to its target (scale 1)."""
    from skimage.transform import SimilarityTransform, warp
    rgb = oriented(rgb, side)
    if frame == "disc_fovea":
        t = SimilarityTransform()
        t.estimate(np.array([DISC_R, FOVEA_R]), np.array([lm["disc"], lm["fovea"]]))
        scale = np.linalg.norm(DISC_R - FOVEA_R) / np.linalg.norm(lm["fovea"] - lm["disc"])
    else:
        t, scale = SimilarityTransform(translation=lm["disc"] - DISC_R), 1.0
    return warp(rgb, t, order=1, mode="constant", cval=0.0).astype(np.float32), scale


def frame_ok_p3(lm, frame):
    return lm["disc"] is not None and (frame == "disc" or lm["status"] == "ok")


def analysis_image_p3(rgb, lm, side, frame):
    if not frame_ok_p3(lm, frame):
        return None
    c, scale = canonical_p3(rgb, lm, side, frame)
    fov512 = field_of_view(c)
    g = c[..., 1].astype(np.float32)
    flat = np.where(fov512, g / background(g, fov512, DISC_R, 2 / 1.5 * lm["radius"] * scale), 0)
    img = flat.reshape(G, 2, G, 2).mean(axis=(1, 3))
    fov = fov512.reshape(G, 2, G, 2).all(axis=(1, 3))
    v = img[fov]
    return np.where(fov, (img - v.mean()) / (v.std() + 1e-6), 0.0).astype(np.float32), fov


# ---------- P4 (Amendment 9.2): disc centre from the bright region, glare rejection, fovea away from the edge ----------

def disc_p4(rgb, fov, r, inner):
    """Disc centre and radius from the bright region around the best of up to 3 candidates; None if none fits."""
    rg = (rgb[..., 0] + rgb[..., 1]).astype(np.float32)
    ratio = ndimage.gaussian_filter(rg, 12) / np.maximum(ndimage.gaussian_filter(fill_outside(rg, fov), 60), 1e-3)
    score = np.where(inner, ratio, -np.inf)
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    for _ in range(3):
        dy, dx = np.unravel_index(np.argmax(score), score.shape)
        peak = score[dy, dx]
        if not np.isfinite(peak) or peak < 1.3:
            return None, "disc_low_contrast"
        near = np.hypot(xx - dx, yy - dy) < 0.25 * r
        lab, _ = ndimage.label(near & fov & (ratio >= (peak + 1.0) / 2))
        blob = lab == lab[dy, dx]
        rad = float(np.sqrt(blob.sum() / np.pi))
        if 0.05 * r <= rad <= 0.22 * r:
            cy, cx = ndimage.center_of_mass(blob)
            return (np.array([cx, cy], float), rad), "ok"
        score = np.where(np.hypot(xx - dx, yy - dy) < max(rad, 0.05 * r) * 1.5, -np.inf, score)
    return None, "disc_shape"


def landmarks_p4(rgb, side):
    rgb = oriented(rgb, side)
    fov = field_of_view(rgb)
    out = {"disc": None, "radius": None, "fovea": None}
    if fov.sum() < 1000:
        return {**out, "status": "no_fov"}
    r = np.sqrt(fov.sum() / np.pi)
    inner = ndimage.binary_erosion(fov, iterations=25)
    found, status = disc_p4(rgb, fov, r, inner)
    if found is None:
        return {**out, "status": status}
    disc, rd = found
    out = {**out, "disc": disc, "radius": rd}
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    d = np.hypot(xx - disc[0], yy - disc[1])
    g = rgb[..., 1].astype(np.float32)
    flat = g / np.maximum(ndimage.gaussian_filter(fill_outside(g, fov & (d > 2 * rd)), 60), 1e-3)
    deep = ndimage.distance_transform_edt(fov) >= 0.25 * r
    region = (d >= 0.40 * r) & (d <= 0.95 * r) & (xx < disc[0]) & (np.abs(yy - disc[1]) < 0.30 * r) & deep
    if not region.any():
        return {**out, "status": "no_region"}
    sm = ndimage.gaussian_filter(np.where(fov, flat, 1.0), 8)
    cand = np.where(region, sm, np.inf)
    fy, fx = np.unravel_index(np.argmin(cand), cand.shape)
    if ndimage.distance_transform_edt(region)[fy, fx] <= 3:
        return {**out, "status": "fovea_at_region_edge"}
    if sm[fy, fx] > 0.9 * np.median(sm[region]):
        return {**out, "status": "fovea_not_dark"}
    return {**out, "fovea": np.array([fx, fy], float), "status": "ok"}


# ---------- P5 (Amendment 9.2): disc centre at disc scale, raw-brightness gate ----------

def disc_p5(rgb, fov, r, inner):
    rg = (rgb[..., 0] + rgb[..., 1]).astype(np.float32)
    bg = np.maximum(ndimage.gaussian_filter(fill_outside(rg, fov), 60), 1e-3)
    ratio12 = ndimage.gaussian_filter(rg, 12) / bg
    ratio25 = ndimage.gaussian_filter(rg, 25) / bg
    raw12 = ndimage.gaussian_filter(rg, 12)
    gate = inner & (raw12 >= np.percentile(raw12[fov], 75))
    score = np.where(gate, ratio25, -np.inf)
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    for _ in range(3):
        dy, dx = np.unravel_index(np.argmax(score), score.shape)
        if not np.isfinite(score[dy, dx]) or ratio12[dy, dx] < 1.3:
            return None, "disc_low_contrast"
        peak = ratio12[dy, dx]
        near = np.hypot(xx - dx, yy - dy) < 0.25 * r
        lab, _ = ndimage.label(near & fov & (ratio12 >= (peak + 1.0) / 2))
        blob = lab == lab[dy, dx] if lab[dy, dx] else np.zeros_like(near)
        rad = float(np.sqrt(blob.sum() / np.pi))
        if rad >= 0.04 * r:
            return (np.array([dx, dy], float), float(np.clip(rad, 0.08 * r, 0.20 * r))), "ok"
        score = np.where(np.hypot(xx - dx, yy - dy) < 0.08 * r, -np.inf, score)
    return None, "disc_shape"


def landmarks_p5(rgb, side):
    import types
    patched = types.SimpleNamespace(disc=disc_p5)
    global disc_p4
    saved, disc_p4 = disc_p4, patched.disc
    try:
        return landmarks_p4(rgb, side)
    finally:
        disc_p4 = saved


# ---------- P6 (Amendment 9.2): PAPILA disc from the expert segmentations; GRAPE = P4 ----------

def polygon(path):
    xy = np.loadtxt(path)
    x, y = xy[:, 0], xy[:, 1]
    a = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * a)
    cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * a)
    return np.array([cx, cy]), float(np.sqrt(abs(a) / np.pi))


def papila_expert_disc(image_path, size=512):
    """Mean expert disc (centre, radius) on the load_base grid, or None if a contour is missing."""
    import os
    from PIL import Image
    base = os.path.splitext(os.path.basename(image_path))[0]
    root = os.path.join(os.path.dirname(os.path.dirname(image_path)), "ExpertsSegmentations", "Contours")
    files = [os.path.join(root, f"{base}_disc_exp{k}.txt") for k in (1, 2)]
    if not all(os.path.exists(f) for f in files):
        return None
    w, h = Image.open(image_path).size
    s = min(w, h)
    off, k = np.array([(w - s) // 2, (h - s) // 2], float), size / s
    cs, rs = zip(*(polygon(f) for f in files))
    return (np.mean(cs, axis=0) - off) * k, float(np.mean(rs)) * k


def landmarks_p6(rgb, side, image_path=None, dataset="GRAPE"):
    if dataset != "PAPILA":
        return landmarks_p4(rgb, side)
    out = {"disc": None, "radius": None, "fovea": None}
    found = papila_expert_disc(image_path) if image_path else None
    if found is None:
        return {**out, "status": "no_expert_contour"}
    disc, rd = found
    if side == "L":                                    # oriented() reflects left eyes
        disc = np.array([rgb.shape[1] - 1 - disc[0], disc[1]])
    return {**out, "disc": disc, "radius": rd, "status": "ok"}


# ---------- P7 (Amendment 9.2): GRAPE = P4 + the P5 raw-brightness gate on disc candidates; PAPILA = P6 ----------

def disc_p7(rgb, fov, r, inner):
    rg = (rgb[..., 0] + rgb[..., 1]).astype(np.float32)
    raw12 = ndimage.gaussian_filter(rg, 12)
    return disc_p4(rgb, fov, r, inner & (raw12 >= np.percentile(raw12[fov], 75)))


def landmarks_p7(rgb, side, image_path=None, dataset="GRAPE"):
    if dataset == "PAPILA":
        return landmarks_p6(rgb, side, image_path, dataset)
    global disc_p4
    saved = disc_p4
    disc_p4 = lambda rgb_, fov_, r_, inner_: saved(rgb_, fov_, r_, inner_ & (  # noqa: E731
        ndimage.gaussian_filter((rgb_[..., 0] + rgb_[..., 1]).astype(np.float32), 12)
        >= np.percentile(ndimage.gaussian_filter((rgb_[..., 0] + rgb_[..., 1]).astype(np.float32), 12)[fov_], 75)))
    try:
        return landmarks_p4(rgb, side)
    finally:
        disc_p4 = saved


# ---------- P8 (Amendment 9.2): clean analysis mask (no notch, no dark rim); landmarks and frames as P7 ----------

def clean_fov(c):
    """Threshold -> largest component -> opening (disc 0.12 r) -> erosion (0.03 r)."""
    m = field_of_view(c)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    r = np.sqrt(m.sum() / np.pi)
    k = int(round(0.12 * r))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    m = ndimage.binary_opening(m, structure=(xx ** 2 + yy ** 2) <= k * k)
    return ndimage.binary_erosion(m, iterations=max(int(round(0.03 * r)), 1))


def analysis_image_p8(rgb, lm, side, frame):
    if not frame_ok_p3(lm, frame):
        return None
    c, scale = canonical_p3(rgb, lm, side, frame)
    fov512 = clean_fov(c)
    g = c[..., 1].astype(np.float32)
    flat = np.where(fov512, g / background(g, fov512, DISC_R, 2 / 1.5 * lm["radius"] * scale), 0)
    img = flat.reshape(G, 2, G, 2).mean(axis=(1, 3))
    fov = fov512.reshape(G, 2, G, 2).all(axis=(1, 3))
    v = img[fov]
    return np.where(fov, (img - v.mean()) / (v.std() + 1e-6), 0.0).astype(np.float32), fov


# ---------- P9 (Amendment 9.2): PAPILA whole photo on a larger canvas, disc to the centre, outside = missing ----------

P9_RAW = 3120.0                       # raw px mapped onto the 512 canvas (field diameter + 1-99% spread of disc dx)
P9_CENTRE = np.array([278.0, 247.0])  # disc target: middle of the range that keeps the 1-99% offsets fully inside
K_P8 = 512 / 1934.0                   # raw -> 512 scale of P0-P8 (centre square of the PAPILA photo)


def papila_expert_disc_raw(image_path):
    """Mean expert disc (centre, radius) in raw photo px, or None if a contour is missing."""
    import os
    base = os.path.splitext(os.path.basename(image_path))[0]
    root = os.path.join(os.path.dirname(os.path.dirname(image_path)), "ExpertsSegmentations", "Contours")
    files = [os.path.join(root, f"{base}_disc_exp{k}.txt") for k in (1, 2)]
    if not all(os.path.exists(f) for f in files):
        return None
    cs, rs = zip(*(polygon(f) for f in files))
    return np.mean(cs, axis=0), float(np.mean(rs))


def canvas_p9(small, disc, side, size=512):
    """Place the (already k-scaled) photo on a size x size canvas with the disc at P9_CENTRE, left eyes reflected.
    Returns (canvas rgb, photo mask): the mask marks where the photo is, so outside is missing by construction."""
    from skimage.transform import SimilarityTransform, warp
    disc = np.asarray(disc, float).copy()
    if side == "L":
        small = small[:, ::-1]
        disc[0] = small.shape[1] - 1 - disc[0]
    t = SimilarityTransform(translation=disc - P9_CENTRE)
    c = warp(small, t, output_shape=(size, size), order=1, mode="constant", cval=0.0).astype(np.float32)
    inside = warp(np.ones(small.shape[:2]), t, output_shape=(size, size), order=0, mode="constant", cval=0.0) > 0.5
    return c, inside


def analysis_image_p9(image_path, side):
    """(image, fov, clipped) on the 256 grid, or None without an expert disc. `clipped`: field reaches the canvas
    edge (acquired field lost there)."""
    from PIL import Image
    found = papila_expert_disc_raw(image_path)
    if found is None:
        return None
    k = 512 / P9_RAW
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    small = np.asarray(im.resize((int(round(w * k)), int(round(h * k))), Image.BOX), np.float32) / 255.0
    disc, rd = found
    c, inside = canvas_p9(small, disc * k, side)
    field = field_of_view(c) & inside
    clipped = bool(field[0].any() or field[-1].any() or field[:, 0].any() or field[:, -1].any())
    fov512 = clean_fov(np.where(inside[..., None], c, 0.0)) & inside
    g = c[..., 1].astype(np.float32)
    size = max(int(round(25 * k / K_P8)), 3)          # same retina per background window as P8
    flat = np.where(fov512, g / background(g, fov512, P9_CENTRE, 2 / 1.5 * rd * k, size=size), 0)
    img = flat.reshape(G, 2, G, 2).mean(axis=(1, 3))
    fov = fov512.reshape(G, 2, G, 2).all(axis=(1, 3))
    v = img[fov]
    return np.where(fov, (img - v.mean()) / (v.std() + 1e-6), 0.0).astype(np.float32), fov, clipped
