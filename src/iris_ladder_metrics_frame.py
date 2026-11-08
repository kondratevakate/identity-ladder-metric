"""Iris frame for the metric panel (docs/PROTOCOL_iris_metrics.md): segmentation, rubber-sheet strip, per-image
masks, ladder units. The preprocessing version is an explicit argument everywhere (no default):
  P2 (current): learned segmentation (Worldcoin open-iris UNet++, `iris_ladder_metrics_models`): pupil and limbus
     circles fitted to the predicted pupil boundary and to the iris boundary where it meets the sclera; strip mask =
     predicted iris AND NOT pupil AND NOT eyelashes, minus specular reflections (robust z > 4 against the iris);
  P2e (Amendment P2e, 8 Nov 2026; REJECTED by Kate, removes little of the lid shadow): P2 with an edge margin. A valid sample is kept only if it lies more than
     EDGE_MARGIN_P2E (3) strip samples (Euclidean, wrapped in angle) from every missing sample and from the pupil end
     of the strip; the grey level is then z-scored again inside the kept samples. Same images, same QC, same units
     as P2 (eligibility unchanged); derived from the P2 frames, never rebuilt separately;
  P1 (rejected by Kate on 7 Nov 2026: crooked limbus and pupil on reflections): classical, described below.

P1 steps per image:
  1. working image for segmentation: `iris_codes.load_grey` (red channel for colour images, grey otherwise; width
     320 px; 5 px median), unchanged;
  2. circles. NIR and UPOL: pupil first (darkest disk-like component with its holes filled, so a specular spot inside
     the pupil no longer breaks it), refined by the integro-differential operator; limbus on the lateral sectors.
     UTIRIS visible light: limbus first (the iris and pupil form one dark disk in the red channel of dark eyes), then
     the pupil inside it by the same operator;
  3. strip: Daugman rubber sheet, 64 radial x 256 angular samples between the two circles (radial share 0.05 to 0.95),
     sampled with bilinear interpolation from the full-resolution channel after a Gaussian anti-alias blur with SD
     half the radial sampling step; angle counter-clockwise from the image +x axis with y up, column 0 at 0 deg;
  4. masks per image (missing, never zero): sample outside the image or on the UPOL black frame; specular
     reflections (robust z > 4 within the strip); eyelids and lashes (robust |z| > 3; inside the upper and lower
     90 deg sectors everything outward of the first run of 3 such rows is masked as eyelid);
  5. grey level z-scored inside each strip's own mask (as faces).
Fellow eye: the other eye's strip mirrored about the vertical image axis, which on the strip is the exact column
re-index theta -> 180 deg - theta (no resampling).

Real images and per-image outputs stay in the private folder (`PRIVATE`); nothing here prints a subject id.
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from iris_calibration import upol_table, utiris_table  # noqa: E402
from iris_codes import WORK_WIDTH, circle_means, integro_differential, load_grey  # noqa: E402

VERSIONS = ("P1", "P2", "P2e")
DERIVED = {"P2e": "P2"}            # version -> the version whose frames it is computed from
EDGE_MARGIN_P2E = 3                  # strip samples: kept samples lie further than this from any missing sample
REJECTED = {"P1": "Kate 7 Nov 2026: crooked limbus circles, pupil found on reflections",
            "P2e": "Kate 8 Nov 2026: keep the lid shadow; a 3-sample edge margin removes little of it"}
QC_RATIO_P2, QC_OFFSET_P2, QC_LIMBUS_COVER = (0.10, 0.80), 0.25, 0.20
LID_EDGE_P2 = 3                      # strip samples masked around predicted lid / lash regions (outer 3/4 of radius)
Z_SHADOW_P2 = 4.0                    # robust z below which a sample connected to the lid is lid shadow
PRIVATE = os.path.join(r"C:\Projects\data", "projects", "identity-ladder-metric", "iris-ladder-metrics-20261107-v1")
H, W = 64, 256                       # strip: radial x angular samples
T0, T1 = 0.05, 0.95                  # radial share between pupil and limbus
N_THETA = 64
OPEN_PUPIL, OPEN_IRIS = 8, 12        # px (working image): opening radius that cuts lashes off the pupil / iris disk
TOPHAT_GREY, TOPHAT_PEAK = 50.0, 120.0   # grey levels (0-255) above the 25 px opening: spot extent, spot peak
Z_REFLECT, Z_OCCLUDE, Z_LID, EYELID_RUN, EYELID_MARGIN = 4.0, 3.0, 2.5, 4, 3   # margin: rows of lid edge
EYELID_SECTORS = ((25.0, 155.0), (205.0, 335.0))   # upper, lower (deg, counter-clockwise, y up)
QC_RATIO, QC_OFFSET, QC_MIN_VALID = (0.15, 0.75), 0.25, 0.40
SETS = ("UTIRIS_NIR", "UTIRIS_VIS", "UPOL")
PRIMARY = "UTIRIS_NIR"


# ---------------------------------------------------------------------------------------------- tables

def image_table(name: str) -> pd.DataFrame:
    """Images of one analysis set, sorted by path (the fixed order used to pick target and repeat)."""
    if name == "UPOL":
        t = upol_table()
    else:
        ut, _ = utiris_table()
        t = ut[ut.sensor == name.split("_")[1]]
    if t.empty:
        raise FileNotFoundError(f"no images for {name}")
    return t.sort_values("path").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- segmentation

def dark_disks(img: np.ndarray, fractions, min_area: float, max_area: float, open_r: int) -> list:
    """Every disk-like dark component (holes filled) at every threshold: [(x, y, radius)]."""
    out = []
    for q in fractions:
        try:
            out += dark_disk(img, (q,), min_area, max_area, "all", open_r)
        except ValueError:
            pass
        c = deepest_dark(img, q, min_area)
        if c is not None:
            out.append(c)
    if not out:
        raise ValueError("no disk-like dark region")
    return out


def deepest_dark(img: np.ndarray, q: float, min_area: float):
    """Candidate at the deepest point of the dark mask (maximum of its Euclidean distance transform), radius = that
    distance. Survives a pupil merged with lashes or the lid line, which breaks the disk-shape test."""
    black = img <= 2
    tissue = ~ndimage.binary_dilation(black, iterations=3)
    content = img[tissue]
    lo, med = np.percentile(content, 0.5), np.median(content)
    dark = ndimage.binary_fill_holes((img < lo + q * (med - lo)) & tissue)
    dt = ndimage.distance_transform_edt(dark)
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    r = float(dt[y, x])
    if np.pi * r * r < min_area * img.size:
        return None
    return float(x), float(y), r


def dark_disk(img: np.ndarray, fractions, min_area: float, max_area: float, pick: str, open_r: int):
    """Disk-like dark component (holes filled) at the first threshold that has one: (x, y, radius).

    pick = "largest" (the iris disk of dark eyes in visible light) or "all" (list of every candidate at the first
    threshold that has one). Components touching the image border or the UPOL black frame zone are never candidates.
    The dark mask is opened with a disk of radius open_r px first, which cuts thin lashes and lid bands off the disk."""
    h, w = img.shape
    black = img <= 2                            # UPOL frames the eye with exact black, which is not tissue
    tissue = ~ndimage.binary_dilation(black, iterations=3)       # its anti-aliased edge is not tissue either
    near_frame = ndimage.binary_dilation(black, iterations=8) if black.any() else black
    content = img[tissue]
    if content.size < 0.1 * img.size:
        raise ValueError("image is almost empty")
    lo, med = np.percentile(content, 0.5), np.median(content)
    disk = np.hypot(*np.mgrid[-open_r:open_r + 1, -open_r:open_r + 1]) <= open_r
    for q in fractions:
        dark = ndimage.binary_fill_holes((img < lo + q * (med - lo)) & tissue)
        lab, _ = ndimage.label(ndimage.binary_opening(dark, structure=disk) if open_r > 0 else dark)
        best, every = None, []
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w:
                continue
            comp = lab[sl] == k
            if (comp & near_frame[sl]).any():
                continue
            area = comp.sum()
            if area < min_area * h * w or area > max_area * h * w:
                continue
            if area / (np.pi * (max(comp.shape) / 2) ** 2) < 0.6:
                continue
            cy, cx = ndimage.center_of_mass(comp)
            cand = (float(area), sl[1].start + cx, sl[0].start + cy, float(np.sqrt(area / np.pi)))
            every.append(cand[1:])
            if best is None or cand[0] > best[0]:
                best = cand
        if pick == "all" and every:
            return every
        if pick == "largest" and best is not None:
            return best[1], best[2], best[3]
    raise ValueError("no disk-like dark region")


def _lateral():
    return np.concatenate([np.linspace(-np.pi / 6, np.pi / 6, N_THETA // 2),
                           np.linspace(5 * np.pi / 6, 7 * np.pi / 6, N_THETA // 2)])


def without_reflections(img: np.ndarray) -> np.ndarray:
    """Segmentation input only (never the strip): specular spots replaced by the local grey minimum over 25 x 25 px,
    so a ring reflex inside the pupil no longer splits it. A spot is a connected component of the white top-hat
    (25 px disk) above TOPHAT_GREY whose top-hat peak reaches TOPHAT_PEAK, closed (3 px) with its enclosed
    centre filled and grown by 5 px (the glow around a reflex). Large bright areas (sclera, skin) are not spots; bright iris fibres stay below
    the peak."""
    tissue = img > 2
    disk = np.hypot(*np.mgrid[-12:13, -12:13]) <= 12
    top = ndimage.white_tophat(img, footprint=disk)
    lab, n = ndimage.label((top > TOPHAT_GREY) & tissue)
    if n == 0:
        return img
    peak = ndimage.maximum(top, lab, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], peak >= TOPHAT_PEAK])
    spot = ndimage.binary_closing(keep[lab], iterations=3)
    spot = ndimage.binary_dilation(ndimage.binary_fill_holes(spot), iterations=5) & tissue
    return np.where(spot, ndimage.grey_erosion(img, size=(25, 25)), img)


def disk_contrast(img: np.ndarray, x: float, y: float, r: float) -> float:
    """Mean grey of the ring 1.1r to 1.5r minus mean grey of the disk 0 to 0.9r (pupil: dark inside, iris around)."""
    th = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    cx, cy = np.array([x]), np.array([y])
    inner = circle_means(img, cx, cy, np.linspace(0, 0.9 * r, 6), th).mean()
    outer = circle_means(img, cx, cy, np.linspace(1.1 * r, 1.5 * r, 5), th).mean()
    return float(outer - inner)


def disk_ratio(img: np.ndarray, x: float, y: float, r: float) -> float:
    """Mean grey of the ring 1.1r to 1.5r over (1 + mean grey of the disk 0 to 0.9r). A pupil is dark inside
    relative to its iris ring; a lash clump against bright skin has a large difference but a smaller ratio."""
    th = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    cx, cy = np.array([x]), np.array([y])
    inner = circle_means(img, cx, cy, np.linspace(0, 0.9 * r, 6), th).mean()
    outer = circle_means(img, cx, cy, np.linspace(1.1 * r, 1.5 * r, 5), th).mean()
    return float(outer / (1.0 + inner))


def disk_darkness(img: np.ndarray, x: float, y: float, r: float) -> float:
    """Minus the mean grey of the disk 0 to 0.9r (the pupil is the darkest disk once reflections are removed)."""
    th = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    return -float(circle_means(img, np.array([x]), np.array([y]), np.linspace(0, 0.9 * r, 6), th).mean())


PUPIL_RULE = disk_ratio   # development (no metric seen): contrast picked lash clumps in NIR, darkness failed in UPOL


def segment(img: np.ndarray, name: str) -> dict:
    """Pupil and limbus circles in working-image pixels."""
    img = without_reflections(img)
    h, w = img.shape
    full = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    if name == "UTIRIS_VIS":
        bx, by, br = dark_disk(img, (0.5, 0.6, 0.7, 0.8), 0.01, 0.35, "largest", OPEN_IRIS)
        lx, ly = np.meshgrid(np.arange(bx - 6, bx + 7, 2.0), np.arange(by - 6, by + 7, 2.0))
        k, ri, si = integro_differential(img, lx.ravel(), ly.ravel(), np.arange(0.75 * br, 1.3 * br + 1, 1.0), _lateral())
        ix, iy = lx.ravel()[k], ly.ravel()[k]
        fx, fy = np.meshgrid(np.arange(ix - 0.1 * ri, ix + 0.1 * ri + 1, 2.0), np.arange(iy - 0.1 * ri, iy + 0.1 * ri + 1, 2.0))
        j, rp, sp = integro_differential(img, fx.ravel(), fy.ravel(), np.arange(max(3, 0.15 * ri), 0.7 * ri, 0.5), full)
        px, py = fx.ravel()[j], fy.ravel()[j]
    else:
        best = None                              # the refined candidate with the largest disk contrast
        cands = dark_disks(img, (0.15, 0.25, 0.35, 0.5), 1.5e-3, 0.25, OPEN_PUPIL)
        if name == "UPOL":                       # the device centres the iris in its circular field of view
            ty, tx = ndimage.center_of_mass(img > 2)
            cands = [c for c in cands if np.hypot(c[0] - tx, c[1] - ty) <= 0.2 * w]
            if not cands:
                raise ValueError("no pupil candidate near the field centre")
        for bx, by, br in cands:
            fx, fy = np.meshgrid(np.arange(bx - 3, bx + 4, 1.0), np.arange(by - 3, by + 4, 1.0))
            j, rp_, _ = integro_differential(img, fx.ravel(), fy.ravel(), np.arange(max(3, 0.7 * br), 1.3 * br + 1, 0.5), full)
            c = PUPIL_RULE(img, fx.ravel()[j], fy.ravel()[j], rp_)
            if best is None or c > best[0]:
                best = (c, fx.ravel()[j], fy.ravel()[j], rp_)
        _, px, py, rp = best
        # UPOL: one fixed optical device that frames the whole iris, so the limbus radius range is set by the frame
        # (0.25 to 0.48 of the width) and the collarette cannot pass as the limbus; UTIRIS: 1.5 to 5 pupil radii
        ri_range = (np.arange(max(1.5 * rp, 0.25 * w), 0.48 * w, 1.0) if name == "UPOL"
                    else np.arange(1.5 * rp, min(5.0 * rp, 0.48 * w), 1.0))
        if len(ri_range) < 5:
            raise ValueError("no room for the limbus search")
        lx, ly = np.meshgrid(np.arange(px - 6, px + 7, 2.0), np.arange(py - 6, py + 7, 2.0))
        k, ri, si = integro_differential(img, lx.ravel(), ly.ravel(), ri_range, _lateral())
        ix, iy = lx.ravel()[k], ly.ravel()[k]
    return {"pupil": (float(px), float(py), float(rp)), "iris": (float(ix), float(iy), float(ri)),
            "ratio": float(rp / ri), "offset": float(np.hypot(ix - px, iy - py) / ri)}


# ---------------------------------------------------------------------------------------------- strip and masks

def load_full(path: str, flip: bool = False) -> np.ndarray:
    """Full-resolution channel (red for colour images, grey otherwise), float, no filtering."""
    with Image.open(path) as im:
        im = im.getchannel("R") if im.mode in ("RGB", "RGBA") else im.convert("L")
        if flip:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return np.asarray(im, dtype=np.float64)


def strip_coords(seg: dict, scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Full-resolution (x, y) of the H x W rubber-sheet samples and the radial sampling step in px."""
    (px, py, rp), (ix, iy, ri) = [tuple(v * scale for v in seg[k]) for k in ("pupil", "iris")]
    th = np.linspace(0, 2 * np.pi, W, endpoint=False)
    t = np.linspace(T0, T1, H)[:, None]
    bxp, byp = px + rp * np.cos(th), py - rp * np.sin(th)
    bxi, byi = ix + ri * np.cos(th), iy - ri * np.sin(th)
    xs, ys = bxp + t * (bxi - bxp), byp + t * (byi - byp)
    step = float(np.median(np.hypot(bxi - bxp, byi - byp)) * (T1 - T0) / (H - 1))
    return xs, ys, step


def robust_z(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    med = np.median(v[m])
    mad = 1.4826 * np.median(np.abs(v[m] - med))
    if not mad > 0:
        raise ValueError("constant strip")
    return (v - med) / mad


def occlusion_mask(v: np.ndarray, inside: np.ndarray) -> tuple[np.ndarray, dict]:
    """Valid samples of one strip: inside the image, not reflection, not eyelid or lashes.

    Robust z against the iris itself: the lateral sectors (within 30 deg of 0 and 180 deg) and the inner three
    quarters of the radius, which eyelids rarely cover. An eyelid covering a quarter of the strip would otherwise
    shift the strip's own median and hide itself."""
    th = np.degrees(np.linspace(0, 2 * np.pi, W, endpoint=False))
    lateral = (np.minimum(th, 360 - th) <= 30) | (np.abs(th - 180) <= 30)
    ref = inside & lateral[None, :] & (np.arange(H) < 0.75 * H)[:, None]
    z = robust_z(v, ref if ref.sum() >= 200 else inside)
    refl = ndimage.binary_dilation((z > Z_REFLECT) & inside, iterations=1)
    occ = (np.abs(z) > Z_OCCLUDE) & inside
    lid = np.zeros_like(occ)
    cand = (np.abs(z) > Z_LID) & inside
    run = ndimage.convolve1d(cand.astype(int), np.ones(EYELID_RUN, int), axis=0, mode="constant", origin=-(EYELID_RUN // 2)) >= EYELID_RUN
    for a, b in EYELID_SECTORS:
        for c in np.flatnonzero((th >= a) & (th <= b)):
            rows = np.flatnonzero(run[:, c])
            if rows.size:
                lid[max(rows[0] - EYELID_MARGIN, 0):, c] = True
    valid = inside & ~refl & ~occ & ~lid
    return valid, {"reflection": float(refl.mean()), "eyelid": float(lid.mean()), "other_occlusion": float((occ & ~lid).mean())}


def strip(path: str, seg: dict, work_width_scale: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """(z-scored strip with NaN outside its mask, valid mask, mask shares) of one image."""
    full = load_full(path)
    xs, ys, step = strip_coords(seg, work_width_scale)
    sd = 0.5 * step
    blurred = ndimage.gaussian_filter(full, sd, mode="nearest") if sd > 0.3 else full
    h, w = full.shape
    inside = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    v = ndimage.map_coordinates(blurred, [ys, xs], order=1, mode="nearest")
    frame = ndimage.binary_dilation(full <= 2, iterations=max(1, int(np.ceil(3 * sd))))
    on_frame = ndimage.map_coordinates(frame.astype(np.float64), [ys, xs], order=0, mode="nearest") > 0
    inside &= ~on_frame
    valid, shares = occlusion_mask(v, inside)
    if valid.mean() < 0.05:
        raise ValueError("strip almost fully masked")
    mu, sdv = v[valid].mean(), v[valid].std()
    x = np.where(valid, (v - mu) / sdv, np.nan).astype(np.float32)
    shares["valid"] = float(valid.mean())
    shares["outside_image"] = float((~inside).mean())
    return x, valid, shares


_WC = []


def worldcoin():
    if not _WC:
        import iris_ladder_metrics_models as MM
        _WC.append(MM.Worldcoin())
    return _WC[0]


def boundary_points(mask: np.ndarray, touching: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of the pixels on the edge of `mask` that touch `touching` (8-neighbourhood)."""
    edge = mask & ~ndimage.binary_erosion(mask)
    ys, xs = np.nonzero(edge & ndimage.binary_dilation(touching, structure=np.ones((3, 3), bool)))
    return xs.astype(np.float64), ys.astype(np.float64)


def segment_p2(path: str) -> dict:
    """P2 circles (original pixels) and the Worldcoin maps on the ISO grid with their transform."""
    import iris_ladder_metrics_models as MM
    im = MM.red_or_grey(path)
    maps, scale, px, py = worldcoin().iso(im)
    m = {k: np.nan_to_num(v) > 0.5 for k, v in maps.items()}
    if not m["pupil"].any():
        raise ValueError("no pupil predicted")
    iris = ndimage.binary_fill_holes(m["iris"] | m["pupil"])
    lab, n = ndimage.label(iris)
    iris = lab == 1 + int(np.argmax(ndimage.sum(iris, lab, range(1, n + 1))))
    # the pupil is the predicted pupil component nearest to the centre of the iris disk (a dark crypt can also be
    # predicted as pupil and can be larger)
    lab, n = ndimage.label(m["pupil"] & iris)
    if n == 0:
        raise ValueError("no pupil predicted inside the iris")
    cy, cx = ndimage.center_of_mass(iris)
    cents = ndimage.center_of_mass(m["pupil"], lab, range(1, n + 1))
    k = int(np.argmin([np.hypot(y - cy, x - cx) for y, x in cents]))
    pupil = ndimage.binary_fill_holes(lab == k + 1)
    sclera = m["eyeball"] & ~iris
    xp, yp = boundary_points(pupil, iris & ~pupil)
    xi, yi = boundary_points(iris, sclera)
    if len(xp) < 12 or len(xi) < 12:
        raise ValueError("too few boundary points")
    pcx, pcy, pr, _ = MM.fit_circle(xp, yp)
    icx, icy, ir, _ = MM.fit_circle(xi, yi)
    ang = np.degrees(np.arctan2(-(yi - icy), xi - icx)) % 360
    cover = float(len(np.unique((ang // 10).astype(int))) / 36)
    back = lambda x, y, r: (x / scale - px, y / scale - py, r / scale)  # noqa: E731
    pupil_c, iris_c = back(pcx, pcy, pr), back(icx, icy, ir)
    return {"pupil": pupil_c, "iris": iris_c, "ratio": pupil_c[2] / iris_c[2],
            "offset": float(np.hypot(iris_c[0] - pupil_c[0], iris_c[1] - pupil_c[1]) / iris_c[2]),
            "limbus_cover": cover, "maps": maps, "iso": (scale, px, py)}


def p2_mask(v: np.ndarray, mp: dict, inside: np.ndarray):
    """P2 strip mask from the model maps sampled on the strip (`mp`) and the strip grey values `v`.

    visible iris = iris > 0.5, pupil < 0.5, eyelashes < 0.5, inside the photograph. Removed from it:
      reflections: robust z > 4 against the visible iris, grown by 1 sample;
      lid shadow: samples darker than robust z < -Z_SHADOW_P2 (4) that are connected (8-neighbour) to a predicted lid or lash
        region in the outer three quarters of the radius (the dark lid margin and lash shadow, which the model
        labels as iris); isolated dark crypts are kept;
      lid edge: LID_EDGE_P2 samples around that lid or lash region.
    Returns (valid, reflection, shadow, visible iris)."""
    iris_vis = (mp["iris"] > 0.5) & (mp["pupil"] < 0.5) & (mp["eyelashes"] < 0.5) & inside
    if iris_vis.sum() < 200:
        raise ValueError("almost no visible iris predicted")
    z = robust_z(v, iris_vis)
    refl = ndimage.binary_dilation((z > Z_REFLECT) & inside, iterations=1)
    lid = inside & ((mp["iris"] <= 0.5) | (mp["eyelashes"] >= 0.5)) & (np.arange(H) >= H // 4)[:, None]
    dark = (z < -Z_SHADOW_P2) & inside
    lab, n = ndimage.label(dark | lid, structure=np.ones((3, 3), bool))
    touch = np.unique(lab[lid])
    shadow = dark & np.isin(lab, touch[touch > 0])
    lid_edge = ndimage.binary_dilation(lid, iterations=LID_EDGE_P2)
    valid = iris_vis & ~refl & ~shadow & ~lid_edge
    return valid, refl, shadow, iris_vis


def strip_p2(path: str, seg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """P2 strip: image samples as P1 (Gaussian anti-alias, bilinear) on the fitted circles; valid where the model
    predicts visible iris (not pupil, not eyelashes) and the sample is not a specular reflection or UPOL frame."""
    full = load_full(path)
    xs, ys, step = strip_coords(seg, 1.0)
    sd = 0.5 * step
    blurred = ndimage.gaussian_filter(full, sd, mode="nearest") if sd > 0.3 else full
    h, w = full.shape
    inside = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    v = ndimage.map_coordinates(blurred, [ys, xs], order=1, mode="nearest")
    frame = ndimage.binary_dilation(full <= 2, iterations=max(1, int(np.ceil(3 * sd))))
    inside &= ~(ndimage.map_coordinates(frame.astype(np.float64), [ys, xs], order=0, mode="nearest") > 0)
    scale, px, py = seg["iso"]
    xi, yi = (xs + px) * scale, (ys + py) * scale
    mp = {k: ndimage.map_coordinates(np.nan_to_num(m, nan=-1.0), [yi, xi], order=1, mode="constant", cval=-1.0)
          for k, m in seg["maps"].items()}
    valid, refl, shadow, iris_vis = p2_mask(v, mp, inside)
    mu, sdv = v[valid].mean(), v[valid].std()
    x = np.where(valid, (v - mu) / sdv, np.nan).astype(np.float32)
    shares = {"valid": float(valid.mean()), "reflection": float((refl & iris_vis).mean()),
              "lid_shadow": float((shadow & iris_vis).mean()),
              "not_iris": float((inside & ~((mp["iris"] > 0.5) & (mp["pupil"] < 0.5))).mean()),
              "eyelashes": float((inside & (mp["eyelashes"] >= 0.5)).mean()), "outside_image": float((~inside).mean())}
    return x, valid, shares


def edge_margin(m: np.ndarray, margin: int) -> np.ndarray:
    """Valid samples further than `margin` strip samples (Euclidean) from every missing sample and from the pupil end
    of the strip (row -1 counts as missing). Wrapped in angle; the limbus end is not an edge (it is the frame end at
    T1, not a mask boundary). Never adds samples."""
    if margin < 1:
        raise ValueError("edge margin must be a positive number of samples")
    pw = W // 2
    pad = np.concatenate([m[:, -pw:], m, m[:, :pw]], axis=1)
    pad = np.vstack([np.zeros((1, pad.shape[1]), bool), pad, pad[-1:]])
    d = ndimage.distance_transform_edt(pad)[1:-1, pw:-pw]
    return m & (d > margin)


def with_edge_margin(fr: dict, margin: int) -> dict:
    """P2e frames from P2 frames: masks shrunk by `edge_margin`, grey level z-scored again inside the kept samples
    (exact, since a z-score of a z-score on a subset equals the z-score of the raw values on that subset)."""
    X, M = fr["X"].copy(), fr["M"].copy()
    tab = fr["table"].copy()
    kept = np.full(len(tab), np.nan)
    for i in np.flatnonzero(tab.ok.to_numpy()):
        m = edge_margin(M[i], margin)
        if m.sum() < 2:
            raise ValueError("edge margin removed almost the whole strip")
        v = X[i][m]
        X[i] = np.where(m, (X[i] - v.mean()) / v.std(), np.nan)
        M[i] = m
        kept[i] = m.mean()
    tab["valid_after_margin"] = kept
    return {"table": tab, "X": X.astype(np.float32), "M": M}


def mirror(a: np.ndarray) -> np.ndarray:
    """Strip of the horizontally flipped image: column j -> (W/2 - j) mod W (theta -> 180 deg - theta)."""
    return a[..., (W // 2 - np.arange(W)) % W]


# ---------------------------------------------------------------------------------------------- per set

def check_version(version: str) -> None:
    if version not in VERSIONS:
        raise ValueError(f"preprocessing version must be one of {VERSIONS}, got {version!r}")


def process(name: str, t: pd.DataFrame, version: str, log_every: int = 50) -> dict:
    """Segment and unwrap every image of a set. Failures are kept as rows with ok = False and a reason."""
    check_version(version)
    if version in DERIVED:
        raise ValueError(f"{version} is derived from {DERIVED[version]} frames; use load_or_build")
    return _process_p2(name, t, log_every) if version == "P2" else _process_p1(name, t, log_every)


def _process_p2(name: str, t: pd.DataFrame, log_every: int) -> dict:
    import time
    n = len(t)
    X = np.full((n, H, W), np.nan, np.float32)
    M = np.zeros((n, H, W), bool)
    seg_rows, t0 = [], time.time()
    for i, p in enumerate(t.path):
        row = {"ok": False, "reason": ""}
        try:
            s = segment_p2(p)
            row.update({"px": s["pupil"][0], "py": s["pupil"][1], "rp": s["pupil"][2], "ix": s["iris"][0],
                        "iy": s["iris"][1], "ri": s["iris"][2], "ratio": s["ratio"], "offset": s["offset"],
                        "limbus_cover": s["limbus_cover"]})
            if not (QC_RATIO_P2[0] <= s["ratio"] <= QC_RATIO_P2[1]):
                row["reason"] = "radius ratio"
            elif s["offset"] > QC_OFFSET_P2:
                row["reason"] = "centre offset"
            elif s["limbus_cover"] < QC_LIMBUS_COVER:
                row["reason"] = "limbus seen on too few angles"
            else:
                x, m, sh = strip_p2(p, s)
                row.update(sh)
                if sh["valid"] < QC_MIN_VALID:
                    row["reason"] = "valid share"
                else:
                    X[i], M[i], row["ok"] = x, m, True
        except ValueError as e:
            row["reason"] = f"segmentation: {e}"
        seg_rows.append(row)
        if (i + 1) % log_every == 0 or i + 1 == n:
            el = time.time() - t0
            print(f"{name} P2: {i + 1}/{n} images, {el:.0f}s, ETA {el / (i + 1) * (n - i - 1):.0f}s", flush=True)
    seg = pd.DataFrame(seg_rows)
    return {"table": pd.concat([t.reset_index(drop=True), seg], axis=1), "X": X, "M": M}


def _process_p1(name: str, t: pd.DataFrame, log_every: int) -> dict:
    import time
    n = len(t)
    X = np.full((n, H, W), np.nan, np.float32)
    M = np.zeros((n, H, W), bool)
    seg_rows, t0 = [], time.time()
    for i, p in enumerate(t.path):
        img = load_grey(p)
        with Image.open(p) as im:
            scale = im.width / WORK_WIDTH
        row = {"ok": False, "reason": ""}
        try:
            s = segment(img, name)
            row.update({"px": s["pupil"][0], "py": s["pupil"][1], "rp": s["pupil"][2], "ix": s["iris"][0],
                        "iy": s["iris"][1], "ri": s["iris"][2], "ratio": s["ratio"], "offset": s["offset"]})
            if not (QC_RATIO[0] <= s["ratio"] <= QC_RATIO[1]):
                row["reason"] = "radius ratio"
            elif s["offset"] > QC_OFFSET:
                row["reason"] = "centre offset"
            else:
                x, m, sh = strip(p, s, scale)
                row.update(sh)
                if sh["valid"] < QC_MIN_VALID:
                    row["reason"] = "valid share"
                else:
                    X[i], M[i], row["ok"] = x, m, True
        except ValueError as e:
            row["reason"] = f"segmentation: {e}"
        seg_rows.append(row)
        if (i + 1) % log_every == 0 or i + 1 == n:
            el = time.time() - t0
            print(f"{name}: {i + 1}/{n} images, {el:.0f}s, ETA {el / (i + 1) * (n - i - 1):.0f}s", flush=True)
    seg = pd.DataFrame(seg_rows)
    return {"table": pd.concat([t.reset_index(drop=True), seg], axis=1), "X": X, "M": M}


def units(tab: pd.DataFrame) -> pd.DataFrame:
    """One unit per eye: target = first image (path order) passing segmentation QC, repeat = the second one.
    Fellow = the unit of the same person's other eye. Not chosen by similarity or image quality."""
    rows = []
    for (person, eye), g in tab[tab.ok].groupby(["person", "eye"], sort=True):
        ix = g.index.to_numpy()
        rows.append({"person": person, "side": eye, "idx": int(ix[0]), "repeat_idx": int(ix[1]) if len(ix) > 1 else -1,
                     "n_images_ok": int(len(ix))})
    u = pd.DataFrame(rows)
    key = {(p, s): k for k, (p, s) in enumerate(zip(u.person, u.side))}
    u["fellow"] = [key.get((p, "L" if s == "R" else "R"), -1) for p, s in zip(u.person, u.side)]
    return u


def cache_path(name: str, version: str) -> str:
    check_version(version)
    return os.path.join(PRIVATE, f"frames_{name}_{version}.npz")


def load_or_build(name: str, version: str) -> dict:
    path = cache_path(name, version)
    if version == "P2e" and not os.path.isfile(path):
        out = with_edge_margin(load_or_build(name, DERIVED[version]), EDGE_MARGIN_P2E)
        np.savez_compressed(path, X=out["X"], M=out["M"])
        out["table"].to_csv(path.replace(".npz", "_table.csv.gz"), index=False)
        return out
    if os.path.isfile(path):
        z = np.load(path, allow_pickle=False)
        tab = pd.read_csv(path.replace(".npz", "_table.csv.gz"), dtype={"person": str})
        return {"table": tab, "X": z["X"], "M": z["M"]}
    os.makedirs(PRIVATE, exist_ok=True)
    out = process(name, image_table(name), version)
    np.savez_compressed(path, X=out["X"], M=out["M"])
    out["table"].to_csv(path.replace(".npz", "_table.csv.gz"), index=False)
    return out


# ---------------------------------------------------------------------------------------------- QC sheet (private)

def qc_sheet(name: str, fr: dict, n_rows: int, path: str, version: str) -> None:
    """Evenly spaced target eyes (unit order, never by quality) with their repeat strip and mirrored fellow strip,
    then up to 6 evenly spaced images that failed numeric QC. Eye cropped to 1.3 limbus radii with both circles,
    strip with mask; light blue = missing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    check_version(version)
    tab, X = fr["table"], fr["X"]
    u = units(tab)
    by_idx = {int(r.idx): r for r in u.itertuples()}
    fails = np.flatnonzero(~tab.ok.to_numpy())
    n_fail = min(6, len(fails))
    rows = list(u.idx.to_numpy()[np.linspace(0, len(u) - 1, n_rows - n_fail).round().astype(int)])
    rows += list(fails[np.linspace(0, len(fails) - 1, n_fail).round().astype(int)]) if n_fail else []
    cmap = matplotlib.colormaps["gray"].copy()
    cmap.set_bad("#9ecae1")
    fig, ax = plt.subplots(n_rows, 4, figsize=(16, 2.6 * n_rows + 0.5), gridspec_kw={"width_ratios": [1.1, 2.4, 2.4, 2.4]})
    for r_, i in enumerate(rows):
        row = tab.iloc[i]
        img = load_full(row.path) if version in ("P2", "P2e") else load_grey(row.path)
        a = ax[r_, 0]
        a.imshow(img, cmap="gray")
        if np.isfinite(row.get("rp", np.nan)):
            a.add_patch(plt.Circle((row.px, row.py), row.rp, fill=False, ec="#e33", lw=1.0))
            a.add_patch(plt.Circle((row.ix, row.iy), row.ri, fill=False, ec="#fd0", lw=1.0))
            a.set_xlim(row.ix - 1.3 * row.ri, row.ix + 1.3 * row.ri)
            a.set_ylim(row.iy + 1.3 * row.ri, row.iy - 1.3 * row.ri)
        a.set_title(("pass" if row.ok else f"fail: {row.reason}") + f", {row.eye} eye", fontsize=6,
                    color="#060" if row.ok else "#a00")
        panels = [("strip (unwrapped iris, mask)", X[i] if row.ok else None)]
        un = by_idx.get(i)
        panels.append(("repeat image of the same eye", X[un.repeat_idx] if un is not None and un.repeat_idx >= 0 else None))
        panels.append(("other eye, mirrored", mirror(X[u.idx[un.fellow]]) if un is not None and un.fellow >= 0 else None))
        for c_, (title, im) in enumerate(panels, start=1):
            a = ax[r_, c_]
            if im is None:
                a.text(0.5, 0.5, "n/a" if c_ > 1 else "no strip", ha="center", va="center", fontsize=7, transform=a.transAxes)
            else:
                a.imshow(im, cmap=cmap, vmin=-2.5, vmax=2.5, aspect="auto")
            if r_ == 0:
                a.set_title(title, fontsize=7)
        for a in ax[r_]:
            a.set_xticks([])
            a.set_yticks([])
        ax[r_, 1].set_ylabel("pupil\n\u2193\nlimbus", fontsize=5)
    ax[-1, 1].set_xticks([0, 64, 128, 192, 255])
    ax[-1, 1].set_xticklabels(["0\u00b0", "90\u00b0 (up)", "180\u00b0", "270\u00b0 (down)", ""], fontsize=6)
    fig.suptitle(f"{name}, preprocessing {version}: segmentation (red pupil, yellow limbus), strips "
                 "(light blue = missing). Private.", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def cvrl_agreement(name: str, version: str, every: int = 4) -> dict:
    """Independent check of the P2 circles: CVRL (Notre Dame) circle regressor on every `every`-th image that
    passed QC. Differences in units of the P2 limbus radius; written to the private folder only."""
    import json
    import iris_ladder_metrics_models as MM
    tab = load_or_build(name, version)["table"]
    ok = tab[tab.ok].iloc[::every]
    cv = MM.CVRL()
    d = []
    for r in ok.itertuples():
        c = cv(MM.red_or_grey(r.path))
        d.append({"pupil_centre": np.hypot(c["pupil"][0] - r.px, c["pupil"][1] - r.py) / r.ri,
                  "pupil_radius": (c["pupil"][2] - r.rp) / r.ri,
                  "limbus_centre": np.hypot(c["iris"][0] - r.ix, c["iris"][1] - r.iy) / r.ri,
                  "limbus_radius": (c["iris"][2] - r.ri) / r.ri})
    d = pd.DataFrame(d)
    out = {"set": name, "version": version, "n_images": int(len(d)), "unit": "P2 limbus radius",
           **{k: {"median_abs": float(d[k].abs().median()), "p95_abs": float(d[k].abs().quantile(0.95)),
                  "share_abs_gt_0.1": float((d[k].abs() > 0.1).mean())} for k in d.columns}, **cv.sha}
    with open(os.path.join(PRIVATE, f"cvrl_agreement_{name}_{version}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--qc-rows", type=int, default=24)
    ap.add_argument("--preprocessing", required=True, choices=VERSIONS)
    ap.add_argument("--cvrl-check", action="store_true", help="P2: compare circles with the CVRL regressor")
    a = ap.parse_args()
    if a.cvrl_check:
        for name in a.sets:
            print(cvrl_agreement(name, a.preprocessing), flush=True)
        return
    for name in a.sets:
        fr = load_or_build(name, a.preprocessing)
        tab = fr["table"]
        u = units(tab)
        print(name, "images", len(tab), "pass", int(tab.ok.sum()), f"({tab.ok.mean():.1%})",
              "fail reasons", tab.loc[~tab.ok, "reason"].fillna("").astype(str).str.split(":").str[0].value_counts().to_dict(),
              "eyes", len(u), "with repeat", int((u.repeat_idx >= 0).sum()), "with fellow", int((u.fellow >= 0).sum()),
              "persons", u.person.nunique(), flush=True)
        qc_sheet(name, fr, a.qc_rows, os.path.join(PRIVATE, f"qc_frame_{name}_{a.preprocessing}.png"), a.preprocessing)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
