"""BACKLOG DRAFT, never run or tested (separate paper; see the protocol). Ladder-based harmonization scorecard on GRAPE cameras (docs/PROTOCOL_harmonization_scorecard.md). Pre-QC.

Each harmonization variant transforms the cached disc-fovea images; five checks with theory targets then show
whether camera effects are gone (1-3) and whether the person's own pattern survived (4-5). Aggregates only.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_fundus as F  # noqa: E402

SEED = 20261102
ROOT = os.path.join(os.path.dirname(__file__), "..")
ALPHA = 0.7          # content_swap: share of the image's own deviation that is kept
BLUR_SIGMA = 3.0
VARIANTS = ("none", "device_shift", "device_location_scale", "content_swap", "blur")


def device_maps(img, fov, dev):
    """Pixelwise mean and SD per camera and pooled, over field-of-view pixels."""
    maps = {}
    for d in np.unique(dev):
        sel = dev == d
        w = fov[sel].astype(np.float64)
        n = np.maximum(w.sum(0), 1)
        m = (img[sel] * w).sum(0) / n
        s = np.sqrt(np.maximum((((img[sel] - m) ** 2) * w).sum(0) / n, 1e-6))
        maps[d] = (m, s)
    w = fov.astype(np.float64)
    n = np.maximum(w.sum(0), 1)
    m = (img * w).sum(0) / n
    s = np.sqrt(np.maximum((((img - m) ** 2) * w).sum(0) / n, 1e-6))
    return maps, (m, s)


def harmonize(variant, img, fov, dev, person, rng):
    if variant == "none":
        return img
    maps, (pm, ps) = device_maps(img, fov, dev)
    out = np.empty_like(img)
    for i in range(len(img)):
        m, s = maps[dev[i]]
        if variant == "device_shift":
            out[i] = img[i] - m + pm
        elif variant in ("device_location_scale", "blur"):
            out[i] = (img[i] - m) / s * ps + pm
        elif variant == "content_swap":
            cand = np.flatnonzero((dev == dev[i]) & (person != person[i]))
            j = rng.choice(cand)
            out[i] = m + ALPHA * (img[i] - m) + np.sqrt(1 - ALPHA ** 2) * (img[j] - m)
    if variant == "blur":
        for i in range(len(out)):
            num = ndimage.gaussian_filter(np.where(fov[i], out[i], 0), BLUR_SIGMA)
            den = ndimage.gaussian_filter(fov[i].astype(np.float32), BLUR_SIGMA)
            out[i] = np.where(fov[i], num / np.maximum(den, 1e-3), 0)
    return np.where(fov, out, 0).astype(np.float32)


def ratio_ci(num, den, clusters, n_boot=2000, seed=SEED):
    """sum(num)/sum(den) with a cluster bootstrap; num and den are per-target arrays."""
    keep = np.isfinite(num) & np.isfinite(den)
    num, den, clusters = num[keep], den[keep], clusters[keep]
    if len(num) == 0:
        return None, [None, None], 0
    up, inv = np.unique(clusters, return_inverse=True)
    gn, gd = np.bincount(inv, num), np.bincount(inv, den)
    rng = np.random.default_rng(seed)
    bs = [gn[s].sum() / gd[s].sum() for s in (rng.integers(0, len(up), len(up)) for _ in range(n_boot))]
    return float(num.sum() / den.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], int(len(num))


def score(img, fov, t, u, patches):
    """Per-target sums over patches for every check; returns aggregates."""
    idx = u.idx.to_numpy()
    n = len(u)
    person, side, dev = u.person.to_numpy(), u.side.to_numpy(), u.device.to_numpy()
    full_ok = t.idx.isin(set(t.idx)) if "idx" in t else None  # noqa: F841  (kept for clarity)
    acc = {k: np.zeros(n) for k in ("V", "dev_mean", "str_same", "mean_same", "str_other", "mean_other",
                                    "visit_same", "Vvs", "visit_other", "Vvo", "fellow", "Vf")}
    cnt = {k: np.zeros(n) for k in ("V", "same", "other", "vs", "vo", "f")}
    # other visits of the same eye from the full table, split by camera (nearest in time)
    pairs = {}
    for i, r in enumerate(u.itertuples()):
        c = t[(t.person == r.person) & (t.side == r.side) & t.ok & ~t.side_conflict & (t.visit != r.visit)]
        for kind, sub in (("same", c[c.device == r.device]), ("other", c[c.device != r.device])):
            if len(sub):
                pairs[i, kind] = int(sub.iloc[np.argmin(np.abs(sub.time.to_numpy() - r.time))].idx)
    fellow = {i: int(np.flatnonzero((person == person[i]) & (side != side[i]))[0])
              for i in range(n) if ((person == person[i]) & (side != side[i])).any()}
    for (y, x, s) in patches:
        win = (slice(y, y + s), slice(x, x + s))
        Xa = img[:, win[0], win[1]].reshape(len(img), -1).astype(np.float64)
        va = fov[:, win[0], win[1]].all(axis=(1, 2))
        X, valid = Xa[idx], va[idx]
        sq = (X ** 2).sum(1)
        D = (sq[:, None] + sq[None] - 2 * X @ X.T) / X.shape[1]
        for i in np.flatnonzero(valid):
            pool = np.flatnonzero((person != person[i]) & (side == side[i]) & valid)
            if pool.size < 10:
                continue
            V = np.mean((X[i] - X[pool].mean(0)) ** 2)
            acc["V"][i] += V; cnt["V"][i] += 1
            same, other = pool[dev[pool] == dev[i]], pool[dev[pool] != dev[i]]
            if same.size >= 5 and other.size >= 5:
                acc["dev_mean"][i] += np.mean((X[i] - X[same].mean(0)) ** 2)
                acc["str_same"][i] += D[i, same].mean(); acc["mean_same"][i] += np.mean((X[i] - X[same].mean(0)) ** 2)
                acc["str_other"][i] += D[i, other].mean(); acc["mean_other"][i] += np.mean((X[i] - X[other].mean(0)) ** 2)
                cnt["same"][i] += 1
            for kind, key, vk in (("same", "visit_same", "Vvs"), ("other", "visit_other", "Vvo")):
                j = pairs.get((i, kind))
                if j is not None and va[j]:
                    acc[key][i] += np.mean((X[i] - Xa[j]) ** 2); acc[vk][i] += V
                    cnt["vs" if kind == "same" else "vo"][i] += 1
            j = fellow.get(i)
            if j is not None and valid[j]:
                acc["fellow"][i] += np.mean((X[i] - X[j]) ** 2); acc["Vf"][i] += V; cnt["f"][i] += 1
    cl = person
    out = {}
    m = cnt["same"] > 0
    e, ci, nn = ratio_ci(acc["dev_mean"][m], acc["V"][m] * (cnt["same"][m] / cnt["V"][m]), cl[m])
    out["1_camera_R2"] = {"value": None if e is None else 1 - e, "ci95": [None if c is None else 1 - c for c in ci[::-1]],
                          "n": nn, "target": 0}
    for kind in ("same", "other"):
        e, ci, nn = ratio_ci(acc[f"str_{kind}"][m], acc[f"mean_{kind}"][m], cl[m])
        out[f"2_stranger_over_mean_{kind}_camera"] = {"value": e, "ci95": ci, "n": nn, "target": "2/(1+1/N)"}
    vdev = {d: float(acc["V"][(dev == d) & (cnt["V"] > 0)].sum() / cnt["V"][(dev == d) & (cnt["V"] > 0)].sum())
            for d in np.unique(dev)}
    out["3_V_by_camera"] = {"max_over_min": max(vdev.values()) / min(vdev.values()), "target": 1,
                            "n_cameras": len(vdev)}
    for key, vk, label in (("visit_same", "Vvs", "4_r_same_eye_same_camera"), ("visit_other", "Vvo", "4_r_same_eye_other_camera"),
                           ("fellow", "Vf", "5_r_fellow_eye")):
        mm = acc[vk] > 0
        e, ci, nn = ratio_ci(acc[key][mm], acc[vk][mm], cl[mm])
        out[label] = {"value": None if e is None else 1 - e / 2,
                      "ci95": [None if c is None else 1 - c / 2 for c in ci[::-1]], "n": nn}
    return out


def main():
    t = F.grape_table()
    z = np.load(os.path.join(F.PRIVATE, "GRAPE_canonical.npz"))
    img, fov, ok, dr = z["img"], z["fov"], z["ok"], z["disc_right"]
    u = F.units(t, ok, dr)
    t = t.assign(idx=np.arange(len(t)), ok=ok, side_conflict=ok & (dr != (t.side == "R")))
    patches, _ = F.patch_positions(fov[u.idx.to_numpy()], np.random.default_rng(F.SEED))
    inc = (t.ok & ~t.side_conflict).to_numpy()
    res = {"status": "pre-QC; demonstration of the scorecard, not a benchmark", "protocol":
           "docs/PROTOCOL_harmonization_scorecard.md", "dataset": "GRAPE", "n_units": len(u),
           "n_images_included": int(inc.sum()), "patches": len(patches), "variants": {}}
    for v in VARIANTS:
        h = img.copy()
        sel = np.flatnonzero(inc)
        h[sel] = harmonize(v, img[sel], fov[sel], t.device.to_numpy()[sel], t.person.to_numpy()[sel],
                           np.random.default_rng(SEED))
        res["variants"][v] = score(h, fov, t, u, patches)
        print(v, json.dumps({k: (round(x["value"], 3) if isinstance(x.get("value"), float) else x.get("max_over_min"))
                             for k, x in res["variants"][v].items()}), flush=True)
    with open(os.path.join(ROOT, "results", "harmonization_scorecard.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, allow_nan=False)


if __name__ == "__main__":
    main()
