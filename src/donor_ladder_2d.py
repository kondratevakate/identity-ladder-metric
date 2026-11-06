"""Donor ladder for 2D images under the general protocol (docs/PROTOCOL_donor_ladder_general.md, section 9).

Fundus instance (PAPILA, GRAPE). Reuses the frame, cache, units, positions and seeded draws of
donor_ladder_fundus, so the original fills reproduce donor_ladder_fundus.json; adds the selection fills with their
stranger and oracle controls, eye sides (R, L, LR), a patch x ring sensitivity grid on the same centres, and
local-error maps with independent person halves. Pre-QC: the disc-fovea landmarks are not visually accepted.
Private rows and maps stay in the data root; the public JSON holds aggregates only.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage, signal

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_fundus as F  # noqa: E402

SEED, G = F.SEED, F.G
PRIVATE = os.path.join(F.DATA, "projects", "identity-ladder-metric", "donor-ladder-2d-20261105-v1")
OUT_JSON = os.path.join(F.ROOT, "results", "donor_ladder_2d.json")
W = 8                                    # block-search half-window, pixels on the 256 grid
K = 16
PATCHES, RINGS = (24, 48, 72), (4, 8, 16, 24)
MAP_WIN = 48
KEEP = {("population", "all"), ("population", "1"), ("population", "16"), ("cohort", "all"), ("knn", "16"),
        ("knn_oracle", "16"), ("knn_fellow", "16")} | {(c, "1") for c in (
            "fellow_eye", "fellow_block", "fellow_block_oracle", "stranger_other_side", "stranger_block",
            "stranger_block_oracle", "other_visit")}


def ring_mask(size, ring):
    m = np.ones((size + 2 * ring,) * 2, bool)
    m[ring:ring + size, ring:ring + size] = False
    return m


def region(img, fov, y, x, size, pad):
    """Square region of edge size + 2 pad starting at (y - pad, x - pad), zero and invalid outside the grid."""
    out = np.zeros((size + 2 * pad,) * 2, np.float32)
    ok = np.zeros(out.shape, bool)
    y0, x0 = y - pad, x - pad
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + out.shape[0], img.shape[0]), min(x0 + out.shape[1], img.shape[1])
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
        ok[ys - y0:ye - y0, xs - x0:xe - x0] = fov[ys:ye, xs:xe]
    return out, ok


def search(win, other, other_ok, size, ring, oracle=False):
    """Best match of the target window (size + 2 ring) inside `other` (size + 2 ring + 2 W): on the ring, or on
    the true patch with `oracle`. Candidates need their whole patch inside the field of view. Returns the patch
    there, or None."""
    m = ~ring_mask(size, ring) if oracle else ring_mask(size, ring)
    o2 = signal.correlate(other.astype(np.float64) ** 2, m.astype(np.float64), mode="valid", method="fft")
    cross = signal.correlate(other.astype(np.float64), np.where(m, win, 0).astype(np.float64), mode="valid",
                             method="fft")
    k = np.zeros(m.shape)
    k[ring:ring + size, ring:ring + size] = 1
    full = signal.correlate(other_ok.astype(np.float64), k, mode="valid", method="fft") >= size * size - 0.5
    if not full.any():
        return None
    d = np.where(full, o2 - 2 * cross, np.inf)
    oy, ox = np.unravel_index(int(np.argmin(d)), d.shape)
    return other[oy + ring:oy + ring + size, ox + ring:ox + ring + size]


def flatten(g, fov, factor=4, size=13):
    """Divide by a background estimate (Amendment 9.1): median over size * factor px, computed at 1/factor
    resolution with the outside of the field of view filled from the nearest inside pixel."""
    n = g.shape[0] // factor
    gs = g[:n * factor, :n * factor].reshape(n, factor, n, factor)
    fs = fov[:n * factor, :n * factor].reshape(n, factor, n, factor)
    cnt = fs.sum(axis=(1, 3))
    small = np.where(cnt > 0, (gs * fs).sum(axis=(1, 3)) / np.maximum(cnt, 1), 0)
    inside = cnt > 0
    if not inside.any():
        return np.zeros_like(g)
    _, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
    bg = ndimage.median_filter(small[iy, ix], size=size, mode="nearest")
    bg = ndimage.zoom(bg, g.shape[0] / n, order=1)[:g.shape[0], :g.shape[1]]
    return np.where(fov, g / np.maximum(bg, 1e-3), 0).astype(np.float32)


def load_flat(path):
    """F.load with the background division before downsampling and z-scoring."""
    rgb = F.L.load_base(path, 512, True)
    lm = F.R.fundus_landmarks(rgb)
    if lm is None:
        return None
    c = F.R.fundus_canonical(rgb)
    disc_right = lm[0][0] > lm[1][0]
    if not disc_right:
        c = c[:, ::-1]
    fov512 = c[..., 0] > 0.1
    g = flatten(c[..., 1].astype(np.float32), fov512).reshape(G, 2, G, 2).mean(axis=(1, 3))
    fov = fov512.reshape(G, 2, G, 2).all(axis=(1, 3))
    v = g[fov]
    g = np.where(fov, (g - v.mean()) / (v.std() + 1e-6), 0.0).astype(np.float32)
    return g, fov, disc_right


def load_arm(t, name, arm):
    if arm == "z":
        return F.load_all(t, os.path.join(F.PRIVATE, f"{name}_canonical.npz"))
    cache = os.path.join(PRIVATE, f"{name}_canonical_{arm}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return z["img"], z["fov"], z["ok"], z["disc_right"]
    n = len(t)
    img, fov = np.zeros((n, G, G), np.float32), np.zeros((n, G, G), bool)
    ok, dr = np.zeros(n, bool), np.zeros(n, bool)
    for i, p in enumerate(t.path):
        r = load_flat(p)
        if r is not None:
            img[i], fov[i], dr[i] = r
            ok[i] = True
        if i % 50 == 0:
            print(f"{name} flat load {i + 1}/{n}", flush=True)
    np.savez_compressed(cache, img=img, fov=fov, ok=ok, disc_right=dr)
    return img, fov, ok, dr


def setup(name, arm="z"):
    t = F.papila_table() if name == "PAPILA" else F.grape_table()
    img, fov, ok, dr = load_arm(t, name, arm)
    u = F.units(t, ok, dr)
    full = t.assign(idx=np.arange(len(t)), ok=ok, side_conflict=ok & (dr != (t.side == "R")))
    patches, n_cand = F.patch_positions(fov[u.idx.to_numpy()], np.random.default_rng(SEED))
    n = len(u)
    person, side = u.person.to_numpy(), u.side.to_numpy()
    fellow = np.full(n, -1)
    for i in range(n):
        j = np.flatnonzero((person == person[i]) & (side != side[i]))
        if j.size:
            fellow[i] = j[0]
    other = np.full(n, -1)
    if name == "GRAPE":
        for i in range(n):
            c = full[(full.person == person[i]) & (full.side == side[i]) & full.ok & ~full.side_conflict
                     & (full.visit != u.visit.iloc[i])]
            if len(c):
                other[i] = c.iloc[np.argmin(np.abs(c.time.to_numpy() - u.time.iloc[i]))].idx
    return t, img, fov, u, patches + [F.METRIC], n_cand, fellow, other


def pools(name, u):
    person, side = u.person.to_numpy(), u.side.to_numpy()
    age, sex, dev = u.age.to_numpy(float), u.sex.astype(str).to_numpy(), u.device.to_numpy()

    def pool(i, cohort=False):
        m = (person != person[i]) & (side == side[i])
        if cohort:
            m &= (sex == sex[i]) & (np.abs(age - age[i]) <= 5)
            if name == "GRAPE":
                m &= dev == dev[i]
        return np.flatnonzero(m)
    draws = {c: [np.random.default_rng([SEED, i, F.DRAWN.index(c)]).permutation(pool(i, c == "cohort"))
                 for i in range(len(u))] for c in ("population", "cohort")}
    return pool, draws


def run_config(name, setup_out, size, ring, ks, primary):
    t, img, fov, u, patches, _, fellow, other = setup_out
    pool, draws = pools(name, u)
    idx = u.idx.to_numpy()
    n = len(u)
    side = u.side.to_numpy()
    rows = []
    for pi, (y0, x0, s0) in enumerate(patches):
        metric = pi == len(patches) - 1
        if metric and size != 48:
            continue                                   # the metric box keeps its size; only the ring varies
        s = s0 if metric else size
        y, x = (y0, x0) if metric else (y0 + s0 // 2 - s // 2, x0 + s0 // 2 - s // 2)
        if y < 0 or x < 0 or y + s > G or x + s > G:
            continue
        reg = [region(img[k], fov[k], y, x, s, ring) for k in idx]
        X = np.stack([r[0][ring:ring + s, ring:ring + s].ravel() for r in reg])
        valid = np.array([r[1][ring:ring + s, ring:ring + s].all() for r in reg])
        rm = ring_mask(s, ring)
        C = np.stack([r[0][rm] for r in reg]).astype(np.float64)
        ring_valid = np.array([r[1][rm].all() for r in reg])
        big = [region(img[k], fov[k], y, x, s, ring + W) for k in idx]
        Xd = X.astype(np.float64)
        for i in range(n):
            if not valid[i]:
                continue
            same = pool(i)
            mu_set = same[valid[same]]
            if mu_set.size < 2:
                continue
            mu = X[mu_set].mean(0)
            xi = X[i]

            def add(c, k, f):
                rows.append((i, pi, c, k, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2)), side[i]))
            for c in ("population", "cohort"):
                order = draws[c][i]
                order = order[valid[order]]
                for k in ks + ("all",):
                    sel = order if k == "all" else order[:k]
                    if k != "all" and sel.size < k:
                        continue
                    if sel.size:
                        add(c, k, X[sel].mean(0))
            cand = mu_set
            if cand.size >= K:
                d_or = ((Xd[cand] - Xd[i]) ** 2).sum(1)
                for k in ks:
                    if cand.size >= k:
                        add("knn_oracle", k, X[cand[np.argsort(d_or, kind="stable")[:k]]].mean(0))
                if ring_valid[i]:
                    cr = cand[ring_valid[cand]]
                    d = ((C[cr] - C[i]) ** 2).sum(1)
                    for k in ks:
                        if cr.size >= k:
                            add("knn", k, X[cr[np.argsort(d, kind="stable")[:k]]].mean(0))
            j = fellow[i]
            if j >= 0 and valid[j]:
                add("fellow_eye", 1, X[j])
                if cand.size >= K:
                    d_f = ((Xd[cand] - Xd[j]) ** 2).sum(1)
                    for k in ks:
                        if cand.size >= k:
                            add("knn_fellow", k, X[cand[np.argsort(d_f, kind="stable")[:k]]].mean(0))
            win = reg[i][0]
            if j >= 0:
                for c, orc in (("fellow_block", False), ("fellow_block_oracle", True)):
                    f = search(win, big[j][0], big[j][1], s, ring, orc)
                    if f is not None:
                        add(c, 1, f.ravel())
            strangers = [q for q in draws["population"][i] if fellow[q] >= 0]
            if strangers:
                q = strangers[0]
                if valid[fellow[q]]:
                    add("stranger_other_side", 1, X[fellow[q]])
                for c, orc in (("stranger_block", False), ("stranger_block_oracle", True)):
                    f = search(win, big[fellow[q]][0], big[fellow[q]][1], s, ring, orc)
                    if f is not None:
                        add(c, 1, f.ravel())
            if other[i] >= 0:
                ro, oko = region(img[other[i]], fov[other[i]], y, x, s, 0)
                if oko.all():
                    add("other_visit", 1, ro.ravel())
    r = pd.DataFrame(rows, columns=["target", "patch", "cohort", "k", "err", "spread", "side"])
    r["k"] = r["k"].astype(str)
    r["person"] = u.person.to_numpy()[r.target]
    r["region"] = np.where(r.patch == len(patches) - 1, "metric_box", "random_patches")
    return r


def views(r):
    return pd.concat([r[r.side == "R"].assign(region=r.region[r.side == "R"] + "/R"),
                      r[r.side == "L"].assign(region=r.region[r.side == "L"] + "/L"),
                      r.assign(region=r.region + "/LR")], ignore_index=True)


def box(x):
    return ndimage.uniform_filter(x.astype(np.float32), MAP_WIN, mode="constant")


def maps(name, setup_out, tag=""):
    """Local-error maps on the canonical frame for fixed-rule donors; pools per side without the target's person;
    independent person halves (pools inside each half). Returns aggregate arrays only."""
    t, img, fov, u, _, _, fellow, other = setup_out
    idx = u.idx.to_numpy()
    person, side = u.person.to_numpy(), u.side.to_numpy()
    age, sex, dev = u.age.to_numpy(float), u.sex.astype(str).to_numpy(), u.device.to_numpy()
    okw = np.stack([box(fov[k]) >= 1 - 1e-4 for k in idx])
    persons = np.unique(person)
    half_of = dict(zip(np.random.default_rng(SEED).permutation(persons), np.arange(len(persons)) % 2))
    half = np.array([half_of[p] for p in person])
    support = okw.mean(0) >= 0.9
    out = {}
    for tag, members in (("all", np.ones(len(u), bool)), ("h0", half == 0), ("h1", half == 1)):
        acc = {d: [np.zeros((G, G)), np.zeros((G, G))] for d in ("fellow_eye", "other_visit", "stranger", "cohort")}
        vsum, vn = np.zeros((G, G)), np.zeros((G, G))
        for i in np.flatnonzero(members):
            p = np.flatnonzero(members & (person != person[i]) & (side == side[i]))
            if p.size < 10:
                continue
            P = img[idx[p]]
            mu = P.mean(0)
            v = box((img[idx[i]] - mu) ** 2)
            ok = okw[i] & support
            vsum += np.where(ok, v, 0)
            vn += ok
            fills = {"stranger": v + box(((P - mu) ** 2).mean(0))}
            coh = p[(sex[p] == sex[i]) & (np.abs(age[p] - age[i]) <= 5) & ((dev[p] == dev[i]) if name == "GRAPE" else True)]
            if coh.size >= 10:
                fills["cohort"] = box((img[idx[i]] - img[idx[coh]].mean(0)) ** 2)
            j = fellow[i]
            if j >= 0:
                fills["fellow_eye"] = np.where(okw[j], box((img[idx[i]] - img[idx[j]]) ** 2), np.nan)
            if other[i] >= 0:
                oo = box(fov[other[i]]) >= 1 - 1e-4
                fills["other_visit"] = np.where(oo, box((img[idx[i]] - img[other[i]]) ** 2), np.nan)
            for d, e in fills.items():
                g = ok & np.isfinite(e)
                acc[d][0] += np.where(g, e, 0)
                acc[d][1] += np.where(g, v, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[tag] = {d: np.where(a[1] > 0, a[0] / a[1], np.nan) for d, a in acc.items()}
            out[tag]["V"] = np.where(vn > 0, vsum / vn, np.nan)
    rel = {}
    for d in ("fellow_eye", "other_visit", "stranger", "cohort"):
        a, b = out["h0"][d], out["h1"][d]
        ok = np.isfinite(a) & np.isfinite(b)
        rel[d] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else None
    np.savez_compressed(os.path.join(PRIVATE, f"{name}_maps{tag}.npz"),
                        **{f"{tag}_{d}": m.astype(np.float32) for tag, mm in out.items() for d, m in mm.items()})
    return {d: {"median_ratio": (float(np.nanmedian(out["all"][d])) if np.isfinite(out["all"][d]).any() else None),
                "split_half_r": rel[d]} for d in rel}


def summary(r, keep=None, **counts):
    if keep is not None:
        r = r[[(c, k) in keep for c, k in zip(r.cohort, r.k)]]
    return F.summarize(views(r), **counts)


def main():
    os.makedirs(PRIVATE, exist_ok=True)
    t0 = time.time()
    ext = os.environ.get("LADDER_EXTERNAL")   # a checkout's `external/` (git-ignored, absent in worktrees)
    if ext:
        F.S.GRAPE_META = os.path.join(ext, *os.path.relpath(F.S.GRAPE_META, os.path.join(
            os.path.dirname(F.S.__file__), "..", "external")).split(os.sep))
    names = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--datasets=")), ["PAPILA", "GRAPE"])
    arm = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--arm=")), "z")   # z or flat (9.1)
    primary_only = "--primary-only" in sys.argv
    tag = "" if arm == "z" else f"_{arm}"
    out_json = OUT_JSON.replace(".json", f"{tag}.json")
    out = {"status": "pre-QC: disc-fovea landmarks not visually accepted", "arm": arm,
           "protocol": "docs/PROTOCOL_donor_ladder_general.md (section 9)", "seed": SEED, "grid": G,
           "block_search_halfwindow_px": W, "datasets": {}}
    if os.path.exists(out_json):
        out["datasets"] = json.load(open(out_json, encoding="utf-8"))["datasets"]
    for name in names:
        so = setup(name, arm)
        u, fellow, other = so[3], so[6], so[7]
        counts = dict(n_units=len(u), n_fellow=int((fellow >= 0).sum()), n_other=int((other >= 0).sum()))
        d = {"counts": counts, "configs": {}}
        r = run_config(name, so, 48, 16, F.KS, True)
        r.to_csv(os.path.join(PRIVATE, f"{name}_rows_p48_r16{tag}.csv.gz"), index=False)
        d["primary"] = summary(r, **counts)
        for p in () if primary_only else PATCHES:
            for ring in RINGS:
                key = f"p{p}_r{ring}"
                rr = r if (p, ring) == (48, 16) else run_config(name, so, p, ring, (1, 16), False)
                d["configs"][key] = summary(rr, KEEP)["regions"]
                print(f"{name} {key} done, {round((time.time() - t0) / 60, 1)} min", flush=True)
        d["maps"] = maps(name, so, tag)
        out["datasets"][name] = d
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(finite(out), fh, indent=1, allow_nan=False)
    print("done", round((time.time() - t0) / 60, 1), "min")


def finite(o):
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [finite(v) for v in o]
    return None if isinstance(o, float) and not np.isfinite(o) else o


if __name__ == "__main__":
    main()
