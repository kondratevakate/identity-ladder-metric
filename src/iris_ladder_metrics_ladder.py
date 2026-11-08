"""Iris donor-averaging curve and error maps, in the fundus format (fig_donor_ladder_fundus, fig_ladder2d_maps_*).

Same P2 frames, units, windows and pools as the iris metric panel (docs/PROTOCOL_iris_metrics.md); the
preprocessing version is passed explicitly. Nothing here selects images by similarity except the kNN-by-context arm,
which is labelled as such.

Curve: for every target eye and every panel window, fills are masked averages of k donor strips (k = 1 ... 64, all).
Arms: population (same side, other people, fixed random order), population both sides (other side mirrored),
kNN by context (donors ranked by the masked squared distance on the ring around the window), own other eye
(mirrored) and own eye, other photo (k = 1). error/V = summed squared error over the summed squared error of the
leave-person-out dataset mean on exactly the same pixels (1 = dataset mean, 2 = one stranger in theory);
spread/V = summed (fill - mean)^2 over the same V. A fill pixel counts only where at least half of its k donors are
valid; pixels outside the target mask, the fill or the mean are missing, never zero.

Maps: masked Gaussian local means (sd = WIN / sqrt(12), wrapped in angle) of the squared error of each donor and of
the dataset mean on the same mask, summed over targets, ratio per strip pixel. Stranger = exact expectation
(target-to-mean error plus the pool variance). Split-half r over independent person halves. Arrays stay private;
the public JSON gets medians and split-half r only.
Usage: python src/iris_ladder_metrics_ladder.py --preprocessing P2 [--sets UTIRIS_NIR,UPOL,UTIRIS_VIS]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import iris_ladder_metrics as IM  # noqa: E402
import iris_ladder_metrics_frame as IF  # noqa: E402

H, W, WIN, RING = IM.H, IM.W, IM.WIN, IM.RING
KS = (1, 2, 4, 8, 16, 32, 64)
ARMS = ("population", "population_both", "knn")
MAP_SD = WIN / np.sqrt(12)                     # Gaussian with the variance of the WIN box (lesson L8)
SEED = IM.SEED
N_BOOT = 2000


# ------------------------------------------------------------------------------------------------ data

def load(name, version):
    IF.check_version(version)
    fr = IF.load_or_build(name, version)
    tab, X, M = fr["table"], fr["X"], fr["M"]
    u = IF.units(tab)
    idx = u.idx.to_numpy()
    valid = M[idx].sum(0) >= IM.COVER * len(u)
    return {"name": name, "X": np.nan_to_num(X), "M": M, "XF": np.nan_to_num(IF.mirror(X)), "MF": IF.mirror(M), "idx": idx,
            "valid": valid, "person": u.person.astype(str).to_numpy(), "side": u.side.to_numpy(),
            "repeat": u.repeat_idx.to_numpy(), "fellow": u.fellow.to_numpy(), "pos": IM.positions(valid),
            "ratio": float(np.nanmedian(tab.loc[tab.ok, "ratio"])) if "ratio" in tab else 0.45}


def masked_mean(P, Mp, need=1):
    c = Mp.sum(0)
    s = (P * Mp).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c >= need, s / np.maximum(c, 1), 0.0), c >= need


def pools(st, ti):
    """Leave-person-out donors: same side (strip as is) and other side (mirrored strip)."""
    same = np.flatnonzero((st["side"] == st["side"][ti]) & (st["person"] != st["person"][ti]))
    other = np.flatnonzero((st["side"] != st["side"][ti]) & (st["person"] != st["person"][ti]))
    return same, other


# ------------------------------------------------------------------------------------------------ curve

def curve_rows(st):
    """Per (target, arm, k): summed squared error, summed spread and summed V on the same pixels."""
    X, M, XF, MF, idx, valid = st["X"], st["M"], st["XF"], st["MF"], st["idx"], st["valid"]
    rows = []
    t0 = time.time()
    for ti in range(len(idx)):
        same, other = pools(st, ti)
        Ps_f, Ms_f = X[idx[same]], M[idx[same]]
        mu_f, mu_ok_f = masked_mean(Ps_f, Ms_f)
        xt_f, mt_f = X[idx[ti]], M[idx[ti]] & valid & mu_ok_f
        ringall_f = valid & M[idx[ti]]
        bothP_f = np.concatenate([Ps_f, XF[idx[other]]])
        bothM_f = np.concatenate([Ms_f, MF[idx[other]]])
        rng = np.random.default_rng([SEED, ti, 11])
        order = {"population": rng.permutation(len(same)), "population_both": rng.permutation(len(bothP_f))}
        acc = {}

        def add(arm, k, f, fm, e, xt, mu):
            e = e & fm
            if e.sum() < 10:
                return
            a = acc.setdefault((arm, k), [0.0, 0.0, 0.0])
            a[0] += float(((xt - f) ** 2)[e].sum())
            a[1] += float(((f - mu) ** 2)[e].sum())
            a[2] += float(((xt - mu) ** 2)[e].sum())

        for (y, x) in st["pos"]:
            # Everything below lives on the window plus its ring only (same pixels as the full strip, cropped).
            by, bx = slice(max(y - RING, 0), y + WIN + RING), slice(max(x - RING, 0), x + WIN + RING)
            win = np.zeros((H, W), bool)
            win[y:y + WIN, x:x + WIN] = True
            win = win[by, bx]
            xt, mu, mu_ok = xt_f[by, bx], mu_f[by, bx], mu_ok_f[by, bx]
            e = win & mt_f[by, bx]
            if e.sum() < IM.MIN_IN_MASK * WIN * WIN:
                continue
            ring = IM.ring_mask(y, x)[by, bx] & ringall_f[by, bx]
            Ps, Ms = Ps_f[:, by, bx], Ms_f[:, by, bx]
            for arm in ARMS:
                if arm == "knn":
                    ov = (Ms & ring).sum((1, 2))
                    d = (((Ps - xt) ** 2) * (Ms & ring)).sum((1, 2)) / np.maximum(ov, 1)
                    d[ov < IM.MIN_RING_VALID * ring.sum()] = np.inf
                    o = np.argsort(d, kind="stable")
                    o = o[np.isfinite(d[o])]
                    P, Mm = Ps[o], Ms[o]
                else:
                    P, Mm = (Ps, Ms) if arm == "population" else (bothP_f[:, by, bx], bothM_f[:, by, bx])
                    P, Mm = P[order[arm]], Mm[order[arm]]
                cs, cm = np.cumsum(P * Mm, 0), np.cumsum(Mm, 0)
                for k in KS:
                    if len(P) < k:
                        continue
                    c = cm[k - 1]
                    fm = c >= max(1, int(np.ceil(k / 2)))
                    add(arm, str(k), np.where(fm, cs[k - 1] / np.maximum(c, 1), 0.0), fm, e, xt, mu)
                if arm == "population":
                    add(arm, "all", mu, mu_ok, e, xt, mu)
            if st["fellow"][ti] >= 0:
                j = idx[st["fellow"][ti]]
                add("fellow_eye", "1", XF[j][by, bx], MF[j][by, bx], e, xt, mu)
            if st["repeat"][ti] >= 0:
                r = st["repeat"][ti]
                add("other_photo", "1", X[r][by, bx], M[r][by, bx], e, xt, mu)
        for (arm, k), (se, ss, sv) in acc.items():
            rows.append((ti, st["person"][ti], arm, k, se, ss, sv))
        if (ti + 1) % 20 == 0 or ti + 1 == len(idx):
            el = time.time() - t0
            print(f"  {st.get('name', '')} curve: {ti + 1}/{len(idx)} eyes, {el:.0f}s, "
                  f"ETA {el / (ti + 1) * (len(idx) - ti - 1):.0f}s", flush=True)
    return rows


def boot(rows, person, seed=SEED):
    persons = np.array(sorted(set(person)))
    pos = {p: i for i, p in enumerate(persons)}
    g = np.zeros((len(persons), 3))
    for p, se, ss, sv in rows:
        g[pos[p]] += (se, ss, sv)
    est = (g[:, 0].sum() / g[:, 2].sum(), g[:, 1].sum() / g[:, 2].sum())
    rng = np.random.default_rng(seed)
    have = np.flatnonzero(g[:, 2] > 0)
    b = np.empty((N_BOOT, 2))
    for i in range(N_BOOT):
        s = have[rng.integers(0, len(have), len(have))]
        b[i] = g[s, 0].sum() / g[s, 2].sum(), g[s, 1].sum() / g[s, 2].sum()
    ci = np.percentile(b, [2.5, 97.5], axis=0)
    return {"err_over_V": float(est[0]), "ci95": [float(ci[0, 0]), float(ci[1, 0])], "spread_over_V": float(est[1]),
            "spread_ci95": [float(ci[0, 1]), float(ci[1, 1])], "n_targets": int(len(rows)), "n_persons": int(len(have))}


def summarise_curve(rows):
    out = {}
    keys = sorted({(r[2], r[3]) for r in rows})
    for arm, k in keys:
        rr = [(r[1], r[4], r[5], r[6]) for r in rows if r[2] == arm and r[3] == k]
        out.setdefault(arm, {})[k] = boot(rr, [r[0] for r in rr])
    for arm, d in out.items():
        ks = [k for k in d if k.isdigit()]
        if len(ks) >= 3:
            kk = np.array([float(k) for k in ks])
            yv = np.array([d[k]["err_over_V"] for k in ks])
            basis = 1 + 1 / kk
            a = float((yv * basis).sum() / (basis ** 2).sum())
            d["fit"] = {"one_minus_R2": a, "R2": 1 - a, "rms_residual": float(np.sqrt(np.mean((yv - a * basis) ** 2)))}
        if arm in ("fellow_eye", "other_photo") and "1" in d:
            d["r_implied"] = 1 - d["1"]["err_over_V"] / 2
    return out


# ------------------------------------------------------------------------------------------------ maps

def local(x, m):
    """Masked Gaussian mean of x over mask m; wrapped in angle (columns), not across the pupil or limbus edge."""
    m = np.asarray(m, bool)
    num = ndimage.gaussian_filter(np.where(m, x, 0.0), MAP_SD, mode=("constant", "wrap"))
    den = ndimage.gaussian_filter(m.astype(np.float64), MAP_SD, mode=("constant", "wrap"))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-3, num / den, np.nan)


MAP_FILLS = ("other_photo", "fellow_eye", "stranger", "lookalike16")
BLOCK = 8                                       # look-alike map: one donor search per 8 x 8 block
PADW = WIN // 2 + RING                          # angular wrap padding for the block windows


def lookalike_fill(xt, mt, Ps, Ms, k=16):
    """Fill made of the k donors closest on the context ring around a WIN window centred on every BLOCK x BLOCK
    block (masked squared distance; ring >= MIN_RING_VALID observed in both). The window is wrapped in angle and
    clipped at the pupil and limbus edges. A block pixel is filled where at least k/2 of the k donors are valid."""
    wrap = lambda a: np.concatenate([a[..., -PADW:], a, a[..., :PADW]], axis=-1)  # noqa: E731
    xw, mw, Pw, Mw = wrap(xt), wrap(mt), wrap(Ps), wrap(Ms)
    F, Fok = np.zeros((H, W)), np.zeros((H, W), bool)
    half = (WIN - BLOCK) // 2
    for y0 in range(0, H, BLOCK):
        for x0 in range(0, W, BLOCK):
            wy, wx = y0 - half, x0 - half + PADW
            ry, rx = slice(max(wy - RING, 0), min(wy + WIN + RING, H)), slice(wx - RING, wx + WIN + RING)
            ring = np.zeros((H, W + 2 * PADW), bool)
            ring[ry, rx] = True
            ring[max(wy, 0):wy + WIN, wx:wx + WIN] = False
            ring = (ring & mw)[ry, rx]
            if ring.sum() < 20:
                continue
            Mr = Mw[:, ry, rx] & ring
            ov = Mr.sum((1, 2))
            d = (((Pw[:, ry, rx] - xw[ry, rx]) ** 2) * Mr).sum((1, 2)) / np.maximum(ov, 1)
            d[ov < IM.MIN_RING_VALID * ring.sum()] = np.inf
            o = np.argsort(d, kind="stable")[:k]
            if not np.isfinite(d[o]).all() or o.size < k:
                continue
            bs = (slice(y0, y0 + BLOCK), slice(x0, x0 + BLOCK))
            f, ok = masked_mean(Ps[o][:, bs[0], bs[1]], Ms[o][:, bs[0], bs[1]], need=k // 2)
            F[bs], Fok[bs] = f, ok
    return F, Fok


def maps(st):
    X, M, XF, MF, idx, valid = st["X"], st["M"], st["XF"], st["MF"], st["idx"], st["valid"]
    persons = np.unique(st["person"])
    half_of = dict(zip(np.random.default_rng(SEED).permutation(persons), np.arange(len(persons)) % 2))
    half = np.array([half_of[p] for p in st["person"]])
    out = {}
    for part, members in (("all", np.ones(len(idx), bool)), ("h0", half == 0), ("h1", half == 1)):
        num = {d: np.zeros((H, W)) for d in MAP_FILLS}
        den = {d: np.zeros((H, W)) for d in MAP_FILLS}
        vsum, vn = np.zeros((H, W)), np.zeros((H, W))
        t0 = time.time()
        targets = np.flatnonzero(members)
        for n_done, ti in enumerate(targets, 1):
            if n_done % 40 == 0:
                el = time.time() - t0
                print(f"  {st.get('name', '')} maps {part}: {n_done}/{len(targets)}, {el:.0f}s, "
                      f"ETA {el / n_done * (len(targets) - n_done):.0f}s", flush=True)
            same, _ = pools(st, ti)
            same = same[members[same]]
            if same.size < 10:
                continue
            Ps, Ms = X[idx[same]], M[idx[same]]
            mu, mu_ok = masked_mean(Ps, Ms)
            xt, mt = X[idx[ti]], M[idx[ti]] & mu_ok
            var, _ = masked_mean((Ps - mu) ** 2, Ms)
            v = local((xt - mu) ** 2, mt)
            ok = np.isfinite(v)
            vsum[ok] += v[ok]
            vn[ok] += 1
            fills = {"stranger": (v + local(var, mt), v)}
            if st["fellow"][ti] >= 0 and members[st["fellow"][ti]]:
                j = idx[st["fellow"][ti]]
                e = mt & MF[j]
                fills["fellow_eye"] = (local((xt - XF[j]) ** 2, e), local((xt - mu) ** 2, e))
            if st["repeat"][ti] >= 0:
                r = st["repeat"][ti]
                e = mt & M[r]
                fills["other_photo"] = (local((xt - X[r]) ** 2, e), local((xt - mu) ** 2, e))
            if same.size >= 16:
                F, Fok = lookalike_fill(xt, mt, Ps, Ms)
                e = mt & Fok
                fills["lookalike16"] = (local((xt - F) ** 2, e), local((xt - mu) ** 2, e))
            for d, (a, b) in fills.items():
                ok = np.isfinite(a) & np.isfinite(b)
                num[d][ok] += a[ok]
                den[d][ok] += b[ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[part] = {d: np.where(valid & (den[d] > 0), num[d] / den[d], np.nan) for d in MAP_FILLS}
            out[part]["V"] = np.where(valid & (vn > 0), vsum / np.maximum(vn, 1), np.nan)
    mean_strip, mean_ok = masked_mean(X[idx], M[idx])
    out["all"]["anatomy"] = np.where(mean_ok, mean_strip, np.nan)
    res = {}
    for d in MAP_FILLS:
        a, b = out["h0"][d], out["h1"][d]
        ok = np.isfinite(a) & np.isfinite(b)
        res[d] = {"median_ratio": float(np.nanmedian(out["all"][d])) if np.isfinite(out["all"][d]).any() else None,
                  "split_half_r": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else None}
    res["valid_px"] = int(valid.sum())
    return out, res


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessing", required=True, choices=IF.VERSIONS)
    ap.add_argument("--sets", default=",".join(IF.SETS))
    a = ap.parse_args()
    if a.preprocessing in IF.REJECTED:
        raise SystemExit(f"{a.preprocessing} was rejected: {IF.REJECTED[a.preprocessing]}")
    d = json.load(open(IM.OUT_JSON, encoding="utf-8"))
    curve, mres = d.get("donor_curve", {}), d.get("maps", {})
    for name in a.sets.split(","):
        t0 = time.time()
        st = load(name, a.preprocessing)
        rows = curve_rows(st)
        curve[name] = summarise_curve(rows)
        curve[name]["counts"] = {"eyes": int(len(st["idx"])), "people": int(len(set(st["person"]))),
                                 "windows": int(len(st["pos"]))}
        print(f"  {name} curve done, maps next", flush=True)
        mm, res = maps(st)
        res["pupil_to_iris_ratio_median"] = st["ratio"]
        mres[name] = res
        np.savez_compressed(os.path.join(IF.PRIVATE, f"maps_{name}_{a.preprocessing}.npz"),
                            **{f"{p}_{k}": v.astype(np.float32) for p, dd in mm.items() for k, v in dd.items()})
        c = curve[name]
        print(f"{name}: {time.time() - t0:.0f}s  population k=1 {c['population']['1']['err_over_V']:.3f}  "
              f"knn k=8 {c['knn'].get('8', {}).get('err_over_V', float('nan')):.3f}  "
              f"maps fellow {res['fellow_eye']['median_ratio']:.2f} (r {res['fellow_eye']['split_half_r']})", flush=True)
    d["donor_curve"] = curve
    d["donor_curve_definition"] = ("error/V and spread/V of masked k-donor averages per panel window, summed over "
                                   "targets on the same pixels as the leave-person-out dataset mean; bootstrap over "
                                   "people (src/iris_ladder_metrics_ladder.py)")
    d["maps"] = mres
    d["maps_definition"] = ("per strip pixel: masked Gaussian (sd = 16/sqrt(12) px, wrapped in angle) local squared "
                            "error of each donor over the dataset mean's on the same mask, summed over targets; "
                            "stranger = exact expectation; arrays private")
    json.dump(d, open(IM.OUT_JSON, "w", encoding="utf-8"), indent=1, allow_nan=False)
    print("updated", IM.OUT_JSON)


if __name__ == "__main__":
    main()
