"""Fundus ladder maps in the face-figure format (general protocol, Amendment 9.5). Pre-QC.

Rows PAPILA (P9) and GRAPE (P8); columns Mean fundus / Another photo / Mirror side / Sibling / Stranger /
16 look-alikes / Same sex and age / Dataset mean, as in `src/donor_ladder_face.py` (frame v6). 128 px frame,
16 px window, look-alikes on a 4 px grid by the 8 px ring, k = 16, 8 strangers. Outside a mask is missing: every
error uses the joint mask (target x donor x valid region) and the dataset-mean reference uses the same pixels.
Aggregates only leave the private folder.
Usage: python src/donor_ladder_2d_facefmt.py [run|figure]   (LADDER_EXTERNAL must point to a checkout's external/)
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import donor_ladder_2d as D  # noqa: E402
import donor_ladder_fundus as F  # noqa: E402
import result_spec  # noqa: E402

SIZE, WIN, STEP, RING, K_LOOK, N_STRANGER, MIN_HAVE = 128, 16, 4, 8, 16, 8, 8
SETS = (("PAPILA", "p9"), ("GRAPE", "p8"))
OUT_JSON = os.path.join(ROOT, "results", "donor_ladder_2d_facefmt.json")
COLS = ("repeat", "mirror", "fellow", "fellow_mirror", "stranger", "lookalike", "sexage")
MIRRORS = {"mirror": "halves", "fellow": "sides", "fellow_mirror": "sides"}   # result spec: direction kinds
AXIS_Y = {"PAPILA": (247.0 + 0.5) / 4 - 0.5, "GRAPE": (256.0 + 0.5) / 4 - 0.5}  # 512-grid row -> 128 frame (9.6)
STRIP = 2.5                                            # excluded either side of the axis, as the face midline


def down2(img, fov):
    """2x2 mean onto the 128 frame; a pixel is valid only if all 4 source pixels are (missing is never zero)."""
    n = img.shape[-1] // 2
    x = img.reshape(img.shape[:-2] + (n, 2, n, 2)).mean(axis=(-3, -1))
    m = fov.reshape(fov.shape[:-2] + (n, 2, n, 2)).all(axis=(-3, -1))
    return np.where(m, x, 0).astype(np.float32), m


def box(x, w=WIN):
    return ndimage.uniform_filter(x.astype(np.float64), w, mode="constant")


def local_error(a, b, m):
    """Masked local mean of the squared difference in a Gaussian window of the 16 px box's standard deviation (as
    the face script from its v3, face Amendment 4.7; Amendment 9.6 here); NaN where < half of the window weight is
    inside the mask."""
    sd = WIN / np.sqrt(12)
    m = m.astype(np.float64)
    num = ndimage.gaussian_filter(((a - b) ** 2) * m, sd, mode="constant")
    den = ndimage.gaussian_filter(m, sd, mode="constant")
    return np.where(den >= 0.5, num / np.maximum(den, 1e-12), np.nan)


def flip_rows(x, y0):
    """Reflect an image about the horizontal line y = y0 (row centres at integers), linear interpolation; for a
    boolean mask a pixel stays valid only if the interpolated mask is 1 (all contributing pixels valid)."""
    from scipy.ndimage import map_coordinates
    yy, xx = np.mgrid[:x.shape[0], :x.shape[1]].astype(np.float64)
    src = np.stack([2 * y0 - yy, xx])
    if x.dtype == bool:
        return map_coordinates(x.astype(np.float64), src, order=1, mode="constant", cval=0.0) > 1 - 1e-6
    return map_coordinates(x.astype(np.float64), src, order=1, mode="constant", cval=0.0).astype(np.float32)


def mirror_fill(xt, mt, y0):
    """Same eye reflected about the disc-fovea axis; the strip of STRIP px either side of the axis is excluded."""
    rows = np.abs(np.arange(xt.shape[0]) - y0) >= STRIP
    return flip_rows(xt, y0), flip_rows(mt, y0) & rows[:, None]


def masked_mean(X, M):
    s, n = (X * M).sum(0), M.sum(0)
    return np.where(n > 0, s / np.maximum(n, 1), 0.0), n


def lookalike_fill(Xt, Mt, Xp, Mp, valid, exclude, ring=RING, step=STEP, k=K_LOOK, min_have=MIN_HAVE):
    """Face `lookalike_fill` with masks (Amendment 7 erratum of the face protocol) plus `exclude` (T x P bool): an
    excluded pool image (the target's own person) is never a neighbour. Returns T fill images, NaN where unfilled."""
    half = WIN // 2
    T = len(Xt)
    acc = np.zeros(Xt.shape, np.float32)
    wsum = np.zeros(Xt.shape, np.float32)
    tent1 = 1 - np.abs(np.arange(WIN) - (WIN - 1) / 2) / (WIN / 2)
    tent = np.outer(tent1, tent1).astype(np.float32)
    r = half + ring
    yy, xx = np.mgrid[-r:r, -r:r]
    in_win = (np.abs(yy + 0.5) < half) & (np.abs(xx + 0.5) < half)
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
            Wt, Wp = Mt[:, ys[sel], xs[sel]], Mp[:, ys[sel], xs[sel]]
            Rt, Rp = Xt[:, ys[sel], xs[sel]] * Wt, Xp[:, ys[sel], xs[sel]] * Wp
            cnt = Wt @ Wp.T
            num = (Rt ** 2) @ Wp.T + Wt @ (Rp ** 2).T - 2 * Rt @ Rp.T
            d = np.where((cnt >= 0.5 * Wt.sum(1, keepdims=True)) & ~exclude, num / np.maximum(cnt, 1), np.inf)
            kk = min(k, d.shape[1] - 1)
            nn = np.argpartition(d, kk, axis=1)[:, :k]
            good = np.isfinite(np.take_along_axis(d, nn, 1)).astype(np.float32)
            wy0, wx0 = cy - half, cx - half
            sy, sx = slice(max(wy0, 0), min(wy0 + WIN, SIZE)), slice(max(wx0, 0), min(wx0 + WIN, SIZE))
            ty, tx = slice(sy.start - wy0, sy.stop - wy0), slice(sx.start - wx0, sx.stop - wx0)
            w = tent[ty, tx] * valid[sy, sx]
            Pm = Mp[:, sy, sx][nn] * good[:, :, None, None]
            have = Pm.sum(1)
            val = (Xp[:, sy, sx][nn] * Pm).sum(1) / np.maximum(have, 1)
            wk = w[None] * (have >= min_have)
            acc[:, sy, sx] += val * wk
            wsum[:, sy, sx] += wk
    covered = (wsum > 1e-6) & valid
    return np.where(covered, acc / np.where(wsum > 1e-6, wsum, 1), np.nan).astype(np.float32)


def boot_ratio(per_person, rng, n=2000):
    keys = sorted(per_person)
    nf = np.array([per_person[k][0] for k in keys])
    df = np.array([per_person[k][1] for k in keys])
    bs = [nf[s].sum() / df[s].sum() for s in (rng.integers(0, len(keys), len(keys)) for _ in range(n))]
    return float(nf.sum() / df.sum()), [float(x) for x in np.percentile(bs, [2.5, 97.5])]


def run_dataset(name, arm):
    t, img, fov, u, _, _, fellow, other = D.setup(name, arm)
    X, M = down2(img, fov)                                   # every image of the table (other visits included)
    idx = u.idx.to_numpy()
    person, side = u.person.to_numpy(), u.side.to_numpy()
    age, sex, dev = u.age.to_numpy(float), u.sex.astype(str).to_numpy(), u.device.to_numpy()
    rng = np.random.default_rng(D.SEED)
    persons = np.unique(person)
    half_of = dict(zip(rng.permutation(persons), np.arange(len(persons)) % 2))
    half = np.array([half_of[p] for p in person])
    out, totals, dir_all = {}, {}, {}
    for h in ("all", 0, 1):
        members = np.ones(len(u), bool) if h == "all" else half == h
        cov_all = M[idx[members]].sum(0)
        valid = cov_all >= 0.9 * members.sum()
        sums = {c: [np.zeros((SIZE, SIZE)), np.zeros((SIZE, SIZE))] for c in COLS}
        vsum, vcnt = np.zeros((SIZE, SIZE)), np.zeros((SIZE, SIZE))
        tot = {c: {} for c in COLS}
        n_t = {c: 0 for c in COLS}
        dirs = {c: {} for c in MIRRORS}                    # direction -> [num, den, targets]
        y0 = AXIS_Y[name]
        upper = (np.arange(SIZE) < y0)[:, None] & np.ones((1, SIZE), bool)
        for s in ("R", "L"):
            S = np.flatnonzero(members & (side == s))
            if len(S) < 20:
                continue
            Xs, Ms = X[idx[S]], M[idx[S]]
            ssum, scnt = (Xs * Ms).sum(0), Ms.sum(0)
            excl = person[S][:, None] == person[S][None, :]
            if not excl.diagonal().all():
                raise result_spec.SpecError("exclusion matrix must exclude each target's own person")
            LF = lookalike_fill(Xs, Ms, Xs, Ms, valid, excl)
            for ti, i in enumerate(S):
                xt, mt = X[idx[i]], M[idx[i]]
                n_others = scnt - mt
                mean = np.where(n_others > 0, (ssum - xt * mt) / np.maximum(n_others, 1), 0.0)
                vt = mt & valid & (n_others > 0)
                eV = local_error(xt, mean, vt)
                fills = {}
                if other[i] >= 0:
                    fills["repeat"] = [(X[other[i]], M[other[i]])]
                fills["mirror"] = [mirror_fill(xt, mt, AXIS_Y[name])]
                j = fellow[i]
                if j >= 0 and members[j]:
                    fills["fellow"] = [(X[idx[j]], M[idx[j]])]
                    fills["fellow_mirror"] = [mirror_fill(X[idx[j]], M[idx[j]], y0)]
                pool = S[person[S] != person[i]]
                result_spec.assert_leave_person_out(person[i], person[pool])
                strangers = np.random.default_rng([D.SEED, int(i)]).choice(pool, N_STRANGER, replace=False)
                fills["stranger"] = [(X[idx[q]], M[idx[q]]) for q in strangers]
                fills["lookalike"] = [(np.nan_to_num(LF[ti]), np.isfinite(LF[ti]))]
                coh = pool[(sex[pool] == sex[i]) & (np.abs(age[pool] - age[i]) <= 5)
                           & ((dev[pool] == dev[i]) if name == "GRAPE" else True)]
                if len(coh) >= 10:
                    cm, cn = masked_mean(X[idx[coh]], M[idx[coh]])
                    fills["sexage"] = [(cm, cn >= 0.5 * len(coh))]
                for c, fl in fills.items():
                    es, nums, dens = [], [], []
                    for f, mf in fl:
                        jm = vt & mf.astype(bool)
                        es.append(local_error(xt, f, jm))
                        nums.append(float((((xt - f) ** 2) * jm).sum()))
                        dens.append(float((((xt - mean) ** 2) * jm).sum()))
                    e = np.nanmean(np.stack(es), 0) if len(es) > 1 else es[0]
                    ok = np.isfinite(e) & np.isfinite(eV)
                    sums[c][0] += np.where(ok, e, 0)
                    sums[c][1] += np.where(ok, eV, 0)
                    if c in MIRRORS:                       # both directions, reported separately
                        f, mf = fl[0]
                        jm = vt & mf.astype(bool)
                        parts = ({"upper vs reflected lower": jm & upper, "lower vs reflected upper": jm & ~upper}
                                 if MIRRORS[c] == "halves" else {f"{side[i]} eye as target": jm})
                        for dn, pm in parts.items():
                            acc = dirs[c].setdefault(dn, [0.0, 0.0, 0])
                            acc[0] += float((((xt - f) ** 2) * pm).sum())
                            acc[1] += float((((xt - mean) ** 2) * pm).sum())
                            acc[2] += 1
                    a, b = tot[c].get(person[i], (0.0, 0.0))
                    tot[c][person[i]] = (a + float(np.mean(nums)), b + float(np.mean(dens)))
                    n_t[c] += 1
                okv = np.isfinite(eV)
                vsum += np.where(okv, eV, 0)
                vcnt += okv
        maps = {c: np.where(valid & (sums[c][1] > 0), sums[c][0] / np.maximum(sums[c][1], 1e-12), np.nan)
                for c in COLS if n_t[c] >= 10}
        maps["V"] = np.where(valid & (vcnt > 0), vsum / np.maximum(vcnt, 1), np.nan)
        out[str(h)] = maps
        if h == "all":
            mean_all, _ = masked_mean(X[idx], M[idx])
            maps["meanfundus"] = np.where(valid, mean_all, np.nan)
            maps["valid"] = valid
            totals = {c: (tot[c], n_t[c]) for c in COLS}
            dir_all = dirs
    res = {}
    for c, (pp, n) in totals.items():
        if n < 10:
            continue
        ratio, ci = boot_ratio(pp, np.random.default_rng(D.SEED))
        a, b = out["0"].get(c), out["1"].get(c)
        r = None
        if a is not None and b is not None:
            ok = np.isfinite(a) & np.isfinite(b)
            r = round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 2) if ok.sum() > 10 else None
        res[c] = {"ratio": round(ratio, 3), "ci95": [round(x, 3) for x in ci], "targets": n, "persons": len(pp),
                  "split_half_r": r}
        if c in MIRRORS:
            res[c]["directions"] = {dn: {"ratio": round(v[0] / v[1], 3), "targets": v[2]}
                                    for dn, v in dir_all[c].items() if v[1] > 0}
    res["valid_px"] = int(out["all"]["valid"].sum())
    res["units"] = int(len(u))
    np.savez_compressed(os.path.join(D.PRIVATE, f"facefmt_{name}_{arm}.npz"),
                        **{k: np.asarray(v, np.float32) for k, v in out["all"].items()})
    return res


def run():
    ext = os.environ.get("LADDER_EXTERNAL")
    if ext:
        F.S.GRAPE_META = os.path.join(ext, *os.path.relpath(F.S.GRAPE_META, os.path.join(
            os.path.dirname(F.S.__file__), "..", "external")).split(os.sep))
    results = {}
    for name, arm in SETS:
        results[name] = {"arm": arm, **run_dataset(name, arm)}
        print(name, json.dumps(results[name]), flush=True)
    meta = {"protocol": "docs/PROTOCOL_donor_ladder_general.md, Amendments 9.5-9.7", "status": "pre-QC",
            "spec": result_spec.SPEC,
            "spec_meta": {"bootstrap_unit": "person", "pools": "leave-person-out", "stranger_rung": "stranger",
                          "mirror_rungs": MIRRORS},
            "scale": "1 = dataset mean, 2 = stranger (squared error, z units, 128 px frame)",
            "window_px": WIN, "lookalike": {"step": STEP, "ring": RING, "k": K_LOOK}, "strangers": N_STRANGER,
            "results": results}
    for flag in result_spec.enforce(meta):              # hard violations stop here, before anything is written
        print("SPEC (soft):", flag, flush=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(OUT_JSON, encoding="utf-8"))["results"]
    cols = [("meanfundus", "Mean fundus"), ("repeat", "Another photo"), ("mirror", "Mirror side**"),
            ("fellow", "Fellow eye"), ("fellow_mirror", "Fellow eye mirrored**"), ("sibling", "Sibling"), ("stranger", "Stranger"), ("lookalike", "16 look-alikes*"),
            ("sexage", "Same sex and age"), ("V", "Dataset mean")]
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(len(SETS), len(cols), figsize=(2.1 * len(cols), 2.6 * len(SETS) + 0.9))
    for r, (name, arm) in enumerate(SETS):
        z = np.load(os.path.join(D.PRIVATE, f"facefmt_{name}_{arm}.npz"))
        valid = z["valid"] > 0.5
        for key, _ in cols:                               # common display mask: a pixel shown in every panel of the
            if key in z.files and key not in ("meanfundus", "valid"):   # row only if every map is defined there
                valid &= np.isfinite(z[key])
        ys, xs = np.flatnonzero(valid.any(1)), np.flatnonzero(valid.any(0))
        crop = (slice(max(ys[0] - 4, 0), ys[-1] + 5), slice(max(xs[0] - 4, 0), xs[-1] + 5))
        under = np.where(valid, z["meanfundus"], np.nan)[crop]
        for c, (key, title) in enumerate(cols):
            a = ax[r, c]
            a.axis("off")
            if r == 0:
                a.set_title(title, fontsize=11)
            a.imshow(under, cmap="gray", vmin=-2.5, vmax=2.5)
            if key == "meanfundus":
                a.text(0.5, -0.08, f"{name} ({arm.upper()})\n(pool mean, no person)", ha="center", va="top",
                       transform=a.transAxes, fontsize=8.5)
                continue
            if key not in z.files or key not in res[name] and key != "V":
                a.text(0.5, 0.5, "not available", ha="center", va="center", transform=a.transAxes, fontsize=9,
                       bbox=dict(fc="white", ec="none", alpha=0.8))
                continue
            m = np.where(valid, z[key], np.nan)[crop]
            if key == "V":
                a.imshow(m, cmap="viridis")
                a.text(0.5, -0.08, "MSE = 1 x mean\n(shown: V, z units)", ha="center", va="top",
                       transform=a.transAxes, fontsize=8.5)
                continue
            s = res[name][key]
            a.imshow(m, cmap="RdBu_r", vmin=0, vmax=2, alpha=0.9)
            r_txt = f"split-half r {s['split_half_r']:.2f}" if s.get("split_half_r") is not None else "split-half r n/a"
            ci = s["ci95"]
            a.text(0.5, -0.08, f"MSE = {s['ratio']:.2f} x mean [{ci[0]:.2f}, {ci[1]:.2f}]\n{r_txt}\nn = {s['targets']}",
                   ha="center", va="top", transform=a.transAxes, fontsize=8.5)
    fig.suptitle("Fundus, every location: pixel error against the dataset mean (pre-QC)", fontsize=13)
    fig.text(0.5, 0.01, "One scale: 1 = dataset mean (white), 2 = stranger; blue beats the mean. * neighbours chosen on "
             "a 4 px grid by the 8 px ring around the 16 px window. ** the same eye reflected top-bottom about the "
             "disc-fovea line (2.5 px strip excluded);\nPAPILA: line through the disc, fovea outside the field. "
             "128 px frame, 16 px window; outside the field "
             "is missing. PAPILA P9, GRAPE P8 (Amendment 9.2); persons left out of their own pools.",
             ha="center", fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "results", "figures", f"fig_ladder2d_facefmt.{ext}"), dpi=200)
    print("figure written")


if __name__ == "__main__":
    (figure if "figure" in sys.argv[1:] else run)()
