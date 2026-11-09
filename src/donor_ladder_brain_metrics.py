"""Tier-1 metric panel on the QTIM brain donor ladder (PROTOCOL_donor_ladder_brain.md, Amendment 10). Pre-QC.

Reads the bilateral region arrays (Amendment 2) and rebuilds the same donors with the same seeds as
`donor_ladder_brain_bilateral.run_side`. Every metric is scored only on voxels inside the brain mask of both the
target and the fill (missing is never zero); a mean fill averages, per voxel, only the pool members valid there.
Squared error over all cube voxels (`mse_allvox`) is kept to reproduce the bilateral rows exactly.
Rows stay in the private folder; the public JSON holds aggregates only.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as BL  # noqa: E402

SEED = B.SEED
PRIVATE_IN = BL.PRIVATE
PRIVATE = os.path.join(B.PROJ, "donor-ladder-brain-metrics-20261106-v3")   # v1 = Amendment 10, v2 = Amendment 11
OUT_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_metrics.json")
BILATERAL_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_bilateral.json")

Z_RANGE = 8.0                     # fixed dynamic range (z units) for PSNR_fixed and the SSIM constants
SSIM_SIGMA = 1.5
LDDT_RADIUS = 2.0
LDDT_THR = (0.25, 0.5, 1.0, 2.0)
KS_MEAN = (2,)                    # Amendment 11: stranger (k = 1), mean of 2, dataset mean
N_STRANGERS = 16
MIN_SHARED = 0.5                  # share of the scaffold (or cube) that must be inside both brains
# higher = closer to the target after orientation; "amplitude" is descriptive and has no orientation
PROTEIN = ("tm_like", "gdt_like", "lddt_rep", "rmse_rep")   # Amendment 12, group 1
REP_MULT = (0.5, 1.0, 2.0, 4.0)                              # multiples of the repeat-scan tolerance
MIN_REP_PAIRS = 5
METRICS = ("mse", "rmse", "mae", "psnr", "psnr_fixed", "ssim", "pearson", "acc_dataset", "acc_cohort", "lddt") + PROTEIN
LOWER_BETTER = {"mse", "rmse", "mae", "rmse_rep", "lpips", "persistence_w1", "lpips_vgg"}   # Amendments 20, 22
DESCRIPTIVE = ("amplitude", "mse_allvox", "n_vox")
TESTS = {"T1": ("repeat", "dataset_mean"), "T2": ("repeat", "shrink_repeat"),
         "T3a": ("repeat", "wrong_detail"), "T3b": ("repeat", "noise")}


# ---------------------------------------------------------------------------------------------------- metrics

def masked_mean(X, W):
    """Per-voxel mean over rows of X using only rows valid there (W bool). Returns (mean, any_valid)."""
    W = W.astype(np.float64)
    n = W.sum(0)
    s = (X.astype(np.float64) * W).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return m.astype(np.float32), n > 0


def _wconv(a, w):
    return ndimage.gaussian_filter(a * w, SSIM_SIGMA, mode="constant", truncate=3.0)


def ssim_masked(x, f, e):
    """3D SSIM by normalised convolution: local statistics weight only the voxels in e, so nothing outside e (filler
    values included) enters; averaged over e."""
    w = e.astype(np.float64)
    x, f = x.astype(np.float64), f.astype(np.float64)
    g = ndimage.gaussian_filter(w, SSIM_SIGMA, mode="constant", truncate=3.0)
    g = np.where(g > 1e-6, g, np.nan)
    mx, mf = _wconv(x, w) / g, _wconv(f, w) / g
    vx = _wconv(x * x, w) / g - mx ** 2
    vf = _wconv(f * f, w) / g - mf ** 2
    cxf = _wconv(x * f, w) / g - mx * mf
    c1, c2 = (0.01 * Z_RANGE) ** 2, (0.03 * Z_RANGE) ** 2
    s = ((2 * mx * mf + c1) * (2 * cxf + c2)) / ((mx ** 2 + mf ** 2 + c1) * (vx + vf + c2))
    return float(np.nanmean(s[e]))


def _offsets(radius=LDDT_RADIUS):
    r = int(np.floor(radius))
    out = []
    for d in np.ndindex(2 * r + 1, 2 * r + 1, 2 * r + 1):
        o = tuple(v - r for v in d)
        if o > (0, 0, 0) and np.sqrt(sum(v * v for v in o)) <= radius:   # one of each +-pair
            out.append(o)
    return out


OFFSETS = _offsets()


def _pair_slices(o, shape):
    a = tuple(slice(max(0, -d), n - max(0, d)) for d, n in zip(o, shape))
    b = tuple(slice(max(0, d), n - max(0, -d)) for d, n in zip(o, shape))
    return a, b


def lddt_like(x, f, e, thr=LDDT_THR):
    """Share of voxel pairs within LDDT_RADIUS (both in e) whose intensity difference is preserved within each
    threshold, averaged over thresholds. Invariant to a constant offset of the fill."""
    hits, total = np.zeros(len(thr)), 0
    for o in OFFSETS:
        a, b = _pair_slices(o, x.shape)
        ok = e[a] & e[b]
        if not ok.any():
            continue
        d = np.abs((f[a] - f[b]) - (x[a] - x[b]))[ok]
        total += d.size
        hits += np.array([(d < t).sum() for t in thr])
    return float((hits / total).mean()) if total else np.nan


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def score(x, f, e, mu_d, mu_c, sigma=None):
    """All tier-1 metrics of fill f against target x on the voxels e (3D bool). mu_d / mu_c: dataset and cohort
    means (3D) for the anomaly correlations and the amplitude ratio. sigma: repeat-scan tolerance (Amendment 12);
    None leaves the protein-style metrics undefined (NaN), never zero."""
    xe, fe = x[e].astype(np.float64), f[e].astype(np.float64)
    if xe.size < 10:
        return None
    d = fe - xe
    mse = float(np.mean(d * d))
    rng_x = float(xe.max() - xe.min())
    out = {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(d))),
           "psnr": float(10 * np.log10(rng_x ** 2 / mse)) if mse > 0 else np.inf,
           "psnr_fixed": float(10 * np.log10(Z_RANGE ** 2 / mse)) if mse > 0 else np.inf,
           "ssim": ssim_masked(x, f, e), "pearson": _corr(xe, fe), "lddt": lddt_like(x, f, e),
           "n_vox": int(e.sum())}
    md = mu_d[e].astype(np.float64)
    out["acc_dataset"] = _corr(xe - md, fe - md)
    out["acc_cohort"] = _corr(xe - mu_c[e], fe - mu_c[e]) if mu_c is not None else np.nan
    nx = np.linalg.norm(xe - md)
    out["amplitude"] = float(np.linalg.norm(fe - md) / nx) if nx > 0 else np.nan
    out.update(protein_like(x, f, e, d, mse, sigma))
    return out


def protein_like(x, f, e, d, mse, sigma):
    """Amendment 12 group 1: TM-, GDT- and lDDT-style scores with the tolerance sigma from repeat scans."""
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        return {k: np.nan for k in PROTEIN}
    ad = np.abs(d)
    return {"tm_like": float(np.mean(1.0 / (1.0 + (ad / sigma) ** 2))),
            "gdt_like": float(np.mean([(ad < m * sigma).mean() for m in REP_MULT])),
            "lddt_rep": lddt_like(x, f, e, tuple(m * sigma for m in REP_MULT)),
            "rmse_rep": float(np.sqrt(mse) / sigma)}


def is_reference_stranger(name):
    """The 16 strangers drawn for c are named str00..str15; the 'stranger' fill itself is a rung."""
    return len(name) == 5 and name.startswith("str") and name[3:].isdigit()


def oriented(m, v):
    return -v if m in LOWER_BETTER else v


# ---------------------------------------------------------------------------------------------------- controls

def shrink(d, mu, lam):
    return (mu + lam * (d - mu)).astype(np.float32)


def wrong_detail(x, mu, s, e):
    """mu + the stranger's residual made orthogonal to (x - mu) on e and scaled to ||x - mu|| on e."""
    rx, rs = (x - mu)[e].astype(np.float64), (s - mu)[e].astype(np.float64)
    nx2 = (rx * rx).sum()
    if nx2 <= 0:
        return None
    rs = rs - (rs @ rx) / nx2 * rx
    ns = np.linalg.norm(rs)
    if ns <= 0:
        return None
    out = mu.astype(np.float64).copy()
    out[e] += rs * np.sqrt(nx2) / ns
    return out.astype(np.float32)


def noise_fill(x, mu, e, rng):
    sd = float(np.std((x - mu)[e]))
    out = mu.astype(np.float64).copy()
    out[e] += rng.normal(scale=sd, size=int(e.sum()))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------------------------------- scaffold

def masked_dist(A, MA, b, mb, n_ref, min_share=MIN_SHARED):
    """Mean squared difference between each row of A and b over voxels valid in both (Amendment 11).
    A: (n, V) candidates, MA: (n, V) bool; b: (V,), mb: (V,) bool; n_ref: voxel count the share is taken of.
    Rows sharing fewer than min_share * n_ref voxels get +inf (never selected)."""
    both = MA & mb[None]
    cnt = both.sum(1)
    diff = np.where(both, A.astype(np.float64) - b.astype(np.float64)[None], 0.0)
    d = (diff * diff).sum(1) / np.maximum(cnt, 1)
    return np.where(cnt >= min_share * n_ref, d, np.inf)


def _xcorr(a, k):
    return signal.correlate(a, k, mode="valid", method="fft")


def block_search_masked(t_win, t_mask, region, r_mask, size, shell, allowed_x, oracle=False,
                        min_share=MIN_SHARED, min_cube=None):
    """Best offset of the target window inside a candidate region, on voxels valid in both (Amendment 11).

    t_win, t_mask: target window (size + 2 shell) and its mask; region, r_mask: candidate region (window + 2W).
    Matched voxels: the scaffold (or the cube itself if oracle) AND the target mask, AND the candidate mask at the
    offset. Distance = mean squared difference over them; an offset needs >= min_share of the matched set shared and
    its cube >= min_cube inside the candidate mask, and its x offset allowed. Returns (cube, cube_mask, offset) or
    None. Filler values outside either mask never enter (they are multiplied by the masks)."""
    min_cube = BL.VALID if min_cube is None else min_cube
    sm = BL.shell_mask(size, shell)
    sel = (~sm if oracle else sm) & t_mask
    n_ref = int(((~sm) if oracle else sm).sum())
    t = np.where(sel, t_win, 0.0).astype(np.float64)
    w = sel.astype(np.float64)
    rm = r_mask.astype(np.float64)
    rv = np.where(r_mask, region, 0.0).astype(np.float64)
    cnt = _xcorr(rm, w)
    ssd = _xcorr(rm, w * t * t) - 2 * _xcorr(rv, w * t) + _xcorr(rv * rv, w)
    cube_k = np.zeros(sm.shape)
    cube_k[~sm] = 1.0
    cube_in = _xcorr(rm, cube_k) / cube_k.sum()
    ok = (cnt >= min_share * n_ref - 1e-6) & (cube_in >= min_cube - 1e-9)
    ok &= np.asarray(allowed_x, bool)[:, None, None]
    if not ok.any():
        return None
    d = np.where(ok, ssd / np.maximum(cnt, 1), np.inf)
    o = np.unravel_index(np.argmin(d), d.shape)
    sl = tuple(slice(a + shell, a + shell + s) for a, s in zip(o, size))
    return region[sl], r_mask[sl], o


# ---------------------------------------------------------------------------------------------------- donors

def rung_lambdas():
    """lambda = r = 1 - ratio / 2 of the repeat and mirror rungs (bilateral JSON, random cubes, LR view)."""
    c = json.load(open(BILATERAL_JSON))["QTIM"]["regions"]["random_patches/LR"]["cohorts"]
    return {"repeat": 1 - c["repeat"]["1"]["err_over_V"] / 2, "mirror": 1 - c["other_hemisphere"]["1"]["err_over_V"] / 2}


def run_position(args):
    """Rows for one position at the Amendment 10 geometry (bilateral region memmaps, BL.SHELL shell, +-BL.W search)."""
    return run_position_at(tuple(args) + ({"shell": BL.SHELL, "W": BL.W, "paths": BL.region_paths(args[0])},))


def run_position_at(args):
    """Rows for one position (both sides). Mirrors donor_ladder_brain_bilateral.run_side for the donors it shares.

    The geometry is explicit (Amendment 21): geom = {"shell": mm, "W": block-search half-window mm, "paths": (region
    volume memmap, region mask memmap)}, regions of (size + 2 shell + 2 W) per side."""
    pi, t, tg_cols, pos, a, lam, tier23, only, geom = args   # tier23: None or a hook name in T23.EXTRAS; only: subset
    if tier23:
        import donor_ladder_brain_metrics_t23 as T23
    tg = pd.DataFrame(tg_cols)
    region, start, size = pos[pi]
    pv, pm = geom["paths"]
    vol, mask = np.load(pv), np.load(pm).astype(bool)
    drawn = BL.side_of(start, size, AFFINE_HOLDER["affine"]) if "affine" in AFFINE_HOLDER else None
    shell, W = geom["shell"], geom["W"]
    n = len(tg)
    fam, sex, age = tg.family.to_numpy(), tg.sex.to_numpy(), tg.age.to_numpy()
    ix = tg["index"].to_numpy()
    row_of = {(p, s): k for k, (p, s) in enumerate(zip(t["person"], t["session"]))}
    win_sl = tuple(slice(W, W + s + 2 * shell) for s in size)
    cube_sl = tuple(slice(W + shell, W + shell + s) for s in size)
    sm = BL.shell_mask(size, shell)
    rows = []
    for side in BL.SIDES:
        fo = BL.own_frame(side, drawn)
        own, other = vol[:, fo], vol[:, 1 - fo]
        cube_all = own[(slice(None),) + cube_sl]                       # (scans, *size)
        mk_all = mask[:, fo][(slice(None),) + cube_sl]
        valid_all = mk_all.reshape(len(cube_all), -1).mean(1) >= BL.VALID
        X, MK, valid = cube_all[ix], mk_all[ix], valid_all[ix]
        Mc = other[(slice(None),) + cube_sl][ix]
        MM = mask[:, 1 - fo][(slice(None),) + cube_sl][ix]
        mvalid = MM.reshape(n, -1).mean(1) >= BL.VALID
        win = own[(slice(None),) + win_sl][ix]
        wmask = mask[:, fo][(slice(None),) + win_sl][ix]
        C, CM = win[:, sm], wmask[:, sm]                               # scaffold values and validity
        Xf, XM = X.reshape(n, -1), MK.reshape(n, -1)
        n_shell, n_cube = int(sm.sum()), Xf.shape[1]
        allowed_x = BL.allowed_offsets(start[0], size[0], a, w=W)
        # repeat-scan tolerance (Amendment 12): mean squared session difference per family, leave-family-out
        rep_msd = {}
        for i in np.flatnonzero(valid):
            j2 = row_of.get((tg.person.iloc[i], "ses-02"))
            if j2 is not None and valid_all[j2]:
                both = MK[i] & mk_all[j2]
                if both.sum() >= 10:
                    rep_msd.setdefault(fam[i], []).append(float(np.mean((X[i][both] - cube_all[j2][both]) ** 2)))
        for i in np.flatnonzero(valid):
            if only is not None and int(i) not in only:
                continue
            others = np.flatnonzero((fam != fam[i]) & valid)
            pairs = [v for f_, vs in rep_msd.items() if f_ != fam[i] for v in vs]
            sigma = float(np.sqrt(np.mean(pairs))) if len(pairs) >= MIN_REP_PAIRS else None
            if others.size < 10:
                continue
            xi, ei = X[i], MK[i]
            mu_plain = X[others].mean(0)
            mu_d, mu_d_ok = masked_mean(X[others], MK[others])
            cpool = others[(sex[others] == sex[i]) & (np.abs(age[others] - age[i]) <= 2)]
            mu_c, mu_c_ok = masked_mean(X[cpool], MK[cpool]) if cpool.size else (None, None)
            pop = np.random.default_rng([SEED, i, pi, B.DRAWN.index("population")]).permutation(others)
            coh = np.random.default_rng([SEED, i, pi, B.DRAWN.index("cohort")]).permutation(cpool)
            fills = {}   # name -> (fill used for metrics, its validity, plain fill for mse_allvox or None)
            fills["stranger"] = (X[pop[0]], MK[pop[0]], X[pop[0]])
            for j, s in enumerate(pop[:N_STRANGERS]):
                fills[f"str{j:02d}"] = (X[s], MK[s], None)
            for k in KS_MEAN:
                m, ok = masked_mean(X[pop[:k]], MK[pop[:k]])
                fills[f"mean_k{k}"] = (m, ok, X[pop[:k]].mean(0))
            fills["dataset_mean"] = (mu_d, mu_d_ok, mu_plain)
            if coh.size:
                fills["cohort_mean"] = (mu_c, mu_c_ok, X[coh].mean(0))
            # scaffold block and oracle block (Amendment 11): single donors, masked distances
            d_s = masked_dist(C[others], CM[others], C[i], CM[i], n_shell)
            d_o = masked_dist(Xf[others], XM[others], Xf[i], XM[i], n_cube)
            for name, d in (("lookalike", d_s), ("lookalike_oracle", d_o)):
                if np.isfinite(d).any():
                    j = others[int(np.argmin(d))]
                    fills[name] = (X[j], MK[j], X[j])
            for name, j, orc in (("mirror_block", i, False), ("stranger_block", int(pop[0]), False),
                                 ("mirror_block_oracle", i, True), ("stranger_block_oracle", int(pop[0]), True)):
                hit = block_search_masked(win[i], wmask[i], other[ix[j]], mask[ix[j], 1 - fo], size, shell,
                                          allowed_x, oracle=orc)
                if hit is not None:
                    fills[name] = (hit[0], hit[1], hit[0])
            if mvalid[i]:
                fills["mirror"] = (Mc[i], MM[i], Mc[i])
            rel = np.flatnonzero((fam == fam[i]) & (np.arange(n) != i) & valid)
            sib = [j for j in rel if age[j] != age[i]]
            twin = [j for j in rel if age[j] == age[i]]
            if sib:
                fills["sibling"] = (X[sib[0]], MK[sib[0]], X[sib[0]])
            if twin:
                fills["cotwin"] = (X[twin[0]], MK[twin[0]], X[twin[0]])
            j2 = row_of.get((tg.person.iloc[i], "ses-02"))
            if j2 is not None and valid_all[j2]:
                fills["repeat"] = (cube_all[j2], mk_all[j2], cube_all[j2])
            # controls built from the truth; validity = target and the donor they start from
            if "repeat" in fills:
                f, ok, _ = fills["repeat"]
                fills["shrink_repeat"] = (shrink(f, mu_d, lam["repeat"]), ok & mu_d_ok, None)
            if "mirror" in fills:
                f, ok, _ = fills["mirror"]
                fills["shrink_mirror"] = (shrink(f, mu_d, lam["mirror"]), ok & mu_d_ok, None)
            e_wd = ei & mu_d_ok & MK[pop[0]]
            wd = wrong_detail(xi, mu_d, X[pop[0]], e_wd)
            if wd is not None:
                fills["wrong_detail"] = (wd, e_wd, None)
            e_nz = ei & mu_d_ok
            fills["noise"] = (noise_fill(xi, mu_d, e_nz, np.random.default_rng([SEED, i, pi, BL.SIDES.index(side), 99])),
                              e_nz, None)
            for name, (f, ok, plain) in fills.items():
                s = score(xi, f, ei & ok, mu_d, mu_c, sigma)
                if s is None:
                    continue
                s["mse_allvox"] = float(np.mean((xi - plain) ** 2)) if plain is not None else np.nan
                if tier23:
                    s.update(T23.EXTRAS[tier23](xi, f, ei & ok, mu_c))
                rows.append({"target": int(i), "patch": pi, "side": side, "drawn": side == drawn,
                             "fill": name, **s})
    return rows


AFFINE_HOLDER = {}


def _init(affine):
    AFFINE_HOLDER["affine"] = affine


# ---------------------------------------------------------------------------------------------------- summary

def boot_mean(tab, col, fam_col="family", reps=2000, seed=SEED):
    """Mean over targets of tab[col] with a 95% interval by bootstrap over families."""
    tab = tab.dropna(subset=[col])
    if tab.empty:
        return None
    g = tab.groupby(fam_col)[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(reps, len(g)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return {"mean": float(s.sum() / c.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_targets": int(c.sum()), "n_families": int(len(g))}


def boot_position(per, m, fill, reps=2000, seed=SEED):
    """(m_fill - m_mean) / (m_repeat - m_mean) as a ratio of means over targets that have all three."""
    w = per[per.fill.isin(["repeat", "dataset_mean", fill])].pivot_table(index=["target", "family"], columns="fill",
                                                                          values=m).dropna()
    if any(c not in w for c in (fill, "repeat", "dataset_mean")) or w.empty:   # e.g. ACC is undefined for the mean
        return None
    w = w.reset_index()
    g = w.groupby("family")[sorted({fill, "repeat", "dataset_mean"})].sum()   # fill may be one of the anchors
    num = (g[fill] - g["dataset_mean"]).to_numpy()
    den = (g["repeat"] - g["dataset_mean"]).to_numpy()
    if abs(den.sum()) < 1e-12:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(reps, len(g)))
    bs = num[idx].sum(1) / den[idx].sum(1)
    return {"position": float(num.sum() / den.sum()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                float(np.percentile(bs, 97.5))],
            "n_targets": int(len(w)), "n_families": int(len(g))}


def calibrated_c(r, metrics):
    """c per (target, patch, side, fill, metric): P(stranger worse) + P(tie) / 2 over the 16 strangers.
    metrics: the panel scored (METRICS for tier 1, T23 metrics for Amendment 20), passed explicitly."""
    strs = r[r.fill.str.match(r"str\d\d$")]
    keys = ["target", "patch", "side"]
    out = []
    for m in metrics:
        sv = strs.assign(v=oriented(m, strs[m])).groupby(keys)["v"].apply(np.asarray)
        fr = r[~r.fill.str.match(r"str\d\d$")][keys + ["fill", m]].copy()
        fr["v"] = oriented(m, fr[m])
        ref = sv.reindex(pd.MultiIndex.from_frame(fr[keys])).to_numpy()
        c = [np.nan if not isinstance(s, np.ndarray) or np.isnan(v) else
             float(((s < v).sum() + 0.5 * (s == v).sum()) / s.size) for s, v in zip(ref, fr["v"].to_numpy())]
        out.append(fr[keys + ["fill"]].assign(metric=m, c=c))
    return pd.concat(out, ignore_index=True)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, k in enumerate(order):
        run = max(run, (len(p) - rank) * p[k])
        adj[k] = min(1.0, run)
    return adj


def tests(per, metrics):
    """T1-T3 per metric on family means of the paired, oriented difference (repeat minus control); Holm over the
    metric x test hypotheses of this call. metrics passed explicitly (tier 1 or Amendment 20)."""
    res, ps = {}, []
    for m in metrics:
        for tname, (a, b) in TESTS.items():
            w = per[per.fill.isin([a, b])].pivot_table(index=["target", "family"], columns="fill", values=m).dropna()
            if a not in w or b not in w:
                res.setdefault(m, {})[tname] = {"status": "not run (no pairs)"}
                continue
            w = w.reset_index()
            w["d"] = oriented(m, w[a]) - oriented(m, w[b])
            fm = w.groupby("family")["d"].mean()
            entry = {"n_targets": int(len(w)), "n_families": int(len(fm)), "mean_diff": float(w["d"].mean()),
                     "share_targets_pass": float((w["d"] > 0).mean())}
            if len(fm) >= 10 and np.any(fm.to_numpy() != 0):
                entry["p_one_sided"] = float(stats.wilcoxon(fm.to_numpy(), alternative="greater").pvalue)
                ps.append((m, tname, entry["p_one_sided"]))
            else:
                entry["status"] = "descriptive only (fewer than 10 families)"
            res.setdefault(m, {})[tname] = entry
    adj = holm([p for _, _, p in ps]) if ps else []
    for (m, tname, _), pa in zip(ps, adj):
        res[m][tname]["p_holm"] = float(pa)
        res[m][tname]["pass_holm_0.05"] = bool(pa < 0.05)
    return {"n_hypotheses_holm": len(ps), "per_metric": res}


TOL = 0.1          # Amendment 11 tolerance: "not better than a stranger" within 10% of the scale
# Amendment 13: reality B ("do not lie") is the reference; Amendment 11's E3-E5 are kept as descriptive (EXPECT_A)
EXPECT = {"E1 mirror closer than stranger": ("upper_lt", "mirror", 1.0),
          "E2 sibling closer than stranger": ("upper_lt", "sibling", 1.0),
          "E3B dataset mean better than stranger": ("upper_lt", "dataset_mean", 1.0),
          "E3B mean of 2 better than stranger": ("upper_lt", "mean_k2", 1.0),
          "E3B averages ordered by k": ("order", ("dataset_mean", "mean_k2", "stranger"), None),
          "E5B wrong detail not better than stranger": ("lower_ge", "wrong_detail", 1.0 - TOL),
          "E6B white noise worse than stranger": ("lower_gt", "noise", 1.0)}
EXPECT_A = {"A-E3a mean of 2 not better than stranger": ("lower_ge", "mean_k2", 1.0 - TOL),
            "A-E3b dataset mean not better than stranger": ("lower_ge", "dataset_mean", 1.0 - TOL),
            "A-E4a repeat blend not rewarded": ("contrast_ge", ("shrink_repeat", "repeat"), -TOL),
            "A-E4b mirror blend not rewarded": ("contrast_ge", ("shrink_mirror", "mirror"), -TOL)}


def ladder_R(per, m, reps=2000, seed=SEED):
    """R(rung) = (mean d(rung) - mean d(repeat)) / (mean d(stranger) - mean d(repeat)), d = distance (lower = closer),
    stranger anchor = per-target mean over the 16 seeded strangers. One shared family resample for all rungs.
    Returns ({fill: {R, ci95, n_targets, n_families}}, {fill: bootstrap array}) or (None, None)."""
    tab = per[["target", "family", "fill", m]].dropna()
    tab = tab.assign(d=-oriented(m, tab[m].to_numpy()))
    strs = tab[tab.fill.map(is_reference_stranger)].groupby(["target", "family"])["d"].mean().reset_index()
    strs["fill"] = "_stranger_anchor"
    tab = pd.concat([tab[~tab.fill.map(is_reference_stranger)][["target", "family", "fill", "d"]], strs],
                    ignore_index=True)
    if "repeat" not in set(tab.fill):
        return None, None
    fams = np.array(sorted(tab.family.unique()))
    fi = {f: k for k, f in enumerate(fams)}
    sums, cnts = {}, {}
    for f, g in tab.groupby("fill"):
        agg = g.groupby("family")["d"].agg(["sum", "count"])
        s_, c_ = np.zeros(len(fams)), np.zeros(len(fams))
        for fam, row in agg.iterrows():
            s_[fi[fam]], c_[fi[fam]] = row["sum"], row["count"]
        sums[f], cnts[f] = s_, c_
    idx = np.random.default_rng(seed).integers(0, len(fams), size=(reps, len(fams)))

    def means(sel=None):
        if sel is None:
            return {f: sums[f].sum() / cnts[f].sum() for f in sums}
        return {f: sums[f][sel].sum(1) / np.maximum(cnts[f][sel].sum(1), 1e-12) for f in sums}
    pt, bs = means(), means(idx)
    den_pt = pt["_stranger_anchor"] - pt["repeat"]
    den_bs = bs["_stranger_anchor"] - bs["repeat"]
    out, arr = {}, {}
    for f in sums:
        if f == "_stranger_anchor":
            continue
        r_pt = (pt[f] - pt["repeat"]) / den_pt
        r_bs = (bs[f] - bs["repeat"]) / den_bs
        arr[f] = r_bs
        out[f] = {"R": float(r_pt), "ci95": [float(np.nanpercentile(r_bs, 2.5)), float(np.nanpercentile(r_bs, 97.5))],
                  "n_targets": int(cnts[f].sum()), "n_families": int((cnts[f] > 0).sum())}
    out["_anchors"] = {"d_repeat": float(pt["repeat"]), "d_stranger": float(pt["_stranger_anchor"]),
                       "n_repeat_targets": int(cnts["repeat"].sum())}
    return out, arr


def expectations(R, arr, table=None):
    """Pass/fail of the expectations (Amendment 13 reality B by default) from the R intervals; None when a rung is
    missing."""
    res = {}
    for name, (kind, fill, thr) in (EXPECT if table is None else table).items():
        if kind == "order":
            if any(f not in R for f in fill):
                res[name] = None
                continue
            vals = [R[f]["R"] for f in fill]
            res[name] = {"value": vals, "pass": bool(all(a < b for a, b in zip(vals, vals[1:])))}
            continue
        if kind == "contrast_ge":
            a, b = fill
            if a not in arr or b not in arr:
                res[name] = None
                continue
            dif = arr[a] - arr[b]
            lo, hi = float(np.nanpercentile(dif, 2.5)), float(np.nanpercentile(dif, 97.5))
            res[name] = {"value": float(R[a]["R"] - R[b]["R"]), "lower95": lo, "upper95": hi, "pass": bool(lo >= thr)}
        else:
            if fill not in R:
                res[name] = None
                continue
            lo, hi = R[fill]["ci95"]
            ok = hi < thr if kind == "upper_lt" else (lo > thr if kind == "lower_gt" else lo >= thr)
            res[name] = {"value": R[fill]["R"], "ci95": [lo, hi], "pass": bool(ok)}
    if "dataset_mean" in R and "stranger" in R:
        res["slope dataset mean minus stranger"] = R["dataset_mean"]["R"] - R["stranger"]["R"]
    return res


def reproduce(r, t, old_path):
    """mse_allvox of the shared donors must equal the rows of the base ladder run in old_path exactly (same target,
    patch, side): the bilateral rows at 16 mm, the Amendment 4 size-grid rows at other sizes (Amendment 21)."""
    old = pd.read_csv(old_path, dtype={"k": str})
    pairs = {"repeat": ("repeat", "1"), "mirror": ("other_hemisphere", "1"), "sibling": ("sibling", "1"),
             "cotwin": ("cotwin", "1"), "stranger": ("population", "1"), "dataset_mean": ("population", "all"),
             "cohort_mean": ("cohort", "all"), **{f"mean_k{k}": ("population", str(k)) for k in KS_MEAN}}
    # changed by design (Amendment 11 scaffold rule): reported, not required to match
    changed = {"lookalike": ("knn", "1"), "lookalike_oracle": ("knn_oracle", "1")}
    out = {}
    for new, (coh, k) in pairs.items():
        a = r[r.fill == new][["target", "patch", "side", "mse_allvox"]]
        b = old[(old.cohort == coh) & (old.k == k)][["target", "patch", "side", "err"]].dropna(subset=["err"])
        b = b[b.patch.isin(r.patch.unique())]
        j = a.merge(b, on=["target", "patch", "side"], how="outer", indicator=True)
        both = j[j._merge == "both"]
        out[new] = {"rows_new": int(len(a)), "rows_old": int(len(b)), "matched": int(len(both)),
                    "max_abs_diff": float(np.abs(both.mse_allvox - both.err).max()) if len(both) else None}
    ok = all(v["rows_new"] == v["rows_old"] == v["matched"] and (v["max_abs_diff"] or 0) < 1e-5 for v in out.values())
    for new_, (coh, k) in changed.items():
        a = r[r.fill == new_][["target", "patch", "side", "mse_allvox"]]
        b = old[(old.cohort == coh) & (old.k == k)][["target", "patch", "side", "err"]]
        b = b[b.patch.isin(r.patch.unique())]
        j = a.merge(b, on=["target", "patch", "side"])
        out[new_ + " (changed by design)"] = {"rows_new": int(len(a)), "rows_old": int(len(b)),
                                              "share_same_donor_proxy": float((np.abs(j.mse_allvox - j.err) < 1e-9).mean())
                                              if len(j) else None}
    return ok, out


def views(r, pos):
    reg = pd.Series([pos[p][0] for p in r.patch.to_numpy()], index=r.index, dtype=object)
    left, right = (r.side == "L").to_numpy(), (r.side == "R").to_numpy()
    parts = [r.assign(view=reg + "/LR"), r[left].assign(view=reg[left] + "/L"),
             r[right].assign(view=reg[right] + "/R")]
    return pd.concat(parts, ignore_index=True)


def ladder_R_2way(per2, m, reps=2000, seed=SEED):
    """Two-way bootstrap of R (Amendment 12): families and positions resampled independently. per2 has one row per
    (target, patch, fill). Returns {fill: array of R over resamples} and the point R per fill."""
    tab = per2[["target", "family", "patch", "fill", m]].dropna()
    tab = tab.assign(d=-oriented(m, tab[m].to_numpy()))
    strs = tab[tab.fill.map(is_reference_stranger)].groupby(["target", "family", "patch"])["d"].mean().reset_index()
    strs["fill"] = "_stranger_anchor"
    tab = pd.concat([tab[~tab.fill.map(is_reference_stranger)][["target", "family", "patch", "fill", "d"]], strs],
                    ignore_index=True)
    if "repeat" not in set(tab.fill):
        return None, None
    fams, pats = np.array(sorted(tab.family.unique())), np.array(sorted(tab.patch.unique()))
    fi = {f: k for k, f in enumerate(fams)}
    pj = {p_: k for k, p_ in enumerate(pats)}
    S, Cn = {}, {}
    for f, g in tab.groupby("fill"):
        agg = g.groupby(["family", "patch"])["d"].agg(["sum", "count"])
        s_, c_ = np.zeros((len(fams), len(pats))), np.zeros((len(fams), len(pats)))
        for (fam_, pat_), row in agg.iterrows():
            s_[fi[fam_], pj[pat_]], c_[fi[fam_], pj[pat_]] = row["sum"], row["count"]
        S[f], Cn[f] = s_, c_
    rng = np.random.default_rng(seed)
    idx_f = rng.integers(0, len(fams), size=(reps, len(fams)))
    idx_p = rng.integers(0, len(pats), size=(reps, len(pats)))
    bs = {f: np.empty(reps) for f in S}
    for k in range(reps):
        for f in S:
            sub_s = S[f][idx_f[k]][:, idx_p[k]].sum()
            sub_c = Cn[f][idx_f[k]][:, idx_p[k]].sum()
            bs[f][k] = sub_s / sub_c if sub_c > 0 else np.nan
    pt = {f: S[f].sum() / Cn[f].sum() for f in S}
    out_arr, out_pt = {}, {}
    for f in S:
        if f == "_stranger_anchor":
            continue
        out_arr[f] = (bs[f] - bs["repeat"]) / (bs["_stranger_anchor"] - bs["repeat"])
        out_pt[f] = (pt[f] - pt["repeat"]) / (pt["_stranger_anchor"] - pt["repeat"])
    return out_pt, out_arr


def calibrate_rows(r):
    """Amendment 12 group 2 per row: Tnorm z against the 16 strangers of the same target, position and side; EVT
    P-value of z under a GEV fitted to the pooled stranger z of that position and side; P(same | z) from Gaussian
    densities of repeat and stranger z per position and side (prior 0.5). Adds columns z_<m>, nlp_<m>, post_<m>."""
    keys = ["target", "patch", "side"]
    ref = r[r.fill.map(is_reference_stranger)]
    out = r.copy()
    for m in METRICS:
        o = oriented(m, ref[m])
        g = ref.assign(o=o).groupby(keys)["o"].agg(["mean", "std"])
        mu = g["mean"].reindex(pd.MultiIndex.from_frame(out[keys])).to_numpy()
        sd = g["std"].reindex(pd.MultiIndex.from_frame(out[keys])).to_numpy()
        z = (oriented(m, out[m]).to_numpy() - mu) / np.where(sd > 0, sd, np.nan)
        out[f"z_{m}"] = z
        nlp = np.full(len(out), np.nan)
        post = np.full(len(out), np.nan)
        is_ref = out.fill.map(is_reference_stranger).to_numpy()
        is_rep = (out.fill == "repeat").to_numpy()
        for (pi, side), idx in out.groupby(["patch", "side"]).indices.items():
            zs = z[idx][is_ref[idx]]
            zs = zs[np.isfinite(zs)]
            zr = z[idx][is_rep[idx]]
            zr = zr[np.isfinite(zr)]
            zi = z[idx]
            if zs.size >= 50:
                try:
                    c_, loc_, sc_ = stats.genextreme.fit(zs)
                    p = stats.genextreme.sf(zi, c_, loc_, sc_)
                    nlp[idx] = -np.log10(np.clip(p, 1e-300, 1.0))
                except Exception:   # noqa: BLE001 - a failed fit leaves NaN (reported as missing), never 0
                    pass
            if zs.size >= 50 and zr.size >= MIN_REP_PAIRS and np.std(zr) > 0:
                ls = stats.norm.logpdf(zi, zs.mean(), zs.std())
                lr = stats.norm.logpdf(zi, zr.mean(), zr.std())
                post[idx] = 1.0 / (1.0 + np.exp(np.clip(ls - lr, -700, 700)))
        out[f"nlp_{m}"] = nlp
        out[f"post_{m}"] = post
    return out


CALIB = tuple(f"{k}_{m}" for m in METRICS for k in ("z", "nlp", "post"))


def summarise(r, cdf, tg, pos):
    fam = tg.family.to_numpy()
    rv = views(r, pos)
    cv = views(cdf, pos)
    out = {}
    for view, rr in rv.groupby("view"):
        per = rr.groupby(["target", "fill"])[list(METRICS) + list(DESCRIPTIVE)].mean().reset_index()
        per["family"] = fam[per.target]
        cc = cv[cv.view == view].groupby(["target", "fill", "metric"])["c"].mean().reset_index()
        cc["family"] = fam[cc.target]
        fills = sorted(f for f in per.fill.unique() if not is_reference_stranger(f))
        v = {"fills": {}}
        for f in fills:
            pf = per[per.fill == f]
            v["fills"][f] = {m: {"raw": boot_mean(pf, m),
                                 "c": boot_mean(cc[(cc.fill == f) & (cc.metric == m)], "c") if m in METRICS else None,
                                 "position_mean0_repeat1": boot_position(per, m, f) if m in METRICS else None}
                             for m in list(METRICS) + ["amplitude", "mse_allvox"]}
        calib_cols = [c_ for c_ in CALIB if c_ in rr.columns]
        if calib_cols:
            pcal = rr.groupby(["target", "fill"])[calib_cols].mean().reset_index()
            pcal["family"] = fam[pcal.target]
            v["calibrated"] = {f: {c_: boot_mean(pcal[pcal.fill == f], c_) for c_ in calib_cols}
                               for f in fills}
        two_way = rr.patch.nunique() > 1
        if two_way:
            per2 = rr.groupby(["target", "patch", "fill"])[list(METRICS)].mean().reset_index()
            per2["family"] = fam[per2.target]
        v["ladder_R"], v["expectations"] = {}, {}
        for m in METRICS:
            R, arr = ladder_R(per, m)
            if R is None:
                continue
            if two_way:          # Amendment 12: intervals and expectations from families x positions
                pt2, arr2 = ladder_R_2way(per2, m)
                for f, e in R.items():
                    if f.startswith("_") or f not in arr2:
                        continue
                    e["ci95_family_only"] = e["ci95"]
                    e["ci95"] = [float(np.nanpercentile(arr2[f], 2.5)), float(np.nanpercentile(arr2[f], 97.5))]
                    e["R_cells"] = float(pt2[f])
                arr = arr2
            v["ladder_R"][m] = R
            v["expectations"][m] = expectations(R, arr)
            v.setdefault("expectations_realityA", {})[m] = expectations(R, arr, EXPECT_A)
        if view.startswith("random_patches/LR"):
            v["tests"] = tests(per, METRICS)
        out[view] = v
    return out


def finite(o):
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--positions", default="", help="comma-separated position indices (debug); default all")
    ap.add_argument("--summary-only", action="store_true", help="re-summarise the saved private rows")
    a = ap.parse_args()
    os.makedirs(PRIVATE, exist_ok=True)
    t0 = time.time()
    t = B.scans()
    pos, affine, n_cand = B.positions(np.random.default_rng(SEED))
    tg = t[t.session == "ses-01"].reset_index()
    lam = rung_lambdas()
    rows_path = os.path.join(PRIVATE, "rows.csv.gz")
    if a.summary_only:
        r = pd.read_csv(rows_path)
    else:
        mc = BL.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
        sel = [int(p) for p in a.positions.split(",")] if a.positions else list(range(len(pos)))
        tcols = t[["person", "session"]].to_dict("list")
        tgcols = tg[["index", "person", "family", "sex", "age"]].to_dict("list")
        # resumable: one file per position; positions already written are skipped (a crash loses one position)
        parts = os.path.join(PRIVATE, "parts")
        os.makedirs(parts, exist_ok=True)
        part_path = lambda pi: os.path.join(parts, f"p{pi:02d}.csv.gz")  # noqa: E731
        todo = [pi for pi in sel if not os.path.exists(part_path(pi))]
        print(f"positions done {len(sel) - len(todo)}/{len(sel)}, to run {len(todo)}", flush=True)
        if todo:
            jobs = [(pi, tcols, tgcols, pos, mc, lam, False, None) for pi in todo]
            with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(affine,)) as ex:
                for k, (pi, part) in enumerate(zip(todo, ex.map(run_position, jobs))):
                    tmp = part_path(pi) + ".tmp"
                    pd.DataFrame(part).to_csv(tmp, index=False, compression="gzip")
                    os.replace(tmp, part_path(pi))
                    print(f"position {pi} ({k + 1}/{len(todo)}) rows {len(part)} {time.time() - t0:.0f}s", flush=True)
        r = pd.concat([pd.read_csv(part_path(pi)) for pi in sel], ignore_index=True)
        if not a.positions:
            r.to_csv(rows_path, index=False)
    ok, rep = reproduce(r, t, os.path.join(PRIVATE_IN, "rows.csv.gz"))
    print(json.dumps({"reproduces_bilateral": ok}), flush=True)
    if a.positions:
        print(json.dumps(rep, indent=1))
        return
    if not ok:
        print(json.dumps(rep, indent=1))
        raise SystemExit("mse_allvox does not reproduce the bilateral rows; no summary written")
    cdf = calibrated_c(r, METRICS)
    r = calibrate_rows(r)
    out = {"status": "pre-QC: affine registration only, MRI registration gate not met (results/registration/status.md)",
           "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendments 10 to 13)", "tier": 1, "seed": SEED,
           "metrics": list(METRICS), "lower_is_better": sorted(LOWER_BETTER), "descriptive": list(DESCRIPTIVE),
           "lambda_shrink": lam, "z_range": Z_RANGE, "ssim_sigma_vox": SSIM_SIGMA, "lddt_radius_vox": LDDT_RADIUS,
           "lddt_thresholds_z": list(LDDT_THR), "n_strangers_for_c": N_STRANGERS,
           "not_run": {"tier2": ["persistence / Betti matching", "Topology Distance"],
                       "tier3": ["MS-SSIM", "LPIPS", "Brain-ID embeddings", "FID", "FWD", "plausibility",
                                 "latent geometry"], "fills": ["block searches (Amendment 9)"]},
           "reproduction": rep,
           "counts": {"n_targets": int(len(tg)), "n_families": int(tg.family.nunique()),
                      "n_repeat_scans": int((t.session == "ses-02").sum()), "n_positions": len(pos)},
           "views": summarise(r, cdf, tg, pos)}
    out["counts"]["minutes_summary_only" if a.summary_only else "minutes"] = round((time.time() - t0) / 60, 1)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(finite(out), fh, indent=1, allow_nan=False)
    print(json.dumps(out["counts"]))


if __name__ == "__main__":
    main()
