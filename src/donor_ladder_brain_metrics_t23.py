"""Metric panel tiers 2 and 3, CPU part, on the frozen boxes (PROTOCOL_donor_ladder_brain.md, Amendment 20). Pre-QC.

Adds to each row of `donor_ladder_brain_metrics.run_position` (same donors, fills and scored voxels E):
MS-SSIM (3 scales, masked), LPIPS 2.5D (AlexNet, central orthogonal slices fully inside E), the amplitude against the
same-sex-and-age mean, and the intensity-persistence distance (cubical sublevel persistence, 1-Wasserstein summed over
dimensions 0-2). Nothing outside E enters any metric: outside voxels are excluded (MS-SSIM weights, LPIPS slice rule)
or set to +inf in the filtration (never born). Missing is NaN, never zero.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain_metrics as M  # noqa: E402

TORCH_HOME = os.path.join(M.B.PROJ, "models-20261107", "torch")
FROZEN = (32, 33)                                      # deep, parietal (Amendment 10 positions)
MS_WEIGHTS = np.array([0.0448, 0.2856, 0.3001])        # Wang et al. via piq 0.8.0, first three scales
MS_WEIGHTS = MS_WEIGHTS / MS_WEIGHTS.sum()
LPIPS_SIDE = 64                                        # short side of the upsampled slice, px
LPIPS_Z = 4.0                                          # z / 4 -> [-1, 1]
T23 = ("ms_ssim", "lpips", "persistence_w1")
T23_LOWER = {"lpips", "persistence_w1"}
T23_DESCRIPTIVE = ("amplitude_cohort", "lpips_slices", "persistence_essential")
_LPIPS = {}


def ssim_terms(x, f, e):
    """Masked SSIM luminance and contrast-structure maps (Amendment 10 constants), each averaged over e."""
    w = e.astype(np.float64)
    x, f = x.astype(np.float64), f.astype(np.float64)
    g = M.ndimage.gaussian_filter(w, M.SSIM_SIGMA, mode="constant", truncate=3.0)
    g = np.where(g > 1e-6, g, np.nan)
    mx, mf = M._wconv(x, w) / g, M._wconv(f, w) / g
    vx = M._wconv(x * x, w) / g - mx ** 2
    vf = M._wconv(f * f, w) / g - mf ** 2
    cxf = M._wconv(x * f, w) / g - mx * mf
    c1, c2 = (0.01 * M.Z_RANGE) ** 2, (0.03 * M.Z_RANGE) ** 2
    lum = (2 * mx * mf + c1) / (mx ** 2 + mf ** 2 + c1)
    cs = (2 * cxf + c2) / (vx + vf + c2)
    return float(np.nanmean(lum[e])), float(np.nanmean(cs[e]))


def downsample(v, e):
    """Halve each axis: mean of the valid voxels of every 2 x 2 x 2 block; a block is valid when >= 4 children are."""
    s = tuple(n - n % 2 for n in v.shape)
    v, e = v[:s[0], :s[1], :s[2]].astype(np.float64), e[:s[0], :s[1], :s[2]]
    shp = (s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2)
    n = e.reshape(shp).sum((1, 3, 5))
    tot = np.where(e, v, 0.0).reshape(shp).sum((1, 3, 5))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, tot / np.maximum(n, 1), 0.0)
    return m, n >= 4


def ms_ssim(x, f, e, scales=3):
    """Masked MS-SSIM: cs at every scale and luminance at the coarsest, clipped at 0, Wang weights renormalised."""
    vals = []
    for k in range(scales):
        if e.sum() < 10:
            return np.nan
        lum, cs = ssim_terms(x, f, e)
        vals.append(max(cs if k < scales - 1 else lum * cs, 0.0))
        if k < scales - 1:
            (x, ex), (f, ef) = downsample(x, e), downsample(f, e)
            e = ex & ef
    return float(np.prod(np.power(vals, MS_WEIGHTS)))


def _lpips(net):
    """LPIPS 0.1 model for net ("alex": Amendment 20, "vgg": Amendment 22), loaded once per process."""
    if net not in ("alex", "vgg"):
        raise ValueError(f"unknown LPIPS net {net!r}")
    if net not in _LPIPS:
        os.environ["TORCH_HOME"] = TORCH_HOME
        import lpips
        import torch
        torch.set_num_threads(1)
        _LPIPS[net], _LPIPS["torch"] = lpips.LPIPS(net=net, version="0.1", verbose=False).eval(), torch
    return _LPIPS[net], _LPIPS["torch"]


def central_slices(v, e):
    """The three central orthogonal slices of a box and their validity."""
    c = [n // 2 for n in v.shape]
    return [(v[c[0]], e[c[0]]), (v[:, c[1]], e[:, c[1]]), (v[:, :, c[2]], e[:, :, c[2]])]


def lpips_25d(x, f, e, net):
    """Mean LPIPS (net passed explicitly) over the central orthogonal slices lying fully inside e; (NaN, 0) when none
    does."""
    pairs = [(a, b) for (a, ea), (b, _) in zip(central_slices(x, e), central_slices(f, e)) if ea.all()]
    if not pairs:
        return np.nan, 0
    model, torch = _lpips(net)
    out = []
    with torch.no_grad():
        for a, b in pairs:
            t = torch.from_numpy(np.clip(np.stack([a, b]) / LPIPS_Z, -1, 1).astype(np.float32))[:, None]
            scale = LPIPS_SIDE / min(a.shape)
            t = torch.nn.functional.interpolate(t, scale_factor=scale, mode="bilinear", align_corners=False)
            t = t.repeat(1, 3, 1, 1)
            out.append(float(model(t[:1], t[1:]).item()))
    return float(np.mean(out)), len(out)


def diagrams(v, e):
    """Finite sublevel cubical persistence intervals in dimensions 0-2 (outside e at +inf) and the essential count."""
    import gudhi
    c = gudhi.CubicalComplex(top_dimensional_cells=np.where(e, v, np.inf).astype(np.float64))
    c.compute_persistence()
    dg, ess = [], 0
    for d in range(3):
        iv = np.asarray(c.persistence_intervals_in_dimension(d), dtype=np.float64).reshape(-1, 2)
        fin = np.isfinite(iv).all(1)
        ess += int((~fin).sum())
        dg.append(iv[fin])
    return dg, ess


def persistence_w1(dx, df):
    from gudhi.wasserstein import wasserstein_distance
    return float(sum(wasserstein_distance(a, b, order=1, internal_p=2) for a, b in zip(dx, df)))


def extra(x, f, e, mu_c, dx=None):
    """Tier 2/3 metrics of fill f against target x on e. dx: the target's diagrams on the same e, if cached."""
    if e.sum() < 10:
        return {}
    lp, n_sl = lpips_25d(x, f, e, net="alex")
    dx = diagrams(x, e)[0] if dx is None else dx
    df, ess = diagrams(f, e)
    out = {"ms_ssim": ms_ssim(x, f, e), "lpips": lp, "lpips_slices": n_sl,
           "persistence_w1": persistence_w1(dx, df), "persistence_essential": ess, "amplitude_cohort": np.nan}
    if mu_c is not None:
        nx = np.linalg.norm((x - mu_c)[e])
        out["amplitude_cohort"] = float(np.linalg.norm((f - mu_c)[e]) / nx) if nx > 0 else np.nan
    return out


SET_SIDE = 24                                          # central square of each central slice (shortest box edge)


def set_crops(x, f, e):
    """Amendment 22 set-level samples: the central SET_SIDE square of each central orthogonal slice of target and fill,
    and whether it lies fully inside e. Returns (x crops, f crops, valid) with shapes (3, S, S), (3, S, S), (3,)."""
    xs, fs, ok = [], [], []
    for (a, ea), (b, _) in zip(central_slices(x, e), central_slices(f, e)):
        r0, c0 = a.shape[0] // 2 - SET_SIDE // 2, a.shape[1] // 2 - SET_SIDE // 2
        if r0 < 0 or c0 < 0:
            raise ValueError(f"slice {a.shape} smaller than the set crop {SET_SIDE}")
        sl = (slice(r0, r0 + SET_SIDE), slice(c0, c0 + SET_SIDE))
        xs.append(a[sl])
        fs.append(b[sl])
        ok.append(bool(ea[sl].all()))
    return np.stack(xs).astype(np.float32), np.stack(fs).astype(np.float32), np.array(ok)


def extra_vgg_set(x, f, e, mu_c):
    """Amendment 22 hook: LPIPS VGG on the Amendment 20 slices plus the set-level crops (keys starting with '_' are
    arrays, split off before the rows are saved)."""
    if e.sum() < 10:
        return {}
    lp, n_sl = lpips_25d(x, f, e, net="vgg")
    xc, fc, ok = set_crops(x, f, e)
    return {"lpips_vgg": lp, "lpips_vgg_slices": n_sl, "_xc": xc, "_fc": fc, "_ok": ok}


def betti_25d(x, f, e):
    """Amendment 24C: Betti matching (2D library 3e7eca26, the fundus / iris settings and formula) on the central
    orthogonal slices lying wholly inside e, as LPIPS; the mean of the matched and unmatched parts over the scored
    slices; (NaN, NaN, 0, []) when none is scored (missing, never 0)."""
    import iris_ladder_metrics_extra as IX
    if IX.BETTI_DIR not in sys.path:
        sys.path.insert(0, IX.BETTI_DIR)
    pairs = [(IX.to_unit(a).astype(np.float64), IX.to_unit(b).astype(np.float64))
             for (a, ea), (b, _) in zip(central_slices(x, e), central_slices(f, e)) if ea.all()]
    if not pairs:
        return np.nan, np.nan, 0, []
    parts = np.array([IX.betti_parts(a, b) for a, b in pairs])
    return float(parts[:, 0].mean()), float(parts[:, 1].mean()), len(pairs), pairs


def extra_betti25d(x, f, e, mu_c):
    m, u, n, pairs = betti_25d(x, f, e)
    out = {"betti_matched": m, "betti_unmatched": u, "betti_slices": n}
    if os.environ.get("BETTI25D_QC") == "1" and pairs:                       # smoke only: slices for the QC sheet
        out["_qc"] = np.array(pairs, np.float32)
    return out


EXTRAS = {"t23": extra, "vgg_set": extra_vgg_set, "betti25d": extra_betti25d}   # tier23 hook name -> function


PRIVATE = os.path.join(M.B.PROJ, "donor-ladder-brain-metrics-20261107-v4")
CHUNK = 40                                             # targets per saved chunk
OUT_JSON = os.path.join(M.B.ROOT, "results", "donor_ladder_brain_metrics_t23.json")
NOT_RUN = {"betti_matching_3d": "needs a C++ build of Betti-Matching-3D (no compiler here; Linux VM)",
           "brain_id_identity": "own amendment; after FastSurfer frees the CPU",
           "fid_fwd_topology_distance": "set-level, own amendment",
           "plausibility_latent": "needs a generative brain model and a GPU"}


def summarise(r, tg, pos, metrics, descriptive):
    """Summary per box view of the metrics given (Amendment 20: T23 and T23_DESCRIPTIVE; Amendment 22: lpips_vgg):
    raw means and c (family bootstrap), R with intervals, the Amendment 13 expectations, and T1-T3 on the L+R view
    of each box with one Holm family over both boxes and these metrics."""
    fam = tg.family.to_numpy()
    cdf = M.calibrated_c(r, metrics)
    rv, cv = M.views(r, pos), M.views(cdf, pos)
    out, ps = {}, []
    for view, rr in rv.groupby("view"):
        per = rr.groupby(["target", "fill"])[list(metrics) + list(descriptive)].mean().reset_index()
        per["family"] = fam[per.target]
        cc = cv[cv.view == view].groupby(["target", "fill", "metric"])["c"].mean().reset_index()
        cc["family"] = fam[cc.target]
        fills = sorted(f for f in per.fill.unique() if not M.is_reference_stranger(f))
        v = {"n_rows": int(len(rr)),
             "missing": {m: int(rr[m].isna().sum()) for m in metrics},
             "fills": {f: {m: {"raw": M.boot_mean(per[per.fill == f], m),
                               "c": M.boot_mean(cc[(cc.fill == f) & (cc.metric == m)], "c") if m in metrics else None}
                           for m in list(metrics) + list(descriptive)} for f in fills},
             "ladder_R": {}, "expectations": {}, "expectations_realityA": {}}
        for m in metrics:
            if per[m].notna().sum() == 0:             # every value missing in this view: no scale, never zero
                v["ladder_R"][m] = None
                continue
            R, arr = M.ladder_R(per, m)
            if R is None:
                continue
            v["ladder_R"][m] = R
            v["expectations"][m] = M.expectations(R, arr)
            v["expectations_realityA"][m] = M.expectations(R, arr, M.EXPECT_A)
        if view.endswith("/LR"):
            v["tests"] = M.tests(per, metrics)
            ps += [(view, m, t, e["p_one_sided"]) for m, d in v["tests"]["per_metric"].items() for t, e in d.items()
                   if "p_one_sided" in e]
        out[view] = v
    for (view, m, t, _), pa in zip(ps, M.holm([p for *_, p in ps]) if ps else []):   # one family over both boxes
        e = out[view]["tests"]["per_metric"][m][t]
        e["p_holm_run"], e["pass_holm_run_0.05"] = float(pa), bool(pa < 0.05)
    return out, len(ps)


def main():
    import argparse
    import json
    import time
    from concurrent.futures import ProcessPoolExecutor

    import nibabel as nib
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0, help="score only the first N targets that have a repeat scan")
    ap.add_argument("--summary-only", action="store_true", help="summarise the saved chunks; run nothing")
    a = ap.parse_args()
    os.makedirs(os.path.join(PRIVATE, "parts"), exist_ok=True)
    t0 = time.time()
    t = M.B.scans()
    pos, affine, _ = M.B.positions(np.random.default_rng(M.SEED))
    assert [pos[p][0] for p in FROZEN] == ["deep", "parietal"], "frozen box positions moved"
    tg = t[t.session == "ses-01"].reset_index()
    only = None
    if a.smoke:
        rep = set(t[t.session == "ses-02"].person)
        only = set(np.flatnonzero(tg.person.isin(rep).to_numpy())[:a.smoke].tolist())
    lam = M.rung_lambdas()
    mc = M.BL.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    tcols = t[["person", "session"]].to_dict("list")
    tgcols = tg[["index", "person", "family", "sex", "age"]].to_dict("list")
    tag = f"smoke{a.smoke}_" if a.smoke else ""
    # resumable chunks of CHUNK targets per box: each chunk is saved on its own, so a crash loses one chunk only
    targets = sorted(only) if only is not None else list(range(len(tg)))
    chunks = [targets[k:k + CHUNK] for k in range(0, len(targets), CHUNK)]
    path = lambda pi, c: os.path.join(PRIVATE, "parts", f"{tag}p{pi:02d}_c{c:02d}.csv.gz")  # noqa: E731
    jobs = [((pi, tcols, tgcols, pos, mc, lam, "t23", set(ch)), pi, c) for pi in FROZEN for c, ch in enumerate(chunks)
            if not os.path.exists(path(pi, c))]
    n_all = len(FROZEN) * len(chunks)
    print(f"chunks done {n_all - len(jobs)}/{n_all}, to run {len(jobs)}", flush=True)
    if a.summary_only:
        if jobs or a.smoke:
            raise SystemExit(f"{len(jobs)} chunks missing (or --smoke given): no summary written")
        r = pd.concat([pd.read_csv(path(pi, c)) for pi in FROZEN for c in range(len(chunks))], ignore_index=True)
        if r.duplicated(["target", "patch", "side", "fill"]).any():
            raise SystemExit("duplicate rows across chunks: no summary written")
        views, n_hyp = summarise(r, tg, pos, T23, T23_DESCRIPTIVE)
        out = {"status": "pre-QC: affine registration only, MRI registration gate not met (results/registration/status.md)",
               "protocol": "docs/PROTOCOL_donor_ladder_brain.md (Amendment 20)", "tier": "2/3 (CPU part)",
               "seed": M.SEED, "boxes": [pos[p][0] for p in FROZEN], "metrics": list(T23),
               "lower_is_better": sorted(T23_LOWER), "descriptive": list(T23_DESCRIPTIVE),
               "ms_ssim_weights": MS_WEIGHTS.tolist(), "lpips": {"net": "alex", "version": "0.1",
               "short_side_px": LPIPS_SIDE, "z_scale": LPIPS_Z, "slices": "central orthogonal, fully inside E"},
               "persistence": "cubical sublevel (gudhi), outside E at +inf, W1 (internal p = 2) summed over H0-H2",
               "missing_rule": "missing is NaN, never zero; counts per view under 'missing'",
               "n_hypotheses_holm_run": n_hyp, "not_run": NOT_RUN,
               "counts": {"n_targets": int(r.target.nunique()), "n_rows": int(len(r)), "n_chunks": n_all},
               "views": views}
        out["counts"]["run"] = ("deep box: 7-8 Nov, 2 workers on a CPU shared with other jobs; parietal box: resumed "
                                "8 Nov after a reboot, 6 chunks in 72 min with 2 workers")
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(M.finite(out), fh, indent=1, allow_nan=False)
        print(f"saved {OUT_JSON}", flush=True)
        return
    with ProcessPoolExecutor(a.workers, initializer=M._init, initargs=(affine,)) as ex:
        for k, ((_, pi, c), part) in enumerate(zip(jobs, ex.map(M.run_position, [j[0] for j in jobs]))):
            pd.DataFrame(part).to_csv(path(pi, c) + ".tmp", index=False, compression="gzip")
            os.replace(path(pi, c) + ".tmp", path(pi, c))
            el = time.time() - t0
            eta = el / (k + 1) * (len(jobs) - k - 1)
            print(f"chunk {k + 1}/{len(jobs)} (box {pi}, targets {c * CHUNK}-{c * CHUNK + CHUNK - 1}): rows {len(part)}, "
                  f"{el / 60:.0f} min, ETA {eta / 60:.0f} min ({time.strftime('%H:%M', time.localtime(time.time() + eta))})",
                  flush=True)
    timing = {"minutes": round((time.time() - t0) / 60, 2), "workers": a.workers, "smoke_targets": a.smoke}
    with open(os.path.join(PRIVATE, f"{tag}timing.json"), "w", encoding="utf-8") as fh:
        json.dump(timing, fh, indent=1)
    print(json.dumps(timing))


if __name__ == "__main__":
    main()
