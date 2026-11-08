"""Iris metric panel, Amendment P3 (docs/PROTOCOL_iris_metrics.md): LPIPS (AlexNet, VGG), Betti matching (matched
and unmatched parts), FID, FWD and the iris-code Hamming distance on the same targets, windows and fills as the panel.

The panel's fills are rebuilt with its own functions (iris_ladder_metrics.fixed_fills / window_fills), so nothing about
targets, pools, seeds or E changes. Rules from the amendment:
- LPIPS: 64 x 64 context centred on the window (rows = the whole 64-row strip, columns wrapped in angle); pixels
  outside the target mask get the target's masked Gaussian local mean (sd 4 px), identically in reference and
  composite; the spatial LPIPS map is averaged over E only; the imputed share of the context is stored.
- Betti matching, FID, FWD: only windows where E covers the whole window (no missing pixel); z clipped to [-3, 3]
  and mapped to [0, 1]. Betti matching, FID and LPIPS-VGG only on the topology subset S (2 wholly valid positions
  per eye, seeded), as the fundus panel.
- Iris-code Hamming distance: log-Gabor code of the whole strip (target vs composite), bits in the window rows and
  window columns +- 8, valid where the target mask holds (inside the window: E).
Rows, set-level values and QC sheets stay in the private folder; the public JSON gets aggregates only (full run).
Usage: python src/iris_ladder_metrics_extra.py smoke|run --preprocessing P2 --threads N [--sets ...] [--every 8]
Needs FUNDUS_METRICS_SRC, BRAIN_METRICS_SRC (as the panel), TORCH_HOME and PYFWD_DIR (the shared pytorchfwd-lib);
no defaults.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import iris_ladder_metrics as IM  # noqa: E402
import iris_ladder_metrics_frame as IF  # noqa: E402

H, W, WIN = IM.H, IM.W, IM.WIN
CTX = 64                               # LPIPS context width (columns); rows = whole strip (H = 64)
IMPUTE_SD = 4.0
BITS_PAD = 8                           # iris-code bits: window columns +- 8
TOPO_PER_UNIT = 2                      # topology subset S per eye (fundus rule, general protocol Amendment 9.15)
SET_RUNGS = ("repeat", "fellow", "lookalike", "lookalike16", "lookalike_oracle", "stranger", "mean_k2", "dataset_mean",
             "wrong_detail", "noise", "shrink_repeat", "shrink_fellow")
BETTI_DIR = os.path.join(os.environ.get("DATA_ROOT", r"C:\Projects\data"), "projects", "identity-ladder-metric",
                         "models-20261107", "betti-matching-2d")
N_BOOT_SET = 200
MIN_SET_WINDOWS = 30

_NETS = {}


def need_env(name):
    v = os.environ.get(name)
    if not v or not os.path.exists(v):
        raise SystemExit(f"{name} must point to an existing path (no default)")
    return v


# ------------------------------------------------------------------------------------------------ helpers

def to_unit(z):
    """Fixed intensity mapping: z clipped to [-3, 3] -> [0, 1]."""
    return (np.clip(z, -3.0, 3.0) + 3.0) / 6.0


def impute(x, m):
    """Target-only filler for network inputs: masked Gaussian local mean (wrapped in angle) outside the mask.
    Never scored; used identically in the reference and the composite."""
    m = np.asarray(m, bool)
    num = ndimage.gaussian_filter(np.where(m, x, 0.0), IMPUTE_SD, mode=("nearest", "wrap"))
    den = ndimage.gaussian_filter(m.astype(np.float64), IMPUTE_SD, mode=("nearest", "wrap"))
    loc = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    return np.where(m, x, loc)


def context_cols(x0):
    """Column indices of the CTX-wide context centred on a window starting at x0 (wrapped)."""
    c0 = x0 + WIN // 2 - CTX // 2
    return np.arange(c0, c0 + CTX) % W


def betti_parts(a, b):
    """Matched and unmatched parts of the Betti-matching loss (library `loss`, split), dims 0 and 1."""
    import BettiMatching as B
    bm = B.BettiMatching(a, b, relative=False, filtration="sublevel")

    def end(v):
        return 1.0 if v == np.inf else 0.0 if v == -np.inf else v
    matched = unmatched = 0.0
    for dim in (0, 1):
        for (i0, _ic, i1) in bm.matched[dim]:
            a0, b0 = bm.CP_0.fine_to_coarse(i0)
            a1, b1 = bm.CP_1.fine_to_coarse(i1)
            matched += 2 * ((a0 - a1) ** 2 + (end(b0) - end(b1)) ** 2)
        for cp, un in ((bm.CP_0, bm.unmatched_0[dim]), (bm.CP_1, bm.unmatched_1[dim])):
            for iv in un:
                p, q = cp.fine_to_coarse(iv)
                unmatched += (p - end(q)) ** 2
    return float(matched), float(unmatched)


def code_hd(code_t, valid_t, comp, mt, e, y, x):
    """Fractional Hamming distance between the target code and the composite's, on the window rows and columns
    +- BITS_PAD, over bits valid in both codes where the target mask holds (inside the window: E)."""
    import iris_codes as IC
    cb, cv = IC.log_gabor_code(comp)
    cols = np.arange(x - BITS_PAD, x + WIN + BITS_PAD) % W
    sel = np.zeros((H, W), bool)
    sel[y:y + WIN, cols] = True
    ok = sel & mt
    inwin = np.zeros((H, W), bool)
    inwin[y:y + WIN, x:x + WIN] = True
    ok &= ~inwin | e
    v = valid_t & cv & ok[..., None]
    if v.sum() < 20:
        return np.nan
    return float((code_t != cb)[v].mean())


def lpips_nets():
    if "alex" not in _NETS:
        need_env("TORCH_HOME")
        import lpips
        for k in ("alex", "vgg"):
            _NETS[k] = lpips.LPIPS(net=k, spatial=True, verbose=False).eval()
    return _NETS


def lpips_batch(refs, comps, es, net):
    """Mean spatial LPIPS over E for each pair; inputs CTX-wide contexts in z units, mapped to [-1, 1]."""
    import torch
    L = lpips_nets()[net]
    r = torch.tensor(np.clip(np.stack(refs), -3, 3) / 3.0, dtype=torch.float32)[:, None].repeat(1, 3, 1, 1)
    c = torch.tensor(np.clip(np.stack(comps), -3, 3) / 3.0, dtype=torch.float32)[:, None].repeat(1, 3, 1, 1)
    vals = []
    with torch.no_grad():
        for s in range(0, len(r), 64):
            vals.append(L(r[s:s + 64], c[s:s + 64]).numpy()[:, 0])
    d = np.concatenate(vals)
    return [float(d[i][es[i]].mean()) if es[i].any() else np.nan for i in range(len(d))]


# ------------------------------------------------------------------------------------------------ per target

def prepare(name, version):
    """The panel's state for one set, complete (build() fills lambda and sigma after its first S_ update)."""
    st = IM.build(name, version)
    IM.S_.update(st)
    return st


def topo_subset(ti, mt, valid, positions, seed=IM.SEED):
    """Topology subset S: TOPO_PER_UNIT positions drawn with [seed, unit, 11] among those whose window lies wholly
    inside the target mask and `valid`."""
    full = [pi for pi, (y, x) in enumerate(positions) if (mt & valid)[y:y + WIN, x:x + WIN].all()]
    if not full:
        return set()
    rng = np.random.default_rng([seed, int(ti), 11])
    return {int(p) for p in rng.choice(full, size=min(TOPO_PER_UNIT, len(full)), replace=False)}


def run_target(ti, st):
    """Per-window rows (LPIPS-AlexNet everywhere; Betti and LPIPS-VGG on S; Hamming everywhere) and window patches
    for the set-level metrics: (rung, patch, in_S, window in [0, 1]) where E covers the whole window."""
    S_ = IM.S_
    fx, pool, Xp, Mp, mu, mu_ok = IM.fixed_fills(ti)
    X, M, idx, valid = S_["X"], S_["M"], S_["idx"], S_["valid"]
    xt_raw, mt_raw = np.nan_to_num(X[idx[ti]]), M[idx[ti]]
    ref_full = impute(xt_raw, mt_raw)
    import iris_codes as IC
    code_t, valid_t = IC.log_gabor_code(ref_full)
    S = topo_subset(ti, mt_raw, valid, S_["pos"])
    rows, patches = [], []
    for pi, (y, x) in enumerate(S_["pos"]):
        if (mt_raw & valid)[y:y + WIN, x:x + WIN].mean() < IM.MIN_IN_MASK:
            continue
        fills, ws, win, xt, mtv = IM.window_fills(ti, pi, fx, pool, Xp, Mp, mu, mu_ok)
        n_tw = int((win & mtv).sum())
        cols = context_cols(x)
        in_s = pi in S
        refs, comps, es, meta = [], [], [], []
        if (win & mtv & mu_ok)[ws].all():
            patches.append(("target", pi, in_s, to_unit(xt[ws])))
        for fname, (f, mf) in fills.items():
            e = win & mtv & mf.astype(bool) & mu_ok
            if e.sum() < max(10, IM.MIN_FILL_SHARE * n_tw):
                continue
            f0 = np.nan_to_num(f)
            comp = np.where(e, f0, ref_full)
            refs.append(ref_full[:, cols])
            comps.append(comp[:, cols])
            es.append(e[:, cols])
            full = bool(e[ws].all())
            row = {"unit": int(ti), "person": S_["person"][ti], "side": S_["side"][ti], "patch": pi, "fill": fname,
                   "topo_subset": in_s, "imputed_share": float(1 - mt_raw[:, cols].mean()), "full_window": full,
                   "betti_matched": np.nan, "betti_unmatched": np.nan, "lpips_vgg": np.nan}
            if full and in_s:
                row["betti_matched"], row["betti_unmatched"] = betti_parts(to_unit(xt[ws]), to_unit(f0[ws]))
            if full and fname in SET_RUNGS:
                patches.append((fname, pi, in_s, to_unit(f0[ws])))
            row["iriscode_hd"] = code_hd(code_t, valid_t, comp, mt_raw, e, y, x)
            meta.append(row)
        if meta:
            la = lpips_batch(refs, comps, es, "alex")
            for i, row in enumerate(meta):
                row["lpips_alex"] = la[i]
            if in_s:
                lv = lpips_batch(refs, comps, es, "vgg")
                for i, row in enumerate(meta):
                    row["lpips_vgg"] = lv[i]
            rows += meta
    return rows, patches


# ------------------------------------------------------------------------------------------------ set level

def inception():
    if "inc" not in _NETS:
        need_env("TORCH_HOME")
        from pytorch_fid.inception import InceptionV3
        _NETS["inc"] = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True,
                                   normalize_input=True).eval()
    return _NETS["inc"]


def fid_features(P):
    import torch
    net = inception()
    out = []
    with torch.no_grad():
        for s in range(0, len(P), 128):
            t = torch.tensor(np.asarray(P[s:s + 128]), dtype=torch.float32)[:, None].repeat(1, 3, 1, 1)
            out.append(net(t)[0][:, :, 0, 0].numpy())
    return np.concatenate(out).astype(np.float64)


def fid_point(F1, F2):
    from pytorch_fid.fid_score import calculate_frechet_distance
    return float(calculate_frechet_distance(F1.mean(0), np.cov(F1, rowvar=False), F2.mean(0),
                                            np.cov(F2, rowvar=False)))


def fid_nuclear(F1, F2):
    """Exact FID with Tr sqrt(C1 C2) = nuclear norm of A1 A2^T (A = centred features / sqrt(n - 1)); n x n work.
    Used for the bootstrap (same formula as the fundus panel)."""
    m1, m2 = F1.mean(0), F2.mean(0)
    A1 = (F1 - m1) / np.sqrt(len(F1) - 1)
    A2 = (F2 - m2) / np.sqrt(len(F2) - 1)
    nuc = np.linalg.svd(A1 @ A2.T, compute_uv=False).sum()
    return float(((m1 - m2) ** 2).sum() + (A1 ** 2).sum() + (A2 ** 2).sum() - 2 * nuc)


def fwd_packets(P):
    """Haar level-2 packets of [0, 1] windows, one grey channel: (packets, n, features)."""
    p = need_env("PYFWD_DIR")
    if p not in sys.path:
        sys.path.insert(0, p)
    import torch
    from pytorchfwd.freq_math import forward_wavelet_packet_transform
    t = torch.from_numpy(np.asarray(P, np.float64))[:, None]
    pk = forward_wavelet_packet_transform(t, "Haar", 2, False)          # (n, P, C, h, w)
    n, npk = pk.shape[:2]
    return pk.permute(1, 0, 2, 3, 4).reshape(npk, n, -1).numpy()


def fwd_value(P1, P2):
    from pytorchfwd.freq_math import calculate_frechet_distance
    return float(np.mean([calculate_frechet_distance(P1[k].mean(0), np.cov(P1[k], rowvar=False),
                                                     P2[k].mean(0), np.cov(P2[k], rowvar=False))
                          for k in range(len(P1))]))


def set_level(patches, person_of, n_boot=N_BOOT_SET, seed=IM.SEED):
    """FID (subset S) and FWD (every full window) per rung on the same windows as the target set; ratio to the
    stranger rung; person bootstrap (resampled people with all their windows)."""
    df = pd.DataFrame([(r, u, pi, s) for (r, (u, pi), s, _a) in patches], columns=["rung", "unit", "patch", "in_s"])
    arr = np.stack([a for *_x, a in patches])
    key = (df.unit.astype(str) + ":" + df.patch.astype(str)).to_numpy()
    person = np.array([person_of[u] for u in df.unit])
    persons = np.array(sorted(set(person_of.values())))
    rng = np.random.default_rng(seed)
    boots = [rng.choice(persons, len(persons)) for _ in range(n_boot)]
    sel_s = df.in_s.to_numpy().astype(bool)
    feats = np.full((len(df), 2048), np.nan)
    if sel_s.any():
        feats[sel_s] = fid_features(arr[sel_s])
    pk = fwd_packets(arr)
    out = {"fid": {}, "fwd": {}}
    for metric, use in (("fid", sel_s), ("fwd", np.ones(len(df), bool))):
        tgt = (df.rung == "target").to_numpy() & use
        for rung in SET_RUNGS:
            rr = (df.rung == rung).to_numpy() & use
            common = np.intersect1d(key[rr], key[tgt])
            if len(common) < MIN_SET_WINDOWS:
                continue
            a = np.flatnonzero(tgt & np.isin(key, common))
            b = np.flatnonzero(rr & np.isin(key, common))
            a, b = a[np.argsort(key[a])], b[np.argsort(key[b])]
            if metric == "fid":
                # Point and bootstrap by the exact nuclear-norm form: sqrtm of the rank-deficient 2048 x 2048
                # covariance took about 4 min per rung on this machine (UPOL smoke, 8 Nov); equal at full rank.
                f = lambda i, j: fid_nuclear(feats[i], feats[j])  # noqa: E731
                val = f(a, b)
            else:
                f = lambda i, j: fwd_value(pk[:, i], pk[:, j])  # noqa: E731
                val = f(a, b)
            pa = person[a]
            bs = []
            for bp in boots:
                sel = np.concatenate([np.flatnonzero(pa == p) for p in bp])
                bs.append(f(a[sel], b[sel]) if len(sel) >= MIN_SET_WINDOWS else np.nan)
            out[metric][rung] = {"value": val, "windows": int(len(a)), "persons": int(len(set(pa))),
                                 "_boot": np.array(bs)}
        ref = out[metric].get("stranger")
        for rung, r in out[metric].items():
            bs = r.pop("_boot")
            r["ci95"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))] if np.isfinite(bs).any() \
                else None
            r["_b"] = bs
        if ref is not None:
            for rung, r in out[metric].items():
                r["over_stranger"] = r["value"] / ref["value"] if ref["value"] > 0 else None
                rat = r["_b"] / ref["_b"]
                r["over_stranger_ci95"] = ([float(np.nanpercentile(rat, 2.5)), float(np.nanpercentile(rat, 97.5))]
                                           if np.isfinite(rat).any() else None)
        for r in out[metric].values():
            r.pop("_b")
    return out


# ------------------------------------------------------------------------------------------------ QC sheet

def qc_sheet(name, st, rows, path, n_targets=4):
    """Evenly spaced targets, one window of S each: target context and composites per rung with LPIPS, Betti
    parts and Hamming distance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S_ = IM.S_
    r = pd.DataFrame(rows)
    show = ["repeat", "fellow", "lookalike16", "stranger", "dataset_mean", "wrong_detail", "noise", "shrink_repeat"]
    units = sorted(r.unit.unique())
    pick = [units[int(i)] for i in np.linspace(0, len(units) - 1, min(n_targets, len(units)))]
    fig, axes = plt.subplots(len(pick), len(show) + 1, figsize=(2.0 * (len(show) + 1), 2.7 * len(pick)), squeeze=False)
    for i, ti in enumerate(pick):
        rr = r[(r.unit == ti) & r.topo_subset.astype(bool)]
        if rr.empty:
            rr = r[r.unit == ti]
        pi = int(sorted(rr.patch.unique())[0])
        y, x = S_["pos"][pi]
        fx, pool, Xp, Mp, mu, mu_ok = IM.fixed_fills(ti)
        fills, ws, win, xt, mtv = IM.window_fills(ti, pi, fx, pool, Xp, Mp, mu, mu_ok)
        mt_raw = S_["M"][S_["idx"][ti]]
        ref = impute(np.nan_to_num(S_["X"][S_["idx"][ti]]), mt_raw)
        cols = context_cols(x)
        ax = axes[i, 0]
        ax.imshow(np.where(mt_raw[:, cols], ref[:, cols], np.nan), cmap="gray", vmin=-2.5, vmax=2.5)
        ax.add_patch(plt.Rectangle((CTX // 2 - WIN // 2 - 0.5, y - 0.5), WIN, WIN, fill=False, ec="#2a78d6", lw=1))
        ax.set_title(f"target eye {i + 1}, window {pi}\n(rows: pupil top, limbus bottom)", fontsize=6.5)
        ax.axis("off")
        for j, fname in enumerate(show):
            ax = axes[i, j + 1]
            ax.axis("off")
            if fname not in fills:
                ax.set_title(f"{fname}\nnot available", fontsize=7)
                continue
            f, mf = fills[fname]
            e = win & mtv & mf.astype(bool) & mu_ok
            comp = np.where(e, np.nan_to_num(f), ref)
            ax.imshow(np.where(mt_raw[:, cols], comp[:, cols], np.nan), cmap="gray", vmin=-2.5, vmax=2.5)
            ax.add_patch(plt.Rectangle((CTX // 2 - WIN // 2 - 0.5, y - 0.5), WIN, WIN, fill=False, ec="#eb6834",
                                       lw=1))
            q = r[(r.unit == ti) & (r.patch == pi) & (r.fill == fname)]
            if len(q):
                q = q.iloc[0]
                bt = (f"Betti m {q.betti_matched:.2f} u {q.betti_unmatched:.2f}" if np.isfinite(q.betti_matched)
                      else "Betti: not scored (window not full)")
                ax.set_title(f"{fname}\nLPIPS A {q.lpips_alex:.3f} V {q.lpips_vgg:.3f}\nHD {q.iriscode_hd:.2f}  {bt}",
                             fontsize=5.6)
            else:
                ax.set_title(f"{fname}\nnot scored (E too small)", fontsize=7)
    elig = r[r.topo_subset.astype(bool)].groupby("fill").full_window.mean()
    fig.suptitle(f"{name} P2, Amendment P3 smoke: target context (blue box = window) and composites (orange box = "
                 f"fill inside E). Topology subset, full-window share: repeat {elig.get('repeat', np.nan):.2f}, "
                 f"fellow {elig.get('fellow', np.nan):.2f}, stranger {elig.get('stranger', np.nan):.2f}. Pre-QC.",
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main

def parse_shard(text):
    """'i/n' -> (i, n) with 0 <= i < n."""
    i, n = (int(v) for v in text.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"shard must be i/n with 0 <= i < n, got {text!r}")
    return i, n


def shard_targets(n_targets, i, n):
    """Targets of shard i of n: every n-th target from i, so every target is in exactly one shard."""
    return np.arange(n_targets)[i::n]


def save_patches(path, patches):
    np.savez_compressed(path, rung=np.array([p[0] for p in patches]), unit=np.array([p[1][0] for p in patches]),
                        patch=np.array([p[1][1] for p in patches]), in_s=np.array([p[2] for p in patches]),
                        arr=np.stack([p[3] for p in patches]).astype(np.float32))


def load_patches(path):
    z = np.load(path, allow_pickle=False)
    return [(str(r), (int(u), int(pi)), bool(ins), a) for r, u, pi, ins, a in
            zip(z["rung"], z["unit"], z["patch"], z["in_s"], z["arr"])]


def finish(name, st, rows, patches, targets, tag, version, t0):
    """Write rows, set-level FID / FWD and the QC sheet for one set, then print the medians."""
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(IF.PRIVATE, f"rows_extra_{name}_{version}_{tag}.csv.gz"), index=False)
    person_of = {int(ti): st["person"][ti] for ti in targets}
    t1 = time.time()
    print(f"{name}: set level on {len(patches)} windows ...", flush=True)
    sl = set_level(patches, person_of)
    with open(os.path.join(IF.PRIVATE, f"setlevel_extra_{name}_{version}_{tag}.json"), "w") as fh:
        json.dump(sl, fh, indent=1)
    qc = os.path.join(IF.PRIVATE, f"qc_extra_{name}_{version}{'' if tag == 'smoke' else '_' + tag}.png")
    qc_sheet(name, st, rows, qc)
    print(f"{name}: done in {time.time() - t0:.0f}s (set level {time.time() - t1:.0f}s), rows {len(r)}, "
          f"topology subset rows {int(r.topo_subset.sum())}, imputed share median {r.imputed_share.median():.2f}; "
          f"qc {qc}", flush=True)
    cols = ["lpips_alex", "lpips_vgg", "betti_matched", "betti_unmatched", "iriscode_hd"]
    print(r.groupby("fill")[cols].median().loc[[f for f in ("repeat", "fellow", "lookalike16", "stranger",
                                                            "dataset_mean", "noise", "shrink_repeat")
                                                if f in set(r.fill)]].round(3).to_string(), flush=True)
    for m in ("fid", "fwd"):
        print(m, {k: round(v["over_stranger"], 2) for k, v in sl[m].items() if v.get("over_stranger") is not None},
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("smoke", "run", "merge"))
    ap.add_argument("--preprocessing", required=True, choices=IF.VERSIONS)
    ap.add_argument("--sets", default=",".join(IF.SETS))
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--threads", type=int, required=True, help="torch CPU threads (shared machine)")
    ap.add_argument("--shard", default="0/1", help="run: i/n, this process takes every n-th target from i")
    ap.add_argument("--shards", type=int, default=0, help="merge: number of shards written by run")
    a = ap.parse_args()
    if a.preprocessing in IF.REJECTED:
        raise SystemExit(f"{a.preprocessing} was rejected: {IF.REJECTED[a.preprocessing]}")
    need_env("TORCH_HOME")
    need_env("PYFWD_DIR")
    sys.path.insert(0, BETTI_DIR)
    import torch
    torch.set_num_threads(a.threads)
    IM.load_shared()
    si, sn = parse_shard(a.shard)
    for name in a.sets.split(","):
        t0 = time.time()
        st = prepare(name, a.preprocessing)
        n_all = len(st["idx"])
        if a.mode == "merge":
            if a.shards < 1:
                raise SystemExit("merge needs --shards n")
            rows, patches, seen = [], [], []
            for i in range(a.shards):
                base = os.path.join(IF.PRIVATE, f"extra_{name}_{a.preprocessing}_shard{i}of{a.shards}")
                rows += pd.read_csv(base + "_rows.csv.gz", dtype={"person": str}).to_dict("records")
                patches += load_patches(base + "_patches.npz")
                seen += list(shard_targets(n_all, i, a.shards))
            done = sorted({int(t) for t in pd.DataFrame(rows).unit.unique()})
            if sorted(seen) != list(range(n_all)) or done != list(range(n_all)):
                raise SystemExit(f"{name}: shards do not cover every target exactly once")
            finish(name, st, rows, patches, np.arange(n_all), "full", a.preprocessing, t0)
            continue
        if a.mode == "smoke":
            targets = np.sort(np.concatenate([st["side_units"][s][::a.every] for s in ("R", "L")]))
            tag = "smoke"
        else:
            targets = shard_targets(n_all, si, sn)
            tag = "full" if sn == 1 else f"shard{si}of{sn}"
        print(f"{name}: panel state ready in {time.time() - t0:.0f}s, {len(targets)} eyes ({tag})", flush=True)
        rows, patches = [], []
        for k, ti in enumerate(targets, 1):
            rr, pp = run_target(int(ti), st)
            rows += rr
            patches += [(rung, (int(ti), pi), ins, arr) for rung, pi, ins, arr in pp]
            el = time.time() - t0
            print(f"  {name} {tag}: {k}/{len(targets)} eyes, {el:.0f}s, ETA {el / k * (len(targets) - k):.0f}s",
                  flush=True)
        if a.mode == "run" and sn > 1:
            base = os.path.join(IF.PRIVATE, f"extra_{name}_{a.preprocessing}_{tag}")
            pd.DataFrame(rows).to_csv(base + "_rows.csv.gz", index=False)
            save_patches(base + "_patches.npz", patches)
            print(f"{name} {tag}: saved {len(rows)} rows, {len(patches)} windows in {time.time() - t0:.0f}s", flush=True)
            continue
        finish(name, st, rows, patches, targets, tag, a.preprocessing, t0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
