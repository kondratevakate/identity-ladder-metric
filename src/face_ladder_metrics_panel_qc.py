"""QC sheets of the Amendment F2 face panel smoke run (docs/PROTOCOL_face_metrics.md). PRIVATE output only.

Sheet 1 per source: the smoke targets (evenly spaced) x rungs; each cell is the fill mosaic on the scored pixels E
(blue = not scored), windows that LPIPS scored (every pixel in E) outlined in green, and under it the target's mean
over windows of the new metrics. Sheet 2: one window of one target, as the metrics see it: the LPIPS inputs (64 px),
the H0 sublevel persistence diagrams and the Betti-matching barcodes of target vs retest, dataset mean and noise.
The PNGs hold face images and stay in the private folder; never copy them into the repository.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import face_ladder_metrics as FM  # noqa: E402
import face_ladder_metrics_panel as P  # noqa: E402
import face_ladder_metrics_qc as Q  # noqa: E402

SHOW = [("ms_ssim", "msssim"), ("lpips", "lpips"), ("persistence", "pers"), ("betti_matched", "bm"),
        ("betti_unmatched", "bu"), ("id_plda", "plda"), ("id_norm", "|dn|"), ("plausibility", "nll")]


def sheet_grid(src, d, pos, lam, rows, sel, path):
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#c9d6e3")
    fig, axes = plt.subplots(len(sel), len(Q.RUNGS), figsize=(1.45 * len(Q.RUNGS), 2.05 * len(sel)))
    for r, ti in enumerate(sel):
        mo = Q.mosaics(src, ti, d["targets"][ti], pos, lam, tuple(d["pool"]), min(FM.POOL_SUB, d["n_pool"]),
                       d["shift_px"], FM.WIN, FM.RING)
        rt = rows[rows.target == ti]
        for c, (k, lab) in enumerate(Q.RUNGS):
            ax = axes[r, c]
            ax.imshow(mo[k], cmap=cmap, vmin=-2.5, vmax=2.5, interpolation="nearest")
            g = rt[rt.fill == k]
            for pk, (y0, x0) in enumerate(pos):
                scored = (g.patch == pk).any() and np.isfinite(g[g.patch == pk].lpips).any()
                ax.add_patch(plt.Rectangle((x0 - .5, y0 - .5), FM.WIN, FM.WIN, fill=False,
                                           lw=0.5 if scored else 0.2, ec="#2ca02c" if scored else "#e8a33d"))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(lab, fontsize=6)
            if len(g):
                txt = [f"{s} {np.nanmean(g[m]):.3g}" if np.isfinite(g[m]).any() else f"{s} -" for m, s in SHOW]
                ax.set_xlabel("\n".join([" ".join(txt[:2]), " ".join(txt[2:5]), " ".join(txt[5:])]), fontsize=4.2)
    fig.suptitle(f"PRIVATE QC {src} v6f, Amendment F2 smoke: fill on scored pixels (blue = not scored); green = "
                 "window scored by LPIPS (all pixels in E); means over the target's windows", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def sheet_window(src, d, pos, lam, path, B):
    """One target (first smoke target) at the window with the most scored rungs near the eyes."""
    T = d["targets"]
    ti = 0
    valid = np.asarray(FM.WORK[f"{src}_valid"])
    best = min(range(len(pos)), key=lambda k: abs(pos[k][0] - 48) + abs(pos[k][1] - 32))
    y0, x0 = pos[best]
    sl = (slice(y0, y0 + FM.WIN), slice(x0, x0 + FM.WIN))
    ring = FM.ring_mask(y0, x0) & valid
    Pw = FM.pool_window(src, {"pool": tuple(d["pool"])}, sl, ring)
    Pw["shift_px"] = d["shift_px"]
    xw, tv, fills = FM.window_fills(src, ti, T[ti], best, sl, ring, lam, Pw, min(FM.POOL_SUB, d["n_pool"]))
    show = [k for k in ("repeat", "stranger", "dataset_mean", "noise") if k in fills]
    fig, axes = plt.subplots(4, len(show) + 1, figsize=(2.3 * (len(show) + 1), 8.4))
    import torch
    for c, k in enumerate(["target"] + show):
        f, ok = (xw, tv) if k == "target" else fills[k][:2]
        e = tv & ok
        t = torch.from_numpy(np.clip(np.asarray(f, np.float32) / P.LPIPS_Z, -1, 1))[None, None]
        up = torch.nn.functional.interpolate(t, size=P.LPIPS_SIDE, mode="bilinear", align_corners=False)[0, 0]
        axes[0, c].imshow(np.where(e, f, np.nan), cmap="gray", vmin=-2.5, vmax=2.5, interpolation="nearest")
        axes[0, c].set_title(f"{k} on E ({int(e.sum())} px)", fontsize=7)
        axes[1, c].imshow(up.numpy() if e.all() else np.full((64, 64), np.nan), cmap="gray", vmin=-1, vmax=1)
        axes[1, c].set_title("LPIPS input 64 px" if e.all() else "LPIPS: not scored (E incomplete)", fontsize=7)
        dg = P.load_fundus(os.environ["FUNDUS_METRICS_SRC"]).h0_diagram(np.asarray(f, np.float64), e)
        axes[2, c].scatter(dg[:, 0], dg[:, 1], s=6)
        axes[2, c].plot([-4, 4], [-4, 4], lw=0.5, c="k")
        axes[2, c].set_title(f"H0 sublevel diagram ({len(dg)} bars)", fontsize=7)
        if k != "target":
            m, u, bm = P.betti_parts(B, xw, np.asarray(f, np.float32), e)
            mt, u0, u1 = bm.get_matching()
            for dim, col in ((0, "C0"), (1, "C3")):
                for (a0, a1) in mt[dim]:
                    axes[3, c].plot([a0[0], a1[0]], [min(a0[1], 1), min(a1[1], 1)], c=col, lw=0.6)
                for iv in u0[dim]:
                    axes[3, c].scatter(iv[0], min(iv[1], 1), c=col, marker="x", s=8)
                for iv in u1[dim]:
                    axes[3, c].scatter(iv[0], min(iv[1], 1), c=col, marker="+", s=8)
            axes[3, c].plot([0, 1], [0, 1], lw=0.5, c="k")
            axes[3, c].set_title(f"Betti matched {m:.3f} unmatched {u:.3f}", fontsize=7)
        else:
            axes[3, c].set_title("lines: matched pairs; x target, + fill\nunmatched (blue H0, red H1)", fontsize=6)
        for r in range(2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle(f"PRIVATE QC {src} v6f: one window (corner {y0},{x0}) of smoke target 0 as the metrics see it",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--limit", type=int, required=True, help="the smoke run's --limit")
    a = ap.parse_args()
    cache = os.path.join(FM.PRIVATE, f"cache{a.version}")
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    FM._init(cache)
    sys.path.insert(0, P.need_env("BETTI_SRC"))
    import BettiMatching as B
    lam = FM.lambdas(a.version)
    rows = pd.read_csv(os.path.join(P.PRIVATE, f"rows{a.version}_n{a.limit}.csv.gz"))
    for src, d in desc.items():
        pos = FM.positions(np.asarray(FM.WORK[f"{src}_valid"]))
        rs = rows[rows.src == src]
        sel = sorted(rs.target.unique())
        sheet_grid(src, d, pos, lam[src], rs, sel, os.path.join(P.PRIVATE, f"qc_panel_{src}{a.version}_private.png"))
        sheet_window(src, d, pos, lam[src], os.path.join(P.PRIVATE, f"qc_window_{src}{a.version}_private.png"), B)
        print(src, "sheets written", flush=True)


if __name__ == "__main__":
    main()
