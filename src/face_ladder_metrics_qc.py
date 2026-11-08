"""QC sheet of the face metric fills (docs/PROTOCOL_face_metrics.md, gate before the full run). PRIVATE output only.

For evenly spaced targets per source, every rung is drawn as it is scored: each window position shows the fill on
the scored pixels E (target mask AND fill validity AND coverage), grey elsewhere. Window-wise rungs (look-alike,
oracle) are assembled from their per-window donors. Reads the cache written by face_ladder_metrics.py. The PNG holds
face images and stays in the private folder; never copy it into the repository.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import face_ladder_metrics as FM  # noqa: E402

RUNGS = [("target", "target"), ("repeat", "retest photo"), ("aug_shift", "same photo, shifted"),
         ("aug_light", "retest light"), ("mirror", "own mirror"), ("sibling", "sibling"),
         ("lookalike", "look-alike k=1"), ("lookalike_oracle", "oracle k=1"), ("stranger", "stranger"),
         ("mean_k2", "mean of 2"), ("cohort_mean", "sex-age mean"), ("dataset_mean", "dataset mean"),
         ("wrong_detail", "wrong detail"), ("noise", "white noise"), ("shrink_repeat", "repeat blend")]


def mosaics(src, ti, d, pos, lam, pool, n_sub, shift_px, win, rw):
    """Fill mosaics of one target; window side win and ring rw explicit (Amendment F3)."""
    valid = np.asarray(FM.WORK[f"{src}_valid"])
    out = {k: np.full((FM.SIZE, FM.SIZE), np.nan, np.float32) for k, _ in RUNGS}
    for pk, (y0, x0) in enumerate(pos):
        sl = (slice(y0, y0 + win), slice(x0, x0 + win))
        ring = FM.ring_mask(y0, x0, FM.SIZE, win, rw) & valid
        if (np.asarray(FM.WORK["M"][d["t"]])[sl] & valid[sl]).mean() < FM.MIN_IN_MASK:
            continue
        P = FM.pool_window(src, {"pool": pool}, sl, ring)
        P["shift_px"] = shift_px
        xw, tv, fills = FM.window_fills(src, ti, d, pk, sl, ring, lam, P, n_sub)
        out["target"][sl] = np.where(tv, xw, np.nan)
        for k, _ in RUNGS[1:]:
            if k in fills:
                f, ok, _ = fills[k]
                out[k][sl] = np.where(tv & ok, f, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--n", type=int, default=6, help="evenly spaced targets per source")
    a = ap.parse_args()
    cache = os.path.join(FM.PRIVATE, f"cache{a.version}")
    desc = json.load(open(os.path.join(cache, "desc.json")))["data"]
    FM._init(cache)
    lam = FM.lambdas(a.version)
    for src, d in desc.items():
        valid = np.asarray(FM.WORK[f"{src}_valid"])
        pos = FM.positions(valid)
        T = d["targets"]
        if src == "imdb":       # show targets with a sibling where possible
            cand = [i for i, t in enumerate(T) if t["sib"] is not None]
        else:
            cand = list(range(len(T)))
        sel = [cand[int(round(k))] for k in np.linspace(0, len(cand) - 1, a.n)]
        fig, axes = plt.subplots(len(sel), len(RUNGS), figsize=(1.25 * len(RUNGS), 1.35 * len(sel)))
        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad("#c9d6e3")
        for r, ti in enumerate(sel):
            mo = mosaics(src, ti, T[ti], pos, lam[src], tuple(d["pool"]), min(FM.POOL_SUB, d["n_pool"]), d["shift_px"],
                          FM.WIN, FM.RING)
            for c, (k, lab) in enumerate(RUNGS):
                ax = axes[r, c]
                ax.imshow(mo[k], cmap=cmap, vmin=-2.5, vmax=2.5, interpolation="nearest")
                for (y0, x0) in pos:
                    ax.add_patch(plt.Rectangle((x0 - .5, y0 - .5), FM.WIN, FM.WIN, fill=False, lw=0.2, ec="#e8a33d"))
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(lab, fontsize=6)
        fig.suptitle(f"PRIVATE QC {src} {a.version}: fills on scored pixels per 16 px window (blue = not scored); "
                     f"{len(pos)} windows; targets evenly spaced", fontsize=7)
        fig.tight_layout()
        out = os.path.join(FM.PRIVATE, f"qc_fills_{src}{a.version}_private.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
