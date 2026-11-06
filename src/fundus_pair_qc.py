"""Visual QC of the pairwise fellow-eye registration and intensity match (general protocol, Amendment 9.12).

4 persons with both eyes per dataset, evenly spaced (not chosen). Per person, top row: the two raw photos as taken,
the target eye on the 128 frame, the fellow eye in the frame only (disc alignment), the fellow eye registered and
intensity-matched, and a stranger registered and matched the same way (control). Bottom row: the leave-person-out
mean and the smoothed squared error of each donor against the target; titles give each donor's error over the joint
valid pixels relative to the mean's error on the same pixels. Real images: private folder only. Pre-QC.
Usage: LADDER_EXTERNAL=<checkout>/external python src/fundus_pair_qc.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import donor_ladder_2d as D  # noqa: E402
import donor_ladder_2d_facefmt as FF  # noqa: E402
import donor_ladder_fundus as F  # noqa: E402
import fundus_register as R  # noqa: E402

N_PERSONS = 4


def raw_small(path, size=220):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    return np.asarray(im)


def ratio(t, tm, f, fm, mean, mm):
    jm = tm & fm & mm
    return float((((t - f) ** 2) * jm).sum() / max((((t - mean) ** 2) * jm).sum(), 1e-9)) if jm.sum() > 50 else np.nan


def err_map(t, f, m):
    return np.where(m, ndimage.gaussian_filter(((t - f) ** 2) * m, 2) / np.maximum(ndimage.gaussian_filter(m * 1.0, 2), 1e-6), np.nan)


def main():
    from multiprocessing import Pool
    ext = os.environ.get("LADDER_EXTERNAL")
    if ext:
        F.S.GRAPE_META = os.path.join(ext, *os.path.relpath(F.S.GRAPE_META, os.path.join(
            os.path.dirname(F.S.__file__), "..", "external")).split(os.sep))
    for name, arm in (("PAPILA", "p9"), ("GRAPE", "p8")):
        t, img, fov, u, _, _, fellow, _ = D.setup(name, arm)
        X, M = FF.down2(img, fov)
        idx, person, side = u.idx.to_numpy(), u.person.to_numpy(), u.side.to_numpy()
        targets = np.flatnonzero((side == "R") & (fellow >= 0))
        targets = targets[np.linspace(0, len(targets) - 1, N_PERSONS).astype(int)]
        cases, jobs = [], []
        for i in targets:
            j = fellow[i]
            pool = np.flatnonzero((side == "R") & (person != person[i]))
            q = np.random.default_rng([D.SEED, int(i), 7]).choice(pool)
            others = (side == "R") & (person != person[i])
            mean, n = FF.masked_mean(X[idx[others]], M[idx[others]])
            cases.append((i, j, q, mean, n >= 0.9 * others.sum()))
            yy, xx = np.mgrid[:FF.SIZE, :FF.SIZE]
            disc = np.hypot(xx - FF.DISC_XY[name][0], yy - FF.DISC_XY[name][1]) <= 12   # disc zone (9.12)
            jobs += [(X[idx[i]], M[idx[i]], X[idx[j]], M[idx[j]], disc),
                     (X[idx[i]], M[idx[i]], X[idx[q]], M[idx[q]], disc)]
        with Pool(8) as p:
            out = p.map(R.register_pair, jobs)
        fig, ax = plt.subplots(2 * N_PERSONS, 6, figsize=(16, 5.6 * N_PERSONS))
        for c, (i, j, q, mean, mm) in enumerate(cases):
            fr, sr = out[2 * c], out[2 * c + 1]
            fr, sr = (fr if fr[0] is not None else None), (sr if sr[0] is not None else None)
            xt, mt = X[idx[i]], M[idx[i]]
            top, bot = ax[2 * c], ax[2 * c + 1]
            top[0].imshow(raw_small(t.path.iloc[idx[i]]))
            top[0].set_title(f"case {c + 1}: right eye, raw", fontsize=9)
            top[1].imshow(raw_small(t.path.iloc[idx[j]]))
            top[1].set_title("left eye (fellow), raw", fontsize=9)
            show = lambda a, x, m: a.imshow(np.where(m, x, np.nan), cmap="gray", vmin=-2.5, vmax=2.5)  # noqa: E731
            show(top[2], xt, mt)
            top[2].set_title("target: right eye, frame", fontsize=9)
            donors = [("fellow, frame only", X[idx[j]], M[idx[j]], None),
                      ("fellow, rigid (vessels) + matched", *(fr[:2] if fr else (None, None)), fr),
                      ("stranger, rigid (vessels) + matched", *(sr[:2] if sr else (None, None)), sr)]
            show(bot[2], mean, mm)
            bot[2].set_title("leave-person-out mean", fontsize=9)
            for k, (lab, f, fm, o) in enumerate(donors, start=3):
                if f is None:
                    top[k].text(0.5, 0.5, "registration rejected" + "\n(counted as missing)", ha="center", transform=top[k].transAxes, color="red")
                    continue
                show(top[k], f, fm)
                extra = f"\nscale {o[4]:.2f}, gain {o[3][0]:.2f}, offset {o[3][1]:+.2f}" if o is not None else ""
                top[k].set_title(lab + extra, fontsize=8.5)
                jm = mt & fm & mm
                bot[k].imshow(err_map(xt, f, jm), cmap="magma", vmin=0, vmax=3)
                bot[k].set_title(f"error vs target: {ratio(xt, mt, f, fm, mean, mm):.2f} x mean", fontsize=9)
            for a in (bot[0], bot[1]):
                a.axis("off")
        for a in ax.ravel():
            a.set(xticks=[], yticks=[])
        fig.suptitle(f"{name}: fellow eye laid over the target - frame only vs rigid by vessels (disc excluded) + intensity-matched "
                     "(stranger = control). Pre-QC, private.", fontsize=12)
        fig.tight_layout()
        outp = os.path.join(D.PRIVATE, f"qc_pair_{name}.png")
        fig.savefig(outp, dpi=90)
        plt.close(fig)
        print(outp, "rejected", [o[1] for o in out if o[0] is None], "of", len(out), flush=True)


if __name__ == "__main__":
    main()
