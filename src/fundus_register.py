"""Groupwise fundus registration to a dataset template (general protocol, Amendment 9.8). Pre-QC.

Arms: R1 = affine to the template, R2 = affine + SyN (ANTsPy `SyN`). Masks of the template and of each image are
used in every stage, so the filler outside a field never acts as an edge; warped masks are valid only where the
linearly warped mask is ~1 (missing is never zero). Template_0 = valid-pixel mean of the input frame, template_k =
valid-pixel mean after iteration k.
Usage: python src/fundus_register.py prototype   (40 evenly spaced images per dataset; QC sheets in the private folder)
"""
import os
import sys

os.environ.setdefault("ANTS_RANDOM_SEED", "1")                  # reproducible metric sampling (tested: identical
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")  # repeats); parallelism is across processes
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KINDS = {"R1": "Affine", "R2": "SyN"}


def valid_mean(X, M):
    n = M.sum(0)
    return np.where(n > 0, (X * M).sum(0) / np.maximum(n, 1), 0.0).astype(np.float32), n


def register_to(template, tmask, img, mask, kind):
    """Warp (img, mask) onto the template; returns (warped image, warped validity)."""
    import ants
    f = ants.from_numpy(template.astype(np.float32))
    m = ants.from_numpy(img.astype(np.float32))
    fm = ants.from_numpy(tmask.astype(np.float32))
    mm = ants.from_numpy(mask.astype(np.float32))
    # identity start: the default centre-of-mass start divides by ~0 on z-scored images (translation ~4e10 px)
    reg = ants.registration(f, m, type_of_transform=KINDS[kind], initial_transform="identity", aff_metric="mattes",
                            mask=fm, moving_mask=mm, mask_all_stages=True)
    w = ants.apply_transforms(f, m, transformlist=reg["fwdtransforms"], interpolator="linear").numpy()
    wm = ants.apply_transforms(f, mm, transformlist=reg["fwdtransforms"], interpolator="linear").numpy() > 0.99
    return np.where(wm, w, 0.0).astype(np.float32), wm


def _one(args):
    T, tmask, x, m, kind = args
    try:
        return register_to(T, tmask, x, m, kind)
    except RuntimeError:
        return None                                   # counted and reported by the caller, never silent


def groupwise(X, M, kind, iters=3, min_share=0.5, log=print, workers=8):
    """Iterative template: register all images to the current template, re-average over the images that registered.
    Returns template, its mask, warped images and masks of the last iteration, and the failed indices (warped mask
    empty for them, so they are missing everywhere)."""
    from multiprocessing import Pool
    T, n = valid_mean(X, M)
    tmask = n >= min_share * len(X)
    W, WM, failed = X, M, []
    for k in range(iters):
        with Pool(workers) as pool:
            out = pool.map(_one, [(T, tmask, x, m, kind) for x, m in zip(X, M)])
        failed = [i for i, o in enumerate(out) if o is None]
        W = np.stack([np.zeros_like(X[i]) if o is None else o[0] for i, o in enumerate(out)])
        WM = np.stack([np.zeros_like(M[i]) if o is None else o[1] for i, o in enumerate(out)])
        T, n = valid_mean(W, WM)
        tmask = n >= min_share * (len(X) - len(failed))
        log(f"{kind} iteration {k + 1}/{iters}: failed {len(failed)} of {len(X)}; template valid px {int(tmask.sum())}")
    return T, tmask, W, WM, failed


def prototype(n_img=40):
    import donor_ladder_2d as D
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, arm in (("PAPILA", "p9"), ("GRAPE", "p8")):
        z = np.load(os.path.join(D.PRIVATE, f"{name}_canonical_{arm}.npz"))
        ok = np.flatnonzero(z["ok"])
        pick = ok[np.linspace(0, len(ok) - 1, n_img).astype(int)]
        X, M = z["img"][pick], z["fov"][pick]
        res = {"R0": (valid_mean(X, M)[0], valid_mean(X, M)[1] >= 0.5 * len(X), X, M, [])}
        for kind in ("R1", "R2"):
            res[kind] = groupwise(X, M, kind, log=lambda s, nm=name: print(nm, s, flush=True))
        np.savez_compressed(os.path.join(D.PRIVATE, f"register_proto_{name}.npz"), pick=pick,
                            **{f"{k}_{a}": v for k, (T, tm, W, WM, _) in res.items()
                               for a, v in (("T", T), ("tmask", tm), ("W", W), ("WM", WM))})
        # QC sheet: 8 evenly spaced images x (R0, R1, R2) with the template's dark-vessel outline over each
        show = np.linspace(0, n_img - 1, 8).astype(int)
        fig, ax = plt.subplots(3, 9, figsize=(22, 7.6))
        for r, kind in enumerate(("R0", "R1", "R2")):
            T, tm, W, WM, failed = res[kind]
            vess = tm & (T < np.percentile(T[tm], 12))
            ax[r, 0].imshow(np.where(tm, T, np.nan), cmap="gray", vmin=-2, vmax=3)
            ax[r, 0].set_ylabel({"R0": "R0 current frame", "R1": "R1 affine", "R2": "R2 affine + SyN"}[kind], fontsize=11)
            ax[r, 0].set_title("template" if r == 0 else "", fontsize=10)
            for c, i in enumerate(show, start=1):
                ax[r, c].imshow(np.where(WM[i], W[i], np.nan), cmap="gray", vmin=-2.5, vmax=2.5)
                ax[r, c].contour(vess, levels=[0.5], colors="#ffcc00", linewidths=0.5)
                if r == 0:
                    ax[r, c].set_title(f"image {i + 1}/{n_img}", fontsize=10)
        for a in ax.ravel():
            a.set(xticks=[], yticks=[])
        fig.suptitle(f"{name}: registration to a groupwise template (yellow = template's darkest 12% = main vessels). "
                     "Pre-QC, private.", fontsize=12)
        fig.tight_layout()
        out = os.path.join(D.PRIVATE, f"qc_register_{name}.png")
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(out, flush=True)


if __name__ == "__main__":  # the guard is required: groupwise() starts worker processes
    if "prototype" in sys.argv[1:]:
        prototype()
