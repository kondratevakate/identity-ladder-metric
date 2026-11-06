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

KINDS = {"R1": ("Affine", {}), "R2": ("SyN", {}),
         "R2b": ("SyN", {"flow_sigma": 6, "total_sigma": 1, "reg_iterations": (20, 10, 0)})}   # softer SyN


def valid_mean(X, M):
    n = M.sum(0)
    return np.where(n > 0, (X * M).sum(0) / np.maximum(n, 1), 0.0).astype(np.float32), n


def register_to(template, tmask, img, mask, kind):
    """Warp (img, mask) onto the template; returns (warped image, warped validity, bending), bending = mean |log det
    Jacobian| of the deformable warp inside the warped field (0 for affine arms)."""
    import ants
    f = ants.from_numpy(template.astype(np.float32))
    m = ants.from_numpy(img.astype(np.float32))
    fm = ants.from_numpy(tmask.astype(np.float32))
    mm = ants.from_numpy(mask.astype(np.float32))
    # identity start: the default centre-of-mass start divides by ~0 on z-scored images (translation ~4e10 px)
    tf, extra = KINDS[kind]
    reg = ants.registration(f, m, type_of_transform=tf, initial_transform="identity", aff_metric="mattes",
                            mask=fm, moving_mask=mm, mask_all_stages=True, **extra)
    w = ants.apply_transforms(f, m, transformlist=reg["fwdtransforms"], interpolator="linear").numpy()
    wm = ants.apply_transforms(f, mm, transformlist=reg["fwdtransforms"], interpolator="linear").numpy() > 0.99
    bend = 0.0
    if tf != "Affine" and wm.any():
        lj = ants.create_jacobian_determinant_image(f, reg["fwdtransforms"][0], do_log=True).numpy()
        bend = float(np.mean(np.abs(lj[wm])))
    return np.where(wm, w, 0.0).astype(np.float32), wm, bend


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
        bends = np.array([np.nan if o is None else o[2] for o in out])
        W = np.stack([np.zeros_like(X[i]) if o is None else o[0] for i, o in enumerate(out)])
        WM = np.stack([np.zeros_like(M[i]) if o is None else o[1] for i, o in enumerate(out)])
        T, n = valid_mean(W, WM)
        tmask = n >= min_share * (len(X) - len(failed))
        log(f"{kind} iteration {k + 1}/{iters}: failed {len(failed)} of {len(X)}; template valid px {int(tmask.sum())}")
    return T, tmask, W, WM, failed, bends


def highpass(x, m, s=6):
    from scipy import ndimage
    num = ndimage.gaussian_filter(x * m, s)
    den = ndimage.gaussian_filter(m.astype(float), s)
    return np.where(m, x - num / np.maximum(den, 1e-6), 0)


def vessel_r(W, WM, edge=6):
    """Per image: correlation of its high-pass (vessel) pattern with the leave-one-out high-pass template, over its
    field minus an edge band, where >= half of the other images are valid."""
    from scipy import ndimage
    H = np.stack([highpass(w, m) for w, m in zip(W, WM)])
    S, N = (H * WM).sum(0), WM.sum(0)
    out = []
    for i in range(len(W)):
        n = N - WM[i]
        T = np.where(n > 0, (S - H[i] * WM[i]) / np.maximum(n, 1), 0)
        m = WM[i] & (n >= 0.5 * (len(W) - 1)) & (ndimage.distance_transform_edt(WM[i]) > edge)
        out.append(float(np.corrcoef(H[i][m], T[m])[0, 1]) if m.sum() > 50 else np.nan)
    return np.array(out)


def prototype(kinds=("R2", "R2b"), n_img=40):
    """Run the arms on n_img evenly spaced images per dataset; QC sheet (R0 + arms) with the template's vessels and
    per-image r and bending; summary printed. Private outputs only."""
    import donor_ladder_2d as D
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage
    for name, arm in (("PAPILA", "p9"), ("GRAPE", "p8")):
        z = np.load(os.path.join(D.PRIVATE, f"{name}_canonical_{arm}.npz"))
        ok = np.flatnonzero(z["ok"])
        pick = ok[np.linspace(0, len(ok) - 1, n_img).astype(int)]
        X, M = z["img"][pick], z["fov"][pick]
        T0, n0 = valid_mean(X, M)
        res = {"R0": (T0, n0 >= 0.5 * len(X), X, M, [], np.zeros(len(X)))}
        for kind in kinds:
            res[kind] = groupwise(X, M, kind, log=lambda s, nm=name: print(nm, s, flush=True))
        rs = {k: vessel_r(v[2], v[3]) for k, v in res.items()}
        for k in res:
            b = res[k][5]
            print(f"{name} {k}: vessel r median {np.nanmedian(rs[k]):.3f} (p10 {np.nanpercentile(rs[k], 10):.3f}); "
                  f"better than R0 in {np.mean(rs[k] > rs['R0']) * 100:.0f}%; bending median {np.nanmedian(b):.3f} "
                  f"(p90 {np.nanpercentile(b, 90):.3f}); failed {len(res[k][4])}", flush=True)
        np.savez_compressed(os.path.join(D.PRIVATE, f"register_proto_{name}_{'_'.join(kinds)}.npz"), pick=pick,
                            **{f"{k}_{a}": v for k, (T, tm, W, WM, _, b) in res.items()
                               for a, v in (("T", T), ("tmask", tm), ("W", W), ("WM", WM), ("bend", b), ("r", rs[k]))})
        show = np.linspace(0, n_img - 1, 8).astype(int)
        rows = list(res)
        fig, ax = plt.subplots(len(rows), 9, figsize=(22, 2.55 * len(rows)))
        for r, k in enumerate(rows):
            T, tm, W, WM, _, b = res[k]
            Th = highpass(T, tm)
            inner = tm & (ndimage.distance_transform_edt(tm) > 6)
            vess = inner & (Th < np.percentile(Th[inner], 8))
            ax[r, 0].imshow(np.where(tm, T, np.nan), cmap="gray", vmin=-2, vmax=3)
            ax[r, 0].contour(vess, levels=[0.5], colors="#ffcc00", linewidths=0.5)
            ax[r, 0].set_ylabel(k, fontsize=12)
            for c, i in enumerate(show, start=1):
                ax[r, c].imshow(np.where(WM[i], W[i], np.nan), cmap="gray", vmin=-2.5, vmax=2.5)
                ax[r, c].contour(vess, levels=[0.5], colors="#ffcc00", linewidths=0.5)
                ax[r, c].set_title(f"img {i + 1}: r {rs[k][i]:.2f}, bend {b[i]:.3f}", fontsize=8.5)
        for a in ax.ravel():
            a.set(xticks=[], yticks=[])
        fig.suptitle(f"{name}: R0 (disc frame) vs {', '.join(kinds)}; yellow = template vessels; r = vessel pattern vs "
                     "leave-one-out template; bend = mean |log det J| of the SyN warp. Pre-QC, private.", fontsize=11)
        fig.tight_layout()
        out = os.path.join(D.PRIVATE, f"qc_register_{name}_{'_'.join(kinds)}.png")
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(out, flush=True)


if __name__ == "__main__":  # the guard is required: groupwise() starts worker processes
    if "prototype" in sys.argv[1:]:
        prototype(tuple(a for a in sys.argv[2:] if a in KINDS) or ("R2", "R2b"))
