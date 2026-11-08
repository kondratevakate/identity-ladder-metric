"""Visual QC of the FastSurfer boxes (Amendment 15; Kate 6 Nov "I want to see everything"). Private figures only.

  rungs   <tid>  : per box, one row per volume (oracle, core rungs, blur): three orthogonal slices through the box
                   centre of the input FastSurfer saw (clipped), its labels, and the oracle labels; title = dice_mean,
                   accuracy and the three worst labels inside the scored voxels.
  oracles <tids> : the oracle input with its labels for many targets, deep box centre, to check conform and
                   segmentation across targets (evenly spaced, never chosen by quality).
  parcels <tid>  : the DKT/aseg labels that overlap each of the 34 positions (2 boxes + 32 random cubes) on that
                   target's oracle labels, with their voxel share; plus pooled ladder R (mse) per view.
Figures and tables go to PRIVATE/qc/ (QTIM volumes; not for git).
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_seg as S  # noqa: E402
import donor_ladder_brain_seg_gcp as G  # noqa: E402

QC = os.path.join(S.PRIVATE, "qc")
MARGIN = 12
LUT_PATH = os.path.join(S.PRIVATE, "FastSurfer_ColorLUT.tsv")


def lut():
    """{label id: (name, rgb 0..1)} from the LUT shipped in the pinned FastSurfer image (copied once)."""
    out = {}
    if not os.path.exists(LUT_PATH):
        return out
    for line in open(LUT_PATH, encoding="utf-8"):
        p = line.split()
        if len(p) >= 5 and p[0].isdigit():
            out[int(p[0])] = (p[1], tuple(int(c) / 255 for c in p[2:5]))
    return out


def label_cmap(L):
    colors = np.zeros((3000, 4))
    rng = np.random.default_rng(0)
    colors[1:, :3] = rng.uniform(0.2, 1, (2999, 3))
    colors[1:, 3] = 1
    for k, (_, rgb) in L.items():
        if k < 3000:
            colors[k, :3] = rgb
    return ListedColormap(colors)


def three_slices(vol, sl, margin=MARGIN):
    """Axial, coronal, sagittal slices through the box centre, cropped to the box plus a margin."""
    c = [(s.start + s.stop) // 2 for s in sl]
    lo = [max(0, s.start - margin) for s in sl]
    hi = [min(n, s.stop + margin) for s, n in zip(sl, vol.shape)]
    ax = vol[lo[0]:hi[0], lo[1]:hi[1], c[2]].T
    co = vol[lo[0]:hi[0], c[1], lo[2]:hi[2]].T
    sa = vol[c[0], lo[1]:hi[1], lo[2]:hi[2]].T
    rects = [(sl[0].start - lo[0], sl[1].start - lo[1], sl[0].stop - sl[0].start, sl[1].stop - sl[1].start),
             (sl[0].start - lo[0], sl[2].start - lo[2], sl[0].stop - sl[0].start, sl[2].stop - sl[2].start),
             (sl[1].start - lo[1], sl[2].start - lo[2], sl[1].stop - sl[1].start, sl[2].stop - sl[2].start)]
    return [ax, co, sa], rects


def worst_labels(pred, oracle, s, L, k=3):
    p, o = pred[s], oracle[s]
    labels = [int(v) for v, c in zip(*np.unique(o, return_counts=True)) if c >= S.MIN_LABEL_VOX and v != 0]
    d = sorted((2 * np.sum((p == v) & (o == v)) / (np.sum(p == v) + np.sum(o == v)), v) for v in labels)
    return ", ".join(f"{L.get(v, (str(v),))[0][:18]} {x:.2f}" for x, v in d[:k])


def fig_rungs(tid, pos):
    base = os.path.join(S.PRIVATE, tid)
    info = json.load(open(os.path.join(base, "fills.json")))
    L = lut()
    cmap = label_cmap(L)
    oracle_lab = np.load(os.path.join(base, "label_fastsurfer_oracle.npy"))
    names = ["oracle"] + [n for n in S.CORE if n in info["volumes"]] + [f"blur_{s}" for s in S.BLUR_SIGMAS]
    for box, pi in S.box_positions(pos).items():
        sl = S.box_slices(pos, pi)
        rows = [n for n in names if n == "oracle" or n.startswith("blur") or n in info["boxes"][box]["fills"]]
        fig, axes = plt.subplots(len(rows), 9, figsize=(16, 2.0 * len(rows)))
        o_sl, rects = three_slices(oracle_lab, sl)
        for r, n in enumerate(rows):
            img = np.asarray(nib.load(os.path.join(G.STAGE_DIR, tid, f"{n}.nii.gz")).dataobj, dtype=np.float32)
            lp = os.path.join(base, f"label_fastsurfer_{n}.npy")
            lab = np.load(lp) if os.path.exists(lp) else None
            i_sl, _ = three_slices(img, sl)
            vmax = np.percentile(img[img > 0], 99.5)
            title = n
            if lab is not None and n != "oracle":
                vfile = f"valid_{box}_{n}.npy" if n in S.CORE else f"valid_{box}_blur.npy"
                s = np.load(os.path.join(base, vfile))
                sc = S.label_scores(lab[sl], oracle_lab[sl], s)
                title = (f"{n}: dice {sc['dice_mean']:.3f}, acc {sc['accuracy']:.3f} | worst: "
                         f"{worst_labels(lab[sl], oracle_lab[sl], s, L)}")
            elif lab is None:
                title = f"{n}: NO LABEL (failed or not run)"
            l_sl = three_slices(lab, sl)[0] if lab is not None else [None] * 3
            for v in range(3):
                for c, (im, kw) in enumerate(((i_sl[v], dict(cmap="gray", vmin=0, vmax=vmax)),
                                              (l_sl[v], dict(cmap=cmap, vmin=0, vmax=2999, interpolation="nearest")),
                                              (o_sl[v], dict(cmap=cmap, vmin=0, vmax=2999, interpolation="nearest")))):
                    a = axes[r, 3 * v + c]
                    a.axis("off")
                    if im is None:
                        continue
                    a.imshow(im, origin="lower", **kw)
                    x, y, w, h = rects[v]
                    a.add_patch(plt.Rectangle((x - .5, y - .5), w, h, fill=False, ec="yellow", lw=.8))
            axes[r, 0].set_title(title, fontsize=7, loc="left")
        for v, name in enumerate(("axial", "coronal", "sagittal")):
            for c, what in enumerate(("input (clipped)", "labels", "oracle labels")):
                axes[0, 3 * v + c].text(0.5, 1.25, f"{name}: {what}", transform=axes[0, 3 * v + c].transAxes,
                                        ha="center", fontsize=7)
        fig.suptitle(f"{tid} {box} box (side {info['boxes'][box]['side']}), FastSurfer seg_only, pre-QC; "
                     f"yellow = box", fontsize=9, y=0.995)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.005, wspace=0.03, hspace=0.28)
        out = os.path.join(QC, f"{tid}_{box}_rungs.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(out)


def fig_oracles(tids, pos):
    L = lut()
    cmap = label_cmap(L)
    sl = S.box_slices(pos, S.box_positions(pos)["deep"])
    big = tuple(slice(max(0, s.start - 40), s.stop + 40) for s in sl)
    fig, axes = plt.subplots(len(tids), 4, figsize=(10, 2.4 * len(tids)))
    axes = np.atleast_2d(axes)
    for r, tid in enumerate(tids):
        img = np.asarray(nib.load(os.path.join(G.STAGE_DIR, tid, "oracle.nii.gz")).dataobj, dtype=np.float32)
        lp = os.path.join(S.PRIVATE, tid, "label_fastsurfer_oracle.npy")
        lab = np.load(lp) if os.path.exists(lp) else None
        (ax_i, co_i, _), _ = three_slices(img, big, 0)
        vmax = np.percentile(img[img > 0], 99.5)
        for c, im in enumerate((ax_i, co_i)):
            axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        if lab is not None:
            (ax_l, co_l, _), _ = three_slices(lab, big, 0)
            for c, im in enumerate((ax_l, co_l)):
                axes[r, 2 + c].imshow(im, cmap=cmap, vmin=0, vmax=2999, interpolation="nearest", origin="lower")
        nlab = len(np.unique(lab)) if lab is not None else 0
        axes[r, 0].set_title(f"{tid} oracle: {nlab} labels" if lab is not None else f"{tid}: NO LABEL", fontsize=8,
                             loc="left")
        for a in axes[r]:
            a.axis("off")
    fig.tight_layout()
    out = os.path.join(QC, "oracles_deep.png")
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print(out)


def table_parcels(tid, pos):
    L = lut()
    lab = np.load(os.path.join(S.PRIVATE, tid, "label_fastsurfer_oracle.npy"))
    rows = []
    for pi, (name, start, size) in enumerate(pos):
        cube = lab[tuple(slice(a, a + s) for a, s in zip(start, size))]
        u, c = np.unique(cube[cube > 0], return_counts=True)
        order = np.argsort(-c)
        tot = int((cube > 0).sum())
        top = [(L.get(int(u[k]), (str(int(u[k])),))[0], round(float(c[k] / max(tot, 1)), 3)) for k in order[:5]]
        rows.append({"position": pi, "name": name, "labelled_voxels": tot, "top_labels": top})
    met = json.load(open(os.path.join(B.ROOT, "results", "donor_ladder_brain_metrics.json"), encoding="utf-8"))
    pooled = {view: {rung: v["R"] for rung, v in met["views"][view]["ladder_R"]["mse"].items()}
              for view in ("deep/L", "parietal/L", "random_patches/L")}
    out = os.path.join(QC, f"{tid}_parcels_per_position.json")
    json.dump({"target": tid, "positions": rows, "ladder_R_mse_pooled": pooled,
               "note": "per-position R of the 32 random cubes is not stored; pooled view only"}, open(out, "w"),
              indent=1)
    print(out)


EDGE_VOX = 2


def edge_masks(shape, k=EDGE_VOX):
    """Box-shaped masks: the k-voxel shell along the box faces and the interior."""
    inner = np.zeros(shape, bool)
    inner[tuple(slice(k, s - k) for s in shape)] = True
    return ~inner, inner


def summary(tid, pos):
    """Go / no-go numbers for one target: per box and rung, aseg and DKT Dice on scored voxels, the same on the box
    edge shell and the interior, and whole-brain Dice outside both boxes (rung vs oracle)."""
    base = os.path.join(S.PRIVATE, tid)
    info = json.load(open(os.path.join(base, "fills.json")))
    oracle = np.load(os.path.join(base, "label_fastsurfer_oracle.npy"))
    pis = S.box_positions(pos)
    outside = np.ones(oracle.shape, bool)
    for pi in pis.values():
        outside[S.box_slices(pos, pi)] = False
    out = {"target": tid, "edge_vox": EDGE_VOX, "volumes": {}}
    for n in info["volumes"]:
        if n == "oracle":
            continue
        lp = os.path.join(base, f"label_fastsurfer_{n}.npy")
        if not os.path.exists(lp):
            out["volumes"][n] = {"status": "no label"}
            continue
        lab = np.load(lp)
        brain = outside & ((lab > 0) | (oracle > 0))
        r = {"status": "ok", "labelled_voxels": int((lab > 0).sum()),
             "outside_boxes": S.label_both(lab, oracle, brain), "boxes": {}}
        for box, pi in pis.items():
            if n in S.CORE and n not in info["boxes"][box]["fills"]:
                continue
            sl = S.box_slices(pos, pi)
            vfile = f"valid_{box}_{n}.npy" if n in S.CORE else f"valid_{box}_blur.npy"
            s = np.load(os.path.join(base, vfile))
            shell, inner = edge_masks(s.shape)
            p, o = lab[sl], oracle[sl]
            r["boxes"][box] = {"all": S.label_both(p, o, s), "edge": S.label_both(p, o, s & shell),
                               "interior": S.label_both(p, o, s & inner)}
        out["volumes"][n] = r
    path = os.path.join(QC, f"{tid}_summary.json")
    json.dump(out, open(path, "w"), indent=1)
    print(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("rungs", "oracles", "parcels", "summary"))
    ap.add_argument("--targets", nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(QC, exist_ok=True)
    pos, affine, _ = B.positions(np.random.default_rng(S.SEED))
    if a.what == "rungs":
        for t in a.targets:
            fig_rungs(t, pos)
    elif a.what == "oracles":
        fig_oracles(a.targets, pos)
    elif a.what == "parcels":
        for t in a.targets:
            table_parcels(t, pos)
    else:
        for t in a.targets:
            summary(t, pos)


if __name__ == "__main__":
    main()
