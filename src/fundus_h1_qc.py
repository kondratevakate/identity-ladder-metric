"""H1 prototype and visual QC (general protocol, Amendment 9.9). Pre-QC; real images stay in the private folder.

40 evenly spaced images per dataset: the same disc detector for PAPILA and GRAPE, one scale per dataset (median disc
radius -> 30 px), disc to the canvas centre, the same field circle and intensity normalisation. Outputs (private):
detector validation against PAPILA expert contours, a QC sheet per dataset, the two means side by side.
Usage: LADDER_EXTERNAL=<checkout>/external python src/fundus_h1_qc.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import donor_ladder_2d as D  # noqa: E402
import donor_ladder_fundus as F  # noqa: E402
import fundus_preprocess as P  # noqa: E402

N_IMG = 40


def run(name):
    t = F.papila_table() if name == "PAPILA" else F.grape_table()
    pick = np.linspace(0, len(t) - 1, N_IMG).astype(int)
    rows = []
    for i in pick:
        path, side = t.path.iloc[i], t.side.iloc[i]
        rgb, inside, k, off = P.load_square(path)
        found, status = P.detect_h1(rgb, inside, side)
        exp = None
        if name == "PAPILA":
            e = P.papila_expert_disc_raw(path)
            if e is not None:
                c = (e[0] + off) * k
                if side == "L":
                    c = np.array([P.H1_SIZE - 1 - c[0], c[1]])
                exp = (c, e[1] * k)
        rows.append(dict(i=int(i), side=side, rgb=rgb, inside=inside, found=found, status=status, expert=exp))
    radii = [r["found"][1] for r in rows if r["found"] is not None]
    scale = P.H1_RADIUS / float(np.median(radii))
    img, fov = np.zeros((N_IMG, P.G, P.G), np.float32), np.zeros((N_IMG, P.G, P.G), bool)
    for j, r in enumerate(rows):
        if r["found"] is not None:
            img[j], fov[j] = P.analysis_image_h1(r["rgb"], r["inside"], r["side"], r["found"][0], r["found"][1], scale)
    val = {}
    if name == "PAPILA":
        both = [r for r in rows if r["found"] is not None and r["expert"] is not None]
        d = np.array([np.linalg.norm(r["found"][0] - r["expert"][0]) / r["expert"][1] for r in both])
        q = np.array([r["found"][1] / r["expert"][1] for r in both])
        val = {"n": len(both), "centre_error_in_expert_radii": {"median": round(float(np.median(d)), 3),
                                                                "p90": round(float(np.percentile(d, 90)), 3),
                                                                "max": round(float(d.max()), 3),
                                                                "beyond_half_radius": int((d > 0.5).sum())},
               "radius_ratio_detected_over_expert": {"median": round(float(np.median(q)), 3),
                                                     "p10": round(float(np.percentile(q, 10)), 3),
                                                     "p90": round(float(np.percentile(q, 90)), 3)}}
    fails = {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}}
    return rows, img, fov, scale, val, fails


def sheet(name, rows, img, fov, scale):
    show = np.linspace(0, N_IMG - 1, 10).astype(int)
    fig, ax = plt.subplots(3, 10, figsize=(22, 7))
    for c, j in enumerate(show):
        r = rows[j]
        a = ax[0, c]
        a.imshow(np.clip(P.oriented(r["rgb"], r["side"]), 0, 1))
        if r["found"] is not None:
            a.add_patch(plt.Circle(r["found"][0], r["found"][1], fill=False, ec="yellow", lw=1))
        if r["expert"] is not None:
            a.add_patch(plt.Circle(r["expert"][0], r["expert"][1], fill=False, ec="cyan", lw=1, ls="--"))
        a.set_title(f"{r['i']} {r['side']}: {r['status']}", fontsize=8, color="black" if r["found"] is not None else "red")
        if r["found"] is not None:
            cc, ins, circle = P.frame_h1(r["rgb"], r["inside"], r["side"], r["found"][0], scale)
            ax[1, c].imshow(np.clip(np.where((ins & circle)[..., None], cc, 0), 0, 1))
            ax[1, c].add_patch(plt.Circle(P.H1_CENTRE, P.H1_RADIUS, fill=False, ec="yellow", lw=0.8))
            ax[2, c].imshow(np.where(fov[j], img[j], np.nan), cmap="gray", vmin=-2.5, vmax=2.5)
    for rr, lab in enumerate(("oriented photo: yellow detected disc" + (", cyan expert" if name == "PAPILA" else ""),
                              "H1 frame (disc to centre, dataset scale)", "H1 analysis image")):
        ax[rr, 0].set_ylabel(lab, fontsize=8)
    for a in ax.ravel():
        a.set(xticks=[], yticks=[])
    fig.suptitle(f"{name} H1 (Amendment 9.9), evenly spaced images; dataset scale {scale:.3f}. Pre-QC, private.",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(D.PRIVATE, f"qc_h1_{name}.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def main():
    ext = os.environ.get("LADDER_EXTERNAL")
    if ext:
        F.S.GRAPE_META = os.path.join(ext, *os.path.relpath(F.S.GRAPE_META, os.path.join(
            os.path.dirname(F.S.__file__), "..", "external")).split(os.sep))
    means, report = {}, {}
    for name in ("PAPILA", "GRAPE"):
        rows, img, fov, scale, val, fails = run(name)
        print(sheet(name, rows, img, fov, scale), flush=True)
        n = fov.sum(0)
        means[name] = (np.where(n > 0, (img * fov).sum(0) / np.maximum(n, 1), np.nan), n / max(len(img), 1))
        report[name] = {"scale": round(scale, 4), "detector_status": fails, "expert_validation": val}
        print(name, json.dumps(report[name]), flush=True)
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.4))
    for c, name in enumerate(("PAPILA", "GRAPE")):
        m, share = means[name]
        ax[0, c].imshow(np.where(share >= 0.5, m, np.nan), cmap="gray", vmin=-1.5, vmax=3)
        ax[0, c].set_title(f"{name} H1 mean ({N_IMG} images)", fontsize=11)
        im = ax[1, c].imshow(share, cmap="viridis", vmin=0, vmax=1)
        ax[1, c].set_title("share of images valid", fontsize=10)
    both = (means["PAPILA"][1] >= 0.5) & (means["GRAPE"][1] >= 0.5)
    diff = np.where(both, means["PAPILA"][0] - means["GRAPE"][0], np.nan)
    ax[0, 2].imshow(diff, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    r = np.corrcoef(means["PAPILA"][0][both], means["GRAPE"][0][both])[0, 1]
    ax[0, 2].set_title(f"PAPILA - GRAPE (where both >= half valid)\nr between means {r:.2f}", fontsize=10)
    fig.colorbar(im, ax=ax[1, :2], fraction=0.03)
    ax[1, 2].axis("off")
    for a in ax.ravel():
        a.set(xticks=[], yticks=[])
    fig.suptitle("H1: the two fundus datasets in one frame (green / background, z units). Pre-QC.", fontsize=12)
    out = os.path.join(D.PRIVATE, "qc_h1_means.png")
    fig.savefig(out, dpi=110)
    print(out, "r between means", round(float(r), 3), flush=True)


if __name__ == "__main__":
    main()
