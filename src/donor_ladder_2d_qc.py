"""Visual QC sheets of the versioned fundus preprocessing (general protocol, Amendment 9.2).

20 images evenly spaced in the dataset table (not chosen by quality), in two sheets of 10. Row 1: raw photo with
the P0 landmarks (red: disc circle, fovea cross) and the version's landmarks (yellow disc circle with the
estimated radius, cyan fovea cross) and its status. Row 2: the version's disc-fovea frame with the target
positions. Row 3: the version's analysis image (green / background, z-scored). Real images: sheets stay in the
private folder and are never committed. Usage: python src/donor_ladder_2d_qc.py PAPILA [P1]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_2d as D  # noqa: E402
import donor_ladder_fundus as F  # noqa: E402
import fundus_preprocess as P  # noqa: E402

N_SHOW, PER_SHEET = 20, 10


def sheet(name, t, picks, page, version):
    fig, axes = plt.subplots(3, PER_SHEET, figsize=(2.0 * PER_SHEET, 6.6))
    stats = []
    for j, i in enumerate(picks):
        ax = axes[0, j]
        if version == "P9":                               # PAPILA whole photo on the larger canvas
            from PIL import Image
            path, side = t.path.iloc[i], t.side.iloc[i]
            found = P.papila_expert_disc_raw(path)
            k = 512 / P.P9_RAW
            im = Image.open(path).convert("RGB")
            small = np.asarray(im.resize((round(im.size[0] * k), round(im.size[1] * k)), Image.BOX), np.float32) / 255
            r = P.analysis_image_p9(path, side) if found is not None else None
            status = "no expert disc" if r is None else ("clipped at canvas edge" if r[2] else "whole field")
            stats.append(status)
            ax.imshow(np.clip(P.oriented(small, side), 0, 1))
            if found is not None:
                d = found[0] * k
                if side == "L":
                    d = np.array([small.shape[1] - 1 - d[0], d[1]])
                ax.add_patch(plt.Circle(d, found[1] * k, fill=False, ec="yellow", lw=1))
            ax.set_title(f"{i} {side}: {status}", fontsize=7, color="black" if r is not None and not r[2] else "red")
            if r is not None:
                c, inside = P.canvas_p9(small, found[0] * k, side)
                axes[1, j].imshow(np.clip(c, 0, 1))
                axes[1, j].contour(inside, levels=[0.5], colors="cyan", linewidths=0.6)
                axes[1, j].plot(*P.P9_CENTRE, "yo", ms=10, mfc="none", mew=1)
                axes[2, j].imshow(np.where(r[1], r[0], np.nan), cmap="gray", vmin=-2.5, vmax=2.5)
            continue
        rgb = F.L.load_base(t.path.iloc[i], 512, True)
        if version in ("P2", "P3", "P4", "P5", "P6", "P7", "P8"):
            side, frame = t.side.iloc[i], P.FRAMES[name]
            fn = {"P2": (P.landmarks_p2, P.frame_ok, P.canonical_p2, P.analysis_image_p2),
                  "P3": (P.landmarks_p3, P.frame_ok_p3, P.canonical_p3, P.analysis_image_p3),
                  "P4": (P.landmarks_p4, P.frame_ok_p3, P.canonical_p3, P.analysis_image_p3),
                  "P5": (P.landmarks_p5, P.frame_ok_p3, P.canonical_p3, P.analysis_image_p3),
                  "P6": (lambda im, sd: P.landmarks_p6(im, sd, t.path.iloc[i], name), P.frame_ok_p3,
                         P.canonical_p3, P.analysis_image_p3),
                  "P7": (lambda im, sd: P.landmarks_p7(im, sd, t.path.iloc[i], name), P.frame_ok_p3,
                         P.canonical_p3, P.analysis_image_p3),
                  "P8": (lambda im, sd: P.landmarks_p7(im, sd, t.path.iloc[i], name), P.frame_ok_p3,
                         P.canonical_p3, P.analysis_image_p8)}[version]
            lm = fn[0](rgb, side)
            ok = fn[1](lm, frame)
            status = lm["status"] if frame == "disc_fovea" else ("ok (disc frame)" if ok else lm["status"])
            stats.append(status)
            ax.imshow(np.clip(P.oriented(rgb, side), 0, 1))
            if lm["disc"] is not None:
                ax.add_patch(plt.Circle(lm["disc"], lm["radius"], fill=False, ec="yellow", lw=1))
            if lm["fovea"] is not None:
                ax.plot(*lm["fovea"], "c+", ms=12, mew=1.6)
            ax.set_title(f"{i} {side}: {status}", fontsize=7, color="black" if ok else "red")
            if ok:
                c, _ = fn[2](rgb, lm, side, frame)
                axes[1, j].imshow(np.clip(c, 0, 1))
                axes[1, j].plot(*P.DISC_R, "yo", ms=10, mfc="none", mew=1)
                if frame == "disc_fovea":
                    axes[1, j].plot(*P.FOVEA_R, "c+", ms=10, mew=1.2)
                img, fov = fn[3](rgb, lm, side, frame)
                axes[2, j].imshow(np.where(fov, img, np.nan), cmap="gray", vmin=-2.5, vmax=2.5)
            else:
                for r in (1, 2):
                    axes[r, j].text(0.5, 0.5, "frame failure\n(left out)", ha="center", va="center", fontsize=7,
                                    color="red", transform=axes[r, j].transAxes)
            continue
        ax.imshow(np.clip(rgb, 0, 1))
        p0 = F.R.fundus_landmarks(rgb)
        if p0 is not None:
            ax.add_patch(plt.Circle(p0[0], 12, fill=False, ec="red", lw=1))
            ax.plot(*p0[1], "r+", ms=9, mew=1.2)
        lm = P.landmarks(rgb)
        stats.append(lm["status"])
        if lm["disc"] is not None:
            ax.add_patch(plt.Circle(lm["disc"], lm["radius"], fill=False, ec="yellow", lw=1))
        if lm["fovea"] is not None:
            ax.plot(*lm["fovea"], "c+", ms=12, mew=1.6)
        ax.set_title(f"{i}: {lm['status']}", fontsize=7, color="black" if lm["status"] == "ok" else "red")
        if lm["status"] == "ok":
            c = P.canonical(rgb, lm)
            axes[1, j].imshow(np.clip(c, 0, 1))
            axes[1, j].plot(*P.DISC_R, "yo", ms=10, mfc="none", mew=1)
            axes[1, j].plot(*P.FOVEA_R, "c+", ms=10, mew=1.2)
            img, fov = P.analysis_image(rgb, lm)
            axes[2, j].imshow(np.where(fov, img, np.nan), cmap="gray", vmin=-2.5, vmax=2.5)
        else:
            for r in (1, 2):
                axes[r, j].text(0.5, 0.5, "frame failure\n(left out)", ha="center", va="center", fontsize=7,
                                color="red", transform=axes[r, j].transAxes)
    for r, lab in enumerate((("oriented raw, " if version in ("P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9") else "raw: P0 red, ") + version + " yellow/cyan", version + " frame + targets",
                             version + " analysis image")):
        axes[r, 0].set_ylabel(lab, fontsize=8)
    for ax in axes.ravel():
        ax.set(xticks=[], yticks=[])
    fig.suptitle(f"{name} fundus preprocessing {version}, QC sheet {page} (evenly spaced images; pre-QC)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(D.PRIVATE, f"qc_{name}_{version}_{page}.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out, stats


def main(name="PAPILA", version="P1"):
    ext = os.environ.get("LADDER_EXTERNAL")
    if ext:
        F.S.GRAPE_META = os.path.join(ext, *os.path.relpath(F.S.GRAPE_META, os.path.join(
            os.path.dirname(F.S.__file__), "..", "external")).split(os.sep))
    t = F.papila_table() if name == "PAPILA" else F.grape_table()
    n_show = int(os.environ.get("QC_N", N_SHOW))
    offset = int(os.environ.get("QC_OFFSET", 0))      # extra sheets: shift the even spacing to new images
    picks = (np.linspace(0, len(t) - 1, n_show).astype(int) + offset) % len(t)
    stats = []
    for page in range(n_show // PER_SHEET):
        out, s = sheet(name, t, picks[page * PER_SHEET:(page + 1) * PER_SHEET], page + 1 + 100 * (offset > 0), version)
        stats += s
        print(out)
    print({k: stats.count(k) for k in sorted(set(stats))})


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) > 1 else ()))
