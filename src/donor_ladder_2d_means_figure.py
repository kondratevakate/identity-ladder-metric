"""Mean fundus per dataset, side by side (general protocol, section 8; visual-qc check of the stacking).

As with the face datasets: if two datasets of one modality give similar means, the stacking is a plausible proxy
for correct alignment. Each mean uses only images valid at each pixel (missing is never zero) and is shown where at
least half of the images are valid. Frames differ by design (Amendment 9.2): PAPILA P9 = disc moved to one point,
whole photo, scale 512/3120 raw; GRAPE P8 = disc-fovea similarity frame. Aggregates only. Pre-QC.
Usage: python src/donor_ladder_2d_means_figure.py   -> results/figures/fig_ladder2d_means.{png,pdf}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import donor_ladder_2d as D  # noqa: E402
import fundus_preprocess as FP  # noqa: E402

SETS = [("PAPILA", "p9", FP.P9_CENTRE / 2, "PAPILA, P9\n(disc to one point, whole photo)"),
        ("GRAPE", "p8", FP.DISC_R / 2, "GRAPE, P8\n(disc-fovea frame)")]


def valid_mean(img, fov):
    """Per-pixel mean over images valid there, the count, and the image count."""
    n = fov.sum(0)
    return np.where(n > 0, (img * fov).sum(0) / np.maximum(n, 1), np.nan), n, len(fov)


def main():
    fig, axes = plt.subplots(2, len(SETS), figsize=(4.2 * len(SETS), 8.0))
    for j, (name, arm, disc, title) in enumerate(SETS):
        z = np.load(os.path.join(D.PRIVATE, f"{name}_canonical_{arm}.npz"))
        img, fov = z["img"][z["ok"]], z["fov"][z["ok"]]
        m, n, total = valid_mean(img, fov)
        a = axes[0, j]
        a.imshow(np.where(n >= 0.5 * total, m, np.nan), cmap="gray", vmin=-1.5, vmax=3)
        a.plot(*disc, "y+", ms=10, mew=1.2)
        a.set_title(f"{title}\nmean of {total} images", fontsize=10)
        a = axes[1, j]
        im = a.imshow(np.where(n > 0, n / total, np.nan), cmap="viridis", vmin=0, vmax=1)
        a.contour(n >= 0.5 * total, levels=[0.5], colors="w", linewidths=0.8)
        a.set_title("share of images valid at each pixel\n(white: half; mean shown inside)", fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046)
    for a in axes.ravel():
        a.set(xticks=[], yticks=[])
    fig.suptitle("Fundus mean per dataset (green / background, z units; outside the field = missing). Pre-QC.",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "..", "results", "figures", "fig_ladder2d_means")
    fig.savefig(out + ".png", dpi=150)
    fig.savefig(out + ".pdf")
    print(os.path.abspath(out + ".png"))


if __name__ == "__main__":
    main()
