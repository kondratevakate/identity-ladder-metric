"""Two candidate page-1 teasers in the shared paper style (fig_style.py); display only, values reused unchanged.

a: the metric map (summary panel A) next to the MSE and FID rows of summary panel B: pixel error sees the person
   but rewards averaging; the set distance rejects averaging but cannot see the person.
b: three map rows of the provenance figure (brain QTIM, retina GRAPE, iris UPOL; aggregate maps only) for the true
   person, the mean of 16 look-alikes and a stranger, next to the metric map.

Inputs: results/cross_modality/fig_metric_summary_R.json; fig/identity_ladder_provenance_v3.pdf and
provenance_infographic_v3.json (map cells and their printed values).
Usage (from paper/cvpr2027): python make_teaser.py   ->  fig/teaser_a.{pdf,png}, fig/teaser_b.{pdf,png}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf as fitz  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

import fig_style as S  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = json.loads((ROOT / "results/cross_modality/fig_metric_summary_R.json").read_text(encoding="utf-8"))
S.apply()
GROUP_MARK = {"mse": "o", "ssim": "o", "tm_like": "o", "lddt": "o", "lpips_alex": "^", "lpips_vgg": "^",
              "betti_matched": "D", "persistence": "D", "fid": "P", "fwd": "P"}
A_LABEL = {"mse": "MSE", "ssim": "SSIM", "tm_like": "TM-like", "lddt": "lDDT", "lpips_alex": "LPIPS AlexNet",
           "lpips_vgg": "LPIPS VGG", "betti_matched": "Betti", "persistence": "Persistence", "fid": "FID", "fwd": "FWD"}
A_OFF = {"mse": (5, -3), "ssim": (-27, -3), "tm_like": (5, -3), "lddt": (5, -3), "lpips_alex": (5, 1),
         "lpips_vgg": (-50, -3), "betti_matched": (-25, -3), "persistence": (5, -2), "fid": (5, -3), "fwd": (5, -3)}


def metric_map(ax, legend=True):
    XLO, XHI, YLO, YHI = -0.08, 1.0, -0.36, 0.66
    ax.axhline(0, color="#cccccc", lw=0.6)
    ax.axvline(0, color="#cccccc", lw=0.6)
    ax.add_patch(plt.Rectangle((0.4, 0.25), XHI - 0.4, YHI - 0.25, color="#1baf7a", alpha=0.10, lw=0))
    ax.text(XHI - 0.02, YHI - 0.02, "both:\nno metric", ha="right", va="top", color="#226b4f", fontweight="bold")
    ax.text(XLO + 0.02, 0.36, "blind to\nthe person", ha="left", va="top", color=S.MUTED)
    ax.text(XHI - 0.02, YLO + 0.02, "rewards\naveraging", ha="right", va="bottom", color=S.MUTED)
    for m, mk in GROUP_MARK.items():
        a = R["panel_A"][m]
        x, y = a["x_stranger_gap"], a["y_shrunk_gap"]
        ax.plot(x, y, marker=mk, ms=S.MS, color="#333333", mec="white", mew=S.EDGE, ls="", zorder=3)
        ax.annotate(A_LABEL[m], (x, y), xytext=A_OFF[m], textcoords="offset points", color=S.INK)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel("sees the person →", labelpad=1)
    ax.set_ylabel("rejects subtle averaging →", labelpad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#f0f0f0", lw=0.5, zorder=0)
    if legend:
        h = [plt.Line2D([], [], marker=mk, ms=S.MS, color="#333333", mec="white", mew=S.EDGE, ls="") for mk in "o^DP"]
        ax.legend(h, ["error and similarity", "LPIPS", "topology", "set level"], loc="lower left", frameon=False,
                  handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)


def strip_row(ax, m, log, xlabel):
    LANE = {k: 0.35 - i * 0.1 for i, k in enumerate(S.ORDER)}
    x0, x1 = (np.log10(0.4), np.log10(300)) if log else (0.0, 2.0)
    tx = np.log10 if log else (lambda v: v)
    ax.add_patch(plt.Rectangle((x0, -0.43), x1 - x0, 0.86, color="#f3f3f3", lw=0, zorder=0))
    for k in S.ORDER:
        v = R[m].get(k)
        if v is None:
            continue
        ax.plot(min(tx(v), x1), LANE[k], marker=S.marker(k), ms=S.MS, color=S.color(k), mec="white", mew=S.EDGE,
                ls="", zorder=3)
    ax.axvline(0 if log else 1, color="#555555", lw=0.6, ls=(0, (3, 3)))
    ax.set_xlim(x0 - 0.02 * (x1 - x0), x1 + 0.04 * (x1 - x0))
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    if log:
        t = [0.5, 1, 2, 5, 10, 30, 100, 300]
        ax.set_xticks(np.log10(t))
        ax.set_xticklabels([str(v) for v in t])
    else:
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_xticklabels(["0\ntarget", "0.5", "1\nstranger", "1.5", "≥2"])
    ax.set_xlabel(xlabel, labelpad=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def teaser_a():
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 2.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.22, left=0.075, right=0.985, top=0.9, bottom=0.3)
    ax = fig.add_subplot(gs[0])
    metric_map(ax)
    right = gs[1].subgridspec(2, 1, hspace=1.25)
    b1 = fig.add_subplot(right[0])
    strip_row(b1, "mse", False, "")
    b1.set_title("MSE, 9 datasets: averages come out closer than the true person", loc="left", pad=3)
    b2 = fig.add_subplot(right[1])
    strip_row(b2, "fid", True, "set distance / stranger set distance (log)")
    b2.set_title("FID, 7 datasets: the mean is far, the true person as far as a stranger", loc="left", pad=3)
    S.letter(fig, 0.005, 0.93, "A")
    S.letter(fig, 0.44, 0.93, "B")
    for keys, yy in ((S.LADDER, 0.05), (S.CONTROLS, -0.01)):
        fig.legend(S.handles(keys), [S.label(k) for k in keys], loc="lower center", bbox_to_anchor=(0.53, yy),
                   ncol=len(keys), frameon=False, handletextpad=0.1, columnspacing=1.2)
    return fig


def map_cells():
    """Raster map cells and printed values from the provenance figure, rows brain, retina, iris; no faces."""
    prov = json.loads((HERE / "provenance_infographic_v3.json").read_text())
    doc = fitz.open(HERE / "fig/identity_ladder_provenance_v3.pdf")
    cells = {}
    for e in prov["panels"]:
        if e["row"] in (0, 2, 3) and e["source_column"] in (1, 4, 5):
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(6, 6), clip=fitz.Rect(e["destination_rect"]), alpha=True)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            m = int(pix.width * 0.05)
            img = img[:, m:pix.width - m]
            cells[e["row"], e["source_column"]] = (img, e["annotation"].split("\n")[0])
    return cells


def teaser_b():
    cells = map_cells()
    fig = plt.figure(figsize=(S.TEXTWIDTH_IN, 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.24, left=0.07, right=0.985, top=0.8, bottom=0.17)
    grid = gs[0].subgridspec(3, 3, hspace=0.38, wspace=0.08)
    cols = [(1, "ref", "True person"), (5, "lookalike16", "Mean of 16\nlook-alikes"), (4, "stranger", "Stranger")]
    rows = [(0, "Brain MRI"), (2, "Retina"), (3, "Iris")]
    for i, (r, name) in enumerate(rows):
        for j, (c, key, lab) in enumerate(cols):
            ax = fig.add_subplot(grid[i, j])
            img, val = cells[r, c]
            ax.imshow(img)
            ax.set_axis_off()
            ax.text(1.0, 0.5, val, transform=ax.transAxes, ha="left", va="center", fontsize=S.SIZE_SMALL)
            if i == 0:
                ax.set_title(lab, fontsize=S.SIZE, pad=3, color=S.INK)
                ax.plot([0.2, 0.8], [1.02, 1.02], transform=ax.transAxes, color=S.color(key), lw=1.6, clip_on=False)
            if j == 0:
                ax.text(-0.08, 0.5, name, transform=ax.transAxes, ha="right", va="center", fontweight="bold")
    cax = fig.add_axes([0.13, 0.075, 0.38, 0.025])
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.4)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(["0 target", "1 dataset mean", "2"])
    cb.outline.set_linewidth(0.4)
    cax.tick_params(length=2, pad=1, labelsize=S.SIZE_SMALL)
    ax = fig.add_subplot(gs[1])
    metric_map(ax, legend=False)
    S.letter(fig, 0.005, 0.93, "A")
    S.title(fig, 0.03, 0.93, "Local squared error / error of the dataset mean")
    S.letter(fig, 0.53, 0.93, "B")
    S.title(fig, 0.555, 0.93, "Where each metric falls")
    return fig


if __name__ == "__main__":
    for name, make in (("teaser_a", teaser_a), ("teaser_b", teaser_b)):
        f = make()
        f.savefig(HERE / "fig" / f"{name}.pdf")
        f.savefig(HERE / "fig" / f"{name}.png", dpi=250)
        print("saved", name, f.get_size_inches().round(2).tolist())
