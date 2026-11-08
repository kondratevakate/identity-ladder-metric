"""Ladder maps of all modalities in one figure: faces, brain MRI and fundus side by side. Pre-QC. Display only.

Rows: faces CelebA and IMDb-WIKI (face pipeline v6f), brain MRI QTIM (axial z = 10 mm), fundus GRAPE (P8) and PAPILA
(P9). Columns as the case figure: mean anatomy, another photo or scan, mirror side, sibling, stranger, 16 look-alikes,
same sex and age, dataset mean. Every map is the error of the donor in units of the dataset mean's error (1 = mean,
2 = stranger), on one diverging scale; the last column shows V itself on its own scale per row. No number is
recomputed: values come from each modality's results JSON. Only aggregate maps (no person) are drawn.
Inputs (read only): face maps `face-ladder-20261105-v1/maps_v6f.npz` and `results/donor_ladder_face_v6f.json` of the
main checkout; brain 2 mm maps (`donor_ladder_brain_maps.PRIVATE`) and their JSONs; fundus
`facefmt_{GRAPE_p8,PAPILA_p9}.npz` and `results/donor_ladder_2d_facefmt.json`.
Output: results/figures/fig_ladder_maps_all_modalities.{png,pdf}.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
DATA = os.environ.get("DATA_ROOT", r"C:\Projects\data")
PROJ = os.path.join(DATA, "projects", "identity-ladder-metric")
MAIN = os.environ.get("LADDER_MAIN", os.path.normpath(os.path.join(ROOT, "..", "..", "identity-ladder-metric")))
FACE_VERSION = "_v6f"                      # face result approved by Kate (face protocol Amendment 10)
OUT = os.path.join(ROOT, "results", "figures", "fig_ladder_maps_all_modalities")

COLS = [("anatomy", "Mean anatomy"), ("repeat", "Another photo\nor scan"), ("mirror", "Mirror side"),
        ("sibling", "Sibling"), ("stranger", "Stranger"), ("lookalike", "16 look-alikes"),
        ("sexage", "Same sex\nand age"), ("V", "Dataset mean\n(V, own scale)")]
NORM = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.4)
INK, MUTED, PANEL = "#1f1e1d", "#6b6a66", "#f3f2ee"


def crop(*arrs, valid, pad=2):
    ys, xs = np.nonzero(valid)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, valid.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, valid.shape[1])
    return [a[y0:y1, x0:x1] for a in arrs]


def face_rows():
    z = np.load(os.path.join(PROJ, "face-ladder-20261105-v1", f"maps{FACE_VERSION}.npz"))
    res = json.load(open(os.path.join(MAIN, "results", f"donor_ladder_face{FACE_VERSION}.json"), encoding="utf-8"))
    rows = []
    for src, label in (("celeba", "CelebA"), ("imdb", "IMDb-WIKI")):
        r = res["results"][src]
        maps = {"repeat": "repeat", "mirror": "mirror", "sibling": "sibling", "stranger": "stranger",
                "lookalike": "lookalike", "sexage": "sexage", "V": "V"}
        m = {k: z[f"{src}_{v}"] for k, v in maps.items() if f"{src}_{v}" in z.files}
        valid = z[f"{src}_valid"] > 0.5
        for k, v in m.items():                       # common display mask: shown only where every map is defined
            valid &= np.isfinite(v)
        stats = {k: (r[k]["ratio"], r[k].get("ci95"), r[k]["targets"]) for k in maps if k in r and k != "V"}
        rows.append({"group": "Faces", "name": label, "under": z[f"{src}_meanface"], "maps": m, "valid": valid,
                     "stats": stats, "origin": "upper"})
    return rows


def brain_row():
    import nibabel as nib
    from nilearn.datasets import load_mni152_template
    from nilearn.image import resample_to_img
    import donor_ladder_brain_maps as P
    from donor_ladder_brain_maps_figure import cut
    ref = nib.load(os.path.join(P.PRIVATE, "V.nii.gz"))
    files = {"repeat": "repeat_ratio.nii.gz", "mirror": "mirror_ratio.nii.gz", "sibling": "sibling_ratio.nii.gz",
             "stranger": "stranger_ratio.nii.gz", "lookalike": "grid_knn16_s4_ratio.nii.gz",
             "sexage": "cohort_ratio.nii.gz", "V": "V.nii.gz"}
    m = {}
    for k, f in files.items():
        img = nib.load(os.path.join(P.PRIVATE, f))
        if f.startswith("grid_"):                    # 8 mm grid: linear upsampling for display (nearest draws squares)
            img = resample_to_img(img, ref, interpolation="continuous", force_resample=True, copy_header=True)
        a = cut(img, "z", 10)
        m[k] = np.where(a == 0, np.nan, a) if k == "V" else a
    under = cut(resample_to_img(load_mni152_template(resolution=1), ref, interpolation="continuous",
                                force_resample=True, copy_header=True), "z", 10)
    valid = np.ones(under.shape, bool)
    for v in m.values():
        valid &= np.isfinite(v)
    fine = json.load(open(os.path.join(ROOT, "results", "donor_ladder_brain_maps.json"), encoding="utf-8"))["maps"]
    grid = json.load(open(os.path.join(ROOT, "results", "donor_ladder_brain_map_grid.json"), encoding="utf-8"))["maps"]
    key = {"repeat": fine["repeat"], "mirror": fine["mirror"], "sibling": fine["sibling"],
           "stranger": fine["stranger"], "sexage": fine["cohort"], "lookalike": grid["knn16_s4"]}
    stats = {k: (v["pooled_ratio"], None, None) for k, v in key.items()}
    return {"group": "Brain MRI", "name": "QTIM\naxial z = 10 mm", "under": under, "maps": m, "valid": valid,
            "stats": stats, "origin": "lower"}


def fundus_rows():
    res = json.load(open(os.path.join(ROOT, "results", "donor_ladder_2d_facefmt.json"), encoding="utf-8"))["results"]
    rows = []
    for name, arm in (("GRAPE", "p8"), ("PAPILA", "p9")):
        z = dict(np.load(os.path.join(PROJ, "donor-ladder-2d-20261105-v1", f"facefmt_{name}_{arm}.npz")))
        m = {"repeat": z.get("repeat"), "mirror": z.get("fellow"), "stranger": z["stranger"],
             "lookalike": z["lookalike"], "sexage": z.get("sexage"), "V": z["V"]}
        m = {k: v for k, v in m.items() if v is not None}
        valid = z["valid"] > 0.5
        for v in m.values():
            valid &= np.isfinite(v)
        r = res[name]
        src = {"repeat": "repeat", "mirror": "fellow", "stranger": "stranger", "lookalike": "lookalike",
               "sexage": "sexage"}
        stats = {k: (r[s]["ratio"], r[s]["ci95"], r[s]["targets"]) for k, s in src.items() if s in r}
        rows.append({"group": "Fundus", "name": f"{name} ({arm.upper()})", "under": z["meanfundus"], "maps": m,
                     "valid": valid, "stats": stats, "origin": "upper"})
    return rows


IRIS_VERSION = "P2"                        # iris frame accepted by Kate on 7 Nov 2026 (docs/PROTOCOL_iris_metrics.md)
IRIS_JSON = os.environ.get("IRIS_JSON", os.path.normpath(os.path.join(ROOT, "..", "identity-ladder-cvpr2027", "results",
                                                                     "iris_ladder_metrics.json")))
IRIS_PRIVATE = os.path.join(PROJ, "iris-ladder-metrics-20261107-v1")
IRIS_T0, IRIS_T1 = 0.05, 0.95              # radial share of the iris strip (pupil to limbus)


def iris_ring(strip, rho, n=161):
    """Iris strip (rows pupil -> limbus, columns angle counter-clockwise from 3 o'clock) drawn on the iris ring."""
    hh, ww = strip.shape
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    t = (np.hypot(xx, yy) - rho) / (1 - rho)
    row = np.rint((t - IRIS_T0) / (IRIS_T1 - IRIS_T0) * (hh - 1)).astype(int)
    col = np.rint((np.arctan2(-yy, xx) % (2 * np.pi)) / (2 * np.pi) * ww).astype(int) % ww
    inside = (row >= 0) & (row < hh)
    out = np.full(t.shape, np.nan)
    out[inside] = strip[row[inside], col[inside]]
    return out


def iris_rows():
    """Iris (UTIRIS near-infrared, UPOL, UTIRIS visible light): maps of src/iris_ladder_metrics_ladder.py (cvpr2027
    worktree) and the spec ratios of results/iris_ladder_metrics.json there. No sibling, no sex or age."""
    d = json.load(open(IRIS_JSON, encoding="utf-8"))
    rows = []
    for name, label in (("UTIRIS_NIR", "UTIRIS near-infrared"), ("UPOL", "UPOL visible light"),
                        ("UTIRIS_VIS", "UTIRIS visible light")):
        z = np.load(os.path.join(IRIS_PRIVATE, f"maps_{name}_{IRIS_VERSION}.npz"))
        rho = d["maps"][name].get("pupil_to_iris_ratio_median", 0.45)
        src = {"repeat": "other_photo", "mirror": "fellow_eye", "stranger": "stranger", "lookalike": "lookalike16",
               "V": "V"}
        m = {k: iris_ring(z[f"all_{v}"], rho) for k, v in src.items() if f"all_{v}" in z.files}
        valid = np.ones(m["V"].shape, bool)
        for v in m.values():
            valid &= np.isfinite(v)
        r = d["results"][name]
        spec = {"repeat": "repeat", "mirror": "fellow", "stranger": "stranger", "lookalike": "lookalike16"}
        stats = {k: (r[s]["ratio"], r[s]["ci95"], r[s]["targets"]) for k, s in spec.items() if s in r}
        rows.append({"group": "Iris", "name": f"{label}\n({IRIS_VERSION}, ring view)",
                     "under": iris_ring(z["all_anatomy"], rho), "maps": m, "valid": valid, "stats": stats,
                     "origin": "upper"})
    return rows


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    rows = face_rows() + [brain_row()] + fundus_rows() + iris_rows()
    nr, nc = len(rows), len(COLS)
    fig = plt.figure(figsize=(1.32 * nc + 1.3, 1.62 * nr + 1.5))
    gs = fig.add_gridspec(nr, nc, left=0.105, right=0.995, top=0.875, bottom=0.12, wspace=0.06, hspace=0.42)
    im_ratio = None
    for i, row in enumerate(rows):
        under, valid = row["under"], row["valid"]
        for j, (key, title) in enumerate(COLS):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=8.2, color=INK, pad=6)
            u = np.where(valid, under, np.nan)
            if key != "anatomy" and key not in row["maps"]:
                ax.set_facecolor(PANEL)
                ax.text(0.5, 0.5, "not available", ha="center", va="center", fontsize=6.6, color=MUTED,
                        transform=ax.transAxes)
                continue
            mp = row["maps"].get(key) if key != "anatomy" else None
            uc, = crop(u, valid=valid)
            lo, hi = np.nanpercentile(uc, [1, 99])
            ax.imshow(uc, cmap="gray", vmin=lo, vmax=hi, origin=row["origin"], interpolation="bilinear")
            if key == "anatomy":
                continue
            mc, = crop(np.where(valid, mp, np.nan), valid=valid)
            if key == "V":
                a, b = np.nanpercentile(mc, [2, 98])
                ax.imshow(mc, cmap="viridis", vmin=a, vmax=b, origin=row["origin"], interpolation="bilinear")
                txt = "1 × mean"
            else:
                im_ratio = ax.imshow(mc, cmap="RdBu_r", norm=NORM, alpha=0.92, origin=row["origin"],
                                     interpolation="bilinear")
                ratio, ci, n = row["stats"].get(key, (None, None, None))
                txt = "" if ratio is None else f"{ratio:.2f} × mean"
                if ci:
                    txt += f"\n[{ci[0]:.2f}, {ci[1]:.2f}]"
            ax.text(0.5, -0.05, txt, ha="center", va="top", fontsize=6.6, color=INK, transform=ax.transAxes)
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.axis("off")
        pos = ax0.get_position()
        fig.text(0.012, (pos.y0 + pos.y1) / 2, f"{row['group']}\n", ha="left", va="center", fontsize=9,
                 weight="bold", color=INK)
        fig.text(0.012, (pos.y0 + pos.y1) / 2 - 0.012, row["name"], ha="left", va="top", fontsize=7.2, color=MUTED)
    # column groups
    def bracket(a, b, text):
        p0 = gs[0, a].get_position(fig)
        p1 = gs[0, b].get_position(fig)
        y = 0.935
        fig.add_artist(plt.Line2D([p0.x0 + 0.004, p0.x0 + 0.004, p1.x1 - 0.004, p1.x1 - 0.004],
                                  [y - 0.008, y, y, y - 0.008], color=MUTED, lw=0.7))
        fig.text((p0.x0 + p1.x1) / 2, y + 0.004, text, ha="center", va="bottom", fontsize=7.6, color=MUTED)
    bracket(1, 4, "Real anatomy from another image or person")
    bracket(5, 7, "Averages of other people")
    cax = fig.add_axes([0.32, 0.055, 0.42, 0.011])
    cb = fig.colorbar(im_ratio, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1, 1.5, 2])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("error of the donor / error of the dataset mean   (blue: beats the mean, white: = mean, "
                 "red: 2 = stranger)", fontsize=7.2, color=INK)
    fig.text(0.5, 0.012, "Pre-QC. Faces v6f, 16 px Gaussian window; brain QTIM session 1, affine to MNI, 16 mm box "
             "window (brain maps not yet switched to the Gaussian), look-alikes on an 8 mm grid; fundus frame only "
             "(Amendment 9.13), 16 px Gaussian window; iris P2 rubber-sheet strip drawn on the ring, 16 px Gaussian "
             "window wrapped in angle, look-alikes per 8 px block, upper lid region not shown (valid in < 90% of "
             "eyes). Persons left out of their own pools. Mirror side: the other face half, hemisphere or eye.", ha="center", fontsize=6.0, color=MUTED, wrap=True)
    fig.suptitle("Every location, every modality: pixel error of each donor in units of the dataset mean",
                 fontsize=11, y=0.985, color=INK)
    fig.savefig(OUT + ".png", dpi=220)
    fig.savefig(OUT + ".pdf")
    print("written", OUT)


if __name__ == "__main__":
    main()
