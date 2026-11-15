"""Atlas v2 (15 Nov): the 9-row aggregate atlas re-laid out in the shared paper style (fig_style.py).

Display assembly only. Map panels are cut unchanged from fig/atlas_all_modalities_full.pdf (the 9-row atlas built by
assemble_atlas.py plus the IXI row; kept unchanged) and placed on a new text-width page: IXI next to QTIM, plain row
and column labels, Helvetica at printed size, one gray style for missing cells, no status codes or internal terms.
Values under the maps are read from the source page text, so they are exactly the printed source values.
No person image is opened.

Usage (from paper/cvpr2027): python assemble_atlas_v2.py
Writes fig/atlas_all_modalities_v2.pdf, .build/atlas_v2_preview.png and atlas_provenance_v2.json.
"""
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pymupdf as fitz
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

import fig_style as S

HERE = Path(__file__).resolve().parent
SRC = HERE / "fig/atlas_all_modalities_full.pdf"
OUT = HERE / "fig/atlas_all_modalities_v2.pdf"

# Source geometry of the original eight rows (same formula as assemble_atlas.py) plus the appended IXI row.
W0, H0 = 853.92, 1041.12
CW = (.995 - .105) * W0 / (8 + 7 * .06)
CH = (.875 - .12) * H0 / (8 + 7 * .42)
XC = [.105 * W0 + j * CW * 1.06 + CW / 2 for j in range(8)]
YC = [.125 * H0 + i * CH * 1.42 + CH / 2 for i in range(8)]
BRAIN_Y = {2: (334, 406), "ixi": (1018, 1090)}

# (source row, group, dataset, columns without data). Display order: QTIM and IXI together.
ROWS = [(0, "Faces", "CelebA", {3}), (1, "Faces", "IMDb-WIKI", set()),
        (2, "Brain MRI", "QTIM", set()), ("ixi", "Brain MRI", "IXI", {1, 3, 6}),
        (3, "Retina", "GRAPE", {3}), (4, "Retina", "PAPILA", {1, 3}),
        (5, "Iris, near-infrared", "UTIRIS", {3, 6}), (6, "Iris, visible light", "UPOL", {3, 6}),
        (7, "Iris, visible light", "UTIRIS", {3, 6})]
COLS = [("Mean\nanatomy", None), ("True person:\nsecond image", "ref"), ("True person:\nopposite side", "ref"),
        ("Sibling", None), ("Stranger", "stranger"), ("Mean of 16\nlook-alikes", "lookalike16"),
        ("Same sex\nand age mean", None), ("Error of the\ndataset mean", "dataset_mean")]
GRAY_FILL = (.94, .94, .93)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb(code):
    return tuple(int(code[i:i + 2], 16) / 255 for i in (1, 3, 5))


def cell_clip(page, row, col):
    """Source cell; its bottom stops above any printed source text inside the cell (short iris rings)."""
    if row in BRAIN_Y:
        y0, y1 = BRAIN_Y[row]
    else:
        y0, y1 = YC[row] - CH / 2, YC[row] + CH / 2
    r = fitz.Rect(XC[col] - CW / 2, y0, XC[col] + CW / 2, y1)
    inside = [b[1] for b in page.get_text("blocks", clip=r) if b[4].strip()]
    if inside:
        r.y1 = min(inside) - 1
    return r


def value_under(page, clip):
    text = page.get_text("text", clip=fitz.Rect(clip.x0, clip.y1, clip.x1, clip.y1 + 24)).strip()
    num = re.search(r"(\d+\.\d+)\s*\S\s*mean", text)
    assert num, (clip, text)
    ci = re.search(r"\[\s*[\d.]+,\s*[\d.]+\s*\]", text)
    return num.group(1) + (("\n" + ci.group(0)) if ci else "")


def main():
    src = fitz.open(SRC)
    page = src[0]
    assert abs(page.rect.width - W0) < 1, "source atlas layout changed"
    doc = fitz.open()
    LW, TOP = 66.0, 34.0                 # label column width, header height
    colw = (S.TEXTWIDTH_PT - LW) / 8
    imw, imh = colw - 4, (colw - 4) * CH / CW
    rowh = imh + 22
    height = TOP + len(ROWS) * rowh + 36
    out = doc.new_page(width=S.TEXTWIDTH_PT, height=height)
    out.insert_font(fontname="helv")
    out.insert_font(fontname="hebo")
    ink, muted = rgb(S.INK), rgb(S.MUTED)

    def put(rect, text, size=S.SIZE, bold=False, color=ink, align=0):
        spare = out.insert_textbox(fitz.Rect(rect), text, fontsize=size, fontname="hebo" if bold else "helv",
                                   color=color, align=align)
        assert spare >= 0, ("label does not fit", text, spare)

    for j, (lab, key) in enumerate(COLS):
        x = LW + j * colw
        put((x - 2, 2, x + colw + 2, TOP - 6), lab, S.SIZE, align=1)
        if key:
            out.draw_line((x + colw * .25, TOP - 4), (x + colw * .75, TOP - 4), color=rgb(S.color(key)), width=1.6)
    record = []
    for i, (row, group, dataset, missing) in enumerate(ROWS):
        y = TOP + i * rowh
        put((0, y + imh / 2 - 12, LW - 2, y + imh / 2), group, S.SIZE, bold=True)
        put((0, y + imh / 2, LW - 2, y + imh / 2 + 12), dataset, S.SIZE, color=muted)
        for j in range(8):
            x = LW + j * colw + 2
            dest = fitz.Rect(x, y, x + imw, y + imh)
            if j in missing:
                out.draw_rect(dest + (4, 2, -4, -2), color=None, fill=GRAY_FILL)
                put((x, y + imh / 2 - 5, x + imw, y + imh / 2 + 6), "not available", S.SIZE_SMALL, color=muted,
                    align=1)
                continue
            clip = cell_clip(page, row, j)
            out.show_pdf_page(dest, src, 0, clip=clip, keep_proportion=True)
            entry = {"row": str(row), "column": j, "source_clip": [round(v, 2) for v in clip]}
            if j not in (0, 7):
                entry["value"] = value_under(page, clip)
                put((x - 3, y + imh + 1, x + imw + 3, y + imh + 21), entry["value"], S.SIZE_SMALL, align=1)
            record.append(entry)
    # color bar
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.4)
    bar = (colormaps["RdBu_r"](np.linspace(0, 1, 512))[None, :, :3] * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(np.repeat(bar, 8, axis=0)).save(buf, format="PNG")
    yb = TOP + len(ROWS) * rowh + 2
    x0, x1 = 150, 420
    out.insert_image(fitz.Rect(x0, yb, x1, yb + 6), stream=buf.getvalue())
    for v in (0, .5, 1, 1.5, 2, 2.4):
        xv = x0 + float(norm(v)) * (x1 - x0)
        put((xv - 12, yb + 7, xv + 12, yb + 19), str(v), S.SIZE_SMALL, align=1)
    put((0, yb + 19, S.TEXTWIDTH_PT, yb + 33),
        "Local squared error divided by the error of the dataset mean: 1 = dataset mean, about 2 = stranger. "
        "The last column has its own scale.", S.SIZE_SMALL, color=muted, align=1)
    doc.set_metadata({"title": "Identity Ladder aggregate maps, all modalities", "author": ""})
    doc.save(OUT, garbage=4, deflate=True)
    (HERE / ".build").mkdir(exist_ok=True)
    out.get_pixmap(matrix=fitz.Matrix(3, 3)).save(HERE / ".build/atlas_v2_preview.png")
    prov = {"generator": "paper/cvpr2027/assemble_atlas_v2.py", "generator_sha256": sha(__file__),
            "source_pdf": "paper/cvpr2027/fig/atlas_all_modalities_full.pdf", "source_pdf_sha256": sha(SRC),
            "method": "Display re-layout of saved aggregate map panels; no map recomputed; values copied from the "
                      "source page text.", "row_order": [f"{g} / {d}" for _, g, d, _ in ROWS], "cells": record,
            "output": {OUT.name: sha(OUT)}}
    (HERE / "atlas_provenance_v2.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(json.dumps({"written": str(OUT), "height_pt": round(height, 1), "cells": len(record)}))


if __name__ == "__main__":
    main()
