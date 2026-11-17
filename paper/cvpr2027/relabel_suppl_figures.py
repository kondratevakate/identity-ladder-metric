"""Supplement figures: remove in-figure titles and footnotes and bring labels to the paper's terms.

Reads the saved figure PDFs (v2) and writes v3 next to them; the maps and numbers are not touched.
Titles and footnotes move to the LaTeX captions; labels use the words of the main text
(second image, opposite side, random person, nearest people, dataset average).

Run from paper/cvpr2027:  python relabel_suppl_figures.py
"""
import fitz

KEEP = dict(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)


def span_at(page, rect):
    """Font size, colour and style of the first text span that overlaps rect."""
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                if fitz.Rect(s["bbox"]).intersects(rect) and s["text"].strip():
                    c = s["color"]
                    col = ((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255)
                    name = s["font"].lower()
                    serif = any(k in name for k in ("times", "nimbus rom", "termes", "cmr", "stix"))
                    bold = "bold" in name or bool(s["flags"] & 16)
                    return s["size"], col, serif, bold
    raise ValueError(f"no span at {rect}")


def edit(page, jobs, align="center"):
    """jobs: (old text, new text or None), applied one after another, so a long phrase is gone before a
    shorter phrase it contains is searched. None removes the text; '|' in the new text starts a new line.
    Every occurrence of the old text is replaced in place."""
    for old, new in jobs:
        hits = page.search_for(old)
        if not hits:
            raise ValueError(f"not found: {old!r}")
        todo = []
        for r in hits:
            todo.append((r, *span_at(page, r)))
            page.add_redact_annot(r + (0, 1, 0, -1), fill=None)  # stay inside the line: neighbours are close
        page.apply_redactions(**KEEP)
        if not new:
            continue
        for r, size, col, serif, bold in todo:
            font = ("tibo" if bold else "tiro") if serif else ("hebo" if bold else "helv")
            width = fitz.get_text_length(new, fontname=font, fontsize=size)
            x = r.x0 if align == "left" else (r.x0 + r.x1) / 2 - width / 2
            page.insert_text((x, r.y1 - 0.22 * size), new,
                             fontname=font, fontsize=size, color=col)


def brain_maps():
    d = fitz.open("fig/brain_gaussian_maps_v2.pdf")
    edit(d[0], [
        ("Real brain, every location: pixel error ranks the dataset mean above related anatomy", None),
        ("One scale: 1 = dataset mean (white), 2 = stranger; blue beats the mean.",
         "One scale: 1 = dataset average (white), 2 = random person; blue is closer than the average."),
        ("Repeat scan", "Second scan"), ("Mirror side", "Opposite side"), ("Stranger", "Random person"),
        ("16 look-alikes*", "16 nearest people*"), ("Dataset mean", "Dataset average"),
    ])
    d.save("fig/brain_gaussian_maps_v3.pdf", garbage=3, deflate=True)


def atlas():
    d = fitz.open("fig/atlas_all_modalities_full_v2.pdf")
    p = d[0]
    # footnote and colour-bar legend go to the caption; the tick numbers of the colour bar stay
    for w in p.get_text("words"):
        x0, y0, x1, y1, text = w[:5]
        in_legend = 250 < x0 < 720 and 984 < y0 < 1010
        in_footnote = y0 > 1115
        tick = text.replace(".", "").isdigit() and y0 < 997  # ticks sit above the legend sentence
        if (in_legend or in_footnote) and not tick:
            p.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=None)
    p.apply_redactions(**KEEP)
    edit(p, [
        ("Every location, every modality: pixel error of each donor in units of the dataset mean", None),
        ("n/a in IXI", None), ("n/a", None),
        ("Mirror side", "Opposite side"), ("Stranger", "Random person"), ("16 look-alikes", "16 nearest people"),
        ("Fundus", "Retina"), ("not available", "no such image"),
        ("Another photo", "Second"), ("or scan", "image"),
        ("Dataset mean", "Dataset average"), ("(V, own scale)", "(own scale)"),
    ])
    d.save("fig/atlas_all_modalities_full_v3.pdf", garbage=3, deflate=True)


def main_figures():
    """Figures 2 and 3 of the main text: the controls carry the names used in the text."""
    d = fitz.open("fig/metric_summary_v7_f8.pdf")
    edit(d[0], [("average + random detail", "wrong detail"), ("average + noise", "noise")], align="left")
    d.save("fig/metric_summary_v8_f8.pdf", garbage=3, deflate=True)
    d = fitz.open("fig/fig_explainer_mse_v3.pdf")
    edit(d[0], [("Average + random detail", "Wrong detail"), ("Average + noise", "Noise"),
                ("scale unit 0.38", "mean of 16: 0.38")])
    d.save("fig/fig_explainer_mse_v4.pdf", garbage=3, deflate=True)


if __name__ == "__main__":
    brain_maps()
    atlas()
    main_figures()
    print("written: brain_gaussian_maps_v3, atlas_all_modalities_full_v3, metric_summary_v8_f8, fig_explainer_mse_v4")
