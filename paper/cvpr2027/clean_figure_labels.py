"""Remove internal version and status labels from two supplement figures (16 Nov 2026).

Display edit only: map panels, colors and numbers are untouched. The original files are kept
(fig/atlas_all_modalities_full.pdf, fig/metric_correlations.pdf); cleaned copies get the suffix _v2.
Labels such as "GRAPE (P8)", "provisional" or "pre-QC" named preprocessing versions and run status;
the paper states alignment quality once, in the supplement text, instead.
"""
from pathlib import Path
import pymupdf

HERE = Path(__file__).resolve().parent

ATLAS_FOOTER = ('Population maps only. Brain: masked Gaussian, sd = 16/sqrt(12) mm; the coarse nearest-person grid is '
                'Gaussian-weighted. IXI: same maps on 100 scans, reference people averaged only inside their masks; no repeat '
                'scans or families (n/a). Missing is unavailable. Anatomy columns are population means or templates. '
                'V has its own scale in each row.')


def line_hits(page, predicate):
    """Return (bbox, text, size) of every text line whose text satisfies predicate."""
    hits = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            text = ''.join(span['text'] for span in line['spans'])
            if predicate(text):
                hits.append((pymupdf.Rect(line['bbox']), text, line['spans'][0]['size']))
    return hits


def redact(page, rects):
    for rect in rects:
        # Inset vertically so that a box never touches the line above or below.
        page.add_redact_annot(pymupdf.Rect(rect.x0, rect.y0 + 2, rect.x1, rect.y1 - 2))
    # Remove text only; keep images and vector graphics under the boxes.
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)


def clean_atlas():
    doc = pymupdf.open(HERE / 'fig/atlas_all_modalities_full.pdf')
    page = doc[0]
    relabel = {'GRAPE (P8)': 'GRAPE', 'PAPILA (P9)': 'PAPILA', '(P2, ring view)': '(ring view)', 'provisional': ''}
    hits = line_hits(page, lambda t: t.strip() in relabel)
    footer = line_hits(page, lambda t: t.startswith('Aggregate-only maps') or t.startswith('unavailable. Anatomy columns'))
    assert len(hits) == 7 and len(footer) == 2, (len(hits), len(footer))
    redact(page, [r for r, _, _ in hits] + [r for r, _, _ in footer])
    for rect, text, size in hits:
        new = relabel[text.strip()]
        if new:
            page.insert_text((rect.x0, rect.y1 - 1.5), new, fontsize=size, fontname='helv')
    top = min(r.y0 for r, _, _ in footer)
    page.insert_textbox(pymupdf.Rect(25, top, page.rect.width - 25, top + 20), ATLAS_FOOTER,
                        fontsize=6, fontname='helv', align=pymupdf.TEXT_ALIGN_CENTER)
    doc.save(HERE / 'fig/atlas_all_modalities_full_v2.pdf', garbage=3, deflate=True)


def clean_correlations():
    doc = pymupdf.open(HERE / 'fig/metric_correlations.pdf')
    page = doc[0]
    hits = line_hits(page, lambda t: 'pre-QC: affine registration only' in t)
    assert len(hits) == 1, len(hits)
    rect, text, size = hits[0]
    redact(page, [rect])
    new = text.replace(' pre-QC: affine registration only', ' Affine registration.')
    page.insert_text((rect.x0, rect.y1 - 2), new, fontsize=size, fontname='helv')
    doc.save(HERE / 'fig/metric_correlations_v2.pdf', garbage=3, deflate=True)


def clean_brain_maps():
    doc = pymupdf.open(HERE / 'fig/brain_gaussian_maps.pdf')
    page = doc[0]
    hits = line_hits(page, lambda t: 'Pre-QC, affine registration only.' in t)
    assert len(hits) == 1, len(hits)
    rect, text, size = hits[0]
    redact(page, [rect])
    new = text.replace('neighbours', 'neighbors').replace('Pre-QC, affine registration only.', 'Affine registration.')
    page.insert_text((rect.x0, rect.y1 - 1), new, fontsize=size, fontname='helv')
    doc.save(HERE / 'fig/brain_gaussian_maps_v2.pdf', garbage=3, deflate=True)


if __name__ == '__main__':
    clean_atlas()
    clean_correlations()
    clean_brain_maps()
    print('Wrote fig/atlas_all_modalities_full_v2.pdf, fig/metric_correlations_v2.pdf and fig/brain_gaussian_maps_v2.pdf')
