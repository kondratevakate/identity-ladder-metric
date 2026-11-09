"""Assemble a source infographic and four saved aggregate heatmap rows.

The decorative bust is synthetic. Every scientific panel is copied unchanged
from the existing aggregate atlas; no participant images or private map arrays
are opened, and no metric or experiment is recomputed.
"""
from pathlib import Path
import hashlib
import io
import json
import re
import subprocess

import numpy as np
import pymupdf as fitz
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
import source_snapshot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_COMMIT = '82ed7fe8a52e557854cf4cd2d02eb47d5803d7cb'
SOURCE_PATH = 'results/figures/fig_ladder_maps_all_modalities.pdf'
OUT = HERE / 'fig/identity_ladder_provenance.pdf'
PNG = HERE / 'fig/identity_ladder_provenance.png'
ASSET = HERE / 'fig/assets/identity_bust_v1.png'
INK = (.09, .14, .19)
MUTED = (.36, .43, .49)
ICE = (.955, .977, .992)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def record(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path.read_bytes())}


def rounded(page, rect, fill, stroke=None, radius=.12, width=.55):
    """Rounded vector cards; radius is a fraction of the short edge."""
    page.draw_rect(rect, color=stroke, fill=fill, width=width, radius=radius)


def main():
    OUT.parent.mkdir(exist_ok=True)
    cache = HERE / '.build'
    cache.mkdir(exist_ok=True)
    full_path = HERE / 'fig/atlas_all_modalities_full.pdf'
    assert full_path.is_file(), 'Run assemble_atlas.py before this figure builder'
    source = fitz.open(full_path)
    page = source[0]
    original_raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', SOURCE_COMMIT + ':' + SOURCE_PATH])
    original = fitz.open(stream=original_raw, filetype='pdf')
    width, height = original[0].rect.width, original[0].rect.height
    assert abs(width - 853.92) < 1 and abs(height - 1041.12) < 1
    cell_w = (.995 - .105) * width / (8 + 7 * .06)
    cell_h = (.875 - .12) * height / (8 + 7 * .42)
    x_centers = [.105 * width + j * cell_w * 1.06 + cell_w / 2 for j in range(8)]
    y_centers = [.125 * height + i * cell_h * 1.42 + cell_h / 2 for i in range(8)]

    # Do not inspect nested forms in the replacement atlas: get_image_info()
    # includes off-clip source objects. The original has unambiguous panel boxes.
    images = {}
    for obj in original[0].get_image_info():
        box = fitz.Rect(obj['bbox'])
        cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        col = min(range(8), key=lambda j: abs(cx - x_centers[j]))
        row = min(range(8), key=lambda i: abs(cy - y_centers[i]))
        if abs(cx - x_centers[col]) < 20 and abs(cy - y_centers[row]) < 20:
            images[row, col] = box

    paths = {
        'brain': ROOT / 'results/donor_ladder_brain_metrics.json',
        'face': ROOT / 'results/face_ladder_metrics.json',
        'fundus': HERE / 'inputs/donor_ladder_2d_facefmt.json',
        'iris': ROOT / 'results/iris_ladder_metrics.json',
    }
    data = {key: (json.loads(path.read_text()) if key == 'fundus' else source_snapshot.load_json(path))
            for key, path in paths.items()}
    b = data['brain']['counts']
    f = data['face']['counts']['celeba']
    g = data['fundus']['results']['GRAPE']
    ir = data['iris']['panel']['UTIRIS_NIR']['counts']
    brain_color, face_color = (.27, .51, .72), (.49, .43, .66)
    fundus_color, iris_color = (.20, .58, .53), (.80, .57, .25)
    cards = [
        {'key': 'brain', 'rect': (10, 43, 176, 104), 'color': brain_color,
         'title': 'Brain MRI  /  QTIM', 'counts': f"{b['n_targets']} targets / {b['n_families']} families",
         'detail': f"{b['n_repeat_scans']} repeat scans; Gaussian maps", 'status': 'P'},
        {'key': 'face', 'rect': (10, 111, 176, 172), 'color': face_color,
         'title': 'Face photos  /  CelebA', 'counts': f"{f['n_targets']} targets / {f['n_families']} clusters",
         'detail': 'v6f frame approved', 'status': 'F'},
        {'key': 'fundus', 'rect': (319, 43, 485, 104), 'color': fundus_color,
         'title': 'Fundus photos  /  GRAPE P8', 'counts': f"{g['units']} eyes / {g['stranger']['persons']} people",
         'detail': f"Repeat: {g['repeat']['targets']} eyes / {g['repeat']['persons']} people", 'status': 'P'},
        {'key': 'iris', 'rect': (319, 111, 485, 172), 'color': iris_color,
         'title': 'Iris images  /  UTIRIS NIR', 'counts': f"{ir['units']} eyes / {ir['persons']} people",
         'detail': 'P2 strip panel; same-eye repeats', 'status': 'P'},
    ]
    doc = fitz.open()
    out = doc.new_page(width=495, height=533)
    out.insert_font(fontname='helv')
    out.insert_font(fontname='hebo')

    def put(rect, text, size=7, bold=False, color=INK, align=0):
        spare = out.insert_textbox(fitz.Rect(rect), text, fontsize=size,
                                   fontname='hebo' if bold else 'helv', color=color, align=align)
        assert spare >= 0, ('Text does not fit', text, spare)

    put((0, 0, 495, 20), 'From observed anatomy to population references', 10.5, True, align=1)
    put((0, 20, 495, 34), 'Source cohorts and controlled donor comparisons', 7.5, color=MUTED, align=1)
    rounded(out, fitz.Rect(0, 34, 495, 181), ICE, (.89, .94, .97), radius=.09)
    if ASSET.is_file():
        out.insert_image(fitz.Rect(177, 35, 318, 181), filename=str(ASSET), keep_proportion=True)
        asset_info = {**record(ASSET), 'role': 'Synthetic decorative illustration, not participant data or a measured anatomy panel.'}
        extra = ASSET.with_suffix('.provenance.json')
        if extra.exists():
            asset_info['generation_record'] = record(extra)
    else:
        # A visibly schematic fallback keeps the scientific assembly usable;
        # the final manuscript should use the supplied decorative asset.
        out.draw_oval(fitz.Rect(225, 46, 274, 109), color=(.78, .85, .89), fill=(.92, .95, .97))
        out.draw_oval(fitz.Rect(204, 113, 292, 201), color=None, fill=(.92, .95, .97))
        asset_info = {'role': 'Schematic fallback; final decorative asset unavailable.', 'missing_path': str(ASSET.relative_to(ROOT))}

    leaders = [
        (brain_color, [(176, 73), (205, 73), (249, 60)]),
        (face_color, [(176, 141), (210, 141), (244, 105)]),
        (fundus_color, [(319, 73), (298, 73), (277, 78)]),
        (iris_color, [(319, 141), (296, 141), (273, 79)]),
    ]
    for color, points in leaders:
        out.draw_polyline(points, color=color, width=.7)
        out.draw_circle(points[-1], 1.7, color=(1, 1, 1), fill=color, width=.5)
    for card in cards:
        box = fitz.Rect(card['rect'])
        rounded(out, box + (0, 1.2, 0, 1.2), (.90, .93, .95), radius=.12, width=0)
        rounded(out, box, (1, 1, 1), (.88, .92, .95), radius=.12)
        out.draw_line((box.x0 + 8, box.y0 + 10), (box.x0 + 8, box.y1 - 10), color=card['color'], width=2.1)
        put((box.x0 + 16, box.y0 + 8, box.x1 - 18, box.y0 + 23), card['title'], 7.6, True)
        put((box.x0 + 16, box.y0 + 26, box.x1 - 7, box.y0 + 40), card['counts'], 7.1)
        put((box.x0 + 16, box.y0 + 42, box.x1 - 7, box.y1 - 2), card['detail'], 6.4, color=MUTED)
        put((box.x1 - 15, box.y0 + 8, box.x1 - 4, box.y0 + 22), card['status'], 7.3, True, color=card['color'], align=1)

    put((0, 186, 495, 204), 'Observed donor differences  /  one primary dataset per modality', 8.6, True)
    labels = ['Mean anatomy', 'Repeat capture', 'Bilateral', 'Unrelated donor', 'Context mean\n16 donors', 'Mean-fill V\nown scale']
    columns = [0, 1, 2, 4, 5, 7]
    for j, label in enumerate(labels):
        x = 69 + j * 70
        put((x, 205, x + 65, 229), label, 6.9, align=1)
    rows = [
        (2, 'Brain MRI', 'QTIM / P', brain_color),
        (0, 'Faces', 'CelebA / F', face_color),
        (3, 'Fundus', 'GRAPE P8 / P', fundus_color),
        (5, 'Iris', 'UTIRIS NIR\nP2 / P', iris_color),
    ]
    panels = []
    for i, (source_row, group, name, color) in enumerate(rows):
        y = 230 + i * 63
        out.draw_line((0, y + 8), (0, y + 37), color=color, width=2.1)
        put((7, y + 7, 68, y + 22), group, 8, True)
        put((7, y + 24, 68, y + 49), name, 6.5, color=MUTED)
        for j, col in enumerate(columns):
            x = 69 + j * 70
            if source_row == 2:
                box = fitz.Rect(x_centers[col] - cell_w / 2, 334, x_centers[col] + cell_w / 2, 406)
            else:
                box = images.get((source_row, col))
            assert box is not None, ('Primary panel unexpectedly unavailable', source_row, col)
            # Render this source clip at high resolution rather than importing
            # its full-page form. That preserves the displayed map composition
            # without embedding off-clip text from unrelated atlas panels.
            pixels = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=box, alpha=True)
            out.insert_image(fitz.Rect(x + 4, y, x + 61, y + 42), pixmap=pixels, keep_proportion=True)
            panel = {'source_row': source_row, 'source_column': col, 'source_clip': list(box)}
            if col not in (0, 7):
                annotation = page.get_text('text', clip=fitz.Rect(x_centers[col] - cell_w / 2, box.y1,
                                                                  x_centers[col] + cell_w / 2, box.y1 + 26)).strip()
                match = re.search(r'(\d+\.\d+)\s*[×x]\s*mean', annotation)
                assert match, (source_row, col, annotation)
                ci = re.search(r'\[\s*[\d.]+,\s*[\d.]+\s*\]', annotation)
                text = match.group(1) + (('\n' + ci.group(0)) if ci else '')
                put((x, y + 44, x + 65, y + 66), text, 6.4, align=1)
                panel['display_annotation'] = text
            panels.append(panel)

    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2.4)
    rgb = (colormaps['RdBu_r'](np.linspace(0, 1, 512))[None, :, :3] * 255).astype('uint8')
    buffer = io.BytesIO()
    Image.fromarray(np.repeat(rgb, 8, axis=0)).save(buffer, format='PNG')
    out.insert_image(fitz.Rect(147, 489, 420, 496), stream=buffer.getvalue())
    for value in (0, .5, 1, 1.5, 2, 2.4):
        x = 147 + float(norm(value)) * 273
        put((x - 11, 498, x + 11, 511), str(value), 6.8, align=1)
    put((15, 515, 490, 530), 'Local error / mean-fill error (Q); white = 1. V: own scale. P: provisional; F: approved frame.',
        6.4, color=MUTED, align=1)

    doc.set_metadata({'title': 'Identity Ladder: data sources and aggregate donor maps', 'author': '',
                      'subject': 'Synthetic decorative bust; measured maps copied from aggregate source PDFs.'})
    doc.save(OUT, garbage=4, deflate=True)
    preview = cache / 'identity_ladder_provenance.png'
    out.get_pixmap(matrix=fitz.Matrix(2, 2)).save(preview)
    out.get_pixmap(matrix=fitz.Matrix(4, 4)).save(PNG)
    proof = {
        'generator': record(Path(__file__)),
        'source_atlas': record(full_path),
        'source_atlas_provenance': record(HERE / 'atlas_provenance.json'),
        'original_source_commit': SOURCE_COMMIT,
        'original_source_path': SOURCE_PATH,
        'original_source_sha256': sha(original_raw),
        'scientific_inputs': {key: (record(path) if key == 'fundus' else source_snapshot.record(path))
                              for key, path in paths.items()},
        'decorative_asset': asset_info,
        'displayed_sources': cards,
        'primary_dataset_selection': {
            'brain': 'QTIM: available MRI panel; Gaussian replacement retained.',
            'face': 'CelebA: primary face set; IMDb-WIKI remains supplementary.',
            'fundus': 'GRAPE P8: repeat acquisition exists; PAPILA remains supplementary.',
            'iris': 'UTIRIS NIR: protocol-primary set; VIS and UPOL remain supplementary.',
        },
        'counts_scope': 'Source-panel counts; endpoint eligibility varies. Fundus repeat count is stated separately. Map scalars summarize pooled source errors, not only the pictured axial slice.',
        'panels': panels,
        'panel_rendering': 'Selected PDF clips rendered at 4x source resolution to avoid hidden off-clip form text; native labels/scalars are redrawn separately. No scientific map array is recomputed.',
        'shared_ratio_scale': {'vmin': 0, 'vcenter': 1, 'vmax': 2.4},
        'measurements': 'Copied aggregate PDF maps and displayed rounded annotations; no scientific map or metric recomputed.',
        'privacy': 'No individual exemplar or participant identifier. Synthetic bust is decorative only.',
        'output': record(OUT),
        'shareable_png': {**record(PNG), 'width_px': 1980, 'height_px': 2132},
    }
    (HERE / 'provenance_infographic.json').write_text(json.dumps(proof, indent=2) + '\n')
    print('Assembled source infographic with four primary datasets; saved maps and values unchanged.')
    print('Figure:', OUT)
    print('Shareable PNG:', PNG)
    print('Preview:', preview)


if __name__ == '__main__':
    main()
