"""GRAPE G1 QC using the shared method, unified protocol Amendment 5. No scoring.

Reviews the transform scoring would use; landmarks are shown for review, never auto-accepted.
"""
import json
import shutil
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.transform import SimilarityTransform

import ladder2d as L
import ladder2d_register as R
import reg_methods as M
from registration_qc_pilot import OUT, PRIVATE, ROOT, ROOTS, dump, sha

RUN = PRIVATE / 'registration-qc-fundus-frame-20261102-v2'
SIZE = 512
BOXES = {'metric': (183, 183, 329, 329), 'arcade_sup': (208, 72, 304, 168), 'arcade_inf': (208, 344, 304, 440)}
COLOURS = {'metric': (255, 180, 0), 'arcade_sup': (0, 200, 255), 'arcade_inf': (255, 0, 200)}


def frame_transform(disc, fovea) -> SimilarityTransform:
    """Canonical -> source similarity, identical to ladder2d_register.fundus_canonical."""
    targets = (R.DISC_R, R.FOVEA_R) if disc[0] > fovea[0] else (R.FOVEA_R, R.DISC_R)
    return SimilarityTransform(matrix=M.similarity_from_two(targets, [disc, fovea]).M)


def flip_x(xy):
    xy = np.array(xy, float)
    xy[..., 0] = SIZE - 1 - xy[..., 0]
    return xy


def box_to_source(box, t, flip_canonical=False, flip_source=False):
    """Box corners in the (possibly reflected) canonical frame -> displayed source coordinates."""
    x0, y0, x1, y1 = box
    c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    if flip_canonical:
        c = flip_x(c)
    s = t(c)
    return flip_x(s) if flip_source else s


def prepare(path, reflected_input=False, fellow=False):
    """Scoring-chain load; a reflected input is explicitly reflected back before landmarks."""
    rgb = L.load_base(str(path), SIZE, True)
    if reflected_input:
        rgb = rgb[:, ::-1].copy()  # simulated reflected input
        rgb = rgb[:, ::-1].copy()  # explicit inverse reflection, the step under test
    working = rgb[:, ::-1].copy() if fellow else rgb
    result = M.g1(working)
    if not result.ok:
        return rgb, None, None, None
    lm = R.fundus_landmarks(working)
    mapping = result.phi.M
    if fellow:
        reflection = np.array([[-1., 0., SIZE - 1], [0., 1., 0.], [0., 0., 1.]])
        mapping = reflection @ mapping  # output -> original unreflected source
        lm = tuple(flip_x(point) for point in lm)
    return rgb, lm, SimilarityTransform(matrix=mapping), result.out


def vessels(rgb):
    """Green-channel black top-hat, top 8% inside the field of view; inspection aid only."""
    g = ndimage.gaussian_filter(rgb[..., 1], 1.0)
    th = ndimage.grey_closing(g, size=(15, 15)) - g
    fov = rgb[..., 0] > 0.1
    if not fov.any():
        return np.zeros(g.shape, bool)
    return (th > np.percentile(th[fov], 92)) & fov


def overlay(a, b):
    va, vb = vessels(a), vessels(b)
    out = np.zeros(a.shape[:2] + (3,), np.uint8)
    out[va, 0] = 255
    out[vb, 1] = 255
    return out


def support(rgb, box):
    x0, y0, x1, y1 = box
    return float((rgb[y0:y1, x0:x1, 0] > 0.1).mean())


def u8(a):
    return (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.dtype != np.uint8 else a


def draw(a, polys=(), points=()):
    im = Image.fromarray(u8(a)).convert('RGB')
    d = ImageDraw.Draw(im)
    for poly, colour in polys:
        d.polygon([tuple(p) for p in np.rint(poly)], outline=colour, width=2)
    for (x, y), kind in points:
        if kind == 'disc':
            d.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 255, 255), width=2)
        else:
            d.line([x - 10, y, x + 10, y], fill=(0, 255, 0), width=2)
            d.line([x, y - 10, x, y + 10], fill=(0, 255, 0), width=2)
    return np.asarray(im)


def sheet(cells, path):
    """Rows of four (image, title) cells, each displayed at 512."""
    rows = [cells[i:i + 4] for i in range(0, len(cells), 4)]
    out = Image.new('RGB', (4 * 512, len(rows) * 550), 'white')
    d = ImageDraw.Draw(out)
    for r, row in enumerate(rows):
        for c, (a, title) in enumerate(row):
            out.paste(Image.fromarray(u8(a)).resize((512, 512), Image.NEAREST), (c * 512, r * 550 + 30))
            d.text((c * 512 + 8, r * 550 + 8), title, fill='black')
    out.save(path)


def black():
    return np.zeros((SIZE, SIZE, 3), np.uint8)


def review_row(row):
    p, mode = row['pair'], row['mode']
    fellow, reflected = mode == 'fellow_mirror', mode == 'reflected_input'
    pa, pb = ROOTS['GRAPE'] / p['anchor'], ROOTS['GRAPE'] / p['candidate']
    a, lma, ta, ca = prepare(pa)
    b, lmb, tb, cb = prepare(pb, reflected_input=reflected, fellow=fellow)
    rec = {'mode': mode, 'source_sha256': [sha(pa), sha(pb)], 'landmarks_found': [lma is not None, lmb is not None]}
    shown_b = b[:, ::-1] if reflected else b  # moving image as it was supplied
    pts_a = [] if lma is None else [(lma[0], 'disc'), (lma[1], 'fovea')]
    pts_b = [] if lmb is None else [((flip_x(lmb[0]) if reflected else lmb[0]), 'disc'),
                                    ((flip_x(lmb[1]) if reflected else lmb[1]), 'fovea')]
    poly_a = [] if ta is None else [(box_to_source(bx, ta), COLOURS[k]) for k, bx in BOXES.items()]
    poly_b = [] if tb is None else [(box_to_source(bx, tb, False, reflected), COLOURS[k]) for k, bx in BOXES.items()]
    canon_polys = [(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), COLOURS[k])
                   for k, (x0, y0, x1, y1) in BOXES.items()]
    def canonical_points(lm, t, flip):
        xy = t.inverse(np.array(lm))
        xy = flip_x(xy) if flip else xy
        return [(xy[0], 'disc'), (xy[1], 'fovea')]

    cells = [(draw(a, poly_a, pts_a), 'Anchor source; disc o, fovea +; boxes mapped back'),
             (draw(shown_b, poly_b, pts_b), f'Moving source as supplied ({mode})'),
             (draw(ca, canon_polys, canonical_points(lma, ta, False)) if ca is not None else black(),
              'Anchor canonical frame'),
             (draw(cb, canon_polys, canonical_points(lmb, tb, False)) if cb is not None else black(),
              'Moving G1 frame' + (' (raw reflected before G1)' if fellow else ''))]
    rec['transforms_canonical_to_source'] = [None if t is None else t.params.tolist() for t in (ta, tb)]
    rec['box_support'] = {}
    for k, (x0, y0, x1, y1) in BOXES.items():
        if ca is None or cb is None:
            cells += [(black(), f'{k}: landmarks missing')] * 4
            continue
        za, zb = ca[y0:y1, x0:x1], cb[y0:y1, x0:x1]
        yy, xx = np.indices(za.shape[:2])
        check = ((xx // 16 + yy // 16) % 2)[..., None].astype(bool)
        cells += [(za, f'{k}: anchor crop'), (zb, f'{k}: moving crop'),
                  (overlay(za, zb), f'{k}: vessels anchor red / moving green / overlap yellow'),
                  (np.where(check, za, zb), f'{k}: checkerboard')]
        rec['box_support'][k] = [support(ca, BOXES[k]), support(cb, BOXES[k])]
    rec['status'] = 'prepared' if all(rec['landmarks_found']) else 'landmarks_missing'
    return rec, cells


def main():
    RUN.mkdir(exist_ok=False)  # never overwrite an earlier run
    manifest_path = OUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    amendment = ROOT / 'docs/PROTOCOL_registration.md'
    dump(RUN / 'plan.json', {'parent_manifest_sha256': sha(manifest_path), 'code_sha256': sha(__file__),
                             'ladder2d_register_sha256': sha(R.__file__), 'reg_methods_sha256': sha(M.__file__),
                             'amendment_sha256': sha(amendment),
                             'boxes_canonical_512': BOXES, 'pilot': 'first saved GRAPE pair per stratum'})
    records = []
    for index in manifest['pilot_indices']:
        row = manifest['rows'][index]
        if row['pair']['dataset'] != 'GRAPE':
            continue
        rec, cells = review_row(row)
        name = f'frame_{len(records):02d}.png'
        sheet(cells, RUN / name)
        rec.update({'index': index, 'stratum': row['stratum'], 'sheet': name, 'sheet_sha256': sha(RUN / name),
                    'visual': 'not_reviewed', 'agent_review': None, 'kate_review': None})
        records.append(rec)
        dump(RUN / 'records.json', records)
    shutil.copy2(__file__, RUN / 'code_snapshot.py')
    html = ['<!doctype html><meta charset="utf-8"><title>GRAPE disc-fovea frame QC</title>',
            '<style>body{font:18px sans-serif;background:#eee;margin:30px}img{width:100%}section{margin-bottom:50px}</style>',
            '<h1>GRAPE disc-fovea frame QC: development pilot</h1>',
            '<p>Gate NOT MET. Check landmarks first: disc o inside the rim, fovea + within about 30 px of the '
            'visible foveal centre. Boxes: orange metric, cyan superior arcade, magenta inferior arcade. '
            'Vessel overlay is an aid, not a landmark. No visual decision is assigned automatically.</p>']
    for r in records:
        html.append(f'<section><h2>{r["stratum"]}</h2><p>{r["status"]}; visual: NOT REVIEWED</p>'
                    f'<img src="{r["sheet"]}"></section>')
    (RUN / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    strata = Counter(r['stratum'] for r in manifest['rows'] if r['pair']['dataset'] == 'GRAPE')
    aggregate = {'gate': 'NOT MET', 'visual_acceptance': 'NOT ESTABLISHED', 'method': 'G1 shared reg_methods.g1, raw reflection',
                 'manifest_sha256': sha(manifest_path), 'pilot_preparation': dict(Counter(r['status'] for r in records)),
                 'strata': [{'stratum': k, 'intended': n, 'reviewed': 0, 'pass': 0, 'fail': 0, 'unreviewable': 0,
                             'not_reviewed': n} for k, n in strata.items()]}
    dump(RUN / 'aggregate.json', aggregate)
    dump(ROOT / 'results/registration/qc/fundus_frame_v2.json', aggregate)
    print(json.dumps({'pilot': len(records), 'preparation': aggregate['pilot_preparation']}))


if __name__ == '__main__':
    main()
