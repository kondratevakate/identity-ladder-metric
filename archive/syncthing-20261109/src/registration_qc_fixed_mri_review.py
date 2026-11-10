"""Render frozen MRI registration outputs without fitting or identity scoring."""
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np

from registration_qc_mri import BOXES
from registration_qc_pilot import PRIVATE, ROOT, dump, sha


def main():
    run = PRIVATE / 'registration-qc-fixed-mri-20261103-v1'
    joint = PRIVATE / 'registration-qc-synthmorph-20261103-v3-raw192'
    complete = json.loads((run / 'complete.json').read_text())
    for method, info in complete.items():
        if info['exit_code'] != 0:
            raise RuntimeError(f'{method}: runtime failure; preserve and inspect before rendering')
        for name, digest in info['outputs'].items():
            if sha(run / method / name) != digest:
                raise RuntimeError('Output drift')
    for name, digest in json.loads((joint / 'complete.json').read_text())['outputs'].items():
        if sha(joint / name) != digest:
            raise RuntimeError('Joint output drift')
    paths = {'template': PRIVATE / 'real-cases-run/template/template_1mm.nii.gz',
        'brain': PRIVATE / 'real-cases-run/template/brain_1mm.nii.gz',
        'rigid': run / 'rigid/moving_to_fixed.nii.gz',
        'affine': run / 'affine/moving_to_fixed.nii.gz',
        'joint': joint / 'moving_to_fixed.nii.gz',
        'syn': PRIVATE / 'registration-qc-mri-refined-20261102-v1/syn.nii.gz'}
    out = run / 'anatomy'
    out.mkdir(exist_ok=False)
    dump(out / 'plan.json', {'code_sha256': sha(__file__),
        'protocol_sha256': sha(ROOT / 'docs/PROTOCOL_registration.md'),
        'input_sha256': {k: sha(p) for k, p in paths.items()},
        'scope': 'existing development outputs only; no fitting or anatomical acceptance',
        'display': 'each volume fixed 0 to own brain P99, all slices; no histogram matching',
        'coverage_limit': 'nonzero intensity fraction is NOT transformed source support',
        'boxes_mm': BOXES})
    shutil.copy2(__file__, out / 'code_snapshot.py')
    images = {k: nib.load(p) for k, p in paths.items()}
    ref = images['template']
    if not np.allclose(ref.affine[:3, :3], np.eye(3)):
        raise RuntimeError('Expected 1mm RAS grid')
    if not all(x.shape == ref.shape and np.allclose(x.affine, ref.affine) for x in images.values()):
        raise RuntimeError('Output grid mismatch')
    arrays = {k: im.get_fdata(dtype=np.float32) for k, im in images.items()}
    brain = arrays.pop('brain') > .5
    diagnostics = {}
    display = {}
    for key, data in arrays.items():
        if not np.isfinite(data).all():
            raise RuntimeError('Nonfinite output')
        p99 = float(np.percentile(data[brain], 99))
        if p99 <= 0:
            raise RuntimeError('Empty brain intensity')
        diagnostics[key] = {'brain_p99': p99,
            'brain_nonzero_fraction': float(np.count_nonzero(data[brain]) / brain.sum()),
            'meaning': 'intensity diagnostic, not anatomical accuracy or source coverage'}
        display[key] = data / p99
    records = []
    for box, bounds in BOXES.items():
        indices = [np.flatnonzero((np.arange(ref.shape[d]) + ref.affine[d, 3] >= lo) &
            (np.arange(ref.shape[d]) + ref.affine[d, 3] < hi)) for d, (lo, hi) in enumerate(bounds)]
        slices = [slice(int(v[0]), int(v[-1]) + 1) for v in indices]
        fig, axes = plt.subplots(3, 8, figsize=(24, 10))
        for row, axis in enumerate([2, 1, 0]):
            at = int(indices[axis][len(indices[axis]) // 2])
            other = [d for d in range(3) if d != axis]
            cut = lambda data: np.take(data, at, axis=axis).T
            crop = lambda data: cut(data)[slices[other[1]], slices[other[0]]]
            a, j = crop(display['affine']), crop(display['joint'])
            yy, xx = np.indices(j.shape)
            panels = [cut(display[k]) for k in ['template', 'rigid', 'affine', 'joint']]
            panels += [crop(display['syn']), a, j, np.where((xx // 4 + yy // 4) % 2, a, j)]
            titles = ['Template', 'SynthMorph rigid', 'SynthMorph affine', 'SynthMorph joint (192)',
                'Prior SyN crop', 'Affine crop', 'Joint crop', 'Affine / joint checker']
            for col, (panel, title) in enumerate(zip(panels, titles)):
                ax = axes[row, col]
                ax.imshow(panel, cmap='gray', origin='lower', vmin=0, vmax=1, interpolation='nearest')
                ax.set_title(title, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                if col < 4:
                    x, y = slices[other[0]], slices[other[1]]
                    ax.add_patch(Rectangle((x.start - .5, y.start - .5), x.stop-x.start,
                        y.stop-y.start, fill=False, ec='orange', lw=1))
            axes[row, 0].set_ylabel(f'{["sagittal", "coronal", "axial"][axis]} {at + ref.affine[axis, 3]:.0f}mm')
        fig.suptitle(f'{box}: one exposed T1w pair; per-volume P99; anatomical acceptance NOT ESTABLISHED')
        fig.tight_layout()
        path = out / f'{box}.png'
        fig.savefig(path, dpi=110); plt.close(fig)
        records.append({'box': box, 'sheet': path.name, 'sha256': sha(path), 'visual': 'NOT REVIEWED'})
    dump(out / 'records.json', {'sheets': records, 'diagnostics': diagnostics})
    dump(ROOT / 'results/registration/qc/fixed_mri_anatomy_v1.json', {
        'development_pairs': 1, 'sheets_prepared': len(records),
        'visual_acceptance': 'NOT ESTABLISHED', 'gate': 'NOT MET',
        'metric_scoring': 'NOT RUN', 'diagnostics': diagnostics})
    print('Two frozen MRI QC sheets rendered; no automatic anatomical pass.')


if __name__ == '__main__':
    main()
