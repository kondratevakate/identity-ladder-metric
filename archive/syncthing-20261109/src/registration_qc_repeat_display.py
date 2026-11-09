"""Supplement frozen MRI repeat QC sheets with a shared brightness window."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from registration_qc_mri_validation import OUT, PRIVATE, TEMPLATE, dump, sha


def main():
    plan = json.loads((OUT / 'plan.json').read_text())
    progress = json.loads((OUT / 'progress.json').read_text())
    rows = json.loads((OUT / 'review_manifest.json').read_text())['rows']
    repeats = [row for row in rows if row['branch'] == 'repeat' and row['availability'] == 'required']
    if len(repeats) != 46:
        raise RuntimeError('Frozen repeat denominator differs from 46')
    destination = OUT / 'repeat_shared_window_cohort_v1'
    destination.mkdir(exist_ok=False)
    reference = nib.load(TEMPLATE)
    mask_path = PRIVATE / 'real-cases-run/template/brain_1mm.nii.gz'
    brain = nib.load(mask_path).get_fdata() > .5
    jobs = {(job['person'], job['session']): job for job in plan['jobs']}
    dump(destination / 'plan.json', {
        'code_sha256': sha(__file__), 'manifest_sha256': sha(OUT / 'review_manifest.json'),
        'brain_mask_sha256': sha(mask_path), 'required_comparisons': len(repeats),
        'window': 'Shared zero to max of both registered scans’ template-brain P99 values',
        'scope': 'Display supplement using the same saved volumes and crop coordinates',
    })
    records = []
    for person in dict.fromkeys(row['target'] for row in repeats):
        pair, digests = [], []
        for visit in ('ses-01', 'ses-02'):
            job = jobs[(person, visit)]
            status = progress['scans'][str(job['index'])]
            path = Path(status['folder']) / 'moving_to_fixed.nii.gz'
            digest = sha(path)
            image = nib.load(path)
            if digest != status['outputs'][path.name] or image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
                raise RuntimeError('Saved repeat volume or grid mismatch')
            pair.append(image.get_fdata(dtype=np.float32))
            digests.append(digest)
        baseline_p99, repeat_p99 = (float(np.percentile(volume[brain], 99)) for volume in pair)
        vmax = max(baseline_p99, repeat_p99)
        for row in (item for item in repeats if item['target'] == person):
            index = row['row']
            original = OUT / 'sheets' / f'row_{index:03d}.png'
            geometry = json.loads(original.with_suffix('.json').read_text())
            crop = tuple(slice(*limits) for limits in geometry['template_crop_slices'])
            first, second = (volume[crop] for volume in pair)
            fig, axes = plt.subplots(3, 4, figsize=(10, 7))
            for panels, axis in zip(axes, (2, 1, 0)):
                a, b = (np.take(volume, volume.shape[axis] // 2, axis=axis).T for volume in (first, second))
                y, x = np.indices(a.shape)
                checker = np.where((x // 4 + y // 4) % 2, a, b)
                for ax, data, title in zip(panels, (a, b, (a + b) / 2, checker),
                                           ('Baseline', 'Repeat', 'Blend', 'Checkerboard')):
                    ax.imshow(data, cmap='gray', vmin=0, vmax=vmax, origin='lower', interpolation='nearest')
                    ax.set_title(title)
                    ax.axis('off')
            fig.suptitle(f'{row["box"]} repeat | shared brightness window')
            fig.tight_layout()
            sheet = destination / original.name
            fig.savefig(sheet, dpi=100)
            plt.close(fig)
            records.append({
                'row': index, 'box': row['box'], 'input_sha256': digests,
                'original_sheet_sha256': sha(original), 'supplement_sha256': sha(sheet),
                'window_max': vmax, 'crop_voxels': int(second.size),
                'repeat_voxels_above_original_window': int(np.count_nonzero(second > baseline_p99)),
                'repeat_voxels_above_shared_window': int(np.count_nonzero(second > vmax)),
            })
    dump(destination / 'complete.json', {'prepared': len(records), 'records': records})
    print(f'Prepared {len(records)}/{len(repeats)} repeat supplements')


if __name__ == '__main__':
    main()
