"""Run a paired, exploratory HACA3+ QC pilot on saved affine volumes."""
import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

from registration_qc_mri_validation import OUT, PRIVATE, ROOT, TEMPLATE, dump, sha


def main():
    release = ROOT / 'external/haca3-plus-4ec722b8'
    models = {}
    for name in ('haca3-plus_model_38538dab', 'fusion_afafc572'):
        pointer = release / 'models' / f'{name}.pt'
        digest = pointer.read_text().split('sha256:')[1].split()[0]
        weight = pointer.with_suffix('.verified.pt')
        if sha(weight) != digest:
            raise RuntimeError('Upstream weight digest mismatch')
        models[name] = weight
    plan = json.loads((OUT / 'plan.json').read_text())
    progress = json.loads((OUT / 'progress.json').read_text())
    rows = json.loads((OUT / 'review_manifest.json').read_text())['rows']
    person = next(row['target'] for row in rows if row['row'] == 374)
    jobs = sorted((job for job in plan['jobs'] if job['person'] == person),
                  key=lambda job: job['session'])
    if [job['session'] for job in jobs] != ['ses-01', 'ses-02']:
        raise RuntimeError('Expected the original development repeat pair')
    target_shape = np.array([192, 224, 192])
    reference = nib.load(TEMPLATE)
    offset = (np.array(reference.shape) - target_shape) // 2
    grid = reference.affine.copy()
    grid[:3, 3] = nib.affines.apply_affine(reference.affine, offset)
    mask_path = PRIVATE / 'real-cases-run/template/brain_1mm.nii.gz'
    brain = nib.load(mask_path).get_fdata() > .5
    out = OUT / 'haca3plus_repeat_pilot_v1'
    resume = sys.argv[1:] == ['--resume']
    if resume:
        if not (out / 'plan.json').exists():
            raise RuntimeError('No prepared pilot to resume')
    else:
        out.mkdir(exist_ok=False)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    sources = []
    for job in ([] if resume else jobs):
        status = progress['scans'][str(job['index'])]
        path = Path(status['folder']) / 'moving_to_fixed.nii.gz'
        if sha(path) != status['outputs'][path.name]:
            raise RuntimeError('Saved affine volume digest mismatch')
        image = nib.load(path)
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
            raise RuntimeError('Unexpected saved affine grid')
        data = image.get_fdata(dtype=np.float32)
        usable = brain & np.isfinite(data) & (data > 0)
        if not np.isfinite(data).all() or not usable.any():
            raise RuntimeError('Invalid N4 input')
        itk = sitk.GetImageFromArray(data.T.copy())
        itk_mask = sitk.GetImageFromArray(usable.T.astype(np.uint8))
        n4 = sitk.N4BiasFieldCorrectionImageFilter()
        n4.Execute(itk, itk_mask)
        corrected = data / np.exp(sitk.GetArrayFromImage(n4.GetLogBiasFieldAsImage(itk)).T)
        padded = np.zeros(tuple(target_shape), dtype=np.float32)
        lo = np.maximum(offset, 0)
        hi = np.minimum(offset + target_shape, data.shape)
        original = tuple(slice(l, h) for l, h in zip(lo, hi))
        adapted = tuple(slice(l, h) for l, h in zip(lo - offset, hi - offset))
        if np.count_nonzero(brain[original]) != np.count_nonzero(brain):
            raise RuntimeError('Model input grid would crop template brain support')
        padded[adapted] = corrected[original]
        prepared = out / f'{job["session"]}_n4_model_grid.nii.gz'
        nib.save(nib.Nifti1Image(padded, grid), prepared)
        sources.append({'session': job['session'], 'input_sha256': sha(path),
                        'prepared_sha256': sha(prepared), 'path': str(prepared)})
    if resume:
        sources = json.loads((out / 'plan.json').read_text())['source_inputs']
        for source in sources:
            if sha(Path(source['path'])) != source['prepared_sha256']:
                raise RuntimeError('Prepared input digest mismatch')
    else:
        dump(out / 'plan.json', {
        'scope': 'Exploratory development repeat pair; QC only',
        'upstream_commit': '4ec722b8b09523cfe94df8c9e9cd6362c6625c3c',
        'code_sha256': sha(__file__), 'protocol_sha256': sha(ROOT / 'docs/PROTOCOL_registration.md'),
        'weights_sha256': {key: sha(value) for key, value in models.items()},
        'source_inputs': sources, 'template_brain_mask_sha256': sha(mask_path),
        'preprocessing': 'SimpleITK default N4 on saved affine 1 mm symmetric grid using positive template-brain voxels, then centered integer crop/pad',
        'model_grid_affine': grid.tolist(), 'target': 'Same baseline T1w appearance for both inputs',
        'runtime': {'torch': torch.__version__, 'simpleitk': sitk.Version_VersionString(),
                    'device': 'CPU', 'threads': 2, 'batches_per_orientation': 56,
                    'background_removal': False},
    })
    env = dict(os.environ, PYTHONPATH=str(release), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
    for source in sources:
        name = source['session']
        output_prefix = out / f'{name}_harmonized.nii.gz'
        result = out / f'{name}_harmonized_harmonized_fusion.nii.gz'
        cmd = [sys.executable, '-u', '-m', 'haca3_plus.test',
               '--in-path', source['path'], '--target-image', sources[0]['path'],
               '--out-path', str(output_prefix), '--harmonization-model',
               str(models['haca3-plus_model_38538dab']), '--fusion-model',
               str(models['fusion_afafc572']), '--num-batches', '56', '--no-bg-removal']
        if not result.exists():
            log_path = out / (f'{name}_resume.log' if resume else f'{name}.log')
            with log_path.open('x', encoding='utf-8') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=2700)
        harmonized = nib.load(result)
        if harmonized.shape != tuple(target_shape) or not np.allclose(harmonized.affine, grid):
            raise RuntimeError('Harmonized output grid mismatch')
        if not np.isfinite(harmonized.get_fdata()).all():
            raise RuntimeError('Nonfinite harmonized output')
        print(f'{name} completed', flush=True)
    dump(out / 'complete.json', {'outputs_sha256': {
        path.name: sha(path) for path in out.glob('*_harmonized_fusion.nii.gz')},
        'completion_code_sha256': sha(__file__),
        'gate': 'NOT MET', 'visual_review': 'NOT REVIEWED'})


if __name__ == '__main__':
    main()
