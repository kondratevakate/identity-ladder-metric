"""Independent RAS affine inverse and field-of-view checks; no anatomical score."""
import json
import shutil
import nibabel as nib
import numpy as np
from registration_qc_pilot import PRIVATE, ROOT, dump, sha


def matrix(path):
    lines = path.read_text().splitlines()
    if not lines[0].startswith('type      = 1'):
        raise RuntimeError('Only LINEAR_RAS_TO_RAS is supported')
    start = lines.index('1 4 4') + 1
    return np.array([[float(x) for x in line.split()] for line in lines[start:start+4]])


def main():
    run = PRIVATE / 'registration-qc-fixed-mri-20261103-v1'
    plan = json.loads((run / 'plan.json').read_text())
    cmd = plan['commands']['rigid']
    moving = next(cmd[i+1].split(':/moving/')[0] for i,v in enumerate(cmd)
                  if v == '-v' and ':/moving/' in cmd[i+1])
    fixed = PRIVATE / 'real-cases-run/template/template_1mm.nii.gz'
    brain = PRIVATE / 'real-cases-run/template/brain_1mm.nii.gz'
    out = run / 'geometry_v2'
    out.mkdir(exist_ok=False)
    dump(out / 'plan.json', {'code_sha256': sha(__file__),
        'inputs_sha256': {'moving': sha(moving), 'fixed': sha(fixed), 'brain': sha(brain)},
        'scope': 'saved RAS matrices only, no fitting; support and inversion are not anatomy acceptance'})
    shutil.copy2(__file__, out / 'code_snapshot.py')
    source, target = nib.load(moving), nib.load(fixed)
    mask = nib.load(brain).get_fdata() > .5
    voxels = np.vstack([np.array(np.where(mask)), np.ones(int(mask.sum()))])
    rows = {}
    for method in ['rigid', 'affine']:
        fwd = run / method / 'forward.lta'
        inv = run / method / 'inverse.lta'
        a, b = matrix(fwd), matrix(inv)
        pull = np.linalg.inv(source.affine) @ b @ target.affine
        coordinates = (pull @ voxels)[:3]
        valid = np.all((coordinates >= 0) & (coordinates <= np.array(source.shape)[:,None]-1), axis=0)
        rows[method] = {
            'transform_sha256': {'forward': sha(fwd), 'inverse': sha(inv)},
            'forward_ras_determinant': float(np.linalg.det(a[:3,:3])),
            'physical_pull_determinant': float(np.linalg.det(b[:3,:3])),
            'inverse_linear_composition_max_abs_unitless': float(np.abs((a @ b)[:3,:3] - np.eye(3)).max()),
            'inverse_translation_composition_norm_mm': float(np.linalg.norm((a @ b)[:3,3])),
            'template_brain_voxels': int(mask.sum()), 'inside_native_fov': int(valid.sum()),
            'native_fov_support_fraction': float(valid.mean()),
            'definitions': 'raw, uncalibrated; determinant positive excludes reflection/collapse; inverse error lower better; support higher better',
            'visual_acceptance': 'NOT ESTABLISHED'}
    dump(out / 'complete.json', rows)
    dump(ROOT / 'results/registration/qc/fixed_mri_geometry_v2.json', rows)
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
