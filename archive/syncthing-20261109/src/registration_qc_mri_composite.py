"""QC-only audit of the saved MRI pull transform, including its affine component.

No registration optimisation, intensity/identity scores or automatic visual acceptance.
Physical coordinate images provide an independent check of the saved image chain.
"""
import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', '2')
import json
import shutil
from pathlib import Path
import numpy as np
from scipy import ndimage
from registration_qc_pilot import PRIVATE, ROOT, sha, dump


def physical_grid(shape, origin, spacing, direction):
    indices = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing='ij'), -1)
    basis = np.asarray(direction) @ np.diag(spacing)
    return (indices @ basis.T + origin).astype(np.float32)


def pull_determinant(coordinates, spacing, direction):
    """det(d moving-physical / d fixed-physical); both use the same LPS basis."""
    derivative = np.stack([np.stack(np.gradient(coordinates[..., k]), -1) for k in range(3)], -2)
    basis = np.asarray(direction) @ np.diag(spacing)
    return np.linalg.det(derivative @ np.linalg.inv(basis))


def reflected_slice(start, stop, length):
    """Discrete half-open reflection, preserving exactly the same voxel count."""
    return [length - stop, length - start]


def main():
    import ants
    from registration_qc_mri import BOXES
    run = PRIVATE / 'real-cases-run'
    source = sorted((run / 'syn_amendment4').glob('scan_*/complete.json'))[0].parent
    repaired = PRIVATE / 'registration-qc-mri-refined-20261102-v1'
    out = PRIVATE / 'registration-qc-mri-composite-20261103-v1'
    meta = json.loads((repaired / 'complete.json').read_text())
    for name, digest in meta['outputs'].items():
        if sha(repaired / name) != digest:
            raise RuntimeError('Saved repair output hash drift')
    transforms = [repaired / 'syn_1Warp.nii.gz', repaired / 'syn_0GenericAffine.mat']
    paths = [source / 'n4.nii.gz', source / 'mask.nii.gz', run / 'template/template_1mm.nii.gz',
             run / 'template/brain_1mm.nii.gz', repaired / 'syn.nii.gz', *transforms]
    out.mkdir(exist_ok=False)
    dump(out / 'plan.json', {'scope': 'one existing development scan; QC only',
        'code_sha256': sha(__file__), 'source_sha256': {str(p): sha(p) for p in paths},
        'transform_order': [p.name for p in transforms], 'invert_flags': [False, False],
        'definitions': {'jacobian': 'det of full fixed-LPS to native-LPS pull map; positive required',
                        'roundtrip': 'native coordinate field sampled back with saved inverse transform'},
        'visual_acceptance': 'not assigned', 'erosion_voxels': 2})
    shutil.copy2(__file__, out / 'code_snapshot.py')
    native = ants.image_read(str(source / 'n4.nii.gz'))
    fixed = ants.image_read(str(paths[2]))
    brain = ants.image_read(str(paths[3])).numpy() > .5
    fwd = [str(p) for p in transforms]
    repeated = ants.apply_transforms(fixed, native, fwd).numpy()
    saved = ants.image_read(str(repaired / 'syn.nii.gz')).numpy()
    difference = np.abs(repeated - saved)
    if not np.allclose(repeated, saved, rtol=1e-5, atol=1e-4):
        raise RuntimeError('Proposed full transform chain does not reproduce saved image')
    xyz = physical_grid(native.shape, native.origin, native.spacing, native.direction)
    coords = []
    for k in range(3):
        coordinate_image = native.new_image_like(xyz[..., k])
        coords.append(ants.apply_transforms(fixed, coordinate_image, fwd).numpy())
    pull = np.stack(coords, -1)
    ones = native.new_image_like(np.ones(native.shape, np.float32))
    support = ants.apply_transforms(fixed, ones, fwd).numpy() > .999
    valid = ndimage.binary_erosion(support, iterations=2) & brain
    determinant = pull_determinant(pull, fixed.spacing, fixed.direction)
    ants.image_write(fixed.new_image_like(determinant.astype(np.float32)), str(out / 'full_pull_jacobian.nii.gz'))
    # Point transforms operate in physical LPS. The inverse image transform stack
    # returns native coordinates to fixed space (the reverse of the pull field).
    import pandas as pd
    sample_indices = np.argwhere(valid)[::500]
    points = pull[tuple(sample_indices.T)]
    inverse = [str(repaired / 'syn_0GenericAffine.mat'), str(repaired / 'syn_1InverseWarp.nii.gz')]
    restored = ants.apply_transforms_to_points(3, pd.DataFrame(points, columns=['x', 'y', 'z']),
                                              inverse, whichtoinvert=[True, False])
    expected = (sample_indices @ (fixed.direction @ np.diag(fixed.spacing)).T + fixed.origin)
    errors = np.linalg.norm(restored[['x', 'y', 'z']].to_numpy() - expected, axis=1)
    # Preserve this as a diagnostic even if inverse consistency fails; no pass is inferred.
    import nibabel as nib
    nii = nib.load(str(paths[2]))
    if not np.allclose(nii.affine[:3, :3], np.eye(3)):
        raise RuntimeError('Expected saved 1mm RAS grid')
    boxes = {}
    for name, bounds in BOXES.items():
        slices = []
        for axis, (low, high) in enumerate(bounds):
            indices = np.flatnonzero((np.arange(nii.shape[axis]) + nii.affine[axis, 3] >= low) &
                                     (np.arange(nii.shape[axis]) + nii.affine[axis, 3] < high))
            slices.append([int(indices[0]), int(indices[-1]) + 1])
        region = tuple(slice(*v) for v in slices)
        selected = determinant[region][valid[region]]
        boxes[name] = {'voxel_slices': slices, 'reflected_source_x_slice': reflected_slice(*slices[0], nii.shape[0]),
                       'intended_voxels': int(np.prod([b-a for a,b in slices])),
                       'valid_brain_voxels': int(selected.size),
                       'nonpositive': int((selected <= 0).sum()),
                       'jacobian_percentiles': np.percentile(selected, [0, 1, 50, 99, 100]).tolist() if selected.size else None}
    report = {'saved_image_reproduced': True, 'max_absolute_resampling_difference': float(difference.max()),
              'valid_brain_voxels': int(valid.sum()), 'full_pull_nonpositive': int((determinant[valid] <= 0).sum()),
              'full_pull_jacobian_percentiles': np.percentile(determinant[valid], [0, 1, 50, 99, 100]).tolist(),
              'roundtrip_points': len(points), 'roundtrip_error_mm_percentiles': np.percentile(errors, [0, 50, 95, 100]).tolist(),
              'boxes': boxes, 'visual_acceptance': 'NOT ESTABLISHED', 'gate': 'NOT MET',
              'limits': 'One development scan. Positive Jacobians and inverse consistency do not prove preservation of individual anatomy.'}
    dump(out / 'audit.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
