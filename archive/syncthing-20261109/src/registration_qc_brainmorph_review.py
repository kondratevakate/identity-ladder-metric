"""Recover official fixed-grid geometry, verify saved input, render BrainMorph QC."""
import json
import shutil
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import torchio as tio
from keymorph.utils import rescale_intensity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from registration_qc_pilot import PRIVATE, ROOT, dump, sha
from registration_qc_mri import BOXES


def main():
    run = PRIVATE / 'registration-qc-brainmorph-20261103-v3'
    complete = json.loads((run / 'complete.json').read_text())
    if complete['exit_code'] != 0:
        raise RuntimeError('Registration did not complete')
    for name, digest in complete['output_files'].items():
        if sha(run / name) != digest:
            raise RuntimeError('Saved output drift')
    fixed = PRIVATE / 'real-cases-run/template/template_1mm.nii.gz'
    aligned_path, = (run / 'outputs').rglob('img_a*affine.npy')
    fixed_path, = (run / 'outputs').rglob('img_f*.npy')
    out = run / 'anatomy'
    out.mkdir(exist_ok=False)
    dump(out / 'plan.json', {'code_sha256': sha(__file__),
        'protocol_sha256': sha(ROOT / 'docs/PROTOCOL_registration.md'),
        'saved_aligned_sha256': sha(aligned_path), 'saved_fixed_sha256': sha(fixed_path),
        'scope': 'reconstruct deterministic fixed preprocessing only, no model fitting or scoring',
        'display': 'each volume own brain P99; template-grid linear display resampling recorded',
        'grid_check': 'saved fixed array must match replayed official preprocessing within 2e-6'})
    shutil.copy2(__file__, out / 'code_snapshot.py')
    transform = tio.Compose([tio.ToCanonical(), tio.Resample(1), tio.Resample('img'),
        tio.CropOrPad((256,256,256), padding_mode=0, include=('img',)),
        tio.CropOrPad((256,256,256), padding_mode=0, include=('seg',)),
        tio.Lambda(rescale_intensity, include=('img',))], include=('img','seg'))
    processed = transform(tio.Subject(img=tio.ScalarImage(fixed)))['img']
    expected = processed.data.numpy()
    actual = np.load(fixed_path, allow_pickle=False)
    error = float(np.max(np.abs(expected - actual)))
    if expected.shape != actual.shape or error > 2e-6:
        raise RuntimeError(f'Cannot establish fixed-grid geometry: replay error {error}')
    aligned = np.load(aligned_path, allow_pickle=False).squeeze()
    if aligned.shape != tuple(processed.spatial_shape) or not np.isfinite(aligned).all():
        raise RuntimeError('Invalid aligned array')
    native = nib.Nifti1Image(aligned, processed.affine)
    nib.save(native, out / 'aligned_official_fixed_grid.nii.gz')
    ref = nib.load(fixed)
    mapped = resample_from_to(native, ref, order=1)
    nib.save(mapped, out / 'aligned_template_grid_display.nii.gz')
    arrays = {'Template': ref.get_fdata(dtype=np.float32),
        'BrainMorph affine': mapped.get_fdata(dtype=np.float32),
        'SynthMorph affine': nib.load(PRIVATE / 'registration-qc-fixed-mri-20261103-v1/affine/moving_to_fixed.nii.gz').get_fdata(dtype=np.float32)}
    brain = nib.load(PRIVATE / 'real-cases-run/template/brain_1mm.nii.gz').get_fdata() > .5
    arrays = {k: v / np.percentile(v[brain], 99) for k,v in arrays.items()}
    records = []
    for box, bounds in BOXES.items():
        indices = [np.flatnonzero((np.arange(ref.shape[d])+ref.affine[d,3]>=lo) &
            (np.arange(ref.shape[d])+ref.affine[d,3]<hi)) for d,(lo,hi) in enumerate(bounds)]
        slices = [slice(int(v[0]),int(v[-1])+1) for v in indices]
        fig, axes = plt.subplots(3,6,figsize=(18,10))
        for row, axis in enumerate([2,1,0]):
            at = int(indices[axis][len(indices[axis])//2]); other = [d for d in range(3) if d != axis]
            for col,(label,data) in enumerate(arrays.items()):
                full = np.take(data,at,axis=axis).T
                crop = full[slices[other[1]],slices[other[0]]]
                for ax,panel,title in [(axes[row,col],full,label), (axes[row,col+3],crop,label+' crop')]:
                    ax.imshow(panel,cmap='gray',origin='lower',vmin=0,vmax=1,interpolation='nearest')
                    ax.set_title(title,fontsize=10); ax.set_xticks([]); ax.set_yticks([])
                x,y = slices[other[0]],slices[other[1]]
                axes[row,col].add_patch(Rectangle((x.start-.5,y.start-.5),x.stop-x.start,y.stop-y.start,
                    fill=False,ec='orange',lw=1))
        fig.suptitle(f'{box}: one exposed T1w pair; anatomical acceptance NOT ESTABLISHED')
        fig.tight_layout(); path = out / f'{box}.png'; fig.savefig(path,dpi=110); plt.close(fig)
        records.append({'box':box,'sha256':sha(path),'sheet':path.name,'visual':'NOT REVIEWED'})
    result = {'fixed_preprocessing_replay_max_abs':error, 'raw_intensity_scale':'min-max 0 to 1; implementation replay error lower better',
        'development_pairs':1,'sheets':records,'visual_acceptance':'NOT ESTABLISHED','gate':'NOT MET','metric_scoring':'NOT RUN'}
    dump(out / 'records.json',result)
    dump(ROOT / 'results/registration/qc/brainmorph_anatomy_v1.json',result)
    print('BrainMorph fixed preprocessing replay verified; two QC sheets rendered.')


if __name__ == '__main__':
    main()
