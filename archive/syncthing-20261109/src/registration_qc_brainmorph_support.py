"""Trace real source FOV through published BrainMorph crop and saved pull grid."""
import json
import shutil
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import torch
import torch.nn.functional as F
import torchio as tio
from registration_qc_pilot import PRIVATE, ROOT, dump, sha
from registration_qc_mri import BOXES


def main():
    run = PRIVATE / 'registration-qc-brainmorph-20261103-v3'
    plan = json.loads((run/'plan.json').read_text())
    cmd = plan['command']; moving = cmd[cmd.index('--moving')+1]
    grid_path, = (run/'outputs').rglob('grid_*affine.npy')
    fixed = PRIVATE/'real-cases-run/template/template_1mm.nii.gz'
    brain = PRIVATE/'real-cases-run/template/brain_1mm.nii.gz'
    out = run/'support'
    out.mkdir(exist_ok=False)
    dump(out/'plan.json', {'code_sha256':sha(__file__), 'grid_sha256':sha(grid_path),
        'moving_sha256':sha(moving),'fixed_sha256':sha(fixed),'brain_sha256':sha(brain),
        'scope':'saved geometry only, no model fitting; full interpolation-kernel support, no anatomical pass',
        'rule':'native ones mask, official geometric preprocessing, saved align_corners=False pull grid; require no border extrapolation',
        'reference':'official KeyMorph align_img uses border padding, which is not real source support'})
    shutil.copy2(__file__, out/'code_snapshot.py')
    torch.set_num_threads(2)
    native = tio.ScalarImage(moving)
    ones = tio.ScalarImage(tensor=torch.ones_like(native.data), affine=native.affine)
    transform = tio.Compose([tio.ToCanonical(),tio.Resample(1),tio.Resample('img'),
        tio.CropOrPad((256,256,256),padding_mode=0,include=('img',))],include=('img',))
    mask = transform(tio.Subject(img=ones))['img']
    grid = torch.from_numpy(np.load(grid_path,allow_pickle=False))[None]
    sampled = F.grid_sample(mask.data[None], grid, mode='bilinear',
        padding_mode='zeros',align_corners=False)[0,0].numpy()
    upper = torch.tensor([1-1/n for n in reversed(mask.spatial_shape)])
    inside = torch.all(torch.abs(grid[0]) <= upper,dim=-1).numpy()
    full = (sampled >= 1-1e-6) & inside
    fixed_grid = nib.load(run/'anatomy/aligned_official_fixed_grid.nii.gz')
    ref = nib.load(fixed)
    mapped = resample_from_to(nib.Nifti1Image(full.astype(np.uint8),fixed_grid.affine),ref,order=0)
    valid = mapped.get_fdata()>.5
    nib.save(mapped,out/'full_kernel_support_template_grid.nii.gz')
    brainmask = nib.load(brain).get_fdata()>.5
    result = {'template_brain_voxels':int(brainmask.sum()),
        'brain_with_full_native_source_support':int(valid[brainmask].sum()),
        'brain_full_support_fraction':float(valid[brainmask].mean()),'boxes':{},
        'definition':'raw fraction, higher better; source support excludes border extrapolation, not an anatomical score',
        'gate':'NOT MET','visual_acceptance':'NOT ESTABLISHED'}
    for name,bounds in BOXES.items():
        indices=[np.flatnonzero((np.arange(ref.shape[d])+ref.affine[d,3]>=lo)&
            (np.arange(ref.shape[d])+ref.affine[d,3]<hi)) for d,(lo,hi) in enumerate(bounds)]
        slices=tuple(slice(int(v[0]),int(v[-1])+1) for v in indices)
        result['boxes'][name]={'voxels':int(valid[slices].size),
            'full_support_voxels':int(valid[slices].sum()),'full_support_fraction':float(valid[slices].mean())}
    dump(out/'complete.json',result)
    dump(ROOT/'results/registration/qc/brainmorph_support_v1.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
