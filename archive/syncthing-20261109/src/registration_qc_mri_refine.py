"""Bounded continuation of the visually inspected one-scan rigid repair."""
import os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='2'
import json
import shutil
import ants
from registration_qc_pilot import PRIVATE, sha, dump


def main():
    run=PRIVATE/'real-cases-run'
    source=sorted((run/'syn_amendment4').glob('scan_*/complete.json'))[0].parent
    rigid=PRIVATE/'registration-qc-mri-rigid-20261102-v1'
    out=PRIVATE/'registration-qc-mri-refined-20261102-v1'; out.mkdir(exist_ok=False)
    native=ants.image_read(str(source/'n4.nii.gz')); mask=ants.image_read(str(source/'mask.nii.gz'))
    fixed=ants.image_read(str(run/'template/template_2mm.nii.gz'))
    fm=ants.image_read(str(run/'template/brain_2mm.nii.gz'))
    moving=ants.resample_image(native*mask,(2,2,2),use_voxels=False,interp_type=0)
    initial=rigid/'rigid_0GenericAffine.mat'
    dump(out/'plan.json',{'code_sha256':sha(__file__),'source_sha256':sha(source/'complete.json'),
         'initial_sha256':sha(initial),'scope':'one development scan; post visual failure; no scores',
         'affine_iterations':[500,250,100,0],'syn_iterations':[60,40,20],
         'syn_step':.1,'syn_flow_sigma':3,'syn_total_sigma':0,'seed':123})
    shutil.copy2(__file__,out/'code_snapshot.py')
    ants.config.set_ants_deterministic(False,seed_value=123)
    aff=ants.registration(fixed=fixed*fm,moving=moving,type_of_transform='Affine',initial_transform=[str(initial)],
        outprefix=str(out/'affine_'),mask=fm,mask_all_stages=True,aff_metric='mattes',
        aff_iterations=(500,250,100,0),aff_shrink_factors=(8,4,2,1),aff_smoothing_sigmas=(3,2,1,0),
        aff_random_sampling_rate=1.0,verbose=False)
    dump(out/'affine_complete.json',{'transforms':[str(p) for p in aff['fwdtransforms']]})
    syn=ants.registration(fixed=fixed*fm,moving=moving,type_of_transform='SyNOnly',initial_transform=aff['fwdtransforms'],
        outprefix=str(out/'syn_'),mask=fm,mask_all_stages=True,syn_metric='CC',syn_sampling=4,
        reg_iterations=(60,40,20),grad_step=.1,flow_sigma=3,total_sigma=0,verbose=False)
    fixed1=ants.image_read(str(run/'template/template_1mm.nii.gz'))
    for name,reg in [('affine',aff),('syn',syn)]:
        ants.image_write(ants.apply_transforms(fixed1,native,reg['fwdtransforms']),str(out/f'{name}.nii.gz'))
        ants.image_write(ants.apply_transforms(fixed1,mask,reg['fwdtransforms'],interpolator='nearestNeighbor'),str(out/f'{name}_mask.nii.gz'))
    dump(out/'complete.json',{'outputs':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.gz','.mat')},'visual':'pending'})
    print('One-scan refinement complete; visual acceptance pending')


if __name__=='__main__': main()
