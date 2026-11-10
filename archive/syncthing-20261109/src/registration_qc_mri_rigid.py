"""One-scan repair diagnostic: explicit brain-centroid initialization, rigid only."""
import os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='2'
import json
import shutil
import numpy as np
import ants
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from registration_qc_pilot import PRIVATE, sha, dump


def main():
    run=PRIVATE/'real-cases-run'; source=sorted((run/'syn_amendment4').glob('scan_*/complete.json'))[0].parent
    out=PRIVATE/'registration-qc-mri-rigid-20261102-v1'; out.mkdir(exist_ok=False)
    fixed=ants.image_read(str(run/'template/template_2mm.nii.gz'))
    fm=ants.image_read(str(run/'template/brain_2mm.nii.gz'))
    native=ants.image_read(str(source/'n4.nii.gz')); nm=ants.image_read(str(source/'mask.nii.gz'))
    moving=ants.resample_image(native*nm,(2,2,2),use_voxels=False,interp_type=0)
    fc=np.array(ants.get_center_of_mass(fm)); mc=np.array(ants.get_center_of_mass(nm))
    t=ants.create_ants_transform(transform_type='Euler3DTransform',dimension=3,center=fc.tolist(),translation=(mc-fc).tolist())
    init=out/'centroid.mat'; ants.write_transform(t,str(init))
    dump(out/'plan.json',{'source_completion_sha256':sha(source/'complete.json'),'code_sha256':sha(__file__),
         'initialization':'binary brain-mask centroids in ANTs LPS physical space; fixed-to-moving sampling',
         'translation_lps':(mc-fc).tolist(),'iterations':[200,100,50,0],
         'scope':'one existing pilot, rigid repair only, not scoring; post visual failure'})
    before=ants.apply_transforms(fixed,moving,[str(init)])
    ants.config.set_ants_deterministic(False,seed_value=123)
    reg=ants.registration(fixed=fixed*fm,moving=moving,type_of_transform='Rigid',initial_transform=[str(init)],
         outprefix=str(out/'rigid_'),mask=fm,mask_all_stages=True,aff_metric='mattes',
         aff_iterations=(200,100,50,0),aff_shrink_factors=(8,4,2,1),aff_smoothing_sigmas=(3,2,1,0),
         aff_random_sampling_rate=1.0,verbose=False)
    after=ants.apply_transforms(fixed,native,reg['fwdtransforms'])
    mask=ants.apply_transforms(fixed,nm,reg['fwdtransforms'],interpolator='nearestNeighbor')
    ants.image_write(after,str(out/'rigid.nii.gz')); ants.image_write(mask,str(out/'rigid_mask.nii.gz'))
    vols=[fixed.numpy(),before.numpy(),after.numpy()]; b=fm.numpy()>.5
    fig,axs=plt.subplots(3,3,figsize=(10,10))
    for c,v in enumerate(vols):
        v=v/max(float(np.percentile(v[b],99)),1e-6)
        for r,axis in enumerate([2,1,0]):
            mid=v.shape[axis]//2
            axs[r,c].imshow(np.take(v,mid,axis=axis).T,cmap='gray',origin='lower',vmin=0,vmax=1)
            axs[r,c].contour(np.take(b,mid,axis=axis).T,levels=[.5],colors=['cyan'],linewidths=.5)
            axs[r,c].axis('off')
    for c,title in enumerate(['Template','Centroid initialization','Repaired rigid']): axs[0,c].set_title(title)
    fig.tight_layout(); fig.savefig(out/'review.png',dpi=130); plt.close(fig)
    dump(out/'complete.json',{'visual':'pending','code_sha256':sha(__file__),
          'outputs':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.gz','.mat','.png')},
          'brain_overlap_fraction':float((mask.numpy()[b]>.5).mean())})
    shutil.copy2(__file__,out/'code_snapshot.py')
    print('One-scan rigid diagnostic complete; visual review required')


if __name__=='__main__': main()
