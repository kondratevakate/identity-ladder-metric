"""Read-only inspection of existing SyN outputs; no registration and no scoring."""
import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','2')
import json
import shutil
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from registration_qc_pilot import PRIVATE, ROOT, sha, dump

BOXES={'deep':((-32,0),(-19,13),(-2,22)), 'parietal':((-45,-13),(-65,-33),(43,67))}


def main():
    run=PRIVATE/'real-cases-run'; files=sorted((run/'syn_amendment4').glob('scan_*/complete.json'))
    if not files: raise RuntimeError('No completed SyN pilot')
    source=Path(os.environ['QC_MRI_SOURCE']) if 'QC_MRI_SOURCE' in os.environ else files[0].parent
    done=source/'complete.json'; meta=json.loads(done.read_text())
    out=PRIVATE/os.environ.get('QC_MRI_OUT','registration-qc-mri-20261102-v1'); out.mkdir(exist_ok=False)
    for name,digest in meta['outputs'].items():
        if sha(source/name)!=digest: raise RuntimeError('Prepared output hash mismatch')
    paths={'syn':source/'syn.nii.gz','affine':source/'affine.nii.gz',
           'template':run/'template/template_1mm.nii.gz','brain':run/'template/brain_1mm.nii.gz',
           'mask':source/'syn_mask.nii.gz'}
    ni={k:nib.load(p) for k,p in paths.items()}
    ref=ni['template']; aff=ref.affine
    if not all(x.shape==ref.shape and np.allclose(x.affine,aff) for x in ni.values()):
        raise RuntimeError('Grid mismatch')
    if not np.allclose(aff[:3,:3],np.eye(3)) or not np.isclose(2*aff[0,3]+ref.shape[0]-1,0):
        raise RuntimeError('Reflection grid invalid')
    data={k:x.get_fdata(dtype=np.float32) for k,x in ni.items()}
    brain=data['brain']>.5; mask=data['mask']>.5
    v=data['syn']; mirror=v[::-1]; audit={'boxes':{},'visual':'pending','source_hashes':{k:sha(p) for k,p in paths.items()}}
    # Display-only scaling; preserve and record each fixed volume scale across panels.
    scales={k:float(np.percentile(data[k][brain],99)) for k in ['syn','affine','template']}
    v=v/scales['syn']; mirror=v[::-1]; template=data['template']/scales['template']
    for name,bounds in BOXES.items():
        indices=[np.flatnonzero((np.arange(ref.shape[d])+aff[d,3]>=lo)&
                               (np.arange(ref.shape[d])+aff[d,3]<hi)) for d,(lo,hi) in enumerate(bounds)]
        sl=tuple(slice(int(z[0]),int(z[-1])+1) for z in indices)
        centres=[int(z[len(z)//2]) for z in indices]
        audit['boxes'][name]={'coverage':float(mask[sl].mean()),'mirror_coverage':float(mask[::-1][sl].mean()),
                              'bounds_mm':bounds,'voxel_slices':[[s.start,s.stop] for s in sl]}
        fig,axes=plt.subplots(3,8,figsize=(24,10))
        for row,axis in enumerate([2,1,0]):
            at=centres[axis]; cut=lambda x: np.take(x,at,axis=axis).T
            rem=[d for d in range(3) if d!=axis]; crop=tuple(sl[d] for d in rem)
            ca=cut(v)[crop[1],crop[0]]; cm=cut(mirror)[crop[1],crop[0]]
            yy,xx=np.indices(ca.shape); check=(xx//4+yy//4)%2
            panels=[cut(template),cut(v),cut(mirror),cut(v),ca,cm,(ca+cm)/2,np.where(check,ca,cm)]
            titles=['Template','SyN','Reflected SyN','Brain contours','Target crop','Mirror crop','Blend','Checker']
            for col,(p,title) in enumerate(zip(panels,titles)):
                ax=axes[row,col]; ax.imshow(p,cmap='gray',vmin=0,vmax=1,origin='lower',interpolation='nearest')
                ax.set_title(title,fontsize=9); ax.set_xticks([]); ax.set_yticks([])
                if col<4:
                    from matplotlib.patches import Rectangle
                    ax.add_patch(Rectangle((sl[rem[0]].start-.5,sl[rem[1]].start-.5),
                                          len(indices[rem[0]]),len(indices[rem[1]]),fill=False,ec='orange'))
                if col==3:
                    ax.contour(cut(brain),levels=[.5],colors=['cyan'],linewidths=.5)
                    ax.contour(cut(mask),levels=[.5],colors=['red'],linewidths=.5)
            axes[row,0].set_ylabel(f'{["sagittal","coronal","axial"][axis]}\n{at+aff[axis,3]:.0f} mm')
        fig.suptitle(f'{name}: RAS grid, displayed axes increase right/up; no visual acceptance')
        fig.tight_layout(); fig.savefig(out/f'{name}.png',dpi=120); plt.close(fig)
    import ants
    domain=ants.image_read(str(run/'template/template_2mm.nii.gz'))
    jac=ants.create_jacobian_determinant_image(domain,str(source/'syn_1Warp.nii.gz'),do_log=False,geom=True)
    j=jac.numpy(); bm=ants.image_read(str(run/'template/brain_2mm.nii.gz')).numpy()>.5
    audit['jacobian']={'nonpositive_in_brain':int((j[bm]<=0).sum()),'brain_voxels':int(bm.sum()),
                       'percentiles':np.percentile(j[bm],[0,1,50,99,100]).tolist(),
                       'scope':'nonlinear displacement only; not full affine composite'}
    audit['display_p99']=scales; audit['code_sha256']=sha(__file__)
    dump(out/'audit.json',audit); shutil.copy2(__file__,out/'code_snapshot.py')
    dump(ROOT/'results/registration/qc/mri.json',{'prepared_scans':1,'boxes_prepared':2,'visual':'pending',
         'hash_verified_outputs':len(meta['outputs']),'grid_and_reflection':'checked',
         'nonpositive_jacobian_voxels':audit['jacobian']['nonpositive_in_brain'],
         'audit_sha256':sha(out/'audit.json'),'gate':'NOT MET'})
    print('MRI QC: hashes/grid/reflection checked; two private three-plane sheets saved; visual pending')


if __name__=='__main__': main()
