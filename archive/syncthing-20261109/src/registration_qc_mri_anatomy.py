"""Render existing affine/SyN anatomy side by side; no registration or scoring."""
import json
import shutil
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from registration_qc_pilot import PRIVATE, ROOT, sha, dump
from registration_qc_mri import BOXES


def main():
    run = PRIVATE / 'real-cases-run'
    source = PRIVATE / 'registration-qc-mri-refined-20261102-v1'
    audit = PRIVATE / 'registration-qc-mri-composite-20261103-v1'
    out = PRIVATE / 'registration-qc-mri-anatomy-20261103-v1'
    complete = json.loads((source / 'complete.json').read_text())
    for name, digest in complete['outputs'].items():
        if sha(source / name) != digest:
            raise RuntimeError('Saved MRI output drift')
    paths = {'template': run / 'template/template_1mm.nii.gz',
             'brain': run / 'template/brain_1mm.nii.gz',
             'affine': source / 'affine.nii.gz', 'syn': source / 'syn.nii.gz',
             'mask': source / 'syn_mask.nii.gz', 'jacobian': audit / 'full_pull_jacobian.nii.gz'}
    out.mkdir(exist_ok=False)
    dump(out / 'plan.json', {'scope': 'existing exposed development scan only; no optimisation, no scores',
         'code_sha256': sha(__file__), 'protocol_sha256': sha(ROOT / 'docs/PROTOCOL_registration.md'),
         'inputs_sha256': {k: sha(p) for k, p in paths.items()}, 'boxes_mm': BOXES,
         'display': 'one 0-to-P99 intensity window shared by affine and SyN; template separate P99; determinant 0-to-2',
         'acceptance': 'not assigned; local sulcal anatomy must not be judged by template similarity alone'})
    shutil.copy2(__file__, out / 'code_snapshot.py')
    images = {k: nib.load(p) for k,p in paths.items()}
    reference = images['template']
    if not all(im.shape == reference.shape and np.allclose(im.affine, reference.affine) for im in images.values()):
        raise RuntimeError('Saved grids disagree')
    data = {k: im.get_fdata(dtype=np.float32) for k,im in images.items()}
    brain = data['brain'] > .5
    scale = float(np.percentile(data['syn'][brain], 99))
    template_scale = float(np.percentile(data['template'][brain], 99))
    affine, syn = data['affine']/scale, data['syn']/scale
    template = data['template']/template_scale
    records = []
    for name,bounds in BOXES.items():
        indices = [np.flatnonzero((np.arange(reference.shape[d])+reference.affine[d,3]>=lo)&
                                  (np.arange(reference.shape[d])+reference.affine[d,3]<hi))
                   for d,(lo,hi) in enumerate(bounds)]
        slices = tuple(slice(int(v[0]),int(v[-1])+1) for v in indices)
        centres = [int(v[len(v)//2]) for v in indices]
        fig, axes = plt.subplots(3,8,figsize=(24,10))
        for row,axis in enumerate([2,1,0]):
            at = centres[axis]
            cut = lambda v: np.take(v,at,axis=axis).T
            remaining = [d for d in range(3) if d != axis]
            crop = lambda v: cut(v)[slices[remaining[1]],slices[remaining[0]]]
            ca, cs, cm = crop(affine),crop(syn),crop(syn[::-1])
            yy,xx = np.indices(cs.shape)
            check = (xx//4+yy//4)%2
            panels = [cut(template),cut(affine),cut(syn),ca,cs,cm,
                      np.where(check,ca,cs),crop(data['jacobian'])]
            titles = ['Template (own P99)','Affine (shared window)','SyN (shared window)',
                      'Affine crop','SyN crop','Exact reflected SyN','Affine/SyN checker','Full pull Jacobian']
            for col,(panel,title) in enumerate(zip(panels,titles)):
                ax = axes[row,col]
                ax.imshow(panel, cmap='viridis' if col==7 else 'gray',vmin=0,vmax=2 if col==7 else 1,
                          origin='lower',interpolation='nearest')
                ax.set_title(title,fontsize=9); ax.set_xticks([]); ax.set_yticks([])
                if col<3:
                    x,y = slices[remaining[0]],slices[remaining[1]]
                    ax.add_patch(Rectangle((x.start-.5,y.start-.5),x.stop-x.start,y.stop-y.start,
                                           fill=False,ec='orange',lw=1))
            axes[row,0].set_ylabel(f'{["sagittal","coronal","axial"][axis]}\n{at+reference.affine[axis,3]:.0f} mm')
        fig.suptitle(f'{name}: same source scan, affine versus SyN; RAS axes increase right/up; NOT ACCEPTED')
        fig.tight_layout()
        path = out / (name+'.png')
        fig.savefig(path,dpi=100); plt.close(fig)
        records.append({'box':name,'sheet':path.name,'sha256':sha(path),'visual':'not_reviewed'})
    dump(out/'records.json', {'sheets':records,'shared_intensity_p99':scale,'template_p99':template_scale})
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Affine / SyN anatomical QC</h1>'+
        '<p>Existing development scan only. No automatic acceptance.</p>'+
        ''.join(f'<h2>{r["box"]}</h2><img style="width:100%" src="{r["sheet"]}">' for r in records),encoding='utf-8')
    print('Two affine/SyN anatomy sheets prepared; visual decisions pending.')


if __name__ == '__main__':
    main()
