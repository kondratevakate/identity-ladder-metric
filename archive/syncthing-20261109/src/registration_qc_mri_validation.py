"""Frozen MRI target cohort, sequential official SynthMorph affine, QC preparation only."""
import argparse
import datetime
import itertools
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull

from registration_qc_mri import BOXES
from registration_qc_fixed_mri_geometry import matrix
from registration_qc_pilot import PRIVATE, ROOT, dump, sha

OUT = PRIVATE / 'registration-qc-mri-validation-20261103-v1'
BIDS = Path('C:/Projects/data/brain_images/siblings')
MANIFEST = PRIVATE / 'registration-qc-mri-manifest-20261103-v1/manifest.json'
SELECTION = PRIVATE / 'real-cases-protocol/selection.json'
SCAN_MANIFEST = PRIVATE / 'real-cases-protocol/scan_manifest.json'
TEMPLATE = PRIVATE / 'real-cases-run/template/template_1mm.nii.gz'


def prepare():
    if sha(MANIFEST) != 'ffb5a2cf66010e7071c1054ba08690a0d5e3eb01ac57c9ddd7c52b8168682461':
        raise RuntimeError('Frozen QC manifest drift')
    if sha(SELECTION) != 'cb264436a1346f6a83b17e7d3abd8f3bc27ac10174ad89c8d7339eec421f7f31':
        raise RuntimeError('Frozen selection drift')
    if sha(SCAN_MANIFEST) != '4016dcf4a30ab95c76170ba714598c435792a260d70a39780dd6701928b0dcb8':
        raise RuntimeError('Frozen source manifest drift')
    selection = json.loads(SELECTION.read_text())
    manifest = json.loads(MANIFEST.read_text())
    targets = {r['target'] for r in selection['set_S'] + selection['set_R']}
    repeats = {r['target'] for r in selection['set_R']}
    files = {r['path']: r['sha256'] for r in json.loads(SCAN_MANIFEST.read_text())['files']}
    pilot = PRIVATE / 'registration-qc-fixed-mri-20261103-v1'
    previous = json.loads((pilot / 'plan.json').read_text())
    jobs = []
    for person, session in selection['scans']:
        if person not in targets or not (session == 'ses-01' or person in repeats and session == 'ses-02'):
            continue
        relative = f'{person}/{session}/anat/{person}_{session}_T1w.nii.gz'
        jobs.append({'index': len(jobs), 'person': person, 'session': session,
            'relative': relative, 'sha256': files[relative],
            'reuse': str(pilot / 'affine') if files[relative] == previous['inputs_sha256']['moving'] else None})
    if len(jobs) != 95 or len({(j['person'], j['session']) for j in jobs}) != 95:
        raise RuntimeError('Unexpected target/repeat population')
    OUT.mkdir(exist_ok=False)
    shutil.copy2(__file__, OUT / 'code_snapshot.py')
    dump(OUT / 'plan.json', {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'code_sha256': sha(__file__), 'protocol_sha256': sha(ROOT / 'docs/PROTOCOL_registration.md'),
        'qc_manifest_sha256': sha(MANIFEST), 'selection_sha256': sha(SELECTION),
        'scan_manifest_sha256': sha(SCAN_MANIFEST), 'template_sha256': sha(TEMPLATE),
        'image_id': previous['image_id'], 'model': 'SynthMorph affine', 'jobs': jobs,
        'parameters': {'extent': 256, 'threads': 2, 'memory': '10g'},
        'scope': '95 target/repeat scans, 334 required comparisons, 72 targets, 68 families; no donor/pool expansion',
        'order': 'original selection scan order; single registration container at a time',
        'fixed_method': 'primary validation candidate only; no per-case model switching or retuning',
        'display': 'one moving P99 window shared by target and repeat; template own P99; never per-slice tuning',
        'review': 'all rows initially NOT REVIEWED; support flags cannot assign visual PASS',
        'geometry': 'linear RAS moving-to-template, inverse for crop mapping; reflect registered array on exact symmetric template x axis',
        'metrics': 'registration geometry/support only; no identity, fidelity or synthesis scoring'})
    dump(OUT / 'review_manifest.json', manifest)
    dump(OUT / 'progress.json', {'scans': {}, 'sheets': {}})
    return json.loads((OUT / 'plan.json').read_text())


def aggregate(plan, progress):
    rows = json.loads((OUT / 'review_manifest.json').read_text())['rows']
    required = [r for r in rows if r['availability'] == 'required']
    summary = {'method': 'SynthMorph affine', 'planned_scans': len(plan['jobs']),
        'completed_scans': sum(s['exit_code'] == 0 for s in progress['scans'].values()),
        'runtime_failed_scans': sum(s['exit_code'] != 0 for s in progress['scans'].values()),
        'required_comparisons': len(required), 'prepared_sheets': len(progress['sheets']),
        'visual_review': 'NOT ESTABLISHED', 'gate': 'NOT MET', 'metric_scoring': 'NOT RUN',
        'strata': [{'box': box, 'branch': branch,
            'intended': sum(r['box'] == box and r['branch'] == branch for r in required),
            'sheets_prepared': sum(str(r['row']) in progress['sheets'] for r in required
                if r['box'] == box and r['branch'] == branch)}
            for box in BOXES for branch in ['template', 'contralateral', 'repeat']],
        'note': 'Preparation counts are not visual acceptance; development and validation phases remain in the frozen manifest.'}
    dump(OUT / 'aggregate.json', summary)
    dump(ROOT / 'results/registration/qc/mri_validation_v1.json', summary)


def render_ready(plan, progress):
    jobs = {(j['person'], j['session']): j for j in plan['jobs']}
    rows = json.loads((OUT / 'review_manifest.json').read_text())['rows']
    reference = nib.load(TEMPLATE)
    if not np.allclose(reference.affine[:3,:3],np.eye(3)) or not np.isclose(2*reference.affine[0,3]+reference.shape[0]-1,0):
        raise RuntimeError('Unexpected reflection grid')
    brain = nib.load(PRIVATE / 'real-cases-run/template/brain_1mm.nii.gz').get_fdata() > .5
    template = reference.get_fdata(dtype=np.float32)
    template /= np.percentile(template[brain],99)
    for target in dict.fromkeys(r['target'] for r in rows):
        job = jobs[(target,'ses-01')]
        status = progress['scans'].get(str(job['index']))
        if not status or status['exit_code'] != 0:
            continue
        relevant = [r for r in rows if r['target'] == target and r['availability'] == 'required' and str(r['row']) not in progress['sheets']]
        if not relevant:
            continue
        repeat_job = jobs.get((target,'ses-02'))
        repeat_status = progress['scans'].get(str(repeat_job['index'])) if repeat_job else None
        repeat_ready = repeat_status and repeat_status['exit_code'] == 0
        if all(r['branch']=='repeat' for r in relevant) and not repeat_ready:
            continue
        folder = Path(status['folder'])
        registered = nib.load(folder / 'moving_to_fixed.nii.gz')
        if registered.shape != reference.shape or not np.allclose(registered.affine,reference.affine):
            raise RuntimeError('Registered grid mismatch')
        raw = registered.get_fdata(dtype=np.float32)
        native = nib.as_closest_canonical(nib.load(BIDS / job['relative']))
        native_data = native.get_fdata(dtype=np.float32)
        scale = float(np.percentile(raw[brain],99))
        if not np.isfinite(raw).all() or scale <= 0:
            raise RuntimeError('Invalid registration intensities')
        moving = raw / scale
        native_data /= scale
        inverse = matrix(folder / 'inverse.lta')
        forward = matrix(folder / 'forward.lta')
        source = nib.load(BIDS / job['relative'])
        pull = np.linalg.inv(source.affine) @ inverse @ reference.affine
        repeat = None
        if repeat_ready:
            repeat = nib.load(Path(repeat_status['folder'])/'moving_to_fixed.nii.gz').get_fdata(dtype=np.float32) / scale
        for row in relevant:
            if row['branch'] == 'repeat' and repeat is None:
                continue
            bounds = BOXES[row['box']]
            indices = [np.flatnonzero((np.arange(reference.shape[d])+reference.affine[d,3]>=lo)&
                (np.arange(reference.shape[d])+reference.affine[d,3]<hi)) for d,(lo,hi) in enumerate(bounds)]
            slices = tuple(slice(int(v[0]),int(v[-1])+1) for v in indices)
            partner = template if row['branch'] == 'template' else moving[::-1] if row['branch'] == 'contralateral' else repeat
            corners = np.array(list(itertools.product(*[(lo-.5,hi-.5) for lo,hi in bounds])))
            corners_h = np.column_stack([corners,np.ones(8)])
            source_corners = (np.linalg.inv(native.affine) @ inverse @ corners_h.T)[:3].T
            centre = np.rint(source_corners.mean(axis=0)).astype(int)
            centre = np.clip(centre,0,np.array(native.shape)-1)
            coordinates = np.vstack([np.indices(raw[slices].shape).reshape(3,-1)+np.array([s.start for s in slices])[:,None],np.ones(raw[slices].size)])
            xyz = (pull @ coordinates)[:3]
            valid = np.all((xyz>=0)&(xyz<=np.array(source.shape)[:,None]-1),axis=0)
            partner_slices = slices
            if row['branch'] == 'contralateral':
                partner_slices = (slice(reference.shape[0]-slices[0].stop,reference.shape[0]-slices[0].start),slices[1],slices[2])
            if row['branch'] == 'template':
                partner_native_data=template; partner_corners=(np.linalg.inv(reference.affine) @ corners_h.T)[:3].T
                partner_valid=np.ones(valid.shape,dtype=bool)
            else:
                if row['branch'] == 'contralateral':
                    partner_source=source; partner_native=native; partner_native_data=native_data
                    partner_pull=pull; partner_inverse=inverse
                    partner_coordinates=coordinates.copy();partner_coordinates[0]=reference.shape[0]-1-partner_coordinates[0]
                    partner_corners_h=corners_h.copy();partner_corners_h[:,0]*=-1
                else:
                    partner_source=nib.load(BIDS/repeat_job['relative']);partner_native=nib.as_closest_canonical(partner_source)
                    partner_native_data=partner_native.get_fdata(dtype=np.float32)/scale
                    partner_inverse=matrix(Path(repeat_status['folder'])/'inverse.lta')
                    partner_pull=np.linalg.inv(partner_source.affine) @ partner_inverse @ reference.affine
                    partner_coordinates=coordinates;partner_corners_h=corners_h
                partner_corners=(np.linalg.inv(partner_native.affine) @ partner_inverse @ partner_corners_h.T)[:3].T
                partner_xyz=(partner_pull @ partner_coordinates)[:3]
                partner_valid=np.all((partner_xyz>=0)&(partner_xyz<=np.array(partner_source.shape)[:,None]-1),axis=0)
            partner_centre=np.rint(partner_corners.mean(axis=0)).astype(int)
            partner_centre=np.clip(partner_centre,0,np.array(partner_native_data.shape)-1)
            native_map = {'source': str(BIDS/job['relative']), 'template_crop_slices': [[s.start,s.stop] for s in slices],
                'source_crop_corners_ras': (inverse @ corners_h.T)[:3].T.tolist(),
                'physical_pull_determinant': float(np.linalg.det(inverse[:3,:3])),
                'inverse_translation_residual_mm': float(np.linalg.norm((forward @ inverse)[:3,3])),
                'crop_full_source_support_fraction': float(valid.mean()), 'crop_voxels': int(valid.size),
                'comparison_full_source_support_fraction':float(partner_valid.mean()),
                'definition': 'raw fraction higher better; requires >=0.95; support alone cannot set visual PASS',
                'target_display_p99': scale, 'reflection': 'x array reversal after individual affine alignment, never new nonlinear fitting'}
            fig, axes = plt.subplots(3,9,figsize=(27,10))
            for axis_row, axis in enumerate([2,1,0]):
                at = int(indices[axis][len(indices[axis])//2]); others = [d for d in range(3) if d != axis]
                cut = lambda v: np.take(v,at,axis=axis).T
                crop = lambda v: cut(v)[slices[others[1]],slices[others[0]]]
                a,b = crop(moving),crop(partner); yy,xx=np.indices(a.shape)
                panels = [np.take(native_data,int(centre[axis]),axis=axis).T,np.take(partner_native_data,int(partner_centre[axis]),axis=axis).T,cut(moving),cut(partner),a,b,(a+b)/2,np.where((xx//4+yy//4)%2,a,b),np.abs(a-b)]
                titles=['Native target; inverse crop','Native comparison; inverse crop','Registered target','Comparison context','Target crop',row['branch']+' crop','50:50 blend','Checkerboard','Display difference (not score)']
                for col,(panel,title) in enumerate(zip(panels,titles)):
                    ax=axes[axis_row,col]; ax.imshow(panel,cmap='gray',origin='lower',vmin=0,vmax=1,interpolation='nearest')
                    ax.set_title(title,fontsize=8);ax.set_xticks([]);ax.set_yticks([])
                    if col in [0,1]:
                        points=(source_corners if col==0 else partner_corners)[:,others];hull=ConvexHull(points)
                        ax.add_patch(Polygon(points[hull.vertices],fill=False,ec='orange',lw=1))
                    elif col in [2,3]:
                        x,y=slices[others[0]],slices[others[1]]
                        ax.add_patch(Rectangle((x.start-.5,y.start-.5),x.stop-x.start,y.stop-y.start,fill=False,ec='orange',lw=1))
                        ax.contour(cut(brain.astype(float)),levels=[.5],colors=['cyan'],linewidths=.45)
                axes[axis_row,0].set_ylabel(f'native voxel plane {centre[axis]}; template {at+reference.affine[axis,3]:.0f}mm')
            fig.suptitle(f'Frozen row {row["row"]}: {row["box"]} / {row["branch"]} / {row["phase"]}; NOT REVIEWED; RAS increases right/up')
            fig.tight_layout(); sheet=OUT/'sheets'/f'row_{row["row"]:03d}.png';sheet.parent.mkdir(exist_ok=True)
            fig.savefig(sheet,dpi=90);plt.close(fig)
            dump(sheet.with_suffix('.json'),native_map)
            progress['sheets'][str(row['row'])]={'path':str(sheet.relative_to(OUT)),'sha256':sha(sheet),'visual':'NOT REVIEWED','phase':row['phase']}
        dump(OUT/'progress.json',progress)
        aggregate(plan,progress)


def run():
    plan=json.loads((OUT/'plan.json').read_text())
    if sha(__file__) != plan['code_sha256'] or sha(TEMPLATE) != plan['template_sha256']:
        raise RuntimeError('Frozen code/template drift')
    lock=OUT/'runner.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        progress=json.loads((OUT/'progress.json').read_text())
        for job in plan['jobs']:
            key=str(job['index'])
            if key in progress['scans']:
                continue
            if sha(BIDS/job['relative']) != job['sha256']:
                raise RuntimeError('Source image hash drift')
            start=time.monotonic()
            if job['reuse']:
                folder=Path(job['reuse']); status=json.loads((folder/'complete.json').read_text())
                for name,digest in status['outputs'].items():
                    if sha(folder/name)!=digest:raise RuntimeError('Reused pilot drift')
            else:
                if subprocess.check_output(['docker','ps','-q']).strip():
                    raise RuntimeError('Existing container workload; refusing overlap')
                folder=OUT/'scans'/f'scan_{job["index"]:03d}';folder.mkdir(parents=True,exist_ok=False)
                name=f'identity-qc-affine-{job["index"]:03d}'
                command=['docker','run','--rm','--name',name,'--cpus=2','--memory=10g',
                    '-v',f'{(BIDS/job["relative"]).resolve()}:/moving/input.nii.gz:ro',
                    '-v',f'{TEMPLATE.parent.resolve()}:/fixed:ro','-v',f'{folder.resolve()}:/out',plan['image_id'],
                    'mri_synthmorph','register','-m','affine','-j','2','-e','256',
                    '-o','/out/moving_to_fixed.nii.gz','-O','/out/fixed_to_moving.nii.gz',
                    '-t','/out/forward.lta','-T','/out/inverse.lta','/moving/input.nii.gz','/fixed/template_1mm.nii.gz']
                dump(folder/'plan.json',{'command':command,'input_sha256':job['sha256'],'parent_plan_sha256':sha(OUT/'plan.json')})
                with (folder/'run.log').open('w',encoding='utf8') as log:
                    try: exit_code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
                    except subprocess.TimeoutExpired:
                        subprocess.run(['docker','stop','-t','5',name],stdout=log,stderr=subprocess.STDOUT)
                        exit_code='timeout'
                status={'exit_code':exit_code,'seconds':time.monotonic()-start,
                    'outputs':{p.name:sha(p) for p in folder.iterdir() if p.suffix in ['.gz','.lta']}}
                dump(folder/'complete.json',status)
            progress['scans'][key]={**status,'folder':str(folder),'reused':bool(job['reuse'])}
            dump(OUT/'progress.json',progress);aggregate(plan,progress)
            render_ready(plan,progress)
            print(json.dumps({'completed_scans':len(progress['scans']),'sheets':len(progress['sheets']),'last_exit_code':status['exit_code']}),flush=True)
        dump(OUT/'preparation_complete.json',{'status':'all planned scans attempted','visual_review':'NOT ESTABLISHED'})
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    if args.action=='prepare': prepare()
    else: run()
