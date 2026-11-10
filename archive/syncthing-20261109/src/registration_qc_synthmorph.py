"""One exposed MRI development target, SynthMorph; no scoring or visual PASS."""
import json
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from registration_qc_pilot import PRIVATE, ROOT, sha, dump


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw',action='store_true'); ap.add_argument('--extent',type=int,choices=[192,256],default=256); args=ap.parse_args()
    source = sorted((PRIVATE/'real-cases-run/syn_amendment4').glob('scan_*/complete.json'))[0].parent
    fixed = PRIVATE/'real-cases-run/template/template_1mm.nii.gz'
    moving = source/'n4.nii.gz'
    if args.raw:
        import real_cases as rc
        rc.use_design('frozen')
        _,relative,expected=rc.scan_jobs(1)[0]
        moving=Path(rc.BIDS)/relative
        if sha(moving)!=expected: raise RuntimeError('Raw input hash drift')
    label='registration-qc-synthmorph-20261103-v3-raw192' if args.raw and args.extent==192 else 'registration-qc-synthmorph-20261103-v2-raw' if args.raw else 'registration-qc-synthmorph-20261103-v1'
    out = PRIVATE/label
    out.mkdir(exist_ok=False)
    container = json.loads(subprocess.check_output(['docker','image','inspect','freesurfer/freesurfer:8.0.0']))[0]
    image = container['Id']
    command = ['docker','run','--rm','--cpus=2','--memory=10g',
        '-v',f'{moving.resolve()}:/moving/input.nii.gz:ro','-v',f'{fixed.parent.resolve()}:/fixed:ro',
        '-v',f'{out.resolve()}:/out',image,'mri_synthmorph','register','-m','joint',
        '-j','2','-e',str(args.extent),'-n','7','-r','0.5','-o','/out/moving_to_fixed.nii.gz',
        '-O','/out/fixed_to_moving.nii.gz','-t','/out/forward.mgz','-T','/out/inverse.mgz',
        '/moving/input.nii.gz','/fixed/template_1mm.nii.gz']
    dump(out/'plan.json', {'protocol_sha256':sha(ROOT/'docs/PROTOCOL_registration.md'),
        'code_sha256':sha(__file__),'inputs_sha256':{'moving':sha(moving),'fixed':sha(fixed)},
        'image_id':image,'image_repo_digests':container.get('RepoDigests'),'raw_input':args.raw,'extent':args.extent,
        'command':command,'scope':'one exposed development target; no automatic visual acceptance'})
    shutil.copy2(__file__,out/'code_snapshot.py')
    fshome=next(e.split('=',1)[1] for e in container['Config']['Env'] if e.startswith('FREESURFER_HOME='))
    model_script = "import glob,hashlib,json; print(json.dumps({p.split('/')[-1]:hashlib.sha256(open(p,'rb').read()).hexdigest() for p in glob.glob("+repr(fshome+'/models/*synthmorph*')+")}))"
    model_info = subprocess.run(['docker','run','--rm','--cpus=1',image,fshome+'/python/bin/python3','-c',model_script],capture_output=True,text=True)
    (out/'model_inventory.log').write_text(model_info.stdout+model_info.stderr,encoding='utf-8')
    start = time.monotonic()
    with (out/'run.log').open('w',encoding='utf-8') as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    status = {'exit_code':result.returncode,'seconds':time.monotonic()-start,
        'outputs':{p.name:sha(p) for p in out.iterdir() if p.name.endswith(('.nii.gz','.mgz'))},
        'visual_acceptance':'NOT ESTABLISHED','metric_scoring':'NOT RUN','gate':'NOT MET'}
    dump(out/'complete.json',status)
    filename='synthmorph_pilot_v3_raw192.json' if args.raw and args.extent==192 else 'synthmorph_pilot_v2_raw.json' if args.raw else 'synthmorph_pilot_v1.json'
    dump(ROOT/'results/registration/qc'/filename,status)
    print(json.dumps(status))


if __name__=='__main__': main()
