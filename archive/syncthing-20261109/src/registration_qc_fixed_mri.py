"""Sequential official SynthMorph rigid/affine pilot; reuse completed joint output."""
import json
import shutil
import subprocess
import time

from registration_qc_pilot import PRIVATE, ROOT, dump, sha


def main():
    source = PRIVATE / 'registration-qc-synthmorph-20261103-v3-raw192'
    previous = json.loads((source / 'plan.json').read_text())
    completed = json.loads((source / 'complete.json').read_text())
    for name, digest in completed['outputs'].items():
        if sha(source / name) != digest:
            raise RuntimeError('Completed joint output drift')
    command = previous['command']
    moving_mount = next(command[i + 1] for i, v in enumerate(command)
                        if v == '-v' and ':/moving/' in command[i + 1])
    moving = moving_mount.split(':/moving/')[0]
    fixed = PRIVATE / 'real-cases-run/template/template_1mm.nii.gz'
    if sha(moving) != previous['inputs_sha256']['moving'] or sha(fixed) != previous['inputs_sha256']['fixed']:
        raise RuntimeError('Frozen input drift')
    if subprocess.check_output(['docker', 'ps', '-q']).strip():
        raise RuntimeError('Existing Docker workload: do not overlap registration')
    out = PRIVATE / 'registration-qc-fixed-mri-20261103-v1'
    out.mkdir(exist_ok=False)
    commands = {}
    for model in ['rigid', 'affine']:
        commands[model] = ['docker', 'run', '--rm', '--cpus=2', '--memory=10g',
            '-v', moving_mount, '-v', f'{fixed.parent.resolve()}:/fixed:ro',
            '-v', f'{(out / model).resolve()}:/out', previous['image_id'],
            'mri_synthmorph', 'register', '-m', model, '-j', '2', '-e', '256',
            '-o', '/out/moving_to_fixed.nii.gz', '-O', '/out/fixed_to_moving.nii.gz',
            '-t', '/out/forward.lta', '-T', '/out/inverse.lta',
            '/moving/input.nii.gz', '/fixed/template_1mm.nii.gz']
    dump(out / 'plan.json', {'protocol_sha256': sha(ROOT / 'docs/PROTOCOL_registration.md'),
        'code_sha256': sha(__file__), 'inputs_sha256': previous['inputs_sha256'],
        'image_id': previous['image_id'], 'commands': commands,
        'joint_reused_from': source.name, 'joint_outputs': completed['outputs'],
        'scope': 'one exposed T1w development pair; official rigid then affine; no tuning',
        'comparability': 'rigid/affine extent 256; completed joint extent 192 is a resource-limited diagnostic',
        'source': 'FreeSurfer bundled mri_synthmorph CLI; https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/',
        'gate': 'unchanged; no automatic anatomy acceptance or identity scoring'})
    shutil.copy2(__file__, out / 'code_snapshot.py')
    shutil.copy2(source / 'model_inventory.log', out / 'model_inventory.log')
    results = {}
    for model, cmd in commands.items():
        folder = out / model
        folder.mkdir()
        start = time.monotonic()
        with (folder / 'run.log').open('w', encoding='utf-8') as log:
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        status = {'exit_code': code, 'seconds': time.monotonic() - start,
            'outputs': {p.name: sha(p) for p in folder.iterdir() if p.suffix in ['.gz', '.lta']},
            'visual_acceptance': 'NOT ESTABLISHED', 'scope': 'one development pair',
            'metric_scoring': 'NOT RUN', 'gate': 'NOT MET'}
        dump(folder / 'complete.json', status)
        results[model] = status
        dump(out / 'progress.json', results)
        print(json.dumps({'method': model, **status}), flush=True)
    dump(out / 'complete.json', results)
    dump(ROOT / 'results/registration/qc/fixed_mri_v1.json', results)


if __name__ == '__main__':
    main()
