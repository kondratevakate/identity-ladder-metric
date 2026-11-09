"""One official BrainMorph affine run on the existing exposed MRI pair."""
import json
import argparse
import os
import shutil
import subprocess
import time

from registration_qc_pilot import PRIVATE, ROOT, dump, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    code = ROOT / 'external/brainmorph-release'
    python = ROOT / 'external/registration-env/Scripts/python.exe'
    weights = PRIVATE / 'registration-models-20261103/foundation-numkey256-numlevels4.pth.tar'
    if sha(weights) != '1d4b0e39aba9de26a21312353f9cb397cb077a4e5935799ac3944f71df32e8cd':
        raise RuntimeError('Checkpoint drift')
    previous = json.loads((PRIVATE / 'registration-qc-synthmorph-20261103-v3-raw192/plan.json').read_text())
    cmd = previous['command']
    moving = next(cmd[i + 1].split(':/moving/')[0] for i, v in enumerate(cmd)
                  if v == '-v' and ':/moving/' in cmd[i + 1])
    fixed = PRIVATE / 'real-cases-run/template/template_1mm.nii.gz'
    if sha(moving) != previous['inputs_sha256']['moving'] or sha(fixed) != previous['inputs_sha256']['fixed']:
        raise RuntimeError('Input drift')
    out = PRIVATE / f'registration-qc-brainmorph-20261103-v{args.attempt}'
    out.mkdir(exist_ok=False)
    command = [str(python), '-m', 'scripts.register', '--num_keypoints', '256', '--variant', 'S',
        '--weights_dir', str(weights.parent), '--moving', moving, '--fixed', str(fixed),
        '--list_of_aligns', 'affine', '--list_of_metrics', '--save_eval_to_disk',
        '--save_dir', str(out / 'outputs')]
    if args.attempt == 3:
        bootstrap = ('import argparse,pathlib,torch,runpy; '
            'torch.serialization.add_safe_globals([argparse.Namespace,'
            '(pathlib.PurePosixPath,"pathlib.PosixPath")]); '
            'runpy.run_module("scripts.register",run_name="__main__")')
        command = [str(python), '-c', bootstrap] + command[3:]
    dump(out / 'plan.json', {'source': json.loads((code / 'SOURCE_COMMIT.json').read_text()),
        'code_sha256': sha(__file__), 'protocol_sha256': sha(ROOT / 'docs/PROTOCOL_registration.md'),
        'official_python_sha256': {str(p.relative_to(code)): sha(p) for p in sorted((code / 'scripts').glob('*.py'))},
        'model_sha256': sha(weights), 'inputs_sha256': previous['inputs_sha256'], 'command': command,
        'preprocessing': 'unmodified official ToCanonical, Resample(1), CropOrPad(256), min-max',
        'selection': 'README example 256-keypoint S model; affine CLI default; no tuning',
        'scope': 'one already-exposed T1w development pair, not independent benchmark validation',
        'metrics': 'empty list; no identity/fidelity scoring; runtime and saved geometry only',
        'resources': 'CPU only, OMP/MKL 2 threads; 1800 second timeout',
        'version_limit': 'BrainMorph MELBA 2025 baseline, not a 2026 venue reproduction'})
    if args.attempt >= 2:
        dump(out / 'dependency_repair.json', {
            'reason': 'PyPI KeyMorph 2.0.1 lacks one_hot_eval_synthseg imported by official BrainMorph',
            'replacement': json.loads((ROOT / 'external/keymorph-release/SOURCE_COMMIT.json').read_text()),
            'scope': 'unmodified official source dependency; same inputs, model, preprocessing and affine command',
            'checkpoint_metadata_compatibility': 'attempt 3 allows argparse.Namespace and maps metadata PosixPath to PurePosixPath; weights_only remains enabled; no weight edits',
            'previous_failures_preserved': ['registration-qc-brainmorph-20261103-v1', 'registration-qc-brainmorph-20261103-v2']})
    shutil.copy2(__file__, out / 'code_snapshot.py')
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
               TORCH_FORCE_WEIGHTS_ONLY_LOAD='1', MPLBACKEND='Agg')
    freeze = subprocess.run([str(python), '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (out / 'environment.txt').write_text(freeze.stdout, encoding='utf8')
    start = time.monotonic()
    with (out / 'run.log').open('w', encoding='utf8') as log:
        try:
            run = subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            exit_code = run.returncode
        except subprocess.TimeoutExpired:
            exit_code = 'timeout'
    status = {'exit_code': exit_code, 'seconds': time.monotonic() - start,
        'output_files': {str(p.relative_to(out)): sha(p) for p in out.glob('outputs/**/*') if p.is_file()},
        'visual_acceptance': 'NOT ESTABLISHED', 'metric_scoring': 'NOT RUN',
        'gate': 'NOT MET', 'development_pairs': 1}
    dump(out / 'complete.json', status)
    dump(ROOT / f'results/registration/qc/brainmorph_pilot_v{args.attempt}.json', status)
    print(json.dumps(status))


if __name__ == '__main__':
    main()
