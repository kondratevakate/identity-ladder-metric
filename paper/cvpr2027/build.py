"""Rebuild the first draft from saved aggregate artifacts; no experiments run."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tectonic', default=shutil.which('tectonic'))
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if not args.tectonic:
        parser.error('Provide --tectonic /path/to/tectonic, or put Tectonic on PATH.')
    (HERE / '.build').mkdir(exist_ok=True)
    for name in ('prepare.py', 'assess_population_audit.py', 'assemble_atlas.py', 'make_provenance_infographic_v2.py', 'catalogue_materials.py'):
        subprocess.run([sys.executable, str(HERE / name)], cwd=ROOT, check=True)
    for stem in ('main', 'supplementary'):
        with (HERE / '.build' / f'{stem}-build.log').open('w') as log:
            subprocess.run([args.tectonic, '-X', 'compile', str(HERE / f'{stem}.tex'), '--keep-logs'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(HERE / 'check_paper.py')], cwd=ROOT, check=True)
    if args.render:
        subprocess.run([sys.executable, str(HERE / 'render_review.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
