"""Read the manuscript's frozen evidence from Git, without duplicate result files.

The checkout also contains newer experiments. Updating the paper's evidence is
a separate review; a build must never silently substitute P3 for the P2 draft.
"""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_COMMIT = 'a48523dccd2b70dae2615717d62d8725e5183528'


def read_bytes(path):
    rel = Path(path).relative_to(ROOT).as_posix()
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', EVIDENCE_COMMIT + ':' + rel])


def load_json(path):
    return json.loads(read_bytes(path))


def record(path):
    rel = Path(path).relative_to(ROOT).as_posix()
    raw = read_bytes(path)
    producer = subprocess.check_output(
        ['git', '-C', str(ROOT), 'log', EVIDENCE_COMMIT, '-1', '--format=%H', '--', rel], text=True).strip()
    return {'path': rel, 'sha256': hashlib.sha256(raw).hexdigest(),
            'git_snapshot': EVIDENCE_COMMIT, 'producing_commit': producer}
