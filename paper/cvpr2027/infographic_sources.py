"""Shared immutable sources for the current manuscript infographic."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_COMMIT = '82ed7fe8a52e557854cf4cd2d02eb47d5803d7cb'
SOURCE_PATH = 'results/figures/fig_ladder_maps_all_modalities.pdf'
ASSET = HERE / 'fig/assets/identity_bust_v1.png'


def record(path):
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
