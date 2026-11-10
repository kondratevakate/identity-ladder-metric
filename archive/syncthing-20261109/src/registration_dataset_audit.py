"""Read-only PAPILA input/annotation audit; aggregate output, no identity metrics."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    base = Path('C:/Projects/data')
    root = base / 'datasets/ophthalmology/papila/figshare-14798004-v2/raw'
    images = list(root.glob('**/FundusImages/*.jpg'))
    overlays = list(root.glob('**/ImagesWithContours/*.jpg'))
    annotations = list(root.rglob('*.txt'))
    canonical = {}
    additional = []
    for path in annotations:
        if re.fullmatch(r'RET\d+(OD|OS)_(cup|disc)_exp\d+\.txt', path.name):
            canonical.setdefault(path.stem, []).append(path)
        else:
            additional.append(path)
    expected = [f'{p.stem}_{structure}_exp{expert}' for p in images
                for structure in ('cup', 'disc') for expert in (1, 2)]
    validity = Counter()
    for key in expected:
        files = canonical.get(key, [])
        if len(files) != 1:
            validity['missing_or_ambiguous'] += 1
            continue
        try:
            points = np.loadtxt(files[0])
            if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3 and np.isfinite(points).all():
                validity['finite_xy_polygon'] += 1
            else:
                validity['unexpected_shape_or_values'] += 1
        except (OSError, ValueError):
            validity['unreadable'] += 1
    report = {
        'dataset': 'PAPILA figshare 14798004 v2',
        'source_images': len(images), 'annotated_overlays': len(overlays),
        'expected_contours': len(expected), 'canonical_contours': sum(map(len, canonical.values())),
        'additional_contour_files': len(additional), 'contour_checks': dict(validity),
        'source_side_counts': dict(Counter(re.search(r'(OD|OS)$', p.stem).group() for p in images)),
        'source_images_with_overlay': sum(('Opht_cont_' + p.name) in {q.name for q in overlays} for p in images),
        'status': 'STRUCTURAL AUDIT ONLY; anatomy and image registration NOT REVIEWED',
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = base / 'projects/identity-ladder-metric/dataset-screen-20261102-v1/papila_structure.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Preserve the previous audit; select a new output version.')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
