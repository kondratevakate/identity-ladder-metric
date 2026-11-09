"""Freeze the original MRI QC population from saved selection metadata, no images."""
import datetime
import json
import shutil
from collections import Counter
from registration_qc_pilot import PRIVATE, ROOT, sha, dump

EXPECTED = 'cb264436a1346f6a83b17e7d3abd8f3bc27ac10174ad89c8d7339eec421f7f31'


def rows_for(selection, pilot_target):
    rows = []
    for group in ('set_S', 'set_R'):
        for entry in selection[group]:
            for box in ('deep', 'parietal'):
                for branch in ('template', 'contralateral', 'repeat'):
                    available = branch != 'repeat' or group == 'set_R'
                    rows.append({'row': len(rows), 'target': entry['target'], 'family': entry['family'],
                        'set': group, 'box': box, 'branch': branch,
                        'availability': 'required' if available else 'not_planned_in_original_selection',
                        'phase': 'development' if entry['target'] == pilot_target else 'validation',
                        'status': 'not_reviewed' if available else 'unavailable',
                        'reason': None if available else 'Original set S has no selected repeat; not a registration failure.'})
    return rows


def main():
    selection_path = PRIVATE / 'real-cases-protocol/selection.json'
    if sha(selection_path) != EXPECTED:
        raise RuntimeError('Original selection hash drift')
    selection = json.loads(selection_path.read_text())
    pilot_target = selection['scans'][0][0]
    rows = rows_for(selection, pilot_target)
    required = [r for r in rows if r['availability'] == 'required']
    target_rows = selection['set_S'] + selection['set_R']
    targets = {r['target'] for r in target_rows}
    families = {r['family'] for r in target_rows}
    if len(rows) != 432 or len(required) != 334 or len(targets) != 72:
        raise RuntimeError('Unexpected original QC population')
    out = PRIVATE / 'registration-qc-mri-manifest-20261103-v1'
    out.mkdir(exist_ok=False)
    manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selection_sha256': EXPECTED, 'protocol_sha256': sha(ROOT / 'docs/PROTOCOL_registration.md'),
        'code_sha256': sha(__file__), 'rows': rows,
        'exposure': 'First existing pilot target already visually inspected. Its rows are development, not blind validation.',
        'scope': 'original design only; no A1.1; donor and pool scans require separate QC'}
    dump(out / 'manifest.json', manifest)
    shutil.copy2(__file__, out / 'code_snapshot.py')
    strata = Counter((r['box'], r['branch']) for r in required)
    aggregate = {'manifest_sha256': sha(out / 'manifest.json'), 'selection_sha256': EXPECTED,
        'code_sha256': sha(__file__), 'gate': 'NOT MET', 'visual_acceptance': 'NOT ESTABLISHED',
        'unique_targets': len(targets), 'unique_families': len(families),
        'family_overlap_between_sets': len({r['family'] for r in selection['set_S']} & {r['family'] for r in selection['set_R']}),
        'required_comparisons': len(required), 'unavailable_repeat_rows': len(rows)-len(required),
        'rows_by_phase': dict(Counter(r['phase'] for r in required)),
        'strata': [{'box': box, 'branch': branch, 'intended': n} for (box, branch), n in strata.items()],
        'review_status': 'Manifest freeze only; existing one-scan review is retained separately and is not counted twice.',
        'scope': 'target QC only. All 244 selected source scans, including donor/pool scans, remain required before scoring.'}
    dump(out / 'aggregate.json', aggregate)
    dump(ROOT / 'results/registration/qc/mri_manifest_v1.json', aggregate)
    print(json.dumps(aggregate))


if __name__ == '__main__':
    main()
