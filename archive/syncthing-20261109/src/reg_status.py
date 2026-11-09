"""One registration status view, built only from saved public aggregates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def counts(strata):
    keys = ('intended', 'reviewed', 'pass', 'fail', 'unreviewable', 'not_reviewed')
    for row in strata:
        if row['reviewed'] != row['pass'] + row['fail'] + row['unreviewable']:
            raise ValueError('Inconsistent reviewed denominator')
        if row['intended'] != row['reviewed'] + row['not_reviewed']:
            raise ValueError('Inconsistent intended denominator')
    return {key: sum(row[key] for row in strata) for key in keys}


def build(root=ROOT):
    root = Path(root)
    result = root / 'results/registration'
    hashes = {}

    def read(relative):
        raw = (result / relative).read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    manifest = read('manifest.json')
    datasets = {}
    for dataset, choice in manifest['choice'].items():
        methods = {}
        for key in manifest['runs']:
            if not key.startswith(dataset + '_'):
                continue
            record = read(key + '.json')
            methods[record['method']] = {
                'frame': record['frame'], 'intended_pairs': record['n_pairs'],
                'dropped': record['n_dropped'], 'eligibility': record['eligibility'],
                'rank': record['rank'], 'accuracy': record['accuracy'],
                'failure': record['failure'], 'evidence': 'completed comparison; visual gate not met',
            }
        datasets[dataset] = {'choice': choice, 'methods': methods,
                             'visual_acceptance': 'NOT ESTABLISHED'}
    development = read('qc/development_latest.json')
    legacy_frames = read('qc/fundus_frame_v1.json')
    current_frames_path = 'qc/fundus_frame_v2.json'
    frames = read(current_frames_path) if (result / current_frames_path).exists() else legacy_frames
    mri = read('qc/mri.json')
    mri_manifest = read('qc/mri_manifest_v1.json') if (result / 'qc/mri_manifest_v1.json').exists() else None
    mri_composite = read('qc/mri_composite_v1.json') if (result / 'qc/mri_composite_v1.json').exists() else None
    pretrained = {name: read('qc/'+name+'.json') for name in
                  ['fundus_pretrained_v1','iris_pretrained_v1','synthmorph_pilot_v1','synthmorph_pilot_v2_raw','synthmorph_pilot_v3_raw192']
                  if (result / ('qc/'+name+'.json')).exists()}
    fixed_mri = {name: read('qc/'+name+'.json') for name in
                 ['fixed_mri_v1', 'fixed_mri_anatomy_v1', 'fixed_mri_geometry_v2',
                  'brainmorph_pilot_v1', 'brainmorph_pilot_v2', 'brainmorph_pilot_v3',
                  'brainmorph_anatomy_v1', 'brainmorph_support_v1']
                 if (result / ('qc/'+name+'.json')).exists()}
    validation = read('qc/mri_validation_v1.json') if (result / 'qc/mri_validation_v1.json').exists() else None
    validation_review = read('qc/mri_validation_review_v1.json') if (result / 'qc/mri_validation_review_v1.json').exists() else None
    history = read('qc/pilot_v1.json')
    state = {
        'protocol': 'docs/PROTOCOL_registration.md',
        'protocol_sha256': hashlib.sha256((root / 'docs/PROTOCOL_registration.md').read_bytes()).hexdigest(),
        'comparison_protocol_sha256': manifest['protocol_sha256'],
        'gate': 'NOT MET', 'identity_metrics_released': False,
        'datasets': datasets, 'development_qc': development,
        'development_counts': counts(development['strata']),
        'fundus_frame_qc': frames, 'fundus_frame_counts': counts(frames['strata']),
        'legacy_fundus_frame_counts': counts(legacy_frames['strata']),
        'fundus_frame_evidence': frames['method'],
        'mri_qc': mri, 'mri_manifest': mri_manifest, 'mri_composite': mri_composite,
        'mri_intended': mri_manifest['required_comparisons'] if mri_manifest else None,
        'pretrained_development': pretrained,
        'fixed_method_mri_development': fixed_mri,
        'mri_fixed_cohort_preparation': validation,
        'mri_fixed_cohort_visual_review': validation_review,
        'historical_pilot_gate': history['gate'], 'source_sha256': hashes,
    }
    lines = ['# Registration: comparison and visual QC', '',
             'Single protocol: `docs/PROTOCOL_registration.md`, Amendment 9. **Gate NOT MET.**', '',
             'Saved method-comparison eligibility is not visual acceptance. No scores were recomputed.', '',
             '| Dataset | Completed methods | Intended pairs per method | Saved recommendation | Visual status |',
             '|---|---|---|---|---|']
    for dataset, item in datasets.items():
        choice = item['choice']
        label = choice['recommended'] or choice['verdict']
        if len(choice['tied']) > 1:
            label += '; tied: ' + ', '.join(choice['tied'])
        denom = sorted({m['intended_pairs'] for m in item['methods'].values()})
        lines.append(f"| {dataset} | {', '.join(item['methods'])} | {', '.join(map(str, denom))} | {label} | NOT ESTABLISHED |")
    lines += ['', '[Frozen comparison and metric definitions](comparison.md). '
              'Q1a: raw error in pixels on the 224 grid, lower better; Q1b fundus Dice higher better; '
              'iris centre error lower better. Per-method JSON retains endpoint-specific denominators and intervals. '
              'Historical c is calibrated, provisional, and not rerun.', '',
              '## Visual evidence, kept separate by method', '',
              '| Evidence | Intended | Reviewed | Pass | Fail | Unreviewable | Not reviewed |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label, c in [('SIFT / iris development history', state['development_counts']),
                     ('Legacy G1-family frame sheets', state['legacy_fundus_frame_counts']),
                     ('Current shared G1 development', state['fundus_frame_counts'])]:
        lines.append('| ' + label + ' | ' + ' | '.join(str(c[k]) for k in
                     ('intended', 'reviewed', 'pass', 'fail', 'unreviewable', 'not_reviewed')) + ' |')
    lines += ['', 'Visual acceptance = pass / intended, raw proportion, higher better; >=90% in every '
              'prespecified stratum, including mirrors. These rows are different methods, not additive reviews.', '',
              f"QTIM: {mri['prepared_scans']} repaired development scan; {mri['pilot_comparisons_reviewed']} "
              f"reviewed box/branch decisions, {mri['pass']} pass, {mri['fail']} fail, "
              f"{mri['unreviewable']} unreviewable. Full MRI denominator: " +
              (str(mri_manifest['required_comparisons']) + ' required comparisons; ' +
               str(mri_manifest['unique_targets']) + ' targets / ' + str(mri_manifest['unique_families']) + ' families.'
               if mri_manifest else 'NOT FROZEN.'), '',
              'Current G1 reviews are development evidence, with Kate review pending; no acceptance is inferred. '
              'Next: repair independently checked disc/fovea localization; independent iris anatomy; '
              'QTIM local anatomical review and remaining-case validation. No identity-metric release.', '']
    if mri_composite:
        lines += ['## MRI full composite geometry: one development scan', '',
                  'Saved image replayed exactly: ' + str(mri_composite['saved_image_reproduced']) + '.',
                  f"Full pull Jacobian: {mri_composite['full_pull_nonpositive']} nonpositive / "
                  f"{mri_composite['valid_brain_voxels']} valid brain voxels; raw unitless determinant, positivity required.",
                  f"Roundtrip error: P95 {mri_composite['roundtrip_error_mm_percentiles'][2]:.6f} mm / "
                  f"{mri_composite['roundtrip_points']} points; raw physical error, lower better.",
                  'These checks do not establish preservation of individual anatomy or visual acceptance. '
                  '[Audit](qc/mri_composite_v1.json); [frozen cohort](qc/mri_manifest_v1.json).', '']
    if pretrained:
        lines += ['## Modality-specific pretrained development (not a release)', '',
                  'Original tables above are preserved. No visual acceptance or identity scoring is inferred.', '']
        for name,item in pretrained.items():
            if 'pilot_rows' in item:
                label=f"{item.get('reviewed_pilot_rows',0)}/{item['pilot_rows']} pilot rows inspected; {item.get('fail_count',0)} fail, {item.get('unreviewable_count',0)} unreviewable; scope: {item.get('scope','feasibility')}"
            else:
                label=f"runtime exit {item['exit_code']}; {len(item['outputs'])} saved outputs; visual acceptance NOT ESTABLISHED"
            lines.append(f'- [{name}](qc/{name}.json): {label}.')
        lines.append('')
    if fixed_mri:
        lines += ['## Sequential fixed MRI methods: exposed development pair', '',
                  'Official SynthMorph rigid and affine completed; joint extent-192 output reused. '
                  'Each method uses the same T1w scan/template. No local landmark acceptance or validation release.', '']
        for name, item in fixed_mri.items():
            if name.startswith('brainmorph_pilot'):
                label = f"runtime exit {item['exit_code']}; {len(item['output_files'])} output files; visual acceptance NOT ESTABLISHED"
            else:
                label = 'saved development evidence; not additive to the frozen visual gate denominator'
            lines.append(f'- [{name}](qc/{name}.json): {label}.')
        lines.append('')
    if validation:
        lines += ['## Fixed MRI cohort: affine preparation in progress', '',
                  f"{validation['completed_scans']}/{validation['planned_scans']} scans completed; "
                  f"{validation['runtime_failed_scans']} runtime failures; "
                  f"{validation['prepared_sheets']}/{validation['required_comparisons']} QC sheets prepared.",
                  'Preparation is not visual acceptance. Original denominators and metric pause remain unchanged. '
                  '[Live aggregate](qc/mri_validation_v1.json).', '']
        if validation_review:
            lines += [f"Agent visual review: {validation_review['reviewed']}/{validation_review['required_comparisons']} comparisons; "
                      f"{validation_review['pass']} pass, {validation_review['fail']} fail, "
                      f"{validation_review['unreviewable']} unreviewable. Kate acceptance PENDING; gate NOT MET.",
                      '[Review by stratum](qc/mri_validation_review_v1.json).', '']
    (result / 'status.json').write_text(json.dumps(state, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (result / 'status.md').write_text('\n'.join(lines), encoding='utf-8')
    return state


if __name__ == '__main__':
    build()
    print(ROOT / 'results/registration/status.md')
