import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
for name in ('grape_calibration', 'iris_calibration', 'iris_codes', 'image_ladder', 'real_atlas_mri', 'real_atlas_iris', 'synthetic_atlas_tissue', 'synthetic_atlas_eyes', 'baseline_v0'):
    data = json.loads((root / 'results' / f'{name}.json').read_text())
    print(name, 'keys:', list(data))
    if name == 'grape_calibration':
        for feature, conditions in data['feature_spaces'].items():
            print('  feature:', feature)
            for key in ('same_eye_other_visit | same camera', 'same_eye_other_visit | different camera', 'fellow_eye_same_visit | same camera'):
                item = conditions.get(key, {})
                print('   ', key, {k: v for k, v in item.items() if k in ('n_pairs', 'n_patients', 'mean_c_auc', 'mean_c_ci95_patient_bootstrap')})
    if name == 'iris_calibration':
        for dataset in ('UPOL', 'UTIRIS_NIR', 'UTIRIS_VIS', 'UTIRIS'):
            if dataset not in data:
                continue
            print(' dataset:', dataset, 'keys:', list(data[dataset]))
            for feature, conditions in data[dataset].get('feature_spaces', {}).items():
                if 'flipped' in feature:
                    continue
                print('  feature:', feature)
                for key, item in conditions.items():
                    if isinstance(item, dict) and 'mean_c_auc' in item:
                        print('   ', key, {k: v for k, v in item.items() if k in ('n_pairs', 'n_people', 'mean_c_auc', 'mean_c_ci95_person_bootstrap')})
    if name == 'iris_codes':
        for key in ('UTIRIS_NIR', 'UPOL_secondary', 'UTIRIS_VIS_secondary'):
            print(' ', key, 'keys:', list(data[key]))
    if name in ('real_atlas_mri', 'real_atlas_iris', 'synthetic_atlas_tissue', 'synthetic_atlas_eyes'):
        for key, value in data.items():
            if key in ('n_cases', 'n_targets', 'n_images', 'n_people', 'n_families', 'n_pairs', 'n_selected', 'selection', 'limitations', 'status', 'case_count'):
                print(' ', key, value)
            elif isinstance(value, (list, dict)):
                print(' ', key, 'length:', len(value))
