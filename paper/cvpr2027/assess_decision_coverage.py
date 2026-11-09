"""Document task coverage and summarize saved code probes; do not run experiments."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    specification=HERE/'decision_coverage.json'
    coverage=json.loads(specification.read_text())
    records={}
    for path in sorted({p for row in coverage['questions'] for p in row['sources']} |
                       {row['path'] for row in coverage['excluded_sources']}):
        source=ROOT/path
        assert source.is_file(),path
        records[path]={'sha256':sha(source),'producing_commit':subprocess.check_output(
            ['git','-C',str(ROOT),'log','-1','--format=%H','--',path],text=True).strip() or None}
    for excluded in coverage['excluded_sources']:
        assert 'WITHDRAWN' in json.loads((ROOT/excluded['path']).read_text())['status']
    probes={}
    for version in ('1.1.2','1.9.0'):
        item=json.loads((ROOT/f'results/brats_panel_checks_tm{version}.json').read_text())
        assert item['torchmetrics']==version and item['inpainting']=='0.0.12'
        probes[version]={key:item[key] for key in ('shape','seed','ssim_dimensionality','psnr_clamping')}
        probes[version]['within_case_rank_tau']=item['per_case_order']['kendall_tau_with_rmse_order']
        probes[version]['n_constructed_fills']=len(item['per_case_order']['fills'])
    manifest={'assessed_on':coverage['assessed_on'],'scope':coverage['assessment_scope'],
              'generator_sha256':sha(Path(__file__)),'coverage_specification_sha256':sha(specification),
              'sources':records,'saved_implementation_probes':probes}
    (HERE/'decision_evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    lines=['# BraTS inpainting: metric-selection questions and experimental coverage','',
           'Author clarification, 9 November 2026: the paper should advance the choice of evaluation for the next BraTS inpainting competition. A useful outcome is a justified narrowing of choices and explicit next tests; naming a winning metric is not required.',
           '',coverage['edition_scope'],'',coverage['coverage_definition'],
           '', '## Claim-first manuscript outline','',
           'Introduction: state the competition decision, distinguish observed-reference recovery from unobserved healthy counterfactual tissue, and define why similarity alone does not fix the evaluation target. Methods: retain the ladder as a candidate-audit instrument. Results: add existing official-evaluator probes before the donor comparisons. Synthesis: map each decision requirement to measured evidence and identify the unfilled model-validation step. Conclusion: report what has become testable rather than announcing an unsupported replacement metric.',
           '', '## Coverage by independent decision question','',
           '| ID | Subtask | Existing coverage | Consequence for selection |',
           '|---|---|---|---|']
    for row in coverage['questions']:
        lines.append('| '+ ' | '.join(row[key] for key in ('id','question','coverage','decision_progress'))+' |')
    for row in coverage['questions']:
        lines.extend(['',f"## {row['id']}: {row['question']}",'',row['evidence'],'',
                      '**Progress:** '+row['decision_progress'],'','**Still needed:** '+row['missing'],'',
                      '**Sources:** '+', '.join('`'+p+'`' for p in row['sources'])+'.'])
    p=probes['1.9.0']
    lines.extend(['','## Concrete implementation progress already available','',
        'These deterministic synthetic probes were previously executed and are inspected here, not rerun. They establish behavior of the recorded implementation, not clinical prevalence or server parity.',
        '', '| Probe | Saved value, torchmetrics 1.9.0 | Meaning |','|---|---|---|',
        f"| SSIM original / joint first-axis permutation | {p['ssim_dimensionality']['ssim_original']:.6f} / {p['ssim_dimensionality']['ssim_after_permuting_first_spatial_axis']:.6f} | Consistent with the documented 4D input being treated as 2D batches/channels |",
        f"| PSNR reported / unclamped-error expression | {p['psnr_clamping']['psnr_reported']:.6f} / {p['psnr_clamping']['psnr_from_unclamped_mse']:.6f} dB | Reference-range clipping changes the evaluated error |",
        f"| Plain PSNR / PSNR_eps rank agreement with RMSE | {p['within_case_rank_tau']['psnr']:.6f} / {p['within_case_rank_tau']['psnr_eps']:.6f} | Exact monotone redundancy applies only when the same error is scored; the plain variant is a counterexample |",
        '', 'The corresponding 1.1.2 source is retained separately. The constructed panel has eleven fills in one synthetic volume; this is not eleven independent patients or a study of model leaderboard stability.',
        '', '## What can be concluded now','',
        'The strongest progress is a set of discriminating controls and implementation specifications. MSE/RMSE quantify reference distortion; a mean advantage cannot by itself identify the wrong ranking under that objective. SSIM requires a precise spatial/implementation contract. Residual correlation measures direction and misses positive amplitude shrinkage; c is a calibration, not new identity information. TM/lDDT-style intensity scores remain provisional candidates. Face ArcFace is an indirect donor reference, and advanced iris metrics do not validate an MRI endpoint. No candidate has yet passed all requirements for selecting the next competition metric.',
        '', '## Next work ordered by value for the decision','',
        'First, freeze the intended output and loss tradeoff, mask/reference rules and evaluator contract. While the registration gate remains unresolved, continue only the authorized QC/source-integration and manuscript work. Then, under a separate frozen protocol consistent with that gate, compare genuinely different inpainting methods and controls on the same held-out brain cases, obtain an independent anatomy/usefulness criterion, and test candidate-method rankings with case-level uncertainty. Freeze a final metric or composite using development cases and validate it on untouched cases. Set stochastic-output and computational rules before seeing final ranks.',
        '', 'For native-grid inpainting outputs, target and prediction already share coordinates; unnecessary between-person registration is not a prerequisite for that future comparison. Donor-ladder registration remains a limitation of the present proxy evidence. No new experiment is launched by this assessment.',
        '', '## Explicit exclusions',''])
    for row in coverage['excluded_sources']:
        lines.append('- `'+row['path']+'`: '+row['reason'])
    lines.extend(['','## Source and verification boundary','',coverage['assessment_scope'],
                  ' Source hashes and extracted probe values are in `decision_evidence_manifest.json`. Coverage judgments are authored scientific assessments, not experimental measurements. Existing F/P labels and documentary reproduction verdicts remain separate.', ''])
    for row in coverage['external_sources']:
        lines.append('- ['+row['role']+']('+row['url']+'), checked '+row['checked_on']+'.')
    (HERE/'BRATS_METRIC_COVERAGE.md').write_text('\n'.join(lines)+'\n')
    print('Assessed 12 decision questions from saved sources; no new experiments or ranking.')

if __name__=='__main__':main()
