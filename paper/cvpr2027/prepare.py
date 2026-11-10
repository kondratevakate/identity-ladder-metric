"""Build paper tables and aggregate-only figures from a frozen Git result snapshot.

No participant data are opened and no experiments or statistical tests are rerun.
Every copied figure and selected result records its source hash and producing commit.
"""
from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess
import source_snapshot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAB = HERE / 'tab'
FIG = HERE / 'fig'
SOURCES = ['donor_ladder_brain_metrics', 'donor_ladder_brain_bilateral',
           'donor_ladder_brain_maps', 'donor_ladder_brain_map_grid', 'donor_ladder_brain_size',
           'donor_ladder_brain_haca3_stage1', 'donor_ladder_2d_p7', 'face_ladder_metrics',
           'iris_ladder_metrics', 'donor_ladder_metrics_correlation', 'population_null_synthetic',
           'pupil_ladder', 'label_robustness', 'brats_panel_checks_tm1.1.2', 'brats_panel_checks_tm1.9.0']
GIT_INPUTS = {
    'fundus_framefmt': ('82ed7fe8a52e557854cf4cd2d02eb47d5803d7cb', 'results/donor_ladder_2d_facefmt.json'),
    'fundus_metrics_v1': ('82ed7fe8a52e557854cf4cd2d02eb47d5803d7cb', 'results/fundus_ladder_metrics.json'),
}


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(path):
    return source_snapshot.record(path)


def tex(text):
    return str(text).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def estimate(item, field='R', status='P', ci=True):
    if item is None:
        return r'\textit{undefined}'
    value = item.get(field)
    if value is None or not math.isfinite(value):
        return r'\textit{unavailable}'
    tail = r'\statuspre' if status == 'P' else r'\statusfinal'
    text = f'{value:.3f}'
    if ci and item.get('ci95'):
        lo, hi = item['ci95']
        assert all(math.isfinite(x) for x in (lo, hi)) and lo <= hi
        return r'\shortstack{' + text + tail + r'\\{\scriptsize [' + f'{lo:.3f}, {hi:.3f}' + r']}}'
    return text + tail


def rows(path, entries):
    lines = [' & '.join(row) + ' ' + chr(92)*2 for row in entries]
    path.write_text('\n'.join(lines) + '\n' + chr(92) + 'bottomrule\n', encoding='utf-8')


def main():
    TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    data = {name: source_snapshot.load_json(ROOT/'results'/f'{name}.json') for name in SOURCES}
    manifest = {'date': '2026-11-09', 'base_commit': git('rev-parse', 'HEAD'),
                'evidence_snapshot': source_snapshot.EVIDENCE_COMMIT,
                'generator': 'paper/cvpr2027/prepare.py', 'generator_sha256': sha(Path(__file__)),
                'scope': 'Saved aggregate evidence only; no raw human images or rows read; no experiments rerun.',
                'sources': {name: source_record(ROOT/'results'/f'{name}.json') for name in SOURCES},
                'selected_records': {}, 'figures': {}}
    inputs = HERE/'inputs'; inputs.mkdir(exist_ok=True)
    for name,(revision,path) in GIT_INPUTS.items():
        raw = subprocess.check_output(['git','-C',str(ROOT),'show',revision+':'+path])
        data[name] = json.loads(raw)
        (inputs/Path(path).name).write_bytes(raw)
        manifest['sources'][name] = {'path':path,'sha256':hashlib.sha256(raw).hexdigest(),
                                    'git_snapshot':revision,'producing_commit':git('log',revision,'-1','--format=%H','--',path)}
    brain = data['donor_ladder_brain_metrics']['views']['random_patches/LR']
    face = data['face_ladder_metrics']
    iris = data['iris_ladder_metrics']
    chosen = manifest['selected_records']

    coverage_path=HERE/'population_audit.json'
    coverage=json.loads(coverage_path.read_text())
    manifest['population_audit']={'path':'paper/cvpr2027/population_audit.json','sha256':sha(coverage_path),
                                   'scope':'Authored qualitative assessment, not an experimental completion percentage.'}
    rows(TAB/'decision_coverage.tex',[[tex(cell) for cell in row] for row in coverage['main_table']])
    full=[]
    for row in coverage['questions']:
        full.append([row['id'],tex(row['question'])+r'\par\emph{'+tex(row['coverage'])+'}',
                     tex(row['evidence'])+r'\par\textbf{Missing:} '+tex(row['missing'])])
    rows(TAB/'decision_questions.tex',full)
    probe_rows=[]
    for version in ('1.1.2','1.9.0'):
        p=data['brats_panel_checks_tm'+version]
        d=p['ssim_dimensionality']; c=p['psnr_clamping']; t=p['per_case_order']['kendall_tau_with_rmse_order']
        probe_rows.append([version,f"{d['ssim_original']:.6f} / {d['ssim_after_permuting_first_spatial_axis']:.6f}",
                           f"{c['psnr_reported']:.3f} / {c['psnr_from_unclamped_mse']:.3f}",
                           f"{t['psnr']:.3f} / {t['psnr_eps']:.3f}"])
        chosen['implementation_probes/'+version]={'ssim_dimensionality':d,'psnr_clamping':c,'rank_tau':t,
                                                 'n_constructed_fills':len(p['per_case_order']['fills'])}
    rows(TAB/'brats_implementation.tex',probe_rows)

    entries = []
    for metric, label in [('mse','MSE'), ('ssim','SSIM'), ('acc_dataset','Residual correlation'),
                          ('tm_like','TM-like'), ('lddt_rep','lDDT-like, repeat tolerance')]:
        values = [brain['ladder_R'][metric].get(rung) for rung in ('dataset_mean','mirror','wrong_detail','noise')]
        entries.append([label] + [estimate(item) for item in values])
        chosen['brain/'+metric] = dict(zip(('dataset_mean','mirror','wrong_detail','noise'),values))
    rows(TAB/'brain_profile.tex', entries)

    entries=[]
    for ds,label in [('celeba','CelebA'),('imdb','IMDb-WIKI')]:
        v=face['views'][ds]['ladder_R']
        for metric,mlabel in [('mse','MSE'),('ssim','SSIM'),('arcface_cos','ArcFace donor reference')]:
            selected={rung:v.get(metric,{}).get(rung) for rung in ('retest','mirror','dataset_mean','lookalike')}
            entries.append([label,mlabel]+[estimate(item,status='F',ci=False) if item is not None
                                           else r'\textit{unavailable}' for item in selected.values()])
            chosen['face/'+ds+'/'+metric]=selected
    rows(TAB/'face_profile.tex', entries)

    entries=[]
    for ds,label in [('UTIRIS_NIR','UTIRIS NIR'),('UTIRIS_VIS','UTIRIS VIS'),('UPOL','UPOL')]:
        selected={r:iris['results'][ds].get(r) for r in ('repeat','fellow','lookalike','stranger')}
        entries.append([label]+[estimate(item,'ratio') for item in selected.values()])
        chosen['iris/'+ds+'/ratio']=selected
    rows(TAB/'iris_ratio.tex', entries)

    entries=[]
    for ds in ('GRAPE','PAPILA'):
        result=data['fundus_framefmt']['results'][ds]
        selected={r:result.get(r) for r in ('repeat','fellow','lookalike','stranger')}
        entries.append([ds+' '+result['arm'].upper()]+[estimate(item,'ratio') if item else r'\textit{unavailable}' for item in selected.values()])
        chosen['fundus/'+ds+'/framefmt']=selected
    rows(TAB/'fundus_ratio.tex',entries)
    chosen['fundus/metric_registry']=data['fundus_metrics_v1']['metrics_confirmatory']
    chosen['iris/metric_registry']=iris['metrics_confirmatory']
    registry_rows=[]
    for metric,item in data['fundus_metrics_v1']['metrics_confirmatory'].items():
        registry_rows.append([tex(metric),tex(item['status']),tex(iris['metrics_confirmatory'].get(metric,{}).get('status','unavailable'))])
    rows(TAB/'metric_registry.tex',registry_rows)

    care=source_snapshot.load_json(ROOT/'results/care_iteration2.json')
    from collections import Counter
    care_counts=Counter(item['status'] for item in care.get('rows',[]))
    rows(TAB/'care_status.tex',[[label,str(care_counts.get(key,0))] for key,label in (
        ('matched','Within the recorded rounding tolerance'),('mismatch','Outside that tolerance'),
        ('blocked_coverage','Blocked by coverage'),('unavailable','No finite available comparison'))])
    manifest['sources']['care_iteration2']=source_record(ROOT/'results/care_iteration2.json')
    chosen['care/aggregation_status_counts']=dict(care_counts)

    # Keep full-source coverage and endpoint-specific counts alongside every selected profile.
    cohort_entries=[]
    b=data['donor_ladder_brain_metrics']['counts']
    cohort_entries.append(['QTIM',str(b['n_targets']),str(b['n_families']),str(b['n_repeat_scans']),r'Affine, gate pending\statuspre'])
    for ds,label in [('celeba','CelebA'),('imdb','IMDb-WIKI')]:
        counts=face['counts'][ds]
        cohort_entries.append([label,str(counts['n_targets']),str(counts['n_families']),r'Endpoint-specific',r'v6f frame approved\statusfinal'])
    for ds,label in [('UTIRIS_NIR','UTIRIS NIR'),('UTIRIS_VIS','UTIRIS VIS'),('UPOL','UPOL')]:
        counts=iris['panel'][ds]['counts']
        cohort_entries.append([label,str(counts['units']),str(counts['persons']),str(counts['eyes_with_repeat']),r'P2 mask QC note\statuspre'])
    for ds in ('GRAPE','PAPILA'):
        result=data['fundus_framefmt']['results'][ds]
        cohort_entries.append([ds+' '+result['arm'].upper(),str(result['units']),str(result['stranger']['persons']),
                               str(result['repeat']['targets']) if result.get('repeat') else r'Unavailable',r'Frame-only\statuspre'])
    rows(TAB/'coverage.tex',cohort_entries)

    tests=[]
    for metric,label in [('mse','MSE'),('ssim','SSIM'),('acc_dataset','Residual correlation'),
                         ('tm_like','TM-like'),('lddt_rep','lDDT-like, repeat tolerance')]:
        selected=brain['tests']['per_metric'][metric]
        cells=[]
        for test in ('T1','T2','T3a','T3b'):
            item=selected[test]
            if 'p_holm' not in item:
                cells.append(r'\notrun')
            else:
                outcome='pass' if item['pass_holm_0.05'] else 'not established'
                cells.append(f"{outcome} ({item['p_holm']:.2g})"+r'\statuspre')
        tests.append([label]+cells)
        chosen['brain/tests/'+metric]=selected
    rows(TAB/'brain_tests.tex',tests)

    supplementary=[]
    for metric, rungs in brain['ladder_R'].items():
        for rung in ('dataset_mean','mirror','repeat','shrink_repeat','wrong_detail','noise'):
            item=rungs.get(rung)
            supplementary.append([tex(metric),tex(rung),estimate(item),
                                  str(item['n_targets']) if item else r'\textit{undefined}',
                                  str(item['n_families']) if item else r'\textit{undefined}'])
    rows(TAB/'brain_full.tex',supplementary)

    # A compact figure uses only scalar aggregate outputs; it contains no human-image exemplars.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.0),sharey=True)
    panels=[('QTIM: provisional',brain['ladder_R'],'R: repeat = 0'),
            ('CelebA: approved frame',face['views']['celeba']['ladder_R'],'R_self: target = 0'),
            ('UTIRIS NIR: P2 panel',iris['panel']['UTIRIS_NIR']['R_self'],'R_self: target = 0')]
    rungs=['dataset_mean','mirror','noise']
    labels=['Dataset mean','Bilateral','Noise']
    colors={'mse':'#2877a0','ssim':'#c36533','acc_dataset':'#657f46'}
    for ax,(title,panel,scale) in zip(axes,panels):
        for k,(metric,color) in enumerate(colors.items()):
            for j,rung in enumerate(rungs):
                actual='fellow' if 'UTIRIS' in title and rung=='mirror' else rung
                item=panel.get(metric,{}).get(actual)
                if not item:
                    continue
                val=item['R'];lo,hi=item['ci95']
                ax.errorbar(val,j+(k-1)*.16,xerr=[[max(0,val-lo)],[max(0,hi-val)]],fmt='o',color=color,
                            capsize=2,markersize=4,label={'mse':'MSE','ssim':'SSIM','acc_dataset':'Residual correlation'}[metric] if j==1 else None)
        ax.axvline(1,color='.55',ls=':',lw=1)
        ax.set_title(title,fontsize=9);ax.set_xlabel(scale+'; stranger = 1',fontsize=8)
        ax.set_xlim(-.05,1.35);ax.set_yticks(np.arange(3),labels);ax.invert_yaxis()
        ax.spines[['top','right']].set_visible(False)
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,1.12),ncol=3,fontsize=8,frameon=False)
    fig.tight_layout()
    fig.savefig(FIG/'audit_profiles.pdf',bbox_inches='tight')
    fig.savefig(HERE/'.build/audit_profiles.png',dpi=180,bbox_inches='tight');plt.close(fig)
    manifest['figures']['audit_profiles.pdf']={'generator':'paper/cvpr2027/prepare.py','generator_sha256':sha(Path(__file__)),
        'assembly_base_commit':manifest['base_commit'],'inputs':[
        manifest['sources'][name] for name in ('donor_ladder_brain_metrics','face_ladder_metrics','iris_ladder_metrics')],
        'sha256':sha(FIG/'audit_profiles.pdf')}

    for original,destination in [('fig_ladder_maps_common.pdf','brain_gaussian_maps.pdf'),
                                  ('fig_metrics_correlation.pdf','metric_correlations.pdf'),
                                  ('fig_size_shell_dependence.pdf','size_shell_sensitivity.pdf')]:
        source=ROOT/'results/figures'/original
        (FIG/destination).write_bytes(source_snapshot.read_bytes(source))
        manifest['figures'][destination]=source_record(source)
        manifest['figures'][destination]['copied_sha256']=sha(FIG/destination)

    (HERE/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for record in manifest['sources'].values():
        raw = (subprocess.check_output(['git','-C',str(ROOT),'show',record['git_snapshot']+':'+record['path']])
               if 'git_snapshot' in record else (ROOT/record['path']).read_bytes())
        assert hashlib.sha256(raw).hexdigest()==record['sha256']
    print('Prepared aggregate-only tables, profiles and copied Gaussian maps; no experiments run.')


if __name__=='__main__':
    main()
