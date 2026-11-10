"""Map all tracked scientific artifacts to manuscript roles without inspecting private data."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def role(path):
    name=path.lower()
    if name in ('results/real_cases.json','results/real_cases_parietal.json'):
        return ('Withdrawn MRI results','Excluded: source explicitly invalidates c/rho/label results after the intensity-scaling failure; not provisional support')
    if 'brats_panel_checks' in name:
        return ('Official evaluator probes','Main implementation audit and supplement; synthetic code behavior, no model ranking or organizer-server validation')
    if 'paper/cvpr2027/' in name:return ('Current manuscript','Main/supplement or assembly record')
    if 'paper/isbi2027/' in name:return ('Related ISBI','Related-work/overlap review; source preserved, not copied as a new contribution')
    if 'care' in name:return ('CARE reproduction','Related work and supplementary aggregation audit; not an image-level reconstruction rerun')
    if 'registration' in name or '/reg_' in name or '/protocol_registration' in name:
        return ('Registration','Methods, source-version interpretation and limitations; unresolved gate retained')
    if 'donor_ladder_brain' in name or 'size_shell' in name:return ('Brain donor ladder','Main brain results and supplementary maps/sensitivity; status per source')
    if 'face' in name and 'drawn_face' not in name:return ('Face donor ladder','Main face panel; historical/synthetic drawings distinguished from measured v6f')
    if 'iris' in name:return ('Iris','Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review')
    if 'fundus' in name or 'donor_ladder_2d' in name or 'ladder2d' in name or 'grape' in name:
        return ('Fundus and earlier 2D','Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence')
    if 'metrics_correlation' in name or 'metrics_blocks' in name or 'metrics_polar' in name:
        return ('Metric comparison','Main profiles and supplementary dependence; do not count monotone variants as independent evidence')
    if 'population_null' in name or 'toy_ladder' in name or 'label_' in name or 'pupil_' in name or 'mixing_' in name or 'synthetic' in name:
        return ('Mechanism controls','Supplementary synthetic controls; toy evidence, no biological validation')
    if 'real_case' in name or 'real_atlas' in name or 'real_detail' in name or 'visual_rebuild' in name:
        return ('Real-case demonstrations','Supplementary preparation history; fixed examples are not cohort validation')
    if 'cross_modality/' in name:return ('Cross-modality exploration','Supplementary exploratory context; no universal fused identity score')
    if 'simon' in name or 'image_ladder' in name or 'baseline_v0' in name or 'grape_repro' in name:
        return ('Earlier pilots','Historical context; acquisition/protocol confounding prevents promotion to current primary results')
    if 'metrics_status' in name:return ('Metric board','Operational generated view; not an additional experiment or independent result')
    return ('Supporting material','Methods/provenance/reference context; review separately before making an empirical claim')


def main():
    tracked=subprocess.check_output(['git','-C',str(ROOT),'ls-files'],text=True).splitlines()
    candidates=[p for p in tracked if (p.startswith(('results/','docs/','paper/isbi2027/'))
                                      and Path(p).suffix.lower() in ('.json','.csv','.md','.pdf','.png','.tex','.bib'))]
    records=[]
    for rel in candidates:
        path=ROOT/rel
        family,placement=role(rel)
        records.append({'path':rel,'family':family,'placement':placement,'bytes':path.stat().st_size,
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    counts=Counter(record['family'] for record in records)
    (HERE/'materials_inventory.json').write_text(json.dumps({'snapshot':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                                                           'scope':'Tracked material inventory; no private participant data opened.',
                                                           'counts':dict(counts),'artifacts':records},indent=2)+'\n')
    lines=['# Materials coverage map','',
           'This inventory maps available tracked materials to text roles. It is an editorial coverage record, not a claim that every historical artifact is a current result. PNG/PDF pairs are duplicate renderings, and JSON/CSV summaries may describe the same run.',
           '', '| Material family | Files | Text role |','|---|---:|---|']
    for family,n in sorted(counts.items()):
        example=next(r for r in records if r['family']==family)
        lines.append(f"| {family} | {n} | {example['placement']} |")
    lines += ['', '## Frozen other-branch inputs','',
              'The bilateral snapshot 82ed7fe supplies results/donor_ladder_2d_facefmt.json, results/fundus_ladder_metrics.json, the published aggregate atlas and its generator. Original JSON bytes are retained in inputs/ and source hashes/commits in evidence_manifest.json and atlas_provenance.json. These are frozen Git artifacts, not the currently running fundus extension.',
              '', '## Material not available in this host snapshot','',
              'The base face pipeline/protocol has now been recovered, and full iris P3 plus newer brain/face/IXI exports are present after the fork update. The manuscript still uses its explicitly pinned a48523d evidence, pending scientific integration review. Private animation and the latest fundus extension remain outside this draft. An available export is not automatically reviewed evidence.',
              '', '## Artifact-by-artifact placement','',
              '| Artifact | Family | Placement / exclusion reason |','|---|---|---|']
    for r in records:
        lines.append(f"| `{r['path']}` | {r['family']} | {r['placement']} |")
    (HERE/'MATERIALS_COVERAGE.md').write_text('\n'.join(lines)+'\n')
    print('Inventoried',len(records),'tracked scientific artifacts across',len(counts),'families.')


if __name__=='__main__':main()
