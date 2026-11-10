"""Record source support for a proposed population audit; no model or metric run."""
from pathlib import Path
import hashlib,json,subprocess
import source_snapshot

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    spec=HERE/'population_audit.json'
    data=json.loads(spec.read_text())
    assert [r['id'] for r in data['questions']]==[f'P{i:02}' for i in range(1,13)]
    sources={p for row in data['questions'] for p in row['sources']}|set(data['excluded_sources'])
    records={}
    for rel in sorted(sources):
        p=ROOT/rel
        assert p.is_file(),rel
        records[rel]=source_snapshot.record(p)
    for rel in data['excluded_sources']:
        assert 'WITHDRAWN' in source_snapshot.load_json(ROOT/rel)['status']
    manifest={'assessed_on':data['assessed_on'],'scope':data['scope'],
              'population_audit_spec_sha256':sha(spec),'generator_sha256':sha(Path(__file__)),
              'sources':records,'generated_model_outputs_evaluated':False}
    (HERE/'population_audit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Evidence coverage for population audits of conditional generation','',data['scope'],'',
           '| ID | Question | Evidence coverage | Missing validation |','|---|---|---|---|']
    for row in data['questions']:
        lines.append('| '+' | '.join(row[k] for k in ('id','question','coverage','missing'))+' |')
    lines.extend(['','The complete scope and sources are defined in population_audit.json. POPULATION_AUDIT_FRAMING.md distinguishes target strata, donor references and acquisition domains. The earlier BraTS decision map is retained as an application-specific history, not the current scope.'])
    (HERE/'POPULATION_AUDIT_COVERAGE.md').write_text('\n'.join(lines)+'\n')
    print('Recorded 12 population-audit questions and source hashes; no generator outputs scored.')

if __name__=='__main__':main()
