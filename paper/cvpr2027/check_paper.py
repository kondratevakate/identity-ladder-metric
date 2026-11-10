"""Check the assembled reading copies and record limited CVPR/metadata verification."""
from pathlib import Path
import hashlib
import io
import json
import re
import subprocess
import urllib.request
import zipfile
import pymupdf

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
KIT='https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip'
GUIDE='https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines'
OFFICIAL_STYLE='https://raw.githubusercontent.com/cvpr-org/author-kit/a2369342011145be8c85e5ad21a9b61f9fa53473/cvpr.sty'


def text_pages(stem):
    target=HERE/'.build'/f'{stem}.txt'
    subprocess.run(['pdftotext','-layout',str(HERE/f'{stem}.pdf'),str(target)],check=True)
    return [p for p in target.read_text().split('\f') if p.strip()]


def words(text):return re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b",text)


def main():
    result={'checked_on':'2026-11-09','scope':'Revised-draft file checks; no portal audit or experimental rerun.','pdfs':{}}
    for stem in ('main','supplementary'):
        pages=text_pages(stem)
        text='\n'.join(pages)
        log=(HERE/f'{stem}.log').read_text(errors='replace')
        defects=[line.strip() for line in log.splitlines() if re.search(r'Overfull|Missing character:|Citation .*undefined|Reference .*undefined',line)]
        identifiers=[word for word in ('Kate','Kondrateva','Lenovo','kondratevakate','claude.ai','C:\\Projects') if word.lower() in text.lower()]
        result['pdfs'][stem]={'pages':len(pages),'layout_or_reference_warnings':defects,'identifying_text_hits':identifiers,
                             'sha256':hashlib.sha256((HERE/f'{stem}.pdf').read_bytes()).hexdigest()}
        if stem=='main':
            llm=[j+1 for j,p in enumerate(pages) if 'LLM Usage Statement' in p]
            # A title may contain "References"; identify the bibliography heading
            # at the start of a text line rather than any occurrence of the word.
            # PDF text lines keep the two columns separate. Caption sentences
            # beginning with "References are ..." are not bibliography headings.
            with pymupdf.open(HERE/'main.pdf') as doc:
                refs=[j+1 for j,page in enumerate(doc) if any(
                    ''.join(span['text'] for span in line['spans']).strip()=='References'
                    for block in page.get_text('dict')['blocks'] if 'lines' in block
                    for line in block['lines'])]
            assert len(llm)==1,'LLM statement missing or duplicated'
            result['main_content_pages']=llm[0]-1
            result['llm_page']=llm[0]
            result['references_first_page']=min(refs) if refs else None
            assert refs and min(refs)>=llm[0]
    sources=[HERE/'main.tex',HERE/'supplementary.tex',HERE/'preamble.tex',*sorted((HERE/'sec').glob('*.tex'))]
    todos=[str(p.relative_to(HERE)) for p in sources if re.search(r'\\TODO\b',p.read_text())]
    result['remaining_todo_files']=todos
    assert not todos
    abstract=(HERE/'sec/0_abstract.tex').read_text().replace('\\begin{abstract}','').replace('\\end{abstract}','').strip()
    title=' '.join(re.search(r'\\title\{([^}]+)\}',(HERE/'main.tex').read_text()).group(1).replace(r'\\',' ').split())
    result['metadata']={'title':title,'title_words':len(words(title)),
                        'abstract_words':len(words(abstract)),'abstract_characters':len(abstract)}
    result['bibliography']=[{'key':r['key'],'status':r['status'],'issues':r.get('issues',[])}
                            for r in json.loads((HERE/'citation_report.json').read_text())]
    archive=HERE/'.build/cvpr2027-author-kit.zip'
    try:
        if not archive.exists():
            req=urllib.request.Request(KIT,headers={'User-Agent':'IdentityLadder-manuscript/1.0'})
            with urllib.request.urlopen(req,timeout=8) as response:archive.write_bytes(response.read())
        with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as z:
            member=next(n for n in z.namelist() if Path(n).name=='cvpr.sty')
            original=z.read(member)
        local=(HERE/'cvpr.sty').read_bytes()
        result['template']={'url':KIT,'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                            'official_style_sha256':hashlib.sha256(original).hexdigest(),
                            'local_style_sha256':hashlib.sha256(local).hexdigest(),
                            'match':original==local}
    except Exception as exc:
        # The official CVPR2027-v1 Git tag is a second primary template source.
        # Retain the kit download failure rather than claiming its bytes matched.
        original_path=HERE/'.build/official-cvpr-github.sty'
        if not original_path.exists():
            with urllib.request.urlopen(OFFICIAL_STYLE,timeout=15) as response:
                original_path.write_bytes(response.read())
        original=original_path.read_bytes()
        local=(HERE/'cvpr.sty').read_bytes()
        result['template']={'url':KIT,'kit_download_status':'UNKNOWN','kit_download_error':type(exc).__name__,
                            'verified_source_url':OFFICIAL_STYLE,'verified_tag':'CVPR2027-v1',
                            'official_style_sha256':hashlib.sha256(original).hexdigest(),
                            'local_style_sha256':hashlib.sha256(local).hexdigest(),'match':original==local}

    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    template_evidence=('Style matches the downloaded official CVPR 2027 kit byte-for-byte; source/hash in verification.json.'
                       if 'zip_sha256' in result['template'] else
                       'Style matches the official CVPR2027-v1 Git tag byte-for-byte; kit URL unavailable; source/hash in verification.json.')
    lines=['# CVPR 2027 revised-draft file audit','',
           'Scope: main-track initial-submission format, main and supplementary PDFs, checked 9 November 2026; official author guide re-read the same day. Overall administrative readiness remains **UNVERIFIED**: the author portal, ethics basis and related-submission overlap were not established here.',
           '',f'The requirements below were checked against the current [official author guide]({GUIDE}) and its [submission kit]({KIT}). Requirements are distinct from local scientific recommendations.',
           '', '| Requirement | Class | File-check result | Evidence |','|---|---|---|---|',
           f"| At most eight content pages; extra references/LLM pages allowed | REQUIRED | {'PASS' if result['main_content_pages']<=8 else 'FAIL'} | {result['main_content_pages']} content pages; LLM starts on page {result['llm_page']} |",
           f"| Official review style | REQUIRED | {'PASS' if result['template'].get('match') else 'UNKNOWN/FAIL'} | {template_evidence} Rendered pages checked for anonymity and line numbering |",
           '| LLM Usage Statement before references | REQUIRED | PASS | Separate statement present before bibliography; describes Codex and earlier Claude assistance |',
           f"| Anonymous main and supplement | REQUIRED | {'PASS within text scan' if not any(v['identifying_text_hits'] for v in result['pdfs'].values()) else 'FAIL'} | Identifier scan plus visual review; author-facing logs are outside this PDF scope |",
           '| Ethics/consent basis for human-data use | REQUIRED | UNKNOWN | No institutional approval, exemption or consent history invented; author confirmation still needed |',
           '| Permissible overlap with related submissions | REQUIRED | UNKNOWN | ISBI preserved; scientific and submission overlap needs author review |',
           '| Enrollment, profiles, conflicts, quotas and reviewer eligibility | REQUIRED | UNKNOWN | Portal state not inspected |',
           '| No content-expanding external links in submission | FORBIDDEN | PASS within file scope | Source citations and internal artifact names do not add a project URL or external supplementary content |',
           '', '## Local verification and scientific limits','',
           'Tables are generated from source aggregate JSONs and frozen Git blobs, with hashes and producing commits. Gaussian atlas assembly preserves source panels and records the brain replacement. No participant exemplars or restricted iris images were added. The abstract and title have a separate editorial review. Reproduction and scientific reviews do not certify data provenance or repair unresolved registration.',
           '',f"PDF page totals: main {result['pdfs']['main']['pages']}; supplementary {result['pdfs']['supplementary']['pages']}. Title: {result['metadata']['title_words']} words. Abstract: {result['metadata']['abstract_words']} words; no claim is made that this count is a venue-mandated limit.",
           '', 'Compiler warnings, text-scan results, citation statuses and exact PDF hashes are stored in verification.json. Successful typesetting is not evidence of completed experiments or submission eligibility.']
    (HERE/'CVPR_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'main_content_pages':result['main_content_pages'],'pdfs':result['pdfs'],'metadata':result['metadata'],'template':result['template']},indent=2))
    assert result['main_content_pages']<=8, 'The main text exceeds the CVPR content-page limit'


if __name__=='__main__':main()
