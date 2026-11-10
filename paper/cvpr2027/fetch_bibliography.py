"""Fetch publisher metadata for the manuscript bibliography, retaining exact source responses."""
import difflib
import json
from pathlib import Path
import re
import unicodedata
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'verification-cache'
BRATS_BIB = r'''@misc{brats2023,
  author = {Kofler, Florian and Meissen, Felix and Steinbauer, Felix and Graf, Robert and Ehrlich, Stefan K. and Reinke, Annika and others},
  title = {{The Brain Tumor Segmentation (BraTS) Challenge: Local Synthesis of Healthy Brain Tissue via Inpainting}},
  year = {2023},
  eprint = {2305.08992},
  archivePrefix = {arXiv},
  note = {Version 3 revised September 2024},
  url = {https://arxiv.org/abs/2305.08992v3}
}'''
DOIS = {
    'wang2004': '10.1109/TIP.2003.819861',
    'blau2018': '10.1109/CVPR.2018.00652',
    'zhang2018lpips': '10.1109/CVPR.2018.00068',
    'care2025': '10.52202/085713-1132',
    'lmsb2025': '10.52202/085713-0202',
    'papila2022': '10.1038/s41597-022-01388-1',
    'grape2023': '10.1038/s41597-023-02424-4',
    'hosseini2010': '10.1109/TIM.2009.2037996',
    'dobes2004': '10.1078/0030-4026-00388',
    'celeba2015': '10.1109/ICCV.2015.425',
    'rothe2018': '10.1007/s11263-016-0940-3',
    'metrics2024': '10.1038/s41592-023-02151-z',
    'daugman2004': '10.1109/TCSVT.2003.818350',
    'daugman2001': '10.1098/rspb.2001.1696',
    'arcface2022': '10.1109/TPAMI.2021.3087709',
    'tmscore2004': '10.1002/prot.20264',
    'lddt2013': '10.1093/bioinformatics/btt473',
}
SEARCH = {
    'brainid2024': 'Brain-ID: Learning Contrast-agnostic Anatomical Representations for Brain Imaging',
    'betti2023': 'Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes',
    'fwd2025': 'Frechet Wavelet Distance: A Domain-Agnostic Metric for Image Generation',
}
OFFICIAL = {
    'betti2023': 'https://proceedings.mlr.press/v202/stucki23a.html',
    'fwd2025': 'https://proceedings.iclr.cc/paper_files/paper/2751-/bibtex',
}
OFFICIAL_FIELDS = {
    'betti2023': {'title': 'Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes',
                 'author': 'Stucki, Nico and Paetzold, Johannes C. and Shit, Suprosanna and Menze, Bjoern and Bauer, Ulrich',
                 'booktitle': 'Proceedings of the 40th International Conference on Machine Learning',
                 'volume': '202', 'pages': '32698--32727', 'year': '2023',
                 'eprint': '2211.15272', 'archivePrefix': 'arXiv',
                 'url': 'https://proceedings.mlr.press/v202/stucki23a.html'},
    'fwd2025': {'title': 'Fréchet Wavelet Distance: A Domain-Agnostic Metric for Image Generation',
               'author': 'Veeramacheneni, Lokesh and Wolter, Moritz and Kuehne, Hilde and Gall, Juergen',
               'booktitle': 'International Conference on Learning Representations', 'year': '2025',
               'eprint': '2312.15289', 'archivePrefix': 'arXiv',
               'url': 'https://proceedings.iclr.cc/paper_files/paper/2025/hash/6605c7d1c1ae1521b328d036df5be1ae-Abstract-Conference.html'},
}


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'IdentityLadder-manuscript/1.0'})
    with urllib.request.urlopen(req, timeout=25) as response:
        return json.load(response)


def normal(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    return re.sub('[^a-z0-9]', '', text.lower())


def tex(text):
    text = str(text).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('–', '--').replace('—', '--')
    return text.replace('š', r'\v{s}').replace('Š', r'\v{S}').replace('\u00a0', ' ')


def bib(key, record):
    kind = 'inproceedings' if record['type'] in ('proceedings-article', 'book-chapter') else 'article'
    authors = ' and '.join(a.get('family', a.get('name', '')) + (', ' + a['given'] if 'given' in a else '') for a in record['author'])
    year = record.get('published-print', record.get('published', record.get('issued')))['date-parts'][0][0]
    fields = {'author': authors, 'title': record['title'][0], 'year': year}
    venue = record.get('container-title', [''])[0]
    fields['booktitle' if kind == 'inproceedings' else 'journal'] = venue
    for source, target in [('volume','volume'), ('issue','number'), ('page','pages'), ('DOI','doi')]:
        if record.get(source):
            fields[target] = record[source]
    if key == 'brainid2024':
        fields['booktitle'] = 'European Conference on Computer Vision (ECCV)'
    return '@' + kind + '{' + key + ',\n' + ',\n'.join('  ' + k + ' = {' + (str(v) if k in ('doi','url','eprint') else tex(v)) + '}' for k,v in fields.items()) + '\n}'


def main():
    CACHE.mkdir(exist_ok=True)
    records = {}
    for key, doi in DOIS.items():
        path = CACHE / (key + '.crossref.json')
        try:
            if not path.exists():
                path.write_text(json.dumps(get('https://api.crossref.org/works/' + urllib.parse.quote(doi, safe=''))))
            records[key] = json.loads(path.read_text())['message']
            print(key, records[key]['title'][0], flush=True)
        except Exception as exc:
            print(key, 'FETCH FAILED:', type(exc).__name__, flush=True)
    for key, title in SEARCH.items():
        if key in OFFICIAL:
            continue
        path = CACHE / (key + '.crossref.json')
        if not path.exists():
            response = get('https://api.crossref.org/works?rows=5&query.title=' + urllib.parse.quote(title))
            candidates = response['message']['items']
            best = max(candidates, key=lambda r: difflib.SequenceMatcher(None, normal(title), normal(r.get('title',[''])[0])).ratio())
            score = difflib.SequenceMatcher(None, normal(title), normal(best['title'][0])).ratio()
            if score < .92:
                print(key, 'NO SAFE TITLE MATCH', score, flush=True)
                continue
            path.write_text(json.dumps({'message':best}))
        records[key] = json.loads(path.read_text())['message']
        print(key, records[key]['title'][0], records[key]['DOI'], flush=True)
    (CACHE/'metadata_index.json').write_text(json.dumps({k:{'title':r['title'][0], 'doi':r['DOI']} for k,r in records.items()},indent=2)+'\n')
    extra=[]
    for key,url in OFFICIAL.items():
        path=CACHE/(key+'.official.bib')
        if not path.exists():
            # Transcribed from the official proceedings page read during source
            # verification when its separate BibTeX download endpoint timed out.
            fields=OFFICIAL_FIELDS[key]
            path.write_text('@inproceedings{'+key+',\n'+',\n'.join('  '+k+' = {'+tex(v)+'}' for k,v in fields.items())+'\n}\n')
        entry=path.read_text().strip()
        if key=='betti2023' and not re.search(r'\beprint\s*=',entry):
            entry=entry[:-1].rstrip().rstrip(',')+',\n  eprint = {2211.15272},\n  archivePrefix = {arXiv}\n}'
        extra.append(entry)
        print(key,'official-proceedings metadata retained (see citation_report.md)',flush=True)
    # Primary arXiv record checked on 9 November 2026; year is first posting,
    # while the separately stated version date identifies the task paper read.
    (CACHE/'brats2023.verified.bib').write_text(BRATS_BIB+'\n')
    (HERE/'main.bib').write_text('% Publisher metadata; see fetch_bibliography.py and citation_report.md.\n\n'+'\n\n'.join([bib(k,r) for k,r in records.items()]+extra+[BRATS_BIB])+'\n')


if __name__ == '__main__':
    main()
