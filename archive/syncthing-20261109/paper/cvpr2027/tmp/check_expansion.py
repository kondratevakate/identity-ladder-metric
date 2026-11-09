import hashlib
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw

here = Path(__file__).resolve().parent.parent
root = here.parent.parent
evidence = json.loads((here / 'evidence.json').read_text())
inventory = json.loads((here / 'artifact_inventory.json').read_text())
for rel, item in inventory['files'].items():
    assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == item['sha256'], rel
for rel, expected in evidence['sources'].items():
    assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == expected, rel
assert inventory['files']['paper/isbi2027/overleaf/main.tex']['sha256'] == evidence['source_isbi_tex_sha256']
assert inventory['files']['paper/isbi2027/overleaf/refs.bib']['sha256'] == evidence['source_isbi_bib_sha256']

current = (here / 'main.tex').read_text()
before = (here / 'tmp/before_text_expansion.tex').read_text()
words = lambda text: len(re.findall(r'\b[A-Za-z]+(?:[-\x27][A-Za-z]+)*\b', text))
numeric_cells = 0
for item in evidence['representation_comparisons'].values():
    for condition in ('same_unit', 'fellow_eye'):
        assert f"{item[condition]['mean_c_auc']:.3f}" in current
        numeric_cells += 1
log = (here / 'main.log').read_text(errors='replace')
assert not re.search(r'Overfull \\[hv]box|undefined references|Citation .* undefined|Missing character:', log)
pages = (here / 'tmp/expanded_layout.txt').read_text().split('\f')
pages = [page for page in pages if page.strip()]
assert len(pages) == 11
assert all('\ufffd' not in page for page in pages)

images = sorted((here / 'tmp').glob('expanded-page-*.png'))
assert len(images) == len(pages)
thumb_w, thumb_h, label_h = 510, 660, 25
for start in range(0, len(images), 6):
    batch = images[start:start + 6]
    sheet = Image.new('RGB', (3 * thumb_w, 2 * (thumb_h + label_h)), '#d8dce1')
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(batch):
        im = Image.open(path).convert('RGB')
        im.thumbnail((thumb_w, thumb_h))
        col, row = index % 3, index // 3
        x, y = col * thumb_w, row * (thumb_h + label_h)
        sheet.paste(im, (x, y + label_h))
        draw.text((x + 8, y + 5), path.stem, fill='black')
    sheet.save(here / f'tmp/contact-{start // 6 + 1}.png')

summary = {
    'source_files_unchanged': len(inventory['files']),
    'new_representation_score_cells_checked': numeric_cells,
    'approximate_tex_word_count_before': words(before),
    'approximate_tex_word_count_after': words(current),
    'approximate_net_word_increase': words(current) - words(before),
    'pdf_pages': len(pages),
    'page_sections': {str(i + 1): [line.strip() for line in page.splitlines() if any(k in line for k in ('Spatial difference atlas', 'Cross-anatomy', 'Photometric controls', 'Conclusion', 'LLM Usage Statement', 'References', 'Table 3:'))] for i, page in enumerate(pages)},
}
(here / 'tmp/expansion_checks.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
