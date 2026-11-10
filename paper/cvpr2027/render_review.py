"""Render both reading copies and a contact sheet for author-side visual review."""
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
OUT = HERE / '.build'


def main():
    OUT.mkdir(exist_ok=True)
    thumbnails = []
    for stem in ('main', 'supplementary'):
        with pymupdf.open(HERE / (stem + '.pdf')) as doc:
            print(stem, len(doc), 'pages')
            for i, page in enumerate(doc, 1):
                target = OUT / f'{stem}-page-{i}.png'
                page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(target)
                with Image.open(target) as im:
                    thumbnails.append((stem, i, ImageOps.contain(im.convert('RGB'), (320, 420))))
    sheet = Image.new('RGB', (4 * 330, ((len(thumbnails) + 3) // 4) * 455), '#dddddd')
    draw = ImageDraw.Draw(sheet)
    for j, (stem, i, im) in enumerate(thumbnails):
        x, y = (j % 4) * 330, (j // 4) * 455
        sheet.paste(im, (x, y + 24))
        draw.text((x + 5, y + 4), f'{stem} {i}', fill='black')
    sheet.save(OUT / 'contact-sheet.png')


if __name__ == '__main__':
    main()
