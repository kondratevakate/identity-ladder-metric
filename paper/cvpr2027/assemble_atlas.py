"""Recompose published aggregate PDF panels; replace the stale brain row with Gaussian maps.

This is figure assembly from immutable Git artifacts, not a rerun of the private-data
map pipeline. Pixel maps are imported unchanged from the saved PDFs. No person image
is opened. Scalar annotations come from the matching aggregate result JSONs.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import pymupdf as fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE_COMMIT='82ed7fe8a52e557854cf4cd2d02eb47d5803d7cb'


def sha(data): return hashlib.sha256(data).hexdigest()


def blob(path):
    return subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE_COMMIT+':'+path])


def compact_atlas(source):
    """Give the main-paper atlas native 7pt labels, retaining all six primary rows."""
    import io
    import numpy as np
    from PIL import Image
    from matplotlib import colormaps
    from matplotlib.colors import TwoSlopeNorm

    page=source[0]
    width,height=page.rect.width,page.rect.height
    cell_w=(.995-.105)*width/(8+7*.06)
    cell_h=(.875-.12)*height/(8+7*.42)
    x_centers=[.105*width+j*cell_w*1.06+cell_w/2 for j in range(8)]
    y_centers=[.125*height+i*cell_h*1.42+cell_h/2 for i in range(8)]
    images={}
    # Original source images have no imported/clipped forms. Inspect their boxes
    # before the replacement: nested forms expose off-clip images to this API.
    original=fitz.open(stream=blob('results/figures/fig_ladder_maps_all_modalities.pdf'),filetype='pdf')
    for item in original[0].get_image_info():
        box=fitz.Rect(item['bbox'])
        cx,cy=(box.x0+box.x1)/2,(box.y0+box.y1)/2
        col=min(range(8),key=lambda j:abs(cx-x_centers[j]))
        row=min(range(8),key=lambda i:abs(cy-y_centers[i]))
        if abs(cx-x_centers[col])<20 and abs(cy-y_centers[row])<20:
            images[row,col]=box
    columns=[0,1,2,4,5,7]
    labels=['Mean anatomy','Repeat capture','Bilateral','Unrelated donor','Context mean\n16 donors','Mean-fill V\nown scale']
    datasets=[('Faces','CelebA'),('Faces','IMDb-WIKI'),('Brain MRI','QTIM\nGaussian; P'),
              ('Fundus','GRAPE P8; P'),('Fundus','PAPILA P9; P'),('Iris','UTIRIS NIR\nP2; P')]
    compact=fitz.open()
    out=compact.new_page(width=495,height=532)
    out.insert_font(fontname='helv')
    out.insert_font(fontname='hebo')
    def put(rect,text,**kwargs):
        spare=out.insert_textbox(rect,text,**kwargs)
        assert spare>=0, ('Text does not fit',text,spare)
    put(fitz.Rect(0,0,495,17),'Donor differences across human anatomy',
                       fontsize=10,fontname='hebo',align=1)
    for j,label in enumerate(labels):
        x=69+j*70
        put(fitz.Rect(x,18,x+65,44),label,fontsize=7,fontname='helv',align=1)
    selected=[]
    for i,(group,dataset) in enumerate(datasets):
        y=45+i*72
        put(fitz.Rect(0,y+13,67,y+27),group,fontsize=8,fontname='hebo')
        put(fitz.Rect(0,y+28,67,y+52),dataset,fontsize=7,fontname='helv')
        for j,col in enumerate(columns):
            x=69+j*70
            box=images.get((i,col))
            if i==2:
                box=fitz.Rect(x_centers[col]-cell_w/2,334,x_centers[col]+cell_w/2,406)
            if box is None:
                out.draw_rect(fitz.Rect(x+4,y,x+61,y+50),color=None,fill=(.95,.95,.94))
                put(fitz.Rect(x,y+19,x+65,y+40),'unavailable',fontsize=6.5,fontname='helv',align=1)
                continue
            out.show_pdf_page(fitz.Rect(x+4,y,x+61,y+50),source,0,clip=box,keep_proportion=True)
            if col in (0,7):
                continue
            # Exact displayed rounded values/intervals from the source annotation.
            bottom=box.y1
            annotation=page.get_text('text',clip=fitz.Rect(x_centers[col]-cell_w/2,bottom,
                                                         x_centers[col]+cell_w/2,bottom+26)).strip()
            match=re.search(r'(\d+\.\d+)\s*[×x]\s*mean',annotation)
            assert match, (i,col,annotation)
            ci=re.search(r'\[\s*[\d.]+,\s*[\d.]+\s*\]',annotation)
            text=match.group(1)+(('\n'+ci.group(0)) if ci else '')
            put(fitz.Rect(x,y+51,x+65,y+73),text,fontsize=6.5,fontname='helv',align=1)
            selected.append({'source_row':i,'source_column':col,'display_annotation':text})
    norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=2.4)
    rgb=(colormaps['RdBu_r'](np.linspace(0,1,512))[None,:,:3]*255).astype('uint8')
    stream=io.BytesIO();Image.fromarray(np.repeat(rgb,8,axis=0)).save(stream,format='PNG')
    out.insert_image(fitz.Rect(145,489,420,497),stream=stream.getvalue())
    for value in (0,.5,1,1.5,2,2.4):
        x=145+float(norm(value))*275
        put(fitz.Rect(x-11,499,x+11,513),str(value),fontsize=7,fontname='helv',align=1)
    put(fitz.Rect(69,516,489,532),'Local squared error / mean-fill error (Q); white = 1. V has a separate scale.',
                       fontsize=7,fontname='helv',align=1)
    compact.set_metadata({'title':'Identity Ladder aggregate Gaussian maps','author':''})
    compact.save(HERE/'fig/atlas_all_modalities.pdf',garbage=4,deflate=True)
    out.get_pixmap(matrix=fitz.Matrix(2,2)).save(HERE/'.build/atlas_preview.png')
    return selected


def main():
    cache=HERE/'.build';cache.mkdir(exist_ok=True)
    source_bytes=blob('results/figures/fig_ladder_maps_all_modalities.pdf')
    source=fitz.open(stream=source_bytes,filetype='pdf')
    brain_path=ROOT/'results/figures/fig_ladder_maps_common.pdf'
    brain=fitz.open(brain_path)
    fine=json.loads((ROOT/'results/donor_ladder_brain_maps.json').read_text())['maps']
    coarse=json.loads((ROOT/'results/donor_ladder_brain_map_grid.json').read_text())['maps']
    boxes=sorted(set(tuple(round(x,2) for x in obj['bbox']) for obj in brain[0].get_image_info()
                     if 125 < obj['bbox'][1] < 130 and 190 < obj['bbox'][3] < 196))
    assert len(boxes)==8, 'The Gaussian source PDF panel layout changed'
    assert abs(source[0].rect.width-853.92)<1 and abs(source[0].rect.height-1041.12)<1
    mapping=[None,'repeat','mirror','sibling','stranger','knn16_s4','cohort',None]
    ratios=[]
    for key in mapping:
        ratios.append(None if key is None else (coarse[key] if key.startswith('knn') else fine[key])['pooled_ratio'])

    # Preserve source panels while removing the superseded row and its text.
    page=source[0]
    page.add_redact_annot(fitz.Rect(0,326,page.rect.width,433),fill=(1,1,1))
    page.add_redact_annot(fitz.Rect(0,1012,page.rect.width,page.rect.height),fill=(1,1,1))
    page.apply_redactions(images=2)
    left,right=.105*page.rect.width,.995*page.rect.width
    cell=(right-left)/(8+7*.06)
    for j,rect in enumerate(boxes):
        x=left+j*cell*1.06
        target=fitz.Rect(x,334,x+cell,406)
        page.show_pdf_page(target,brain,0,clip=fitz.Rect(rect),keep_proportion=True)
        label='' if j==0 else ('1 x mean' if j==7 else f'{ratios[j]:.2f} x mean')
        page.insert_textbox(fitz.Rect(x,410,x+cell,426),label,fontname='helv',fontsize=6.5,align=1)
    page.insert_textbox(fitz.Rect(7,358,85,379),'Brain MRI',fontsize=9,fontname='hebo')
    page.insert_textbox(fitz.Rect(7,380,85,416),'QTIM\nGaussian, z = 10 mm\nprovisional',fontsize=7,fontname='helv')
    footer=('Aggregate-only maps. Brain: masked Gaussian, sd = 16/sqrt(12) mm; coarse look-alike grid is Gaussian-weighted. '
            'Fundus P8/P9: provisional frame-only results. Iris: P2 panel, not full P3. Missing is unavailable. '
            'Anatomy columns are population means or templates. V has its own scale in each row.')
    page.insert_textbox(fitz.Rect(8,1015,page.rect.width-8,1038),footer,fontsize=6,fontname='helv',align=1)
    source.set_metadata({'title':'Identity Ladder aggregate Gaussian maps','author':'','subject':'Aggregate-only, version-labeled spatial maps'})
    full=HERE/'fig/atlas_all_modalities_full.pdf'
    source.save(full,garbage=4,deflate=True)

    selected=compact_atlas(source)
    provenance={
        'assembly_generator':'paper/cvpr2027/assemble_atlas.py',
        'assembly_generator_sha256':sha(Path(__file__).read_bytes()),
        'assembly_base_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'source_commit':SOURCE_COMMIT,
        'source_pdf':'results/figures/fig_ladder_maps_all_modalities.pdf',
        'source_pdf_sha256':sha(source_bytes),
        'source_generator':'src/ladder_maps_all_figure.py',
        'source_generator_sha256':sha(blob('src/ladder_maps_all_figure.py')),
        'replacement_pdf':str(brain_path.relative_to(ROOT)),
        'replacement_pdf_sha256':sha(brain_path.read_bytes()),
        'replacement_producing_commit':subprocess.check_output(['git','-C',str(ROOT),'log','-1','--format=%H','--',str(brain_path.relative_to(ROOT))],text=True).strip(),
        'replacement_row':2,'replacement_source_slice':'MNI axial z=10 mm',
        'replacement_scalar_annotations':ratios,
        'shared_diverging_scale':{'vmin':0,'vcenter':1,'vmax':2.4},
        'method':'PDF-panel assembly from saved aggregate maps; private-data maps were not recomputed.',
        'changes':'Replace brain row/scalar annotations and stale footer. Main version redraws labels/legend at native size, retaining six primary rows and columns anatomy/repeat/bilateral/stranger/context/V. Full version retains all eight rows and columns.',
        'compact_source_annotations':selected,
        'outputs':{p.name:sha(p.read_bytes()) for p in (HERE/'fig/atlas_all_modalities.pdf',full)},
    }
    (HERE/'atlas_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Assembled Gaussian brain row with existing provisional fundus and P2 iris panels; no experiments rerun.')


if __name__=='__main__':main()
