"""Journal-style source diagram and saved heatmaps; display operations only.

Scientific arrays/scalars/colour limits are unchanged. A shared exterior mask
removes only border-connected black/white display background from the six MRI
panels, preserves their internal pixels, and applies an identical tight crop.
"""
from collections import deque
from pathlib import Path
import io
import json
import re
import subprocess

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageFilter
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm

import make_provenance_infographic as source_spec
import source_snapshot

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'fig/identity_ladder_provenance_v2.pdf'
PNG=HERE/'fig/identity_ladder_provenance_v2.png'
INK=(.08,.10,.13)
MUTED=(.38,.40,.43)
RULE=(.82,.84,.86)


def record(path):return source_spec.record(path)


def render_clip(page, box, local_canvas=False):
    """A fixed local canvas avoids subpixel-dependent source clip dimensions."""
    if not local_canvas:
        pix=page.get_pixmap(matrix=fitz.Matrix(4,4),clip=box,alpha=True)
        return Image.frombytes('RGBA',(pix.width,pix.height),pix.samples)
    temporary=fitz.open()
    local=temporary.new_page(width=box.width,height=box.height)
    local.show_pdf_page(local.rect,page.parent,page.number,clip=box,keep_proportion=False)
    pix=local.get_pixmap(matrix=fitz.Matrix(4,4),alpha=True)
    image=Image.frombytes('RGBA',(pix.width,pix.height),pix.samples)
    temporary.close()
    return image


def mri_display_panels(images):
    arrays=[np.asarray(im).copy() for im in images]
    assert len({a.shape for a in arrays})==1
    # Pixels informative in any of the six reference/control panels protect
    # the same location in every panel, including dark reference anatomy.
    empty=[]
    for a in arrays:
        rgb=a[:,:,:3]
        empty.append((rgb.max(axis=2)<=2)|(rgb.min(axis=2)>=253)|(a[:,:,3]==0))
    candidate=np.logical_and.reduce(empty)
    h,w=candidate.shape
    exterior=np.zeros((h,w),bool)
    queue=deque()
    for y,x in ([(0,x) for x in range(w)]+[(h-1,x) for x in range(w)]+
                [(y,0) for y in range(h)]+[(y,w-1) for y in range(h)]):
        if candidate[y,x] and not exterior[y,x]:
            exterior[y,x]=True;queue.append((y,x))
    while queue:
        y,x=queue.popleft()
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<h and 0<=xx<w and candidate[yy,xx] and not exterior[yy,xx]:
                exterior[yy,xx]=True;queue.append((yy,xx))
    support=~exterior
    assert support.any() and exterior.any()
    # Add a subtle outline outside the retained support, never over the data.
    support_image=Image.fromarray((support*255).astype('uint8'))
    dilation=np.asarray(support_image.filter(ImageFilter.MaxFilter(5)))>0
    outline=dilation&exterior
    ys,xs=np.nonzero(dilation)
    bbox=(max(0,int(xs.min())-2),max(0,int(ys.min())-2),
          min(w,int(xs.max())+3),min(h,int(ys.max())+3))
    processed=[]
    for a in arrays:
        result=a.copy()
        result[exterior,3]=0
        result[outline,:3]=170
        result[outline,3]=150
        assert np.array_equal(a[support],result[support]),'Retained MRI data pixels changed'
        processed.append(Image.fromarray(result).crop(bbox))
    support_image.save(HERE/'.build/brain_display_support_v2.png')
    kept_dark=int(((arrays[0][:,:,:3].max(axis=2)<25)&support&(arrays[0][:,:,3]>0)).sum())
    proof={'method':'Shared exterior display mask, not a measurement/acquisition mask.',
           'definition':'Border-connected pixels that are black/white/transparent in all six aligned source panels; enclosed dark anatomy remains retained.',
           'black_threshold_rgb':2,'white_threshold_rgb':253,'source_shape_px':[w,h],
           'common_crop_bbox_px':bbox,'exterior_pixels':int(exterior.sum()),
           'internal_dark_reference_pixels_retained':kept_dark,
           'retained_rgba_pixels_unchanged':True,
           'outline':'Two-source-pixel grey outline placed outside retained support only.',
           'scope':'Transparency and uniform display crop only; no statistic, array, colour normalization or eligibility changed.'}
    return processed,proof


def png_stream(image):
    buffer=io.BytesIO();image.save(buffer,format='PNG');return buffer.getvalue()


def main():
    (HERE/'.build').mkdir(exist_ok=True)
    full=HERE/'fig/atlas_all_modalities_full.pdf'
    source=fitz.open(full);page=source[0]
    original_raw=subprocess.check_output(['git','-C',str(ROOT),'show',source_spec.SOURCE_COMMIT+':'+source_spec.SOURCE_PATH])
    original=fitz.open(stream=original_raw,filetype='pdf')
    width,height=original[0].rect.width,original[0].rect.height
    cell_w=(.995-.105)*width/(8+7*.06)
    cell_h=(.875-.12)*height/(8+7*.42)
    xs=[.105*width+j*cell_w*1.06+cell_w/2 for j in range(8)]
    ys=[.125*height+i*cell_h*1.42+cell_h/2 for i in range(8)]
    boxes={}
    for obj in original[0].get_image_info():
        box=fitz.Rect(obj['bbox']);cx=(box.x0+box.x1)/2;cy=(box.y0+box.y1)/2
        col=min(range(8),key=lambda j:abs(cx-xs[j]));row=min(range(8),key=lambda i:abs(cy-ys[i]))
        if abs(cx-xs[col])<20 and abs(cy-ys[row])<20:boxes[row,col]=box
    columns=[0,1,2,4,5,7]
    rows=[(2,'Brain MRI','QTIM','P'),(0,'Faces','CelebA','F'),
          (3,'Fundus','GRAPE','P'),(5,'Iris','UTIRIS NIR','P')]
    panel_images={};panels=[]
    for i,(row,_,_,_) in enumerate(rows):
        for j,col in enumerate(columns):
            box=(fitz.Rect(xs[col]-cell_w/2,334,xs[col]+cell_w/2,406) if row==2 else boxes[row,col])
            panel_images[i,j]=render_clip(page,box,local_canvas=row==2)
            entry={'row':i,'source_row':row,'source_column':col,'source_clip':list(box)}
            if col not in (0,7):
                annotation=page.get_text('text',clip=fitz.Rect(xs[col]-cell_w/2,box.y1,xs[col]+cell_w/2,box.y1+26)).strip()
                number=re.search(r'(\d+\.\d+)\s*[×x]\s*mean',annotation)
                assert number,(row,col,annotation)
                interval=re.search(r'\[\s*[\d.]+,\s*[\d.]+\s*\]',annotation)
                entry['annotation']=number.group(1)+(('\n'+interval.group(0)) if interval else '')
            panels.append(entry)
    cleaned,mri_proof=mri_display_panels([panel_images[0,j] for j in range(6)])
    for j,image in enumerate(cleaned):panel_images[0,j]=image

    paths={'brain':ROOT/'results/donor_ladder_brain_metrics.json',
           'face':ROOT/'results/face_ladder_metrics.json',
           'fundus':HERE/'inputs/donor_ladder_2d_facefmt.json',
           'iris':ROOT/'results/iris_ladder_metrics.json'}
    data={key:(json.loads(p.read_text()) if key=='fundus' else source_snapshot.load_json(p)) for key,p in paths.items()}
    b=data['brain']['counts'];f=data['face']['counts']['celeba']
    g=data['fundus']['results']['GRAPE'];ir=data['iris']['panel']['UTIRIS_NIR']['counts']
    colours=[(.27,.51,.66),(.49,.43,.61),(.20,.53,.48),(.73,.52,.23)]
    blocks=[{'name':'Brain MRI (QTIM)','rect':(8,23,172,75),'colour':colours[0],
             'counts':f"{b['n_targets']} targets, {b['n_families']} families",'detail':f"{b['n_repeat_scans']} repeat scans"},
            {'name':'Faces (CelebA)','rect':(8,85,172,137),'colour':colours[1],
             'counts':f"{f['n_targets']} targets",'detail':'Repeat photographs available'},
            {'name':'Fundus (GRAPE)','rect':(328,23,493,75),'colour':colours[2],
             'counts':f"{g['units']} eyes, {g['stranger']['persons']} people",'detail':f"Repeat: {g['repeat']['targets']} eyes, {g['repeat']['persons']} people"},
            {'name':'Iris (UTIRIS NIR)','rect':(328,85,493,137),'colour':colours[3],
             'counts':f"{ir['units']} eyes, {ir['persons']} people",'detail':'Same-eye repeat acquisitions'}]
    document=fitz.open();out=document.new_page(width=495,height=541)
    out.insert_font(fontname='helv');out.insert_font(fontname='hebo')
    def put(rect,text,size=7.2,bold=False,colour=INK,align=0):
        spare=out.insert_textbox(fitz.Rect(rect),text,fontsize=size,fontname='hebo' if bold else 'helv',color=colour,align=align)
        assert spare>=0,('Label does not fit',text,spare)
    put((0,0,14,23),'a',12,True)
    put((18,3,495,23),'Source cohorts',9.5,True)
    assert source_spec.ASSET.is_file()
    out.insert_image(fitz.Rect(188,9,311,141),filename=str(source_spec.ASSET),keep_proportion=True)
    leaders=[(colours[0],[(172,43),(211,43),(248,31)]),
             (colours[1],[(172,105),(210,105),(247,75)]),
             (colours[2],[(328,43),(301,43),(273,49)]),
             (colours[3],[(328,105),(300,105),(269,49)])]
    for colour,points in leaders:
        out.draw_polyline(points,color=colour,width=.6)
        out.draw_circle(points[-1],1.35,color=(1,1,1),fill=colour,width=.4)
    for block in blocks:
        x,y,x1,y1=block['rect']
        out.draw_line((x,y+4),(x+8,y+4),color=block['colour'],width=1.8)
        put((x+13,y-2,x1,y+14),block['name'],8.1,True)
        put((x+13,y+17,x1,y+31),block['counts'],7.2)
        put((x+13,y+33,x1,y1),block['detail'],6.6,colour=MUTED)
    out.draw_line((0,145),(495,145),color=RULE,width=.45)
    put((0,151,14,175),'b',12,True)
    put((18,154,495,174),'Donor-reference error maps',9.5,True)
    labels=['Reference\nanatomy','Repeat\nacquisition','Opposite\nside','Unrelated\ndonors',
            'Context-selected\nmean (16 donors)','Mean-fill error V\n(row-scaled)']
    for j,label in enumerate(labels):
        x=69+j*70
        put((x,173,x+66,197),label,7.1,align=1)
    for i,(_,group,dataset,status) in enumerate(rows):
        y=202+i*72
        put((1,y+9,68,y+25),group,8.6,True)
        put((1,y+28,68,y+49),dataset+' ('+status+')',6.9,colour=MUTED)
        for j in range(6):
            x=69+j*70
            out.insert_image(fitz.Rect(x+2,y,x+64,y+50),stream=png_stream(panel_images[i,j]),keep_proportion=True)
            entry=next(p for p in panels if p['row']==i and p['source_column']==columns[j])
            entry['destination_rect']=[x+2,y,x+64,y+50]
            if 'annotation' in entry:put((x,y+52,x+66,y+74),entry['annotation'],7,align=1)
    norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=2.4)
    rgb=(colormaps['RdBu_r'](np.linspace(0,1,512))[None,:,:3]*255).astype('uint8')
    out.insert_image(fitz.Rect(146,498,421,505),stream=png_stream(Image.fromarray(np.repeat(rgb,8,axis=0))))
    for value in (0,.5,1,1.5,2,2.4):
        x=146+float(norm(value))*275
        put((x-12,507,x+12,520),str(value),7.1,align=1)
    put((8,523,493,538),'Q: local squared error / mean-fill error. V: row-scaled. P: provisional; F: source-frame approval.',7.1,colour=MUTED,align=1)
    document.set_metadata({'title':'Source cohorts and population-reference error maps','author':'',
                           'subject':'Journal-style figure; decorative synthetic bust; measured aggregate maps.'})
    document.save(OUT,garbage=4,deflate=True)
    out.get_pixmap(matrix=fitz.Matrix(4,4)).save(PNG)
    out.get_pixmap(matrix=fitz.Matrix(2,2)).save(HERE/'.build/identity_ladder_provenance_v2.png')
    proof={'generator':record(Path(__file__)),'source_specification_module':record(Path(source_spec.__file__)),
           'source_atlas':record(full),'source_atlas_provenance':record(HERE/'atlas_provenance.json'),
           'source_commit':source_spec.SOURCE_COMMIT,'original_source_pdf':source_spec.SOURCE_PATH,
           'scientific_inputs':{key:(record(path) if key=='fundus' else source_snapshot.record(path)) for key,path in paths.items()},
           'decorative_asset':{**record(source_spec.ASSET),'role':'Previously generated decorative bust; unchanged asset.',
                               'generation_record':record(source_spec.ASSET.with_suffix('.provenance.json'))},
           'source_cohorts':blocks,'panels':panels,'mri_background_display':mri_proof,
           'layout_changes':'White canvas; a/b panels; source text with leaders instead of dashboard cards; smaller bust; maps 50pt high instead of42pt; labels7pt.',
           'numeric_colour_scale':{'vmin':0,'vcenter':1,'vmax':2.4,'changed':False},
           'spatial_scale':'Aspect preserved; one common MRI crop. Rows have display scales, not a common physical scale.',
           'measurement_changes':False,'generated_scientific_images':False,
           'output':record(OUT),'shareable_png':record(PNG)}
    (HERE/'provenance_infographic_v2.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'outputs':[str(OUT),str(PNG)],'mri_display':mri_proof},indent=2))


if __name__=='__main__':main()
