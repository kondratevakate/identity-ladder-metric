"""v3 (15 Nov): v2 in the shared paper style (fig_style.py): capital panel letters, shared rung names and
colors on the column headers, no status codes. v2 is kept unchanged.

Journal-style source diagram and saved heatmaps; display operations only.

Scientific arrays/scalars/color limits are unchanged. A shared exterior mask
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
from PIL import Image, ImageDraw, ImageFilter
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm

import fig_style as S
import infographic_sources as source_spec
import source_snapshot

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'fig/identity_ladder_provenance_v3.pdf'
PNG=HERE/'fig/identity_ladder_provenance_v3.png'
TEASER=HERE/'fig/teaser_maps.pdf'
DATASETS=HERE/'fig/datasets_column.pdf'
TEASER_AB=HERE/'fig/teaser_ab.pdf'
TEASER_V2=HERE/'fig/teaser_v2.pdf'
TEASER_V3=HERE/'fig/teaser_v3.pdf'
TEASER_V4=HERE/'fig/teaser_v4.pdf'
TEASER_V5=HERE/'fig/teaser_v5.pdf'


def rgb_hex(code):return tuple(int(code[i:i+2],16)/255 for i in (1,3,5))
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
    dilation=np.asarray(support_image.filter(ImageFilter.MaxFilter(3)))>0
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
    support_image.save(HERE/'.build/brain_display_support_v3.png')
    kept_dark=int(((arrays[0][:,:,:3].max(axis=2)<25)&support&(arrays[0][:,:,3]>0)).sum())
    proof={'method':'Shared exterior display mask, not a measurement/acquisition mask.',
           'definition':'Border-connected pixels that are black/white/transparent in all six aligned source panels; enclosed dark anatomy remains retained.',
           'black_threshold_rgb':2,'white_threshold_rgb':253,'source_shape_px':[w,h],
           'common_crop_bbox_px':bbox,'exterior_pixels':int(exterior.sum()),
           'internal_dark_reference_pixels_retained':kept_dark,
           'retained_rgba_pixels_unchanged':True,
           'outline':'Two-source-pixel gray outline placed outside retained support only.',
           'scope':'Transparency and uniform display crop only; no statistic, array, color normalization or eligibility changed.'}
    return processed,proof


def recolor_viridis(image):
    """Display-only recolor of a saved viridis map into the red half of the maps' RdBu palette.

    Each non-gray pixel within 70 RGB units of a viridis color is mapped back to its viridis position t in [0, 1]
    and redrawn as RdBu_r(0.5 + t/2): white at the lowest value of that map, dark red at its highest. Pixels far
    from viridis (background, outline, anatomy) are left unchanged. The map keeps its own scale.
    """
    a=np.asarray(image.convert('RGBA')).astype(np.int32)
    lut=(colormaps['viridis'](np.linspace(0,1,256))[:,:3]*255).astype(np.int32)
    new=(colormaps['RdBu_r'](0.5+np.linspace(0,1,256)/2)[:,:3]*255).astype(np.uint8)
    rgb=a[:,:,:3].reshape(-1,3)
    best=np.empty(len(rgb),np.int32);dist=np.full(len(rgb),1<<30)
    for k in range(256):
        d=((rgb-lut[k])**2).sum(1)
        better=d<dist;dist[better]=d[better];best[better]=k
    gray=(rgb.max(1)-rgb.min(1))<28
    hit=(dist<=70**2)&(~gray)&(a[:,:,3].reshape(-1)>0)
    out=a.astype(np.uint8).reshape(-1,4).copy()
    out[hit,:3]=new[best[hit]]
    return Image.fromarray(out.reshape(a.shape))


def v_ranges():
    """2nd and 98th percentile of the dataset-average error map V (MSE of the dataset average, z units) as drawn in
    the atlas rows (src/ladder_maps_all_figure.py colors V from these limits); read from the saved private maps."""
    import sys as _sys
    _sys.path.insert(0,str(ROOT/'src'))
    import ladder_maps_all_figure as L
    def rng(V,valid,maps):
        for m in maps:valid=valid&np.isfinite(m)
        a,b=np.nanpercentile(np.where(valid,V,np.nan),[2,98]);return float(a),float(b)
    z=np.load(Path(L.PROJ)/'face-ladder-20261105-v1'/'maps_v6f.npz')
    face=[z['celeba_'+k] for k in ('repeat','mirror','sibling','stranger','lookalike','sexage','V') if 'celeba_'+k in z.files]
    out={'CelebA':rng(z['celeba_V'],z['celeba_valid']>0.5,face)}
    g=dict(np.load(Path(L.PROJ)/'donor-ladder-2d-20261105-v1'/'facefmt_GRAPE_p8.npz'))
    out['GRAPE']=rng(g['V'],g['valid']>0.5,[g[k] for k in ('repeat','fellow','stranger','lookalike','sexage','V') if k in g])
    d=json.loads(Path(L.IRIS_JSON).read_text());rho=d['maps']['UPOL'].get('pupil_to_iris_ratio_median',0.45)
    zi=np.load(Path(L.IRIS_PRIVATE)/'maps_UPOL_P2.npz')
    m={k:L.iris_ring(zi['all_'+v],rho) for k,v in dict(repeat='other_photo',mirror='fellow_eye',stranger='stranger',lookalike='lookalike16',V='V').items() if 'all_'+v in zi.files}
    out['UPOL']=rng(m['V'],np.ones(m['V'].shape,bool),list(m.values()))
    return out


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
          (3,'Retina','GRAPE','P'),(6,'Iris','UPOL','P')]
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
    brain_v2_used={}
    import brain_row_v2 as BR
    if BR.available():   # v2 brain maps reach the brain edge (Kate, 15 Nov); look-alike column stays v1 (8 mm grid)
        bcells,bratios=BR.cells()
        for j,key in ((0,'anatomy'),(1,'repeat'),(2,'mirror'),(3,'stranger'),(5,'V')):panel_images[0,j]=bcells[key]
        for e in panels:
            if e['row']==0 and e['source_column'] in (1,2,4):
                e['annotation']=f"{bratios[{1:'repeat',2:'mirror',4:'stranger'}[e['source_column']]]:.2f}"
                e['source']='donor_ladder_brain_maps_v2'
        brain_v2_used[0]=True

    paths={'brain':ROOT/'results/donor_ladder_brain_metrics.json',
           'face':ROOT/'results/face_ladder_metrics.json',
           'fundus':HERE/'inputs/donor_ladder_2d_facefmt.json',
           'iris':ROOT/'results/iris_ladder_metrics.json'}
    data={key:(json.loads(p.read_text()) if key=='fundus' else source_snapshot.load_json(p)) for key,p in paths.items()}
    b=data['brain']['counts'];f=data['face']['counts']['celeba']
    g=data['fundus']['results']['GRAPE'];ir=data['iris']['panel']['UPOL']['counts']
    colors=[(.27,.51,.66),(.49,.43,.61),(.20,.53,.48),(.73,.52,.23)]
    blocks=[{'name':'Brain MRI (QTIM)','rect':(8,23,172,75),'color':colors[0],
             'counts':f"{b['n_targets']} targets, {b['n_families']} families",'detail':f"{b['n_repeat_scans']} with a second scan"},
            {'name':'Faces (CelebA)','rect':(8,85,172,137),'color':colors[1],
             'counts':f"{f['n_targets']} targets",'detail':'Second photographs available'},
            {'name':'Retina (GRAPE)','rect':(328,23,493,75),'color':colors[2],
             'counts':f"{g['units']} eyes, {g['stranger']['persons']} people",'detail':f"Second visit: {g['repeat']['targets']} eyes, {g['repeat']['persons']} people"},
            {'name':'Iris (UPOL)','rect':(328,85,493,137),'color':colors[3],
             'counts':f"{ir['units']} eyes, {ir['persons']} people",'detail':'Same eye captured again'}]
    document=fitz.open();out=document.new_page(width=495,height=541)
    out.insert_font(fontname='helv');out.insert_font(fontname='hebo')
    def put(rect,text,size=7.2,bold=False,color=INK,align=0):
        spare=out.insert_textbox(fitz.Rect(rect),text,fontsize=size,fontname='hebo' if bold else 'helv',color=color,align=align)
        assert spare>=0,('Label does not fit',text,spare)
    put((0,3,14,23),'A',9,True)
    put((14,4,495,23),'Datasets',7.5,True)
    assert source_spec.ASSET.is_file()
    out.insert_image(fitz.Rect(188,9,311,141),filename=str(source_spec.ASSET),keep_proportion=True)
    leaders=[(colors[0],[(172,43),(211,43),(248,31)]),
             (colors[1],[(172,105),(210,105),(247,75)]),
             (colors[2],[(328,43),(301,43),(273,49)]),
             (colors[3],[(328,105),(300,105),(269,49)])]
    for color,points in leaders:
        out.draw_polyline(points,color=color,width=.6)
        out.draw_circle(points[-1],1.35,color=(1,1,1),fill=color,width=.4)
    for block in blocks:
        x,y,x1,y1=block['rect']
        out.draw_line((x,y+4),(x+8,y+4),color=block['color'],width=1.8)
        put((x+13,y-2,x1,y+14),block['name'],7.5,True)
        put((x+13,y+15,x1,y+29),block['counts'],7)
        put((x+13,y+28,x1,y1),block['detail'],6.3,color=MUTED)
    out.draw_line((0,145),(495,145),color=RULE,width=.45)
    put((0,154,14,175),'B',9,True)
    put((14,155,495,174),'Where each fill is closer or further than the dataset mean (squared error)',7.5,True)
    labels=['Mean\nanatomy','True person:\nsecond image','True person:\nopposite side','Stranger',
            'Mean of 16\nlook-alikes','Error of the\ndataset mean']
    keys=[None,'ref','ref','stranger','lookalike16','dataset_mean']
    def rgb(code):return tuple(int(code[i:i+2],16)/255 for i in (1,3,5))
    for j,label in enumerate(labels):
        x=69+j*70
        put((x,173,x+66,195),label,7,align=1)
        if keys[j]:out.draw_line((x+18,196),(x+48,196),color=rgb(S.color(keys[j])),width=1.6)
    for i,(_,group,dataset,status) in enumerate(rows):
        y=202+i*72
        put((1,y+9,68,y+25),group,7.5,True)
        put((1,y+24,68,y+45),dataset,7,color=MUTED)
        for j in range(6):
            x=69+j*70
            image=panel_images[i,j]
            if j==5 and not brain_v2_used.get(i):image=recolor_viridis(image)
            iw,ih=image.size;scale=min(62/iw,50/ih);dw,dh=iw*scale,ih*scale
            dest=fitz.Rect(x+33-dw/2,y+25-dh/2,x+33+dw/2,y+25+dh/2)   # explicit aspect, centered in the 62x50 slot
            out.insert_image(dest,stream=png_stream(image))
            entry=next(p for p in panels if p['row']==i and p['source_column']==columns[j])
            entry['destination_rect']=[round(v,2) for v in dest]
            if 'annotation' in entry:put((x,y+51,x+66,y+73),entry['annotation'],6.3,align=1)
    norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=2.4)
    rgb=(colormaps['RdBu_r'](np.linspace(0,1,512))[None,:,:3]*255).astype('uint8')
    out.insert_image(fitz.Rect(146,498,421,505),stream=png_stream(Image.fromarray(np.repeat(rgb,8,axis=0))))
    for value in (0,.5,1,1.5,2,2.4):
        x=146+float(norm(value))*275
        put((x-12,507,x+12,520),str(value),7,align=1)
    put((8,523,493,538),'Local squared error divided by the error of the dataset mean: 1 = dataset mean, about 2 = stranger.',7,color=MUTED,align=1)
    document.set_metadata({'title':'Source cohorts and population-reference error maps','author':'',
                           'subject':'Journal-style figure; decorative synthetic bust; measured aggregate maps.'})
    document.save(OUT,garbage=4,deflate=True)
    out.get_pixmap(matrix=fitz.Matrix(4,4)).save(PNG)
    out.get_pixmap(matrix=fitz.Matrix(2,2)).save(HERE/'.build/identity_ladder_provenance_v3.png')
    # Teaser A|B (Kate, 15 Nov): datasets as a narrow left column, maps as the right column; descriptive labels.
    tab=fitz.open();tq=tab.new_page(width=495,height=352)
    tq.insert_font(fontname='helv');tq.insert_font(fontname='hebo')
    def tput(rect,text,size,bold=False,color=INK,align=0):
        r=fitz.Rect(rect);r.y1+=3
        spare=tq.insert_textbox(r,text,fontsize=size,fontname='hebo' if bold else 'helv',color=color,align=align)
        assert spare>=0,('Label does not fit',text,spare)
    tput((0,0,14,18),'A',9,True);tput((12,2,120,30),'Datasets and their\ntargets (hidden regions)',7.5,True)
    bw=96;bh=bw*132/123;bx0,by0=6,34
    tq.insert_image(fitz.Rect(bx0,by0,bx0+bw,by0+bh),filename=str(source_spec.ASSET),keep_proportion=True)
    for (color,points),block in zip(leaders,blocks):
        px,py=points[-1];end=(bx0+(px-188)/123*bw,by0+(py-9)/132*bh)
        tq.draw_circle(end,2.2,color=(1,1,1),fill=color,width=.5)
    for k,block in enumerate(blocks):
        y=by0+bh+8+k*46
        tq.draw_circle((4,y+5.5),2.6,color=(1,1,1),fill=block['color'],width=.4)
        tput((10,y,120,y+12),block['name'],7.2,True)
        tput((10,y+11,120,y+23),block['counts'],6.8)
        tput((10,y+22,120,y+44),block['detail'],6.3,color=MUTED)
    X0=128;LW=46;CWB=(495-X0-LW)/6
    tput((X0-4,0,X0+10,18),'B',9,True)
    tput((X0+8,2,495,30),'Squared error of each fill of the hidden region, divided by the error of the dataset mean:\nbelow 1 (blue) the fill is closer to the target than the dataset mean, above 1 (red) further',7,True)
    heads=['Average\nanatomy','Same person:\nsecond image','Same person:\nother side','Stranger','Mean of 16\nlook-alike\nstrangers','Error of the\ndataset mean\n(own scale)']
    for j,label in enumerate(heads):
        x=X0+LW+j*CWB
        tput((x-3,30,x+CWB+3,59),label,6.5,align=1)
        if keys[j]:tq.draw_line((x+CWB*.25,62),(x+CWB*.75,62),color=rgb_hex(S.color(keys[j])),width=1.6)
    RH=62
    for i,(_,group,dataset,status) in enumerate(rows):
        y=68+i*RH
        tput((X0,y+10,X0+LW,y+22),group,7.2,True)
        tput((X0,y+22,X0+LW,y+33),dataset,6.8,color=MUTED)
        for j in range(6):
            x=X0+LW+j*CWB
            image=panel_images[i,j]
            if j==5 and not brain_v2_used.get(i):image=recolor_viridis(image)
            iw,ih=image.size;sc=min((CWB-4)/iw,42/ih);dw,dh=iw*sc,ih*sc
            cx=x+CWB/2
            tq.insert_image(fitz.Rect(cx-dw/2,y+21-dh/2,cx+dw/2,y+21+dh/2),stream=png_stream(image))
            entry=next(p for p in panels if p['row']==i and p['source_column']==columns[j])
            if 'annotation' in entry:tput((x-3,y+42,x+CWB+3,y+61),entry['annotation'],6.1,align=1)
    yb=68+4*RH+2;x0b,x1b=X0+LW+20,495-20
    tq.insert_image(fitz.Rect(x0b,yb,x1b,yb+6),stream=png_stream(Image.fromarray(np.repeat(rgb,8,axis=0))))
    for value in (0,.5,1,1.5,2,2.4):
        xv=x0b+float(norm(value))*(x1b-x0b)
        tput((xv-12,yb+7,xv+12,yb+19),str(value),6.5,align=1)
    tput((X0,yb+18,495,yb+30),'1 = dataset mean, about 2 = a stranger. Numbers: mean over the region [95% interval].',6.3,color=MUTED,align=1)
    tab.set_metadata({'title':'Datasets and population-reference error maps','author':''})
    tab.save(TEASER_AB,garbage=4,deflate=True)
    # Teaser v2 (Kate, 15 Nov): one panel, compact; each row names the data type, the shown dataset with its size
    # and the other datasets of that modality in the evaluation; one plain header over the fills.
    # Counts are those of tab/dataset_card_short.tex (people or families, and targets).
    row_text=[('Brain MRI','3D T1-weighted scans','1 mm voxels','QTIM: 221 people','also IXI: 100 people'),
              ('Faces','Photographs, aligned','128 x 128 pixels','CelebA: 921 people','also IMDb-WIKI: 678 people'),
              ('Retina','Fundus photographs','128 x 128 pixels','GRAPE: 134 people, 239 eyes','also PAPILA: 244 people'),
              ('Iris','Eye images','64 x 256 pixel strip','UPOL: 64 people, 128 eyes','also UTIRIS NIR, VIS: 79 people')]
    tv=fitz.open();H2=318;tw=tv.new_page(width=495,height=H2)
    tw.insert_font(fontname='helv');tw.insert_font(fontname='hebo')
    def wput(rect,text,size,bold=False,color=INK,align=0):
        r=fitz.Rect(rect);r.y1+=6
        spare=tw.insert_textbox(r,text,fontsize=size,fontname='hebo' if bold else 'helv',color=color,align=align)
        assert spare>=0,('Label does not fit',text,spare)
    LW2=112;CW2=(495-LW2)/6;MH=40;RH2=54;Y0=58
    heads2=['Dataset\naverage image','A second image\nof the same\nperson','The same\nperson\'s other\nside, mirrored',
            'A random\nstranger','Average of the\n16 most similar\nstrangers','Where the dataset\naverage itself\nerrs (own scale)']
    gx0,gx1=LW2+CW2,LW2+5*CW2
    wput((0,0,495,10),'Local MSE of each fill divided by the local MSE of the dataset average: blue = closer to the target than the average, red = further',7,True,align=1)
    wput((gx0,13,gx1,22),'The hidden region is filled with:',6.6,True,align=1)
    tw.draw_line((gx0+4,24),(gx1-4,24),color=RULE,width=.5)
    for j,label in enumerate(heads2):
        x=LW2+j*CW2
        wput((x-2,26,x+CW2+2,52),label,6.3,align=1)
        if keys[j]:tw.draw_line((x+CW2*.3,55),(x+CW2*.7,55),color=rgb_hex(S.color(keys[j])),width=1.5)
    for i,(group,kind,res,shown,also) in enumerate(row_text):
        y=Y0+i*RH2
        wput((0,y,LW2-4,y+9),group,7,True)
        wput((0,y+8.5,LW2-4,y+17),kind,6.2,color=MUTED)
        wput((0,y+16.5,LW2-4,y+25),res,6.2,color=MUTED)
        wput((0,y+26,LW2-4,y+34),shown,6.2)
        wput((0,y+34.5,LW2-4,y+43),also,6.2,color=MUTED)
        for j in range(6):
            x=LW2+j*CW2
            image=panel_images[i,j]
            if j==5 and not brain_v2_used.get(i):image=recolor_viridis(image)
            iw,ih=image.size;sc=min((CW2-6)/iw,MH/ih);dw,dh=iw*sc,ih*sc
            cx=x+CW2/2
            tw.insert_image(fitz.Rect(cx-dw/2,y+MH/2-dh/2,cx+dw/2,y+MH/2+dh/2),stream=png_stream(image))
            entry=next(p for p in panels if p['row']==i and p['source_column']==columns[j])
            if 'annotation' in entry:
                wput((x,y+MH,x+CW2,y+MH+8),'region mean '+entry['annotation'].split('\n')[0],6,align=1)
    yb=Y0+4*RH2+1;x0b,x1b=LW2+CW2*0.5,LW2+CW2*4.5
    tw.insert_image(fitz.Rect(x0b,yb,x1b,yb+5),stream=png_stream(Image.fromarray(np.repeat(rgb,8,axis=0))))
    ticks={0:'0: identical\nto the target',1:'1: as close as\nthe dataset average',2:'2: as far as\na random stranger'}
    for value,lab in ticks.items():
        xv=x0b+float(norm(value))*(x1b-x0b)
        wput((xv-45,yb+6,xv+45,yb+20),lab,6,align=1)
    wput((LW2,yb+26,495,yb+34),'MSE in a Gaussian window around each pixel, over all targets of the dataset. Region mean: the same ratio over the whole region.',6,color=MUTED,align=1)
    tv.set_metadata({'title':'Population-reference error maps','author':''})
    tv.save(TEASER_V2,garbage=4,deflate=True)
    # Teaser v3 (Kate, 15 Nov): same content, landscape: lower cells, two-line row labels, one-line legend.
    row3=[('Brain MRI','3D T1 MRI, 1 mm voxels','QTIM: 221 people','also IXI: 100 people'),
          ('Faces','Photographs, 128 x 128 px','CelebA: 921 people','also IMDb-WIKI: 678 people'),
          ('Retina','Fundus photos, 128 x 128 px','GRAPE: 134 people, 239 eyes','also PAPILA: 244 people'),
          ('Iris','Eye images, 64 x 256 px strip','UPOL: 64 people, 128 eyes','also UTIRIS NIR, VIS: 79 people')]
    t3=fitz.open();H3=245;t3p=t3.new_page(width=495,height=H3)
    t3p.insert_font(fontname='helv');t3p.insert_font(fontname='hebo')
    def put3(rect,text,size,bold=False,color=INK,align=0):
        r=fitz.Rect(rect);r.y1+=6
        spare=t3p.insert_textbox(r,text,fontsize=size,fontname='hebo' if bold else 'helv',color=color,align=align)
        assert spare>=0,('Label does not fit',text,spare)
    LW3=118;CW3=(495-LW3)/6;MH3=32;RH3=41;Y3=57
    heads3=['Dataset\naverage image','Second image\nof the same person',"Same person's\nother side, mirrored",
            'A random\nstranger','Average of the 16\nmost similar\nstrangers','Where the dataset\naverage errs\n(own scale)']
    face=panel_images[1,0].convert('RGBA');fw,fh=face.size
    bx=(int(fw*.18),int(fh*.25),int(fw*.42),int(fh*.25)+int(fw*.24))       # hidden square over the left eye
    hid=face.copy();ImageDraw.Draw(hid).rectangle(bx,fill=(150,150,150,255))
    fil=face.copy();mx=(fw-bx[2],bx[1],fw-bx[0],bx[3])                        # fill: the mirrored other side
    fil.paste(face.crop(mx).transpose(Image.FLIP_LEFT_RIGHT),bx[:2])
    ImageDraw.Draw(fil).rectangle(bx,outline=(43,95,184,255),width=max(2,fw//60))
    ImageDraw.Draw(hid).rectangle(bx,outline=(43,95,184,255),width=max(2,fw//60))
    for k,(im,lab) in enumerate(((face,'target'),(hid,'hide a region'),(fil,'fill it, compare'))):
        x=k*40;sc=min(36/im.size[0],40/im.size[1]);dw,dh=im.size[0]*sc,im.size[1]*sc
        t3p.insert_image(fitz.Rect(x+18-dw/2,0,x+18+dw/2,dh),stream=png_stream(im))
        put3((x-3,dh+1,x+39,dh+7),lab,5.4,align=1)
        if k<2:put3((x+35,14,x+42,22),'>',7,color=MUTED)
    gx0,gx1=LW3+CW3,LW3+5*CW3
    put3((gx0,10,gx1,19),'The hidden region is filled with:',6.8,True,align=1)
    t3p.draw_line((gx0+4,22),(gx1-4,22),color=RULE,width=.5)
    for j,label in enumerate(heads3):
        x=LW3+j*CW3
        put3((x-3,24,x+CW3+3,46),label,6,align=1)
        if keys[j]:t3p.draw_line((x+CW3*.3,53),(x+CW3*.7,53),color=rgb_hex(S.color(keys[j])),width=1.4)
    for i,(group,kind,shown,also) in enumerate(row3):
        y=Y3+i*RH3
        put3((0,y,LW3-4,y+8),group,6.8,True)
        put3((0,y+8,LW3-4,y+15.5),kind,5.9,color=MUTED)
        put3((0,y+16,LW3-4,y+23.5),shown,5.9)
        put3((0,y+24,LW3-4,y+31.5),also,5.9,color=MUTED)
        for j in range(6):
            x=LW3+j*CW3
            image=panel_images[i,j]
            if j==5 and not brain_v2_used.get(i):image=recolor_viridis(image)
            iw,ih=image.size;sc=min((CW3-6)/iw,MH3/ih);dw,dh=iw*sc,ih*sc
            cx=x+CW3/2
            t3p.insert_image(fitz.Rect(cx-dw/2,y+MH3/2-dh/2,cx+dw/2,y+MH3/2+dh/2),stream=png_stream(image))
            entry=next(p for p in panels if p['row']==i and p['source_column']==columns[j])
            if 'annotation' in entry:
                put3((x,y+MH3,x+CW3,y+MH3+7),'mean '+entry['annotation'].split(chr(10))[0],5.9,align=1)
    yb=Y3+4*RH3+2;x0b,x1b=LW3+CW3*0.5,LW3+CW3*4.5
    t3p.insert_image(fitz.Rect(x0b,yb,x1b,yb+5),stream=png_stream(Image.fromarray(np.repeat(rgb,8,axis=0))))
    for value,lab in {0:'0 = identical to the target',1:'1 = as close as the dataset average',2:'2 = as far as a stranger'}.items():
        xv=x0b+float(norm(value))*(x1b-x0b)
        put3((xv-60,yb+6,xv+60,yb+13),lab,5.9,align=1)
    # Explanations and the takeaway belong in the LaTeX caption, not on the image (Kate, 15 Nov).
    t3.set_metadata({'title':'Population-reference error maps','author':''})
    t3.save(TEASER_V3,garbage=4,deflate=True)
    # Teaser v4 (Kate, 15 Nov, after design review): A = method sketch, B = result grid; short headers, no color
    # dashes, row labels = modality + dataset with n, the dataset-average error column set apart with its own scale.
    t4=fitz.open();H4=254;W4=495;t4p=t4.new_page(width=W4,height=H4)
    t4p.insert_font(fontname='helv');t4p.insert_font(fontname='hebo')
    def put4(rect,text,size,bold=False,color=INK,align=0):
        r=fitz.Rect(rect);r.y1+=9
        spare=t4p.insert_textbox(r,text,fontsize=size,fontname='hebo' if bold else 'helv',color=color,align=align)
        assert spare>=0,('Label does not fit',text,spare)
    dark=(.33,.33,.33)
    # ---- A: hide and fill, on the CelebA population average face
    put4((0,0,12,12),'A',9,True)
    AW=46
    for k,(im,lab) in enumerate(((face,'1  Target'),(hid,'2  Hide'+chr(10)+'a region'),(fil,'3  Fill and'+chr(10)+'compare'))):
        y=22+k*72;sc=min(38/im.size[0],44/im.size[1]);dw,dh=im.size[0]*sc,im.size[1]*sc
        t4p.insert_image(fitz.Rect(AW/2-dw/2,y,AW/2+dw/2,y+dh),stream=png_stream(im))
        put4((-5,y+dh+2,AW+5,y+dh+16),lab,6.5,align=1)
    t4p.draw_line((AW+8,4),(AW+8,H4-4),color=RULE,width=.5)
    # ---- B: result grid
    BX=AW+14;put4((BX,0,BX+12,12),'B',9,True)
    LW4=58;GAP=10;CW4=(W4-BX-LW4-GAP)/6;MH4=40;RH4=45;Y4=43
    xs4=[BX+LW4+j*CW4+(GAP if j==5 else 0) for j in range(6)]
    heads4=['Dataset'+chr(10)+'average','Same person','Mirrored side','Random person','16 nearest'+chr(10)+'people','Error of the'+chr(10)+'average']
    fx0,fx1=xs4[1],xs4[4]+CW4
    put4((fx0,4,fx1,12),'Fill of the hidden region',7,True,align=1)
    t4p.draw_line((fx0+6,15),(fx1-6,15),color=RULE,width=.5)
    for j,label in enumerate(heads4):
        put4((xs4[j]-2,17,xs4[j]+CW4+2,32),label,7,align=1)
    sep=xs4[5]-GAP/2
    t4p.draw_line((sep,18),(sep,Y4+4*RH4),color=RULE,width=.6)
    rows4=[('Brain MRI','QTIM, n = 221','1 mm voxels'),('Faces','CelebA, n = 921','128 x 128 px'),
           ('Retina','GRAPE, n = 134','128 x 128 px'),('Iris','UPOL, n = 64','64 x 256 px strip')]
    for i,(group,ds,res) in enumerate(rows4):
        y=Y4+i*RH4
        put4((BX,y+MH4/2-13,BX+LW4,y+MH4/2-5),group,7,True)
        put4((BX,y+MH4/2-4,BX+LW4,y+MH4/2+4),ds,6.5,color=dark)
        put4((BX,y+MH4/2+4.5,BX+LW4,y+MH4/2+12.5),res,6.5,color=dark)
        for j in range(6):
            image=panel_images[i,j]
            if j==5 and not brain_v2_used.get(i):image=recolor_viridis(image)
            iw,ih=image.size;sc=min((CW4-6)/iw,MH4/ih)*(0.8 if i in (2,3) else 1.0);dw,dh=iw*sc,ih*sc   # retina, iris smaller
            cx=xs4[j]+CW4/2
            t4p.insert_image(fitz.Rect(cx-dw/2,y+MH4/2-dh/2,cx+dw/2,y+MH4/2+dh/2),stream=png_stream(image))
            entry=next(p for p in panels if p['row']==i and p['source_column']==columns[j])
            # no numbers under the maps (Kate, 15 Nov); region values are in the paper tables
    # ---- legends: shared ratio scale under the fills, own scale under the last column
    yb=Y4+4*RH4+2;x0b,x1b=xs4[1]+6,xs4[4]+CW4-6
    t4p.insert_image(fitz.Rect(x0b,yb,x1b,yb+5),stream=png_stream(Image.fromarray(np.repeat(rgb,8,axis=0))))
    for value,lab in {0:'0  Identical',1:'1  Dataset average',2:'2  Random person'}.items():
        xv=x0b+float(norm(value))*(x1b-x0b)
        t4p.draw_line((xv,yb+5),(xv,yb+7.5),color=dark,width=.5)
        put4((xv-40,yb+8,xv+40,yb+15),lab,6.5,align=1)
    put4((x0b,yb+17,x1b,yb+24),'MSE of the fill, relative to the dataset average',6.5,True,align=1)
    red=(colormaps['RdBu_r'](0.5+np.linspace(0,1,256)[None,:]/2)[:,:,:3]*255).astype('uint8')
    rx0,rx1=xs4[5]+6,xs4[5]+CW4-6
    t4p.insert_image(fitz.Rect(rx0,yb,rx1,yb+5),stream=png_stream(Image.fromarray(np.repeat(red,8,axis=0))))
    put4((rx0-6,yb+8,rx0+20,yb+15),'low',6.5)
    put4((rx1-20,yb+8,rx1+6,yb+15),'high',6.5,align=2)
    t4.set_metadata({'title':'Hide, fill and compare: population error maps','author':''})
    t4.save(TEASER_V4,garbage=4,deflate=True)
    # Teaser v5 (Kate and Paula, 16 Nov): v4 layout at the same aspect, printed at 0.72 textwidth, so fonts are larger;
    # row labels give targets, people and how many have a second image; the last column prints each row's color range.
    # Counts: tab/coverage.tex (QTIM 23 with repeat; GRAPE 171; UPOL 128) and results/face_ladder_metrics.json
    # (CelebA n_repeat_targets = 921). Ranges: 2nd and 98th percentile of the displayed V map, the limits of its color.
    vr=v_ranges()
    if brain_v2_used.get(0):vr['QTIM']=bratios['V_range']
    t5=fitz.open();H5=254;W5=495;t5p=t5.new_page(width=W5,height=H5)
    t5p.insert_font(fontname='helv');t5p.insert_font(fontname='hebo')
    def put5(rect,text,size,bold=False,color=INK,align=0):
        r=fitz.Rect(rect);r.y1+=9
        spare=t5p.insert_textbox(r,text,fontsize=size,fontname='hebo' if bold else 'helv',color=color,align=align)
        assert spare>=0,('Label does not fit',text,spare)
    dark=(.30,.30,.30);F=8.0;FS=7.4
    put5((0,0,14,13),'A',10,True)
    AW=50
    for k,(im,lab) in enumerate(((face,'1  Target'),(hid,'2  Hide'+chr(10)+'a region'),(fil,'3  Fill and'+chr(10)+'compare'))):
        y=20+k*76;sc=min(40/im.size[0],44/im.size[1]);dw,dh=im.size[0]*sc,im.size[1]*sc
        t5p.insert_image(fitz.Rect(AW/2-dw/2,y,AW/2+dw/2,y+dh),stream=png_stream(im))
        put5((-6,y+dh+1,AW+6,y+dh+18),lab,FS,align=1)
    t5p.draw_line((AW+7,4),(AW+7,H5-4),color=RULE,width=.5)
    BX=AW+12;put5((BX,0,BX+14,13),'B',10,True)
    LW5=92;GAP=8;CW5=(W5-BX-LW5-GAP)/6;MH5=40;RH5=45;Y5=42
    xs5=[BX+LW5+j*CW5+(GAP if j==5 else 0) for j in range(6)]
    heads5=['Dataset'+chr(10)+'average','Same'+chr(10)+'person','Mirrored'+chr(10)+'side','Random'+chr(10)+'person','16 nearest'+chr(10)+'people','Error of the'+chr(10)+'average']
    fx0,fx1=xs5[1],xs5[4]+CW5
    put5((fx0,1,fx1,10),'Fill of the hidden region',F,True,align=1)
    t5p.draw_line((fx0+6,13),(fx1-6,13),color=RULE,width=.5)
    for j,label in enumerate(heads5):
        put5((xs5[j]-3,16,xs5[j]+CW5+3,33),label,FS,align=1)
    sep=xs5[5]-GAP/2
    t5p.draw_line((sep,18),(sep,Y5+4*RH5),color=RULE,width=.6)
    rows5=[('Brain MRI',['QTIM, 221 people','23 with a second scan','1 mm voxels'],'QTIM'),
           ('Faces',['CelebA, 921 people','all with a second photo','128 x 128 px'],'CelebA'),
           ('Retina',['GRAPE, 239 eyes','of 134 people','171 with a second visit','128 x 128 px'],'GRAPE'),
           ('Iris',['UPOL, 128 eyes','of 64 people','all with a second photo','64 x 256 px strip'],'UPOL')]
    for i,(group,lines,key) in enumerate(rows5):
        y=Y5+i*RH5;step=8.6;yl=y+MH5/2-(len(lines)+1)*step/2
        put5((BX,yl,BX+LW5,yl+8),group,F,True)
        for q,line in enumerate(lines):
            put5((BX,yl+step*(q+1),BX+LW5,yl+step*(q+1)+8),line,FS,color=dark)
        for j in range(6):
            image=panel_images[i,j]
            if j==5 and not brain_v2_used.get(i):image=recolor_viridis(image)
            iw,ih=image.size;sc=min((CW5-6)/iw,(MH5-(8 if j==5 else 0))/ih)*(0.8 if i in (2,3) else 1.0);dw,dh=iw*sc,ih*sc
            cx=xs5[j]+CW5/2;cy=y+(MH5-(8 if j==5 else 0))/2
            t5p.insert_image(fitz.Rect(cx-dw/2,cy-dh/2,cx+dw/2,cy+dh/2),stream=png_stream(image))
        lo,hi=vr[key]
        put5((xs5[5]-4,y+MH5-6,xs5[5]+CW5+4,y+MH5+2),f'{lo:.2f} to {hi:.2f}',FS,color=dark,align=1)
    yb=Y5+4*RH5+3;x0b,x1b=xs5[1]+6,xs5[4]+CW5-6
    t5p.insert_image(fitz.Rect(x0b,yb,x1b,yb+5),stream=png_stream(Image.fromarray(np.repeat(rgb,8,axis=0))))
    for value,lab in {0:'0  Identical',1:'1  Dataset average',2:'2  Random person'}.items():
        xv=x0b+float(norm(value))*(x1b-x0b)
        t5p.draw_line((xv,yb+5),(xv,yb+7.5),color=dark,width=.5)
        put5((xv-45,yb+8,xv+45,yb+16),lab,FS,align=1)
    put5((x0b,yb+17,x1b,yb+25),'MSE of the fill, relative to the dataset average',FS,True,align=1)
    red=(colormaps['RdBu_r'](0.5+np.linspace(0,1,256)[None,:]/2)[:,:,:3]*255).astype('uint8')
    rx0,rx1=xs5[5]+3,xs5[5]+CW5-3
    t5p.insert_image(fitz.Rect(rx0,yb,rx1,yb+5),stream=png_stream(Image.fromarray(np.repeat(red,8,axis=0))))
    put5((rx0,yb+8,rx0+26,yb+16),'min',FS)
    put5((rx1-26,yb+8,rx1,yb+16),'max',FS,align=2)
    put5((xs5[5]-8,yb+17,W5,yb+25),'MSE, z units',FS,True,align=1)
    t5.set_metadata({'title':'Hide, fill and compare: population error maps','author':''})
    t5.save(TEASER_V5,garbage=4,deflate=True)
    print(json.dumps({'teaser_v5_V_ranges':{k:[round(a,4),round(b,4)] for k,(a,b) in vr.items()}}))
    # Teaser (Kate, 15 Nov): panel B alone at full text width, without its letter and title row.
    teaser=fitz.open();tp=teaser.new_page(width=495,height=541-168)
    tp.show_pdf_page(tp.rect,document,0,clip=fitz.Rect(0,168,495,541))
    teaser.set_metadata({'title':'Population-reference error maps','author':''})
    teaser.save(TEASER,garbage=4,deflate=True)
    # Datasets panel re-laid out for one column (Kate, 15 Nov): bust in the middle, two cohorts on each side.
    CW,CH=236,108
    col=fitz.open();cp=col.new_page(width=CW,height=CH)
    cp.insert_font(fontname='helv');cp.insert_font(fontname='hebo')
    def cput(rect,text,size,bold=False,color=INK,align=0):
        spare=cp.insert_textbox(fitz.Rect(rect),text,fontsize=size,fontname='hebo' if bold else 'helv',color=color,align=align)
        assert spare>=0,('Label does not fit',text,spare)
    bw=72;bh=bw*132/123;bx0=(CW-bw)/2;by0=(CH-bh)/2
    cp.insert_image(fitz.Rect(bx0,by0,bx0+bw,by0+bh),filename=str(source_spec.ASSET),keep_proportion=True)
    def to_col(pt):return (bx0+(pt[0]-188)/123*bw,by0+(pt[1]-9)/132*bh)
    slots=[(0,2,'left'),(0,56,'left'),(CW-80,2,'right'),(CW-80,56,'right')]
    for block,(x,y,side),(_,points) in zip(blocks,slots,leaders):
        end=to_col(points[-1])
        start=(x+80,y+6) if side=='left' else (x,y+6)
        knee=(start[0]+(6 if side=='left' else -6),start[1])
        cp.draw_polyline([start,knee,end],color=block['color'],width=.5)
        cp.draw_circle(end,1.1,color=(1,1,1),fill=block['color'],width=.3)
        al=0 if side=='left' else 2
        cp.draw_line((x if side=='left' else x+72,y+1),(x+8 if side=='left' else x+80,y+1),color=block['color'],width=1.6)
        cput((x,y+3,x+80,y+16),block['name'],7,True,align=al)
        cput((x,y+15,x+80,y+34),block['counts'],6.5,align=al)
        cput((x,y+26,x+80,y+50),block['detail'],6.3,color=MUTED,align=al)
    col.set_metadata({'title':'Source cohorts','author':'','subject':'Decorative synthetic bust; cohort counts.'})
    col.save(DATASETS,garbage=4,deflate=True)
    proof={'generator':record(Path(__file__)),'source_specification_module':record(Path(source_spec.__file__)),
           'source_atlas':record(full),'source_atlas_provenance':record(HERE/'atlas_provenance.json'),
           'source_commit':source_spec.SOURCE_COMMIT,'original_source_pdf':source_spec.SOURCE_PATH,
           'scientific_inputs':{key:(record(path) if key=='fundus' else source_snapshot.record(path)) for key,path in paths.items()},
           'decorative_asset':{**record(source_spec.ASSET),'role':'Previously generated decorative bust; unchanged asset.',
                               'generation_record':record(source_spec.ASSET.with_suffix('.provenance.json'))},
           'source_cohorts':blocks,'panels':panels,'mri_background_display':mri_proof,
           'layout_changes':'White canvas; a/b panels; source text with leaders instead of dashboard cards; smaller bust; maps 50pt high instead of42pt; labels7pt.',
           'numeric_color_scale':{'vmin':0,'vcenter':1,'vmax':2.4,'changed':False},
           'spatial_scale':'Aspect preserved; one common MRI crop. Rows have display scales, not a common physical scale.',
           'measurement_changes':False,'generated_scientific_images':False,
           'output':record(OUT),'shareable_png':record(PNG)}
    (HERE/'provenance_infographic_v3.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'outputs':[str(OUT),str(PNG)],'mri_display':mri_proof},indent=2))


if __name__=='__main__':main()
