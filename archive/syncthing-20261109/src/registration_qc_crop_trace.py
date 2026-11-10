"""Trace the historical centre-square 224/64 crop, without scoring or re-fitting."""
import json
import cv2
import numpy as np
from registration_qc_pilot import OUT, ROOTS, canvas, marked, sheet, dump, sha


def historical_box(geometry):
    h,w=geometry['native_shape']; s=min(h,w)
    left,top=(w-s)//2,(h-s)//2
    # Continuous pixel-boundary coordinates; cv2 resize uses pixel centres.
    lo=np.array([left,top])+80*s/224-.5
    hi=np.array([left,top])+144*s/224-.5
    native=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])
    scale=np.array(geometry['scale_xy']); offset=np.array(geometry['offset_xy'])
    return (native+.5)*scale-.5+offset


def extract(a,poly):
    # Preview interpolation only; not identical to historical BOX image intensities.
    x0,y0=poly[0]; x1,y1=poly[2]
    xx,yy=np.meshgrid(x0+(np.arange(146)+.5)*(x1-x0)/146,
                      y0+(np.arange(146)+.5)*(y1-y0)/146)
    return cv2.remap(a,xx.astype('float32'),yy.astype('float32'),cv2.INTER_LINEAR)


def main():
    dest=OUT/'historical_crop_trace'; dest.mkdir(exist_ok=False)
    manifest=json.loads((OUT/'manifest.json').read_text())
    records=json.loads((OUT/'pilot_records.json').read_text())
    trace=[]
    for i,rec in enumerate(records):
        if 'source_geometry' not in rec: continue
        row=manifest['rows'][rec['index']]; pair=row['pair']; ds=pair['dataset']
        pa,pb=[ROOTS[ds]/pair[k] for k in ('anchor','candidate')]
        if [sha(pa),sha(pb)]!=rec['source_sha256']:
            raise RuntimeError('source changed')
        a,va,ga=canvas(pa); raw,vb,gb=canvas(pb)
        if row['mode']=='reflected_input': raw=raw[:,::-1].copy()
        b=raw[:,::-1].copy() if row['mode'] in ('reflected_input','fellow_mirror') else raw
        poly=historical_box(ga); z=np.zeros_like(a); moving=raw
        source_polygon=None
        if rec['moving_to_anchor'] is not None:
            t=np.array(rec['moving_to_anchor'])
            z=cv2.warpAffine(b,t,(512,512))
            src=cv2.transform(poly.astype('float32')[None],cv2.invertAffineTransform(t))[0]
            if row['mode'] in ('reflected_input','fellow_mirror'): src[:,0]=511-src[:,0]
            moving=marked(raw,src); source_polygon=src.tolist()
        ca,cz=extract(a,poly),extract(z,poly)
        yy,xx=np.indices(ca.shape[:2]); check=((xx//16+yy//16)%2)[...,None]
        edges=np.zeros_like(a)
        edges[:,:,1]=cv2.Canny(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),60,140)
        edges[:,:,0]=cv2.Canny(cv2.cvtColor(z,cv2.COLOR_RGB2GRAY),60,140)
        name=f'crop_{i:02d}.png'
        sheet([marked(a,poly),moving,marked(z,poly),edges,ca,cz,
               cv2.addWeighted(ca,.5,cz,.5,0),np.where(check,ca,cz)],
              ['Full anchor: historical metric crop','Full moving: inverse-mapped crop',
               'Aligned candidate (black if rejected)','Edges: anchor green / moving red',
               'Anchor historical crop geometry','Moving at corresponding geometry',
               '50:50 blend','Checkerboard'],dest/name)
        trace.append({'stratum':rec['stratum'],'anchor_polygon':poly.tolist(),
                      'moving_polygon':source_polygon,'sheet':name,'sha256':sha(dest/name),
                      'visual':'not_reviewed'})
    dump(dest/'trace.json',{'parent_manifest_sha256':sha(OUT/'manifest.json'),
                          'code_sha256':sha(__file__),'records':trace})
    import shutil
    shutil.copy2(__file__,dest/'code_snapshot.py')
    html=['<!doctype html><meta charset="utf-8"><style>body{font:18px sans-serif;margin:25px}img{width:100%}</style>',
          '<h1>Historical centre-square crop: coordinate trace</h1>',
          '<p>Exact source extent of [80:144,80:144] after centre-square resize to 224. Preview interpolation differs from historical BOX intensity processing. Later canonical iris frames are not validated by this trace. No visual acceptance assigned.</p>']
    for r in trace: html.append(f'<h2>{r["stratum"]}</h2><img src="{r["sheet"]}">')
    (dest/'index.html').write_text('\n'.join(html),encoding='utf-8')
    print(json.dumps({'historical_crop_sheets':len(trace)}))


if __name__=='__main__': main()
