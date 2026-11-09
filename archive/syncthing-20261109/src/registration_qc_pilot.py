"""Independent QC-only development pilot. No scoring, no automatic visual PASS.

Raw frames, source paths and pair decisions remain in the private data root.
No dependency on the concurrently edited ladder2d registration/figure modules.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('C:/Projects/data')
PRIVATE = DATA / 'projects/identity-ladder-metric'
SELECTION = PRIVATE / 'run-2026-11-02-2d-ladder'
OUT = PRIVATE / 'registration-qc-independent-20261102-v1'
ROOTS = {
    'GRAPE': DATA / 'datasets/ophthalmology/grape/figshare-c6406319-v1/raw/CFPs',
    'UPOL': DATA / 'datasets/biometrics/upol/version-unverified/raw',
    'UTIRIS': DATA / 'datasets/biometrics/utiris/v1/raw',
}
EXPECTED = {
    'grape': 'd3c236d60d125f886ea93ed2804a815b0007e12ccc1082c01b36c2d1a7a4708a',
    'iris': 'c3a41f09ac7d622c8215788b99c2d67ce1ecfc729d05ba33aab1b319240e107c',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def canvas(path):
    # Preserve entire field and aspect ratio, without the historical centre crop.
    with Image.open(path) as im:
        a = np.asarray(im.convert('RGB'))
    h, w = a.shape[:2]
    s = 512 / max(h, w)
    nw, nh = round(w*s), round(h*s)
    ox, oy = (512-nw)//2, (512-nh)//2
    b = np.zeros((512, 512, 3), np.uint8)
    b[oy:oy+nh, ox:ox+nw] = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    valid = np.zeros((512, 512), np.uint8)
    valid[oy:oy+nh, ox:ox+nw] = 1
    return b, valid, {'native_shape': [h,w], 'scale_xy': [nw/w,nh/h], 'offset_xy': [ox,oy]}


def fit(a, b, va, vb):
    """Moving -> anchor proper similarity; rejected estimates never become identity fallback."""
    sift = cv2.SIFT_create(nfeatures=2000)
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    ka, da = sift.detectAndCompute(ga, va*255)
    kb, db = sift.detectAndCompute(gb, vb*255)
    if da is None or db is None:
        return None, [], {'status':'no_descriptors'}
    matches = [m for pair in cv2.BFMatcher().knnMatch(db, da, k=2)
               if len(pair)==2 for m,n in [pair] if m.distance < .7*n.distance]
    if len(matches)<12:
        return None, [], {'status':'insufficient_matches', 'matches':len(matches)}
    src = np.float32([kb[m.queryIdx].pt for m in matches])
    dst = np.float32([ka[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(20261102)
    t, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                           ransacReprojThreshold=3, maxIters=2000)
    if t is None:
        return None, [], {'status':'no_model'}
    scale = float(np.hypot(t[0,0],t[1,0]))
    angle = float(np.degrees(np.arctan2(t[1,0],t[0,0])))
    n = int(inliers.sum())
    meta = {'status':'candidate_only', 'matches':len(matches), 'inliers':n,
            'scale':scale, 'angle_deg':angle}
    if n<12 or not .8<=scale<=1.25 or abs(angle)>20:
        meta['status']='transform_rejected'
        return None, [], meta
    return t, dst[inliers.ravel().astype(bool)].tolist(), meta


def marked(a, polygon, points=()):
    a = a.copy()
    cv2.polylines(a, [np.rint(polygon).astype(np.int32)], True, (255,180,0),2)
    for x,y in points:
        cv2.circle(a,(round(x),round(y)),3,(0,255,255),1)
    return a


def sheet(images, titles, path):
    result = Image.new('RGB', (4*512,2*550), 'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(result)
    for i,(a,title) in enumerate(zip(images,titles)):
        x,y=(i%4)*512,(i//4)*550
        result.paste(Image.fromarray(a).resize((512,512)),(x,y+30))
        draw.text((x+8,y+8),title,fill='black')
    result.save(path)


def main():
    cv2.setNumThreads(2)
    # Exclusive directory creation preserves previous pilots and avoids shared ownership.
    OUT.mkdir(exist_ok=False)
    rows=[]
    for kind,want in EXPECTED.items():
        p=SELECTION/f'selection_{kind}.json'
        if sha(p)!=want:
            raise RuntimeError('frozen selection hash mismatch')
        for pair in json.loads(p.read_text())['pairs']:
            branch='fellow_mirror' if pair['rung'].startswith('fellow') else 'ordinary'
            for mode in ([branch] if branch=='fellow_mirror' else ['ordinary','reflected_input']):
                rows.append({'pair':pair,'mode':mode,'visual':'not_reviewed'})
    groups={}
    for i,r in enumerate(rows):
        p=r['pair']; key=' | '.join([p['dataset'],p['rung'],r['mode']])
        r['stratum']=key
        groups.setdefault(key,[]).append(i)
    pilot=[idx[0] for idx in groups.values()]
    # Freeze complete denominator and pilot membership before opening any pixels.
    dump(OUT/'manifest.json', {'rows':rows,'pilot_indices':pilot,
                              'selection_hashes':EXPECTED,'code_sha256':sha(__file__),
                              'protocol_sha256':sha(ROOT/'docs/PROTOCOL_registration.md'),
                              'opencv':cv2.__version__})
    reports=[]
    for num,idx in enumerate(pilot):
        r=rows[idx]; p=r['pair']; ds=p['dataset']
        rec={'stratum':r['stratum'],'index':idx,'visual':'not_reviewed'}
        try:
            pa,pb=[ROOTS[ds]/p[k] for k in ('anchor','candidate')]
            a,va,ma=canvas(pa); raw,vraw,mb=canvas(pb)
            b,vb=raw.copy(),vraw.copy()
            # Test explicit reflection handling, not unlabelled negative determinant fitting.
            if r['mode']=='reflected_input':
                raw=raw[:,::-1].copy(); vraw=vraw[:,::-1].copy()
                b=raw[:,::-1].copy(); vb=vraw[:,::-1].copy()
            elif r['mode']=='fellow_mirror':
                b=raw[:,::-1].copy(); vb=vraw[:,::-1].copy()
            t,points,info=fit(a,b,va,vb)
            rec.update(info)
            rec.update({'source_sha256':[sha(pa),sha(pb)],'source_geometry':[ma,mb],
                        'moving_to_anchor':None if t is None else t.tolist()})
            # Pilot crop is diagnostic, explicitly not the old metric crop.
            # Side iris region avoids an all-pupil centred square.
            x0,y0=(310,183) if ds!='GRAPE' else (183,183)
            x1,y1=x0+146,y0+146
            poly=np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
            rec['diagnostic_box_xyxy']=[x0,y0,x1,y1]
            z=np.zeros_like(a)
            if t is not None:
                z=cv2.warpAffine(b,t,(512,512),flags=cv2.INTER_LINEAR)
                support=cv2.warpAffine(vb,t,(512,512),flags=cv2.INTER_NEAREST)
                rec['crop_support_fraction']=float(support[y0:y1,x0:x1].mean())
                inv=cv2.invertAffineTransform(t)
                src=cv2.transform(poly[None],inv)[0]
                if r['mode'] in ('fellow_mirror','reflected_input'):
                    src[:,0]=511-src[:,0]
                moving=marked(raw,src)
            else:
                moving=raw
            ca,cz=a[y0:y1,x0:x1],z[y0:y1,x0:x1]
            yy,xx=np.indices(ca.shape[:2]); check=((xx//16+yy//16)%2)[...,None]
            edge=np.zeros_like(a)
            edge[:,:,1]=cv2.Canny(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),60,140)
            edge[:,:,0]=cv2.Canny(cv2.cvtColor(z,cv2.COLOR_RGB2GRAY),60,140)
            name=f'qc_{num:02d}.png'
            sheet([marked(a,poly,points),moving,marked(z,poly),edge,ca,cz,
                   cv2.addWeighted(ca,.5,cz,.5,0),np.where(check,ca,cz)],
                  ['Anchor full field; crop; fitted points','Moving full field; mapped crop',
                   'Aligned candidate (black if rejected)','Edges: anchor green / moving red',
                   'Anchor diagnostic crop','Moving diagnostic crop','50:50 blend','Checkerboard'],OUT/name)
            rec.update({'sheet':name,'sheet_sha256':sha(OUT/name)})
        except Exception as e:
            rec.update({'status':'preparation_error','error_type':type(e).__name__})
        reports.append(rec)
        dump(OUT/'pilot_records.json',reports)
    aggregate={'gate':'NOT MET','visual_acceptance':'NOT ESTABLISHED',
               'manifest_sha256':sha(OUT/'manifest.json'),'strata':[],
               'pilot_preparation':dict(Counter(r['status'] for r in reports))}
    for key,indices in groups.items():
        aggregate['strata'].append({'stratum':key,'intended':len(indices),'reviewed':0,
                                   'pass':0,'fail':0,'unreviewable':0,'not_reviewed':len(indices)})
    dump(OUT/'aggregate.json',aggregate)
    dump(ROOT/'results/registration/qc/pilot_v1.json',aggregate)
    html=['<!doctype html><meta charset="utf-8"><title>Private registration QC</title>',
          '<style>body{font:18px sans-serif;background:#eee;margin:30px}img{width:100%}section{margin-bottom:50px}</style>',
          '<h1>Private registration QC: development pilot</h1>',
          '<p>No visual PASS assigned. Diagnostic crops are not the historical metric crops. Cyan points are fitted features, not independent validation landmarks. Rejected transforms show black panels, never unregistered fallback.</p>']
    for r in reports:
        html.append(f'<section><h2>{r["stratum"]}</h2><p>{r["status"]}; visual: NOT REVIEWED</p>')
        if 'sheet' in r: html.append(f'<img src="{r["sheet"]}">')
        html.append('</section>')
    (OUT/'index.html').write_text('\n'.join(html),encoding='utf-8')
    print(json.dumps({'intended':len(rows),'pilot':len(reports),'preparation':aggregate['pilot_preparation']}))


if __name__=='__main__':
    main()
