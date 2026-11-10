"""Post-review GRAPE pilot: retinal features, exclude camera boundary; no scoring."""
import json
import shutil
import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from registration_qc_pilot import OUT, PRIVATE, ROOTS, canvas, fit, marked, sheet, sha, dump


def feature(a,valid):
    mask=binary_erosion((a[...,0]>25)&(valid>0),iterations=25).astype('uint8')
    g=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(a[...,1])
    return np.repeat(g[...,None],3,axis=2),mask


def main():
    dest=PRIVATE/'registration-qc-fundus-20261102-v2'; dest.mkdir(exist_ok=False)
    manifest=json.loads((OUT/'manifest.json').read_text()); records=[]
    dump(dest/'plan.json',{'parent_manifest_sha256':sha(OUT/'manifest.json'),'code_sha256':sha(__file__),
         'posthoc':'v1 fitted camera boundary; now erode FOV by 25 pixels and use green-channel CLAHE; same seven pilot rows',
         'fit_thresholds':'unchanged from v1; no nonlinear deformation'})
    for index in manifest['pilot_indices']:
        row=manifest['rows'][index]; p=row['pair']
        if p['dataset']!='GRAPE': continue
        a,va,ga=canvas(ROOTS['GRAPE']/p['anchor']); raw,vb,gb=canvas(ROOTS['GRAPE']/p['candidate'])
        b=raw[:,::-1].copy() if row['mode']=='fellow_mirror' else raw.copy()
        if row['mode']=='fellow_mirror': vb=vb[:,::-1].copy()
        if row['mode']=='reflected_input': raw=raw[:,::-1].copy()
        fa,ma=feature(a,va); fb,mb=feature(b,vb)
        t,points,info=fit(fa,fb,ma,mb)
        poly=np.float32([[183,183],[329,183],[329,329],[183,329]])
        z=np.zeros_like(a); moving=raw
        if t is not None:
            z=cv2.warpAffine(b,t,(512,512))
            src=cv2.transform(poly[None],cv2.invertAffineTransform(t))[0]
            if row['mode'] in ('fellow_mirror','reflected_input'): src[:,0]=511-src[:,0]
            moving=marked(raw,src)
        ca,cz=a[183:329,183:329],z[183:329,183:329]
        yy,xx=np.indices(ca.shape[:2]); check=((xx//16+yy//16)%2)[...,None]
        edges=np.zeros_like(a);edges[:,:,1]=cv2.Canny(fa[...,0],60,140);edges[:,:,0]=cv2.Canny(cv2.cvtColor(z,cv2.COLOR_RGB2GRAY),60,140)
        name=f'fundus_{len(records):02d}.png'
        sheet([marked(a,poly,points),moving,marked(z,poly),edges,ca,cz,cv2.addWeighted(ca,.5,cz,.5,0),np.where(check,ca,cz)],
              ['Anchor; interior retinal matches','Moving; inverse crop','Registered candidate','Edges (different enhancement)',
               'Anchor crop','Moving crop','Blend','Checker'],dest/name)
        records.append({'index':index,'stratum':row['stratum'],**info,'visual':'pending','sheet':name,'sheet_sha256':sha(dest/name),
                        'transform':None if t is None else t.tolist()})
    dump(dest/'records.json',records);shutil.copy2(__file__,dest/'code_snapshot.py')
    from collections import Counter
    print(json.dumps(dict(Counter(x['status'] for x in records))))


if __name__=='__main__':main()
