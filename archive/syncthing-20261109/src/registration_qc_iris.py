"""Post-review iris pilot: circle-based full-field geometry, no metric scoring."""
import json
import os
import shutil
import cv2
import numpy as np
from scipy import ndimage
import iris_codes as ic
from registration_qc_pilot import PRIVATE, OUT, ROOTS, canvas, marked, sheet, dump, sha


def pupil(gray, valid):
    candidates=[]
    region=gray[128:384,128:384]
    for q in [5,10,15,20,25,30,35]:
        binary=((gray<np.percentile(region,q)) & (valid>0)).astype('uint8')*255
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=cv2.contourArea(contour); perimeter=cv2.arcLength(contour,True)
            if not perimeter or not 1000<area<18000: continue
            circularity=4*np.pi*area/perimeter**2
            m=cv2.moments(contour); x=m['m10']/m['m00']; y=m['m01']/m['m00']
            radius=np.sqrt(area/np.pi)
            if circularity<.65 or np.hypot(x-255.5,y-255.5)>150: continue
            candidates.append((area*circularity,x,y,radius))
    circles=cv2.HoughCircles(np.uint8(gray),cv2.HOUGH_GRADIENT,1,40,param1=80,param2=20,minRadius=18,maxRadius=70)
    if circles is not None:
        for x,y,r in circles[0]:
            if np.hypot(x-255.5,y-255.5)<130:
                candidates.append((-np.hypot(x-255.5,y-255.5),float(x),float(y),float(r)))
    if not candidates: raise ValueError('no central pupil circle')
    yy,xx=np.indices(gray.shape); ranked=[]
    for _,x,y,r in candidates:
        rr=np.hypot(xx-x,yy-y)
        inner=gray[rr<.65*r]; outer=gray[(rr>1.15*r)&(rr<1.45*r)]
        contrast=float(np.median(outer)-np.median(inner))
        ranked.append((contrast-.02*np.hypot(x-255.5,y-255.5),x,y,r))
    _,x,y,r=max(ranked)
    return x,y,r


def frame(path):
    a,valid,geo=canvas(path)
    # Dark pupil blob only; do not fit eyelashes or camera-frame keypoints.
    gray=a[...,0].astype(float)  # visible iris has better pupil contrast in red; NIR channels equal
    px,py,rp=pupil(ndimage.median_filter(gray,size=5),valid)
    # Lateral arcs avoid upper/lower eyelids. This estimated contour needs visual review.
    angles=np.deg2rad(np.r_[np.arange(-25,26,2),np.arange(155,206,2)])
    radii=np.arange(max(rp*1.7,35),min(rp*6,245),1.)
    xx=px+radii[:,None]*np.cos(angles); yy=py+radii[:,None]*np.sin(angles)
    values=ndimage.map_coordinates(ndimage.gaussian_filter(gray,2),[yy.ravel(),xx.ravel()],order=1,mode='nearest').reshape(xx.shape)
    gradient=np.gradient(values.mean(1))
    ri=float(radii[np.argmax(gradient)])
    scale=180/ri
    t=np.float64([[scale,0,255.5-scale*px],[0,scale,255.5-scale*py]])
    c=cv2.warpAffine(a,t,(512,512),flags=cv2.INTER_LINEAR)
    support=cv2.warpAffine(valid,t,(512,512),flags=cv2.INTER_NEAREST)
    return a,c,support,t,{'pupil':[px,py,rp],'limbus':[px,py,ri],
                         'ratio':rp/ri,'source_sha256':sha(path),'source_geometry':geo}


def main():
    dest=PRIVATE/os.environ.get('QC_IRIS_OUT','registration-qc-iris-20261102-v6'); dest.mkdir(exist_ok=False)
    manifest=json.loads((OUT/'manifest.json').read_text()); records=[]
    dump(dest/'plan.json',{'parent_manifest_sha256':sha(OUT/'manifest.json'),'code_sha256':sha(__file__),
         'segmentation_code_sha256':sha(ic.__file__),'pilot_indices':manifest['pilot_indices'],
         'posthoc':'v6 ranks all circle/blob candidates by radial pupil contrast in red/NIR after v5 eyelash false positive; same six pairs',
         'canonical_radius':180,'rotation':'not corrected in this step','nonlinear_warp':False})
    for index in manifest['pilot_indices']:
        r=manifest['rows'][index]; p=r['pair']; ds=p['dataset']
        if ds=='GRAPE': continue
        rec={'index':index,'stratum':r['stratum'],'visual':'pending'}
        try:
            raw,a,va,ta,ma=frame(ROOTS[ds]/p['anchor'])
            moving,b,vb,tb,mb=frame(ROOTS[ds]/p['candidate'])
            if r['mode']=='fellow_mirror': b=b[:,::-1].copy(); vb=vb[:,::-1].copy()
            if r['mode']!='fellow_mirror':
                from scipy.optimize import least_squares
                ag=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(float)/255
                bg=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY).astype(float)/255
                yy,xx=np.mgrid[0:512:3,0:512:3]; rr=np.hypot(xx-255.5,yy-255.5)
                keep=(rr>75)&(rr<155)&(abs(yy-255.5)<65)
                xs,ys=xx[keep],yy[keep]; target=ag[ys,xs]
                target=(target-target.mean())/max(target.std(),.01)
                def residual(z):
                    dx,dy,ang,logscale=z; co,si=np.cos(ang),np.sin(ang); sc=np.exp(logscale)
                    u=sc*(co*(xs-255.5)-si*(ys-255.5))+255.5+dx
                    v=sc*(si*(xs-255.5)+co*(ys-255.5))+255.5+dy
                    val=ndimage.map_coordinates(bg,[v,u],order=1,mode='constant')
                    return (val-val.mean())/max(val.std(),.01)-target
                fit=least_squares(residual,[0,0,0,0],bounds=([-20,-20,-.26,-.14],[20,20,.26,.14]),
                                  loss='soft_l1',max_nfev=100,diff_step=.001)
                dx,dy,ang,ls=fit.x; co,si=np.cos(ang)*np.exp(ls),np.sin(ang)*np.exp(ls)
                inv=np.array([[co,-si,255.5+dx-co*255.5+si*255.5],[si,co,255.5+dy-si*255.5-co*255.5]])
                correction=cv2.invertAffineTransform(inv)
                # Compose original-to-canonical with correction, resample original once.
                tb=(np.vstack([correction,[0,0,1]])@np.vstack([tb,[0,0,1]]))[:2]
                b=cv2.warpAffine(moving,tb,(512,512))
                _,source_valid,_=canvas(ROOTS[ds]/p['candidate'])
                vb=cv2.warpAffine(source_valid,tb,(512,512),flags=cv2.INTER_NEAREST)
                rec['lateral_refinement_parameters']=fit.x.tolist()
            # Reflected-input branch carries an explicit inverse reflection, no anatomical flip.
            if r['mode']=='reflected_input': moving=moving[:,::-1].copy()
            poly=np.float32([[311,183],[457,183],[457,329],[311,329]])
            source_a=cv2.transform(poly[None],cv2.invertAffineTransform(ta))[0]
            source_b=poly.copy()
            if r['mode']=='fellow_mirror': source_b[:,0]=511-source_b[:,0]
            source_b=cv2.transform(source_b[None],cv2.invertAffineTransform(tb))[0]
            if r['mode']=='reflected_input': source_b[:,0]=511-source_b[:,0]
            context_a=marked(raw,source_a); context_b=marked(moving,source_b)
            for img,m,flip in [(context_a,ma,False),(context_b,mb,r['mode']=='reflected_input')]:
                for key,color in [('pupil',(0,255,255)),('limbus',(0,255,0))]:
                    x,y,rad=m[key]
                    cv2.circle(img,(round(511-x if flip else x),round(y)),round(rad),color,2)
            ca,cb=a[183:329,311:457],b[183:329,311:457]
            yy,xx=np.indices(ca.shape[:2]); check=((xx//16+yy//16)%2)[...,None]
            name=f'iris_{len(records):02d}.png'
            sheet([context_a,context_b,marked(a,poly),marked(b,poly),ca,cb,
                  cv2.addWeighted(ca,.5,cb,.5,0),np.where(check,ca,cb)],
                  ['Anchor full: estimated circles + crop','Moving full: circles + inverse crop',
                   'Anchor canonical','Moving canonical (explicit side rule)',
                   'Anchor crop','Moving crop','50:50 blend','Checkerboard'],dest/name)
            rec.update({'sheet':name,'sheet_sha256':sha(dest/name),'anchor':ma,'moving':mb,
                        'anchor_support':float(va[183:329,311:457].mean()),
                        'moving_support':float(vb[183:329,311:457].mean()),'status':'prepared'})
        except Exception as e: rec.update({'status':'preparation_error','error_type':type(e).__name__})
        records.append(rec)
    dump(dest/'records.json',records); shutil.copy2(__file__,dest/'code_snapshot.py')
    print(json.dumps({'prepared':sum(r['status']=='prepared' for r in records),'intended_pilot':len(records)}))


if __name__=='__main__': main()
