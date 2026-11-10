"""Modality-specific pretrained QC on the frozen exposed pilot; no identity scores."""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from registration_qc_pilot import PRIVATE, ROOT, OUT, ROOTS, sha, dump

MODELS = PRIVATE/'registration-models-20261103'
CHECKPOINTS = {'SuperRetina.pth':'df9d28098d1d95347f96c6a2a425db3f9ab1683778173a22eca96f0e863758cf',
    'iris_semseg_upp_scse_mobilenetv2.onnx':'1228a1c760791749fceeedf9322e24f098aac80234368e1beee156e1e8fa57e7'}


def iris_preprocess(gray):
    """Official Open-IRIS default: 640 width, 480 height, ImageNet channel scaling."""
    a = cv2.resize(gray,(640,480)).astype(np.float64)/255
    a = np.repeat(a[...,None],3,axis=2)
    return ((a-[.485,.456,.406])/[.229,.224,.225]).transpose(2,0,1)[None].astype(np.float32)


def iris_postprocess(prediction, shape):
    return cv2.resize(prediction[0].transpose(1,2,0).astype(np.float32),
                      (shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)


def run_iris(dest, manifest):
    import onnxruntime as ort
    import registration_qc_fundus_frame as F
    opts=ort.SessionOptions(); opts.intra_op_num_threads=2; opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(MODELS/'iris_semseg_upp_scse_mobilenetv2.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    cache={}; records=[]
    for index in manifest['pilot_indices']:
        row=manifest['rows'][index]; pair=row['pair']
        if pair['dataset']=='GRAPE': continue
        cells=[]; roles=[]
        for role in ('anchor','candidate'):
            path=ROOTS[pair['dataset']]/pair[role]
            key=str(path)
            if key not in cache:
                gray=np.asarray(Image.open(path).convert('L'))
                out=session.run(None,{session.get_inputs()[0].name:iris_preprocess(gray)})[0]
                probs=iris_postprocess(out,gray.shape)
                cache[key]=(gray,probs)
            gray,probs=cache[key]
            masks=probs>.5
            # Native coordinate space; no canonical crop or normalization change.
            rgb=np.repeat(gray[...,None],3,axis=2)
            overlay=rgb.copy()
            for channel,colour in [(0,(255,160,0)),(1,(0,255,0)),(2,(0,180,255)),(3,(255,0,180))]:
                contours,_=cv2.findContours(masks[...,channel].astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay,contours,-1,colour,1)
            visible=masks[...,1] & ~masks[...,2] & ~masks[...,3]
            cells += [(rgb,role+' native (no warp)'),(overlay,'eyeball orange / iris green / pupil cyan / lashes magenta'),
                      (visible.astype(np.uint8)*255,'predicted visible iris; not ground truth'),
                      ((probs[...,1]*255).astype(np.uint8),'iris probability (not identity score)')]
            name=f'row_{len(records):02d}_{role}.npz'
            np.savez_compressed(dest/name,probabilities=probs)
            roles.append({'role':role,'source_sha256':sha(path),'probabilities_sha256':sha(dest/name),
                'native_shape':list(gray.shape),'mask_pixels':[int(masks[...,k].sum()) for k in range(4)],
                'visible_iris_pixels':int(visible.sum())})
        sheet=f'iris_{len(records):02d}.png'; F.sheet(cells,dest/sheet)
        records.append({'index':index,'stratum':row['stratum'],'sheet':sheet,'sheet_sha256':sha(dest/sheet),
            'mode':row['mode'],'roles':roles,'visual':'not_reviewed','registration':'NOT RUN',
            'domain':'NIR intended modality' if pair['dataset']=='UTIRIS' else 'visible-light out-of-domain feasibility'})
        dump(dest/'records.json',records)
    return records


def run_fundus(dest,manifest):
    import torch
    import registration_qc_fundus_frame as F
    sys.path.insert(0,str(ROOT/'external/SuperRetina'))
    from model.super_retina import SuperRetina
    from common.common_util import pre_processing,simple_nms,remove_borders,sample_keypoint_desc
    from torchvision import transforms
    torch.set_num_threads(2)
    model=SuperRetina(); model.load_state_dict(torch.load(MODELS/'SuperRetina.pth',map_location='cpu',weights_only=True)['net']); model.eval()
    transform=transforms.Compose([transforms.Resize((768,768)),transforms.ToTensor()])
    cache={}; records=[]
    def features(img):
        inp=pre_processing((np.clip(img[...,1],0,1)*255).astype(np.uint8))
        tensor=transform(Image.fromarray((inp*255).astype(np.uint8)))[None]
        with torch.inference_mode(): detector,descriptor=model(tensor)
        score=simple_nms(detector,10)[0,0]
        kp=torch.nonzero(score>.01); values=score[tuple(kp.T)]
        kp,_=remove_borders(kp,values,4,768,768); kp=torch.flip(kp,[1]).float()
        desc=sample_keypoint_desc(kp[None],descriptor,8)[0].T.numpy()
        return kp.numpy()*(512/768),desc
    for index in manifest['pilot_indices']:
        row=manifest['rows'][index]; pair=row['pair']
        if pair['dataset']!='GRAPE': continue
        ap,bp=ROOTS['GRAPE']/pair['anchor'],ROOTS['GRAPE']/pair['candidate']
        a,_,ta,ca=F.prepare(ap)
        b,_,tb,cb=F.prepare(bp,reflected_input=row['mode']=='reflected_input',fellow=row['mode']=='fellow_mirror')
        rec={'index':index,'stratum':row['stratum'],'mode':row['mode'],'source_sha256':[sha(ap),sha(bp)],'visual':'not_reviewed'}
        if ca is None or cb is None:
            rec['status']='G1_failure'; records.append(rec); continue
        for key,img in [(('a',sha(ap),False),ca),(('b',sha(bp),row['mode']=='fellow_mirror'),cb)]:
            if key not in cache: cache[key]=features(img)
        ka,da=cache[('a',sha(ap),False)]; kb,db=cache[('b',sha(bp),row['mode']=='fellow_mirror')]
        matches=[m for v in cv2.BFMatcher(cv2.NORM_L2).knnMatch(db,da,k=2) if len(v)==2 for m,n in [v] if m.distance<.9*n.distance]
        src=np.float32([kb[m.queryIdx] for m in matches]); dst=np.float32([ka[m.trainIdx] for m in matches])
        rec.update({'keypoints':[len(ka),len(kb)],'ratio_matches':len(matches),'fits':{}})
        for kind in ('homography','similarity'):
            cv2.setRNGSeed(20261103)
            mat,inside=(None,None)
            if len(matches)>=4:
                if kind=='homography': mat,inside=cv2.findHomography(src,dst,cv2.LMEDS)
                else:
                    affine,inside=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=500)
                    if affine is not None: mat=np.vstack([affine,[0,0,1]])
            fit={'status':'no_fit','transform':None,'inliers':int(inside.sum()) if inside is not None else 0}
            if mat is not None and np.isfinite(mat).all():
                fit.update(status='geometric_candidate',transform=mat.tolist())
                if kind=='similarity':
                    scale=np.hypot(mat[0,0],mat[1,0]); rotation=np.degrees(np.arctan2(mat[1,0],mat[0,0]))
                    fit.update(scale=float(scale),rotation_deg=float(rotation))
                    if fit['inliers']<12 or not .8<=scale<=1.25 or abs(rotation)>=20: fit['status']='rejected_bounds'
                warped=cv2.warpPerspective(cb,mat,(512,512))
                x0,y0,x1,y1=F.BOXES['metric']; za,zb=ca[y0:y1,x0:x1],warped[y0:y1,x0:x1]
                yy,xx=np.indices(za.shape[:2]); check=((xx//16+yy//16)%2)[...,None]
                poly=np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
                inverse_poly=cv2.perspectiveTransform(poly[None],np.linalg.inv(mat))[0]
                back=tb(inverse_poly)
                fit['candidate_crop_source_xy']=back.tolist()
                fit['metric_fov_support']=float((warped[y0:y1,x0:x1,0]>.1).mean())
                cells=[(F.draw(a,[(ta(poly),(255,180,0))]),'anchor source; fixed crop mapped back'),
                       (F.draw(b,[(back,(255,180,0))]),'candidate source; inverse-mapped crop'),
                       (ca,'anchor shared G1'),(cb,'candidate shared G1'),
                       (warped,'candidate after '+kind),(F.overlay(ca,warped),'vessel overlay: anchor red / moving green'),
                       ((ca+warped)/2,'full blend'),(np.where((((np.indices(ca.shape[:2]).sum(axis=0)//16)%2)[...,None]),ca,warped),'full alternating bands'),
                       (za,'fixed anchor metric crop'),(zb,'registered moving metric crop'),((za+zb)/2,'crop blend'),(np.where(check,za,zb),'crop checker')]
                sheet=f'fundus_{len(records):02d}_{kind}.png'; F.sheet(cells,dest/sheet)
                fit.update(sheet=sheet,sheet_sha256=sha(dest/sheet))
            rec['fits'][kind]=fit
        rec['status']='prepared'; records.append(rec); dump(dest/'records.json',records)
    return records


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('modality',choices=['fundus','iris']); args=ap.parse_args()
    filename='SuperRetina.pth' if args.modality=='fundus' else 'iris_semseg_upp_scse_mobilenetv2.onnx'
    if sha(MODELS/filename)!=CHECKPOINTS[filename]: raise RuntimeError('Model hash drift')
    manifest_path=OUT/'manifest.json'; manifest=json.loads(manifest_path.read_text())
    dest=PRIVATE/f'registration-qc-{args.modality}-pretrained-20261103-v1'; dest.mkdir(exist_ok=False)
    dump(dest/'plan.json',{'code_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'docs/PROTOCOL_registration.md'),
        'manifest_sha256':sha(manifest_path),'model_sha256':CHECKPOINTS[filename],
        'scope':'same exposed development pilot only; no scoring; no auto PASS'})
    shutil.copy2(__file__,dest/'code_snapshot.py')
    records=run_fundus(dest,manifest) if args.modality=='fundus' else run_iris(dest,manifest)
    aggregate={'modality':args.modality,'pilot_rows':len(records),'visual_acceptance':'NOT ESTABLISHED','gate':'NOT MET',
        'model_sha256':CHECKPOINTS[filename],'plan_sha256':sha(dest/'plan.json'),
        'records_sha256':sha(dest/'records.json'),'status':'prepared; visual review pending'}
    dump(dest/'aggregate.json',aggregate)
    dump(ROOT/f'results/registration/qc/{args.modality}_pretrained_v1.json',aggregate)
    print(json.dumps(aggregate))


if __name__=='__main__':main()
