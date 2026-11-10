"""Coordinate/registration checks, not visual acceptance tests."""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from registration_qc_pilot import canvas, fit


def test_nested_pupil_is_not_hidden_by_dark_frame():
    from registration_qc_iris import pupil
    a=np.zeros((512,512),np.uint8)
    cv2.circle(a,(250,260),180,110,-1)
    cv2.circle(a,(245,255),32,20,-1)
    cv2.circle(a,(250,250),5,255,-1)
    x,y,r=pupil(a.astype(float),np.ones_like(a))
    assert np.hypot(x-245,y-255)<4
    assert abs(r-32)<5


def test_fundus_features_exclude_camera_edge():
    from registration_qc_fundus import feature
    a=np.zeros((512,512,3),np.uint8)
    cv2.circle(a,(256,256),220,(100,80,50),-1)
    _,mask=feature(a,np.ones((512,512),np.uint8))
    assert mask[256,256]==1
    assert mask[256,475]==0


def test_full_field_keeps_aspect_and_native_extent(tmp_path):
    p=tmp_path/'test.png'
    Image.fromarray(np.full((100,200,3),128,np.uint8)).save(p)
    a,m,g=canvas(p)
    assert a.shape==(512,512,3)
    assert m.sum()==512*256
    assert g['offset_xy']==[0,128]
    assert np.all(a[128:384]==128)


def test_blank_fit_has_no_identity_fallback():
    a=np.zeros((512,512,3),np.uint8)
    t,points,meta=fit(a,a,np.ones((512,512),np.uint8),np.ones((512,512),np.uint8))
    assert t is None and not points
    assert meta['status']=='no_descriptors'


def test_similarity_direction_and_reflection_equivalence():
    rng=np.random.default_rng(9)
    a=np.zeros((512,512,3),np.uint8)
    for x,y,r in rng.integers([40,40,3],[470,470,12],size=(150,3)):
        cv2.circle(a,(int(x),int(y)),int(r),tuple(int(v) for v in rng.integers(50,256,3)),-1)
    forward=np.float64([[1,0,9],[0,1,-6]])
    b=cv2.warpAffine(a,forward,(512,512))
    m=np.ones((512,512),np.uint8)
    t,_,_=fit(a,b,m,m)
    assert t is not None
    assert np.allclose(t,cv2.invertAffineTransform(forward),atol=.15)
    reflected=b[:,::-1].copy()
    recovered,_,_=fit(a,reflected[:,::-1].copy(),m,m)
    assert np.array_equal(t,recovered)
    # Half-open interval reflection is [W-stop,W-start), not W-1-stop.
    assert np.array_equal(b[:,136:200],b[:,::-1][:,512-200:512-136][:,::-1])
