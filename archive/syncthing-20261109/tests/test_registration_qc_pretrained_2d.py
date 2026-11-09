import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from registration_qc_pretrained_2d import iris_preprocess,iris_postprocess


def test_iris_input_uses_documented_width_height_and_channel_normalization():
    gray=np.full((100,200),255,np.uint8)
    tensor=iris_preprocess(gray)
    assert tensor.shape==(1,3,480,640)
    assert tensor.dtype==np.float32
    assert np.allclose(tensor[0,:,200,300],[(1-.485)/.229,(1-.456)/.224,(1-.406)/.225])


def test_iris_probability_resize_preserves_classes_and_nearest_values():
    pred=np.zeros((1,4,2,3),np.float32)
    pred[0,1,0,0]=.8
    pred[0,3,1,2]=.6
    output=iris_postprocess(pred,(4,6))
    assert output.shape==(4,6,4)
    assert np.allclose(output[:2,:2,1],.8)
    assert np.allclose(output[-2:,-2:,3],.6)
    assert not output[...,2].any()
