import cv2
import numpy as np

import nckuh_dmsa_fullframe_relative_geometry_probe_20261102 as study


def test_relative_geometry_is_approximately_scale_invariant():
    frame=np.zeros((100,120,3),np.uint8)
    cv2.ellipse(frame,(60,55),(48,35),0,0,360,(80,80,80),-1)
    mask=np.zeros((100,120),np.uint8)
    cv2.ellipse(mask,(60,55),(25,12),15,0,360,1,-1)
    first,fallback=study.relative_geometry(frame,mask)
    enlarged=cv2.resize(frame,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    enlarged_mask=cv2.resize(mask,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    second,enlarged_fallback=study.relative_geometry(enlarged,enlarged_mask)
    assert not fallback and not enlarged_fallback
    stable=np.arange(len(first))!=13
    np.testing.assert_allclose(first[stable],second[stable],rtol=.08,atol=.02)
    assert abs(first[13]-second[13])<.20
