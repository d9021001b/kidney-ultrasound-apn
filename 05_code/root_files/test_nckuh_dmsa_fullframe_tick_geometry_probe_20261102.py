import cv2
import numpy as np

import nckuh_dmsa_fullframe_tick_geometry_probe_20261102 as study


def test_depth_tick_spacing_recovers_periodic_scale():
    image=np.zeros((240,300,3),np.uint8)
    for y in range(35,215,30):
        cv2.circle(image,(25,y),1,(255,255,255),-1)
    assert study.depth_tick_spacing(image)==30
