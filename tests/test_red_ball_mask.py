import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ball_tracker import red_ball_mask


class RedBallMaskTests(unittest.TestCase):
    def test_dark_red_and_antialiased_red_are_kept_but_gray_is_not(self):
        rgb=np.array([[[55,2,2],[180,125,125],[150,150,150],[255,255,255]]],dtype=np.uint8)
        np.testing.assert_array_equal(red_ball_mask(rgb,np.ones((1,4))),[[True,True,False,False]])

    def test_invalid_depth_and_weak_red_tint_are_rejected(self):
        rgb=np.array([[[255,0,0],[255,0,0],[150,140,140],[255,0,0]]],dtype=np.uint8)
        np.testing.assert_array_equal(red_ball_mask(rgb,np.array([[np.inf,.01,1.,4.]])),[[False,False,False,False]])
