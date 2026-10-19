import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from closure_trigger import contact_closure_step,anticipated_closure_fraction,incoming_hand_tilt
import numpy as np


class ClosureTriggerTests(unittest.TestCase):
    def test_short_actual_contact_starts_closure_without_camera_input(self):
        streak,close=contact_closure_step(0,.16,.5);self.assertTrue(close)

    def test_no_contact_does_not_close(self):
        streak,close=contact_closure_step(0,0,.5)
        self.assertEqual(streak,0);self.assertFalse(close)

    def test_reset_warmup_is_not_a_catch(self):
        streak,close=contact_closure_step(4,.5,.1)
        self.assertEqual(streak,0);self.assertFalse(close)

    def test_visual_preparation_is_early_partial_and_requires_alignment(self):
        self.assertAlmostEqual(anticipated_closure_fraction(.06,.02,.02),.1)
        self.assertEqual(anticipated_closure_fraction(.2,.02,.02),0.)
        self.assertEqual(anticipated_closure_fraction(.06,.1,.02),0.)
        self.assertEqual(anticipated_closure_fraction(.06,.02,.15),0.)
        self.assertLessEqual(anticipated_closure_fraction(0,.02,0),.2)

    def test_palm_faces_front_and_side_incoming_velocity(self):
        front=incoming_hand_tilt([-1.,0,-2.]);left=incoming_hand_tilt([0,-1.,-2.]);right=incoming_hand_tilt([0,1.,-2.])
        self.assertGreater(front[1],0);self.assertLess(left[0],0);self.assertGreater(right[0],0)
        self.assertLessEqual(np.linalg.norm(incoming_hand_tilt([-3.,-3.,-1.])),.55+1e-12)
