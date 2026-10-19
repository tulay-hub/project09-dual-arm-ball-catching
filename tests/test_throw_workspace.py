import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from throw_workspace import sphere_in_view,predict_landing
from ballistics import release_velocity


class ThrowWorkspaceTests(unittest.TestCase):
    def test_full_ball_must_fit_and_above_camera_is_rejected(self):
        camera=np.array([.36,0,1.9])
        self.assertTrue(sphere_in_view(np.array([[.36,.18,1.2]]),.028,camera,np.eye(3)))
        self.assertFalse(sphere_in_view(np.array([[.36,.4,1.4]]),.028,camera,np.eye(3)))
        self.assertFalse(sphere_in_view(np.array([[.36,0,2.0]]),.028,camera,np.eye(3)))

    def test_vision_ballistic_prediction_recovers_offset_target(self):
        start=np.array([.18,.12,1.2]);goal=np.array([.43,.25,.927])
        velocity=release_velocity(start,goal,.52)
        tau,point=predict_landing(start,velocity,goal[2])
        self.assertAlmostEqual(tau,.52);np.testing.assert_allclose(point,goal,atol=1e-12)

    def test_unreachable_height_is_rejected(self):
        self.assertIsNone(predict_landing([.2,0,1.],[0,0,-1],1.5))
