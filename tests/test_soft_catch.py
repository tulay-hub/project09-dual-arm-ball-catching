import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from soft_catch import reception_target,reception_joint_velocity,SOFT_SETTINGS
from policy_contract import resolve_grip_profile


class SoftCatchTests(unittest.TestCase):
    def test_follow_is_bounded_in_world_space_and_in_each_step(self):
        anchor=np.array([20.3,-11.2,.91]);previous=anchor.copy();ball=anchor+np.array([.2,.1,-.1])
        for i in range(12):
            target,velocity=reception_target(anchor,previous,ball,[-1.,.1,-2.],0.,i/60,np.zeros(3),1/60)
            self.assertLessEqual(float(np.linalg.norm(target-anchor)),.03500001)
            self.assertLessEqual(float(np.linalg.norm(target-previous)),.35/60+1e-9)
            self.assertLessEqual(float(np.linalg.norm(velocity)),.35000001)
            previous=target

    def test_stale_camera_or_finished_window_keeps_final_position(self):
        anchor=np.zeros(3);previous=np.array([.01,0,-.01])
        for age,elapsed in [(.061,.05),(.01,.181),(.01,-.01)]:
            target,velocity=reception_target(anchor,previous,np.ones(3),np.ones(3),age,elapsed,np.zeros(3),1/60)
            np.testing.assert_array_equal(target,previous);np.testing.assert_array_equal(velocity,0.)

    def test_feedforward_stays_inside_arm_drive_and_joint_braking_bounds(self):
        j=torch.zeros(2,3,7);j[:,:,:3]=torch.eye(3);q=torch.zeros(2,7);q[1,0]=.999
        limits=torch.tensor([[[-1.,1.]]*7]*2)
        velocity=reception_joint_velocity(torch.full((2,7),2.8),j,torch.ones(2,3)*3.,q,limits)
        self.assertLessEqual(float(velocity.abs().max()),3.)
        self.assertLessEqual(float(velocity[1,0]),0.)
        self.assertTrue(bool(torch.isfinite(velocity).all()))

    def test_saved_grip_profile_is_used_for_playback(self):
        self.assertEqual(resolve_grip_profile(None,{'grip_profile':'soft_catch'}),'soft_catch')
        self.assertEqual(resolve_grip_profile(None,{},'soft_catch'),'fast')
        self.assertEqual(resolve_grip_profile('fast',{'grip_profile':'soft_catch'}),'fast')
        self.assertLessEqual(SOFT_SETTINGS['closure_duration_s'],.25)

