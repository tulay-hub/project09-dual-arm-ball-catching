import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from predictive_tracking import position_priority_delta,early_arrival_velocity
from response_control import arrival_joint_velocity
from exposure_tracking import ExposureTrackingBatch
from ballistics import projectile
from ballistics import GRAVITY
from policy_contract import resolve_tracking_profile


class PredictiveTrackingTests(unittest.TestCase):
    def test_tactile_mode_prevents_gravity_bias_on_a_held_ball_and_resets(self):
        batch=ExposureTrackingBatch(2,filtered_state=True);center=np.array([.36,.18,.91])
        for i in range(120):batch.update(i/60,fetch_measurements=lambda ids:[center for _ in ids])
        self.assertGreater(float(np.linalg.norm(batch.position[0]-center)),.05)
        batch.set_contact_mode([True,False])
        for i in range(120,140):batch.update(i/60,fetch_measurements=lambda ids:[center for _ in ids])
        np.testing.assert_allclose(batch.position[0],center,atol=1e-8)
        np.testing.assert_allclose(batch.velocity[0],0.,atol=1e-8)
        self.assertGreater(float(np.linalg.norm(batch.position[1]-center)),.05)
        batch.reset([0]);self.assertFalse(batch.contact_mode[0])
        np.testing.assert_array_equal(batch.trackers[0].filter.gravity,GRAVITY)

    def test_playback_defaults_to_saved_controller_but_explicit_override_wins(self):
        self.assertEqual(resolve_tracking_profile(None,{'tracking_profile':'predictive'}),'predictive')
        self.assertEqual(resolve_tracking_profile(None,{},default='predictive'),'fast')
        self.assertEqual(resolve_tracking_profile('fast',{'tracking_profile':'predictive'}),'fast')
        self.assertEqual(resolve_tracking_profile(None,None,default='predictive'),'predictive')

    def test_other_joints_take_over_when_one_joint_hits_its_bound(self):
        j=torch.zeros(1,6,7);j[0,0,:2]=1;j[0,1,2]=1;j[0,2,3]=1
        q=torch.zeros(1,7);q[0,0]=.99;limits=torch.tensor([[[-1.,1.]]*7])
        error=torch.tensor([[.1,.05,-.03]])
        delta=position_priority_delta(j,error,torch.zeros(1,3),q,limits)
        self.assertLessEqual(float(delta[0,0]),.005001)
        self.assertGreater(float(delta[0,1]),.09)
        torch.testing.assert_close((j[:,:3]@delta[:,:,None]).squeeze(-1),error,atol=2e-5,rtol=0.)
        self.assertTrue(bool(((q+delta)>=limits[:,:,0]).all()))
        self.assertTrue(bool(((q+delta)<=limits[:,:,1]).all()))

    def test_orientation_cannot_take_away_a_reachable_position(self):
        j=torch.zeros(1,6,7);j[0,:3,:3]=torch.eye(3);j[0,3:,:3]=torch.eye(3)
        delta=position_priority_delta(j,torch.tensor([[.05,-.04,.03]]),torch.ones(1,3),torch.zeros(1,7),torch.tensor([[[-2.,2.]]*7]))
        torch.testing.assert_close(delta[:,:3],torch.tensor([[.05,-.04,.03]]),atol=1e-4,rtol=0.)

    def test_early_motion_is_stronger_but_never_drives_outward_at_limit(self):
        delta=torch.ones(2,7)*.18;ttc=torch.ones(2)*.4;q=torch.zeros(2,7)
        limits=torch.tensor([[[-1.,1.]]*7]*2);q[1,0]=.999
        new=early_arrival_velocity(delta,ttc,q,limits)
        old=arrival_joint_velocity(delta,ttc,torch.zeros(2,dtype=torch.bool))
        self.assertGreater(float(new[0,0]),float(old[0,0])*2)
        self.assertLessEqual(float(new[1,0]),0.)
        self.assertLessEqual(float(new.abs().max()),3.)

    def test_filtered_position_and_velocity_share_the_measurement_time(self):
        batch=ExposureTrackingBatch(1,filtered_state=True)
        p=np.array([.8,.1,1.]);v=np.array([-1.,.2,1.8])
        for i,t in enumerate([0.,.017,.034,.051,.068]):
            measured=projectile(p,v,t)+np.array([.001*(-1)**i,0.,0.])
            batch.update(t,fetch_measurements=lambda ids:[measured])
        state=batch.trackers[0].state(.068)[0]
        np.testing.assert_allclose(batch.position[0],state[:3])
        np.testing.assert_allclose(batch.velocity[0],state[3:])
        self.assertGreater(float(np.linalg.norm(batch.position[0]-measured)),1e-5)

if __name__=='__main__':unittest.main()
