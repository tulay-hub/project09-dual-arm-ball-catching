import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from response_control import arrival_joint_velocity
from closure_trigger import anticipated_closure_fraction


class ResponseControlTests(unittest.TestCase):
    def test_extended_preclose_respects_visual_and_alignment_gates(self):
        options=dict(window_s=.2,alignment_m=.04,maximum_fraction=.45)
        self.assertAlmostEqual(anticipated_closure_fraction(.1,.035,.05,**options),.225)
        self.assertEqual(anticipated_closure_fraction(.1,.041,.05,**options),0.)
        self.assertEqual(anticipated_closure_fraction(.1,.03,.11,**options),0.)
        self.assertEqual(anticipated_closure_fraction(.201,.03,.05,**options),0.)
        self.assertAlmostEqual(anticipated_closure_fraction(0.,.03,.05,**options),.45)

    def test_later_arrival_reduces_required_velocity(self):
        delta=torch.tensor([[.12,-.12],[.12,-.12]])
        v=arrival_joint_velocity(delta,torch.tensor([.12,.30]),torch.tensor([False,False]))
        self.assertGreater(v[0,0],v[1,0]);self.assertAlmostEqual(v[0,0].item(),2.,places=5)

    def test_zero_arrival_and_large_error_respect_drive_limit(self):
        v=arrival_joint_velocity(torch.tensor([[5.,-5.]]),torch.tensor([0.]),torch.tensor([False]))
        torch.testing.assert_close(v,torch.tensor([[3.,-3.]]));self.assertTrue(torch.isfinite(v).all())

    def test_missing_prediction_and_captured_state_use_finite_horizon(self):
        v=arrival_joint_velocity(torch.tensor([[.15],[.15]]),torch.tensor([-1.,.01]),torch.tensor([False,True]))
        torch.testing.assert_close(v,torch.ones_like(v))

if __name__=='__main__':unittest.main()
