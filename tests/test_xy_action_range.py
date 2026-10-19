import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from policy_contract import filter_residual_actions,XY_ACTION_INDEX,XY_RESIDUAL_SCALE_M


class XYActionRangeTests(unittest.TestCase):
    def test_both_hands_can_exceed_the_original_20mm_range(self):
        raw=torch.full((3,14),4.);raw[:,XY_ACTION_INDEX]=torch.tensor([3.,-2.,-3.,2.])
        previous=torch.zeros_like(raw)
        for _ in range(40):requested,previous=filter_residual_actions(raw,previous)
        torch.testing.assert_close(previous[:,XY_ACTION_INDEX]*XY_RESIDUAL_SCALE_M,
            torch.tensor([[.06,-.04,-.06,.04]]).expand(3,4))
        other=[i for i in range(14) if i not in XY_ACTION_INDEX]
        torch.testing.assert_close(previous[:,other],torch.ones(3,len(other)))
        torch.testing.assert_close(requested[:,XY_ACTION_INDEX],raw[:,XY_ACTION_INDEX])

    def test_200mm_cap_and_faster_predictive_slew(self):
        raw=torch.zeros(1,14);raw[:,XY_ACTION_INDEX]=torch.tensor([100.,-100.,200.,-200.])
        previous=torch.zeros_like(raw)
        requested,first=filter_residual_actions(raw,previous,xy_slew_per_step=.5)
        torch.testing.assert_close(first[:,XY_ACTION_INDEX],torch.tensor([[.5,-.5,.5,-.5]]))
        for _ in range(25):requested,previous=filter_residual_actions(raw,previous,xy_slew_per_step=.5)
        torch.testing.assert_close(requested[:,XY_ACTION_INDEX],torch.tensor([[10.,-10.,10.,-10.]]))
        torch.testing.assert_close(previous[:,XY_ACTION_INDEX]*XY_RESIDUAL_SCALE_M,torch.tensor([[.2,-.2,.2,-.2]]))

    def test_inside_original_range_and_legacy_actions_are_unchanged(self):
        torch.manual_seed(123)
        for size in [8,14]:
            raw=torch.randn(100,size)
            if size==14:raw[:,XY_ACTION_INDEX]=raw[:,XY_ACTION_INDEX].clamp(-1,1)
            previous=torch.rand_like(raw)*2-1
            requested,applied=filter_residual_actions(raw,previous)
            torch.testing.assert_close(requested,raw.clamp(-1,1))
            torch.testing.assert_close(applied,previous+(raw.clamp(-1,1)-previous).clamp(-.1,.1))

    def test_nonfinite_inputs_are_rejected(self):
        for invalid in [float('nan'),float('inf'),-float('inf')]:
            raw=torch.zeros(1,14);raw[0,4]=invalid
            with self.assertRaises(RuntimeError):filter_residual_actions(raw,torch.zeros_like(raw))

if __name__=='__main__':unittest.main()
