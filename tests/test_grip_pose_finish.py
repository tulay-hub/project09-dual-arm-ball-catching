import sys,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from grip_pose_finish import contact_rotation_target,settling_joint_velocity


class GripPoseFinishTests(unittest.TestCase):
    def test_first_touch_keeps_the_valid_planned_rotation_instead_of_freezing(self):
        actual=Rotation.from_rotvec([0.,.1,0.]).as_matrix();planned=np.array([0.,.3,0.]);reference=np.eye(3)
        result=contact_rotation_target(actual,reference,planned,.03,.02)
        np.testing.assert_allclose(result,planned,atol=1e-8)
        for t,age in [(-1.,.02),(.03,.11)]:
            np.testing.assert_allclose(contact_rotation_target(actual,reference,planned,t,age),[0.,.1,0.],atol=1e-8)

    def test_large_rotation_correction_is_bounded(self):
        result=contact_rotation_target(np.eye(3),np.eye(3),[0.,1.,0.],.02,0.)
        self.assertAlmostEqual(float(np.linalg.norm(result)),.35)

    def test_contact_settling_is_temporary_and_respects_drive_and_joint_limits(self):
        base=torch.ones(4,7)*.8;delta=torch.ones(4,7)*.12;q=torch.zeros(4,7);q[3,0]=.999
        limits=torch.tensor([[[-1.,1.]]*7]*4)
        result=settling_joint_velocity(base,delta,torch.tensor([0.,.13,-1.,0.]),torch.tensor([True,True,False,True]),q,limits)
        self.assertGreater(float(result[0,0]),float(base[0,0]))
        torch.testing.assert_close(result[1:3],base[1:3])
        self.assertLessEqual(float(result[3,0]),0.)
        self.assertLessEqual(float(result.abs().max()),3.)

