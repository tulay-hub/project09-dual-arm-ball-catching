import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from arm_tracking import joint_body_jacobian,joint_gravity_forces


class ArmTrackingTests(unittest.TestCase):
    def test_floating_base_columns_and_body_rows(self):
        jac=np.arange(1*4*6*9).reshape(1,4,6,9)
        np.testing.assert_array_equal(joint_body_jacobian(jac,2,4,3,[0,2]),jac[:,2][:,:,[6,8]])
        force=np.arange(9).reshape(1,9)
        np.testing.assert_array_equal(joint_gravity_forces(force,3),[[6,7,8]])

    def test_fixed_base_root_is_omitted(self):
        jac=np.arange(1*3*6*3).reshape(1,3,6,3)
        np.testing.assert_array_equal(joint_body_jacobian(jac,2,4,3,[0,2]),jac[:,1][:,:,[0,2]])

    def test_unexpected_shape_is_rejected(self):
        with self.assertRaises(ValueError):joint_body_jacobian(np.zeros((1,4,6,5)),2,4,3,[0,2])
