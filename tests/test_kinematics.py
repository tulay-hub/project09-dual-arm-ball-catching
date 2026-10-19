"""Check URDF IK and velocity contract against independently perturbed FK."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kinematics import RobotKinematics, ROOT
from ballistics import BallisticFilter,release_velocity,projectile
from interception import InterceptionPlanner


class KinematicsTests(unittest.TestCase):
    def test_ready_palms_and_limits(self):
        cfg=json.loads((ROOT/'config/scene.json').read_text())
        kin=RobotKinematics(base_height=cfg['base_height_m'])
        for side in ['left','right']:
            q=cfg['ready'][side]['joint_positions_rad']
            p,R=kin.palm(side,q)
            np.testing.assert_allclose(p,cfg['ready'][side]['palm_position_world_m'],atol=2e-5)
            np.testing.assert_allclose(R@np.array([1,0,0]),[0,0,1],atol=2e-5)
            self.assertTrue(all(kin.limits[n][0]<v<kin.limits[n][1] for n,v in q.items()))

    def test_jacobian_predicts_combined_joint_motion(self):
        cfg=json.loads((ROOT/'config/scene.json').read_text());kin=RobotKinematics()
        q=cfg['ready']['left']['joint_positions_rad'];p,_=kin.palm('left',q)
        dq=np.array([.2,-.1,.15,.3,-.2,.1,-.1])*1e-4
        changed={n:q[n]+dq[i] for i,n in enumerate(kin.arm_names('left'))}
        moved,_=kin.palm('left',changed)
        np.testing.assert_allclose(moved-p,kin.palm_jacobian('left',q)[:3]@dq,atol=1e-8)

    def test_interception_accepts_feasible_throw_and_rejects_far_throw(self):
        cfg=json.loads((ROOT/'config/scene.json').read_text());kin=RobotKinematics()
        planner=InterceptionPlanner(kin,cfg['hand_rotation_world'])
        q=dict(cfg['arm_ready_joint_positions_rad'])
        seed=np.array([q[n] for n in kin.arm_names('left')])
        rotation=Rotation.from_rotvec([0,.22,0]).as_matrix()@np.array(cfg['hand_rotation_world'])
        prepared,pe,re=kin.solve_palm('left',np.array(cfg['ready']['left']['palm_position_world_m']),rotation,seed)
        self.assertLess(pe,.005);self.assertLess(re,.08);q.update(prepared)
        f=BallisticFilter(position_std=.001)
        initial=np.array([.6,.18,1.]);velocity=release_velocity(initial,[.36,.18,.928],.4)
        for t in [0,.016,.033]:f.observe(t,projectile(initial,velocity,t))
        plan=planner.solve(f,.033,q)
        self.assertIsNotNone(plan)
        self.assertLessEqual(plan.peak_joint_speed,3.)
        np.testing.assert_allclose(plan.ball_position,projectile(initial,velocity,plan.time),atol=1e-8)
        unreachable=BallisticFilter()
        for t in [0,.016,.033]:unreachable.observe(t,[10.,10.,1.-.5*9.81*t*t])
        self.assertIsNone(planner.solve(unreachable,.033,q))

    def test_gravity_compensation_matches_potential_energy_gradient(self):
        cfg=json.loads((ROOT/'config/scene.json').read_text());kin=RobotKinematics()
        q=cfg['arm_ready_joint_positions_rad'];torque=kin.gravity_torques(q)
        def potential(joints):
            T=kin.forward(joints)
            return sum(mass*9.81*(T[name][:3,3]+T[name][:3,:3]@com)[2]
                       for name,(mass,com) in kin.mass_com.items())
        for name in ['Left_Elbow_Pitch_Joint','Right_Shoulder_Pitch_Joint','lh_index_mcp_pitch']:
            plus=dict(q);minus=dict(q);plus[name]=q.get(name,0)+1e-6;minus[name]=q.get(name,0)-1e-6
            finite=(potential(plus)-potential(minus))/2e-6
            self.assertAlmostEqual(torque[name],finite,places=6)


if __name__=='__main__':unittest.main()
