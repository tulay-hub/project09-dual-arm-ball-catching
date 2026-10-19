"""Original STL finger postures must fit the sphere without adding rubber pads."""
import sys,json,unittest
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kinematics import RobotKinematics,ROOT
from grasp import SphereGraspPlanner


class GraspGeometryTests(unittest.TestCase):
    def setUp(self):
        cfg=json.loads((ROOT/'config/scene.json').read_text())
        self.kin=RobotKinematics();self.planner=SphereGraspPlanner(self.kin,cfg['hand_mesh_bounds_local_m'])

    def test_both_hands_use_original_mesh_contacts_and_joint_limits(self):
        for side in ['left','right']:
            grasp=self.planner.solve(side,[.048,0,.078])
            self.assertTrue(grasp['geometry_feasible'])
            self.assertGreaterEqual(grasp['usable_contacts'],4)
            self.assertFalse(hasattr(self.planner,'boxes'))
            for contact in grasp['contacts'].values():self.assertGreater(contact['gap_m'],-.004)
            for name,q in grasp['targets_rad'].items():
                self.assertGreaterEqual(q,self.kin.limits[name][0]);self.assertLessEqual(q,self.kin.limits[name][1])

    def test_sphere_beyond_fingertips_is_not_feasible(self):
        grasp=self.planner.solve('left',[.048,0,.16])
        self.assertTrue(grasp is None or not grasp['geometry_feasible'])


if __name__=='__main__':unittest.main()
