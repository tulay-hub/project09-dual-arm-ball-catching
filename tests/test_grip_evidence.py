import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from grip_evidence import opposing_contacts,stable_grasp,retention_contacts,stable_retention


class GripEvidenceTests(unittest.TestCase):
    def evidence(self,vectors=None,points=None,separations=None):
        return opposing_contacts(['lh_thumb_distal','lh_index_distal','lh_middle_distal'],
            np.array([[0,0,1],[0,0,-.6],[0,0,-.6]]) if vectors is None else vectors,
            np.array([[0,0,-.028],[0,0,.028],[.01,0,.026]]) if points is None else points,
            [1,1,1],[0,1,2],np.zeros(3) if separations is None else separations,[0,0,0],.028,'lh_')

    def test_true_opposing_contacts_are_valid(self):
        self.assertTrue(self.evidence()['valid'])

    def test_large_same_direction_forces_do_not_make_a_grasp(self):
        self.assertFalse(self.evidence(vectors=np.array([[0,0,100],[0,0,80],[0,0,80]]))['valid'])

    def test_opposite_forces_with_points_on_same_side_are_rejected(self):
        self.assertFalse(self.evidence(points=np.array([[0,0,.028],[0,0,.028],[.01,0,.026]]))['valid'])

    def test_penetration_is_rejected(self):
        self.assertFalse(self.evidence(separations=np.array([-.01,0,0]))['valid'])

    def test_support_or_unsettled_fingers_are_not_stable_grasps(self):
        self.assertFalse(stable_grasp(True,False,.01,.001,.01,.03))
        self.assertFalse(stable_grasp(True,True,.3,.001,.01,.03))
        self.assertFalse(stable_grasp(True,True,.01,.015,.01,.03))
        self.assertFalse(stable_grasp(True,True,.01,.001,1.,.03))
        self.assertTrue(stable_grasp(True,True,.01,.001,.01,.03))

    def test_palm_support_is_accepted_without_enclosure(self):
        result=retention_contacts(['lh_hand_base_link'],[[0,0,.5]],[[0,0,-.028]],
            [1],[0],[0],[0,0,0],.028,'lh_')
        self.assertTrue(result['valid']);self.assertTrue(stable_retention(True,.01,.001,.03))

    def test_retention_rejects_sliding_and_fake_contact(self):
        self.assertFalse(stable_retention(True,.3,.001,.03))
        self.assertFalse(stable_retention(True,.01,.02,.03))
        result=retention_contacts(['lh_hand_base_link'],[[0,0,.5]],[[0,0,-.028]],
            [0],[0],[0],[0,0,0],.028,'lh_')
        self.assertFalse(result['valid'])
