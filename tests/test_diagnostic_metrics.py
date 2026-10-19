import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnostic_metrics import EpisodeMetrics,wilson

class DiagnosticMetricsTests(unittest.TestCase):
    def frame(self,grip,hold,done=False):
        a=np.array([grip]);zero=np.zeros((1,8));components=np.array([[0.,0.,2.*grip,15.*(done and grip),0.,0.]])
        return dict(grip=a,thumb=a,fingers=np.array([True]),opposition=np.array([.2]),distance=np.array([.04]),speed=np.array([.1]),hold=np.array([hold]),success=np.array([done and grip]),failed=np.array([False]),timeout=np.array([done and not grip]),done=np.array([done]),side=np.array([0]),visible=np.array([True]),capture_time=np.array([.4]),grasp_start=np.array([.8]),reward=components.sum(-1),reward_components=components,actions=zero,previous_actions=zero,requested_actions=zero)
    def test_interrupted_hold_is_counted_before_reset(self):
        c=EpisodeMetrics(1,'mean',0);c.consume(self.frame(True,12));c.consume(self.frame(False,0));c.consume(self.frame(True,8,True));r=c.records[0]
        self.assertEqual(r['max_hold_steps'],12);self.assertEqual(r['hold_interruptions'],1);self.assertEqual(r['break_criteria_failures'],[1,0,0,0,0]);self.assertEqual(r['return'],19)
    def test_auto_reset_episode_is_excluded(self):
        c=EpisodeMetrics(1,'mean',0);c.consume(self.frame(True,30,True));c.consume(self.frame(False,0,True));self.assertEqual(len(c.records),1);self.assertEqual(c.records[0]['outcome'],'success')
    def test_reward_components_must_match_and_confidence_handles_edges(self):
        c=EpisodeMetrics(1,'mean',0);d=self.frame(True,1);d['reward']+=1
        with self.assertRaises(AssertionError):c.consume(d)
        self.assertEqual(wilson(0,0),[0.,1.]);lo,hi=wilson(0,100);self.assertEqual(lo,0.);self.assertGreater(hi,0.)

if __name__=='__main__':unittest.main()
