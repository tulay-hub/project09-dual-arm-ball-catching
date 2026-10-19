"""Long-time regression: camera float32 accumulation must not age a new image."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from exposure_tracking import ExposureTrackingBatch,SynchronousSimulationClock
from test_exposure_tracking import sphere_frame

class SimulationClockTests(unittest.TestCase):
    def setUp(self):
        self.clock=SynchronousSimulationClock(1,600)
        self.batch=ExposureTrackingBatch(1)
        self.calls=0
    def fetch(self,ids):
        self.calls+=1;rgb,depth,K=sphere_frame([.02,-.01,.7])
        return rgb[None],depth[None],K[None],np.zeros((1,3)),np.array([[0,0,0,1.]])

    def test_new_capture_after_1000_seconds_does_not_use_drifting_sensor_clock(self):
        stamp=np.float32(0);dt=np.float32(1/600)
        for _ in range(600_000):stamp=np.float32(stamp+dt)
        self.assertGreater(1000.-float(stamp),.15)
        self.clock.record_capture(600_000)
        self.batch.update(self.clock.timestamps,self.fetch,self.clock.control_time(600_000))
        self.assertTrue(self.batch.visible[0]);self.assertEqual(self.batch.age[0],0.)
        # A following real exposure remains distinct, with the correct 60Hz interval.
        self.clock.record_capture(600_010)
        self.batch.update(self.clock.timestamps,self.fetch,self.clock.control_time(600_010))
        self.assertTrue(self.batch.visible[0]);self.assertEqual(self.batch.samples,2)
        self.assertAlmostEqual(self.batch.trackers[0].last_time-1000.,1/60)

    def test_stepping_without_capture_still_expires_last_image(self):
        self.clock.record_capture(600_000)
        self.batch.update(self.clock.timestamps,self.fetch,self.clock.control_time(600_000))
        self.batch.update(self.clock.timestamps,self.fetch,self.clock.control_time(600_120))
        self.assertFalse(self.batch.visible[0]);self.assertAlmostEqual(self.batch.age[0],.2)
        self.assertEqual(self.calls,1);self.assertEqual(self.batch.samples,1)

    def test_partial_captures_and_duplicate_render_keep_timestamp_identity(self):
        clock=SynchronousSimulationClock(2,600);clock.record_capture(600_000)
        original=clock.timestamps.copy();clock.record_capture(600_000)
        np.testing.assert_array_equal(clock.timestamps,original)
        clock.record_capture(600_010,ids=[0])
        np.testing.assert_allclose(clock.timestamps,[1000.+1/60,1000.])

    def test_unrecorded_or_backward_capture_is_rejected(self):
        with self.assertRaises(RuntimeError):_ = self.clock.timestamps
        self.clock.record_capture(20)
        with self.assertRaises(ValueError):self.clock.record_capture(19)

if __name__=='__main__':unittest.main()
