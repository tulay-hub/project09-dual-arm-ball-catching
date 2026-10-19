"""Regression: reset one env must not hide another env's unchanged valid exposure."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from exposure_tracking import ExposureTrackingBatch


def sphere_frame(center):
    n=128;K=np.array([[180.,0,64],[0,180.,64],[0,0,1.]])
    v,u=np.mgrid[:n,:n];rays=np.stack([(u+.5-64)/180,(v+.5-64)/180,np.ones_like(u)],-1)
    center=np.asarray(center);a=(rays*rays).sum(-1);b=-2*(rays*center).sum(-1)
    disc=b*b-4*a*(center@center-.028**2);inside=disc>=0
    depth=np.full((n,n),np.inf);depth[inside]=(-b[inside]-np.sqrt(disc[inside]))/(2*a[inside])
    rgb=np.zeros((n,n,3),dtype=np.uint8);rgb[inside]=[240,10,10]
    return rgb,depth,K


class ExposureTrackingTests(unittest.TestCase):
    def setUp(self):
        self.batch=ExposureTrackingBatch(2)
        self.centers=np.array([[.02,-.01,.70],[-.015,.008,.74]])
        self.hidden=set();self.calls=[]

    def fetch(self,ids):
        self.calls.append(ids.tolist());frames=[sphere_frame(self.centers[e]) for e in ids]
        rgb=np.stack([f[0] for f in frames])
        for row,e in enumerate(ids):
            if e in self.hidden:rgb[row]=0
        return rgb,np.stack([f[1] for f in frames]),np.stack([f[2] for f in frames]),np.zeros((len(ids),3)),np.tile([0,0,0,1.],(len(ids),1))

    def test_duplicate_render_preserves_visibility_position_and_age(self):
        self.batch.update(0.,self.fetch)
        before=self.batch.position.copy()
        ids=self.batch.update(0.,self.fetch)
        self.assertEqual(len(ids),0);self.assertEqual(self.calls,[[0,1]])
        self.assertTrue(self.batch.visible.all());np.testing.assert_allclose(self.batch.age,0)
        np.testing.assert_allclose(self.batch.position,before)
        self.assertEqual(self.batch.samples,2);self.assertEqual(self.batch.accepted,2)
        np.testing.assert_allclose(before,self.centers,atol=1e-5)

    def test_partial_reset_reprocesses_only_that_environment(self):
        self.batch.update(1.,self.fetch);before=self.batch.trackers[1].filter.P.copy()
        self.batch.reset([0]);self.centers[0]=[-.01,.01,.62]
        ids=self.batch.update(1.,self.fetch)
        np.testing.assert_array_equal(ids,[0]);self.assertEqual(self.calls,[[0,1],[0]])
        self.assertTrue(self.batch.visible.all());self.assertEqual(self.batch.samples,3)
        np.testing.assert_allclose(self.batch.position,self.centers,atol=1e-5)
        np.testing.assert_array_equal(self.batch.trackers[1].filter.P,before)

    def test_new_occluded_exposure_invalidates_once_and_retains_last_position(self):
        self.batch.update(0.,self.fetch);before=self.batch.position.copy();self.hidden.add(0)
        self.batch.update([.02,0],self.fetch,control_time=.02)
        self.assertFalse(self.batch.visible[0]);self.assertTrue(self.batch.visible[1])
        np.testing.assert_allclose(self.batch.age,[.02,.02]);np.testing.assert_allclose(self.batch.position,before)
        self.batch.update([.02,0],self.fetch,control_time=.02)
        self.assertEqual(self.batch.samples,3);self.assertEqual(self.batch.accepted,2)
        np.testing.assert_allclose(self.batch.age,[.02,.02])

    def test_frozen_camera_expires_without_reprocessing_duplicate(self):
        self.batch.update(0.,self.fetch)
        self.batch.update(0.,self.fetch,control_time=.20)
        self.assertFalse(self.batch.visible.any());np.testing.assert_allclose(self.batch.age,.20)
        self.assertEqual(self.calls,[[0,1]]);self.assertEqual(self.batch.samples,2)

    def test_independent_exposure_times_and_identical_pixels_are_supported(self):
        self.batch.update(0.,self.fetch)
        ids=self.batch.update([.02,0],self.fetch,control_time=.02)
        np.testing.assert_array_equal(ids,[0]);self.assertEqual(self.batch.samples,3)
        self.assertTrue(self.batch.visible.all());np.testing.assert_allclose(self.batch.age,[0,.02])

    def test_backward_exposure_does_not_clear_current_valid_state(self):
        self.batch.update(1.,self.fetch)
        self.batch.update(.9,self.fetch,control_time=1.01)
        self.assertTrue(self.batch.visible.all());self.assertEqual(self.batch.samples,2)
        np.testing.assert_allclose(self.batch.age,.01)

    def test_nonfinite_timestamp_is_rejected_before_fetch_or_state_change(self):
        self.batch.update(0.,self.fetch)
        with self.assertRaises(ValueError):self.batch.update([np.nan,0],self.fetch)
        self.assertEqual(self.calls,[[0,1]]);self.assertTrue(self.batch.visible.all())

if __name__=='__main__':unittest.main()
