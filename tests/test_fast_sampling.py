"""Numerical regression of batched reads and sparse rendered-image processing."""
import sys,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.optimize._numdiff import approx_derivative
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fast_sampling import contact_masks,reduce_contact_forces,sparse_red_pixels
from ball_tracker import BallTracker,red_ball_mask
from exposure_tracking import ExposureTrackingBatch


class FastSamplingTests(unittest.TestCase):
    def test_gpu_mask_matches_numpy_with_invalid_depth_and_edges(self):
        rng=np.random.default_rng(73)
        rgb=rng.integers(0,256,(4,48,64,3),dtype=np.uint8)
        depth=rng.uniform(0,4,(4,48,64)).astype('float32')
        depth[0,0,:4]=[np.nan,np.inf,.03,3.]
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        expected=[]
        for e in range(4):
            v,u=np.nonzero(red_ball_mask(rgb[e],depth[e]))
            expected.extend(np.c_[np.full(len(u),e),u,v,depth[e,v,u]])
        for device in devices:
            actual=sparse_red_pixels(torch.tensor(rgb,device=device),torch.tensor(depth,device=device)).cpu().numpy()
            np.testing.assert_array_equal(actual,np.array(expected,dtype='float32'))

    def test_force_reduction_uses_sensor_names_instead_of_sensor_index(self):
        names=['rh_index_distal','lh_hand_base_link','lh_thumb_ip','rh_hand_base_link','lh_middle_dip']
        masks=contact_masks([names,names[::-1]],'cpu')
        values=torch.tensor([[[1.,2.,3.],[4.,0.,0.],[0.,5.,0.],[0.,0.,6.],[3.,4.,0.]]]*2)
        total,features,peak=reduce_contact_forces(values,*masks)
        for e,order in enumerate([names,names[::-1]]):
            norms=np.linalg.norm(values[e].numpy(),axis=1)
            expected=[sum(norms[i] for i,n in enumerate(order) if n.startswith(p)) for p in ['lh_','rh_']]
            np.testing.assert_allclose(total[e],expected)
            for j,(p,f) in enumerate((p,f) for p in ['lh_','rh_'] for f in ['thumb','index','middle','ring','pinky']):
                self.assertAlmostEqual(features[e,j].item(),max([norms[i] for i,n in enumerate(order) if n.startswith(p+f)]+[0.]),places=5)
        self.assertEqual(peak.item(),6.)

    def test_sparse_fit_preserves_point_threshold(self):
        tracker=BallTracker(radius=.02)
        for count in [0,11]:
            a=np.arange(count,dtype=float)
            self.assertIsNone(tracker.measure_sparse(a,a,np.ones(count),np.eye(3),np.zeros(3),[0,0,0,1]))

    def test_analytic_derivative_matches_numerical_derivative(self):
        rng=np.random.default_rng(29);points=rng.normal(size=(40,3));center=np.array([.1,.2,1.])
        delta=center-points;jac=delta/np.linalg.norm(delta,axis=1,keepdims=True)
        numeric=approx_derivative(lambda x:np.linalg.norm(points-x,axis=1)-.02,center)
        np.testing.assert_allclose(jac,numeric,atol=1e-9)

    def test_compacted_measurement_path_preserves_duplicate_and_partial_reset(self):
        batch=ExposureTrackingBatch(2,radius=.02);calls=[]
        centers=np.array([[.1,.2,.8],[-.1,.2,.8]])
        def fetch(ids):
            calls.append(ids.tolist());return centers[ids]
        batch.update(0.,fetch_measurements=fetch)
        before=batch.trackers[1].filter.P.copy()
        batch.update(0.,control_time=.02,fetch_measurements=fetch)
        self.assertTrue(batch.visible.all());self.assertEqual(calls,[[0,1]])
        batch.reset([0]);centers[0]=[.2,.1,.9]
        batch.update(0.,control_time=.02,fetch_measurements=fetch)
        self.assertEqual(calls,[[0,1],[0]])
        np.testing.assert_array_equal(batch.trackers[1].filter.P,before)
        np.testing.assert_allclose(batch.position,centers)

    def test_compacted_occlusion_does_not_overwrite_last_measurement(self):
        batch=ExposureTrackingBatch(2,radius=.02)
        batch.update(0.,fetch_measurements=lambda ids:[np.array([.1,.2,.8]) for _ in ids])
        before=batch.position.copy()
        batch.update(.02,fetch_measurements=lambda ids:[None,np.array([.1,.2,.81])])
        self.assertFalse(batch.visible[0]);self.assertTrue(batch.visible[1])
        np.testing.assert_array_equal(batch.position[0],before[0])
        batch.update(.02,control_time=.2,fetch_measurements=lambda ids:self.fail('duplicate must not fetch'))
        self.assertFalse(batch.visible.any())

if __name__=='__main__':unittest.main()
