"""Independent physical examples for projectile, exposure timestamps and RGB-D sphere fitting."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ballistics import BallisticFilter,release_velocity,projectile,quintic
from ball_tracker import BallTracker


class BallisticsTests(unittest.TestCase):
    def test_release_reaches_target_and_returns_at_equal_height(self):
        p=np.array([.36,.18,.93]);target=np.array([.36,-.18,.93])
        v=release_velocity(p,target,.4)
        np.testing.assert_allclose(v,[0.,-.9,1.962],atol=1e-12)
        np.testing.assert_allclose(projectile(p,v,.4),target,atol=1e-12)
        self.assertAlmostEqual(projectile(p,v,.2)[2],1.1262)
        with self.assertRaises(ValueError):release_velocity(p,target,0)

    def test_irregular_exposures_and_occlusion_predict_current_state(self):
        f=BallisticFilter(position_std=.001)
        p=np.array([.6,.1,1.4]);v=np.array([-.5,-.3,1.2])
        for t in [0.,.014,.032,.049,.07,.101]:
            self.assertTrue(f.observe(t,projectile(p,v,t)))
        x,P=f.predict(.18)
        np.testing.assert_allclose(x[:3],projectile(p,v,.18),atol=1e-8)
        np.testing.assert_allclose(x[3:],v+[0,0,-9.81*.18],atol=1e-8)
        self.assertGreater(np.trace(P[:3,:3]),np.trace(f.P[:3,:3]))
        self.assertFalse(f.observe(.07,p))
        self.assertFalse(f.observe(.20,[np.nan,0,0]))
        f.reset();self.assertIsNone(f.predict(.3))

    def test_quintic_endpoint_velocity_is_preserved(self):
        q0=np.array([.2,-.1]);v0=np.array([.3,0]);q1=np.array([.7,.2]);v1=np.array([1.,-.2])
        for t,q,v in [(0,q0,v0),(.5,q1,v1)]:
            actual,velocity,acc=quintic(t,.5,q0,v0,q1,v1)
            np.testing.assert_allclose(actual,q,atol=1e-12)
            np.testing.assert_allclose(velocity,v,atol=1e-12)
            np.testing.assert_allclose(acc,0,atol=1e-11)

    def test_sphere_fit_recovers_center_from_visible_surface(self):
        # Ray/sphere intersections give an independent depth image of a known sphere.
        n=128;K=np.array([[180,0,64],[0,180,64],[0,0,1.]])
        v,u=np.mgrid[:n,:n]
        rays=np.stack([(u+.5-64)/180,(v+.5-64)/180,np.ones_like(u)],axis=-1)
        center=np.array([.02,-.01,.7]);radius=.028
        a=np.sum(rays*rays,axis=-1);b=-2*np.sum(rays*center,axis=-1)
        disc=b*b-4*a*(center@center-radius**2)
        visible=disc>=0
        depth=np.full((n,n),np.inf)
        depth[visible]=(-b[visible]-np.sqrt(disc[visible]))/(2*a[visible])
        rgb=np.zeros((n,n,3),dtype=np.uint8);rgb[visible]=[240,10,10]
        tracker=BallTracker(radius)
        measured=tracker.observe(rgb,depth,K,[0,0,0],[0,0,0,1],0.)
        np.testing.assert_allclose(measured,center,atol=1e-5)


if __name__=='__main__':unittest.main()
