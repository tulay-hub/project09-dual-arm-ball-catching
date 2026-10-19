"""RGB-D sphere-center fit; inputs are images/calibration, not simulator ball state."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from ballistics import BallisticFilter


def red_ball_mask(rgb,depth):
    """Include dark red and antialiased red edges, reject neutral backgrounds."""
    r,g,b=np.asarray(rgb)[...,:3].astype(np.float32).transpose(2,0,1)
    z=np.asarray(depth).squeeze()
    return ((r>40)&(r-g>25)&(r-b>25)&(r>1.25*g)&(r>1.25*b)
            &np.isfinite(z)&(z>.03)&(z<3.))


class BallTracker:
    def __init__(self, radius=.028, position_std=.0015):
        self.radius=radius
        self.filter=BallisticFilter(position_std=position_std)
        self.last_measurement=None
        self.last_time=None
        self.visible=False

    def reset(self):
        self.filter.reset()
        self.last_measurement=None
        self.last_time=None
        self.visible=False

    def observe(self, rgb, depth, K, camera_position, camera_quaternion_xyzw, acquisition_time):
        z=np.asarray(depth).squeeze()
        v,u=np.nonzero(red_ball_mask(rgb,z))
        center=self.measure_sparse(u,v,z[v,u],K,camera_position,camera_quaternion_xyzw)
        return self.observe_world(center,acquisition_time)

    def measure_sparse(self,u,v,depths,K,camera_position,camera_quaternion_xyzw):
        """Fit exactly the same ordered red pixels after GPU mask compaction."""
        if len(u)<12:return None
        near=np.quantile(depths,.1)
        selected=depths<near+2*self.radius
        v,u,depths=v[selected],u[selected],depths[selected]
        if len(u)<12:return None
        stride=max(1,len(u)//400)
        u,v,depths=u[::stride],v[::stride],depths[::stride]
        K=np.asarray(K)
        points=np.c_[(u+.5-K[0,2])*depths/K[0,0],(v+.5-K[1,2])*depths/K[1,1],depths]
        return self.measure_points(points,near,camera_position,camera_quaternion_xyzw)

    def measure_points(self,points,near,camera_position,camera_quaternion_xyzw):
        surface=np.median(points,axis=0)
        guess=surface+self.radius*surface/np.linalg.norm(surface)
        def residuals(center):return np.linalg.norm(points-center,axis=1)-self.radius
        def jacobian(center):
            delta=center-points
            return delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)
        fit=least_squares(residuals,guess,jac=jacobian,loss='soft_l1',f_scale=.002,max_nfev=30)
        residual=np.median(np.abs(residuals(fit.x)))
        if residual>.008 or fit.x[2]<near:return None
        return Rotation.from_quat(camera_quaternion_xyzw).apply(fit.x)+np.asarray(camera_position)

    def observe_world(self,center,acquisition_time):
        self.visible=False
        if center is None:return None
        if self.filter.observe(acquisition_time,center):
            self.last_measurement=center
            self.last_time=acquisition_time
            self.visible=True
            return center
        return None

    def state(self, control_time):
        return self.filter.predict(control_time)
