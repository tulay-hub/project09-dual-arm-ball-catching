"""Bounded candidate interception search; dynamics and collision gates remain required."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from ballistics import quintic


@dataclass
class Interception:
    side: str
    time: float
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    joints: dict
    joint_velocity: np.ndarray
    peak_joint_speed: float


class InterceptionPlanner:
    def __init__(self, kinematics, palm_rotation, ball_radius=.028, max_joint_speed=3., max_joint_accel=20.):
        self.kin = kinematics
        self.rotation = np.asarray(palm_rotation)
        self.radius = ball_radius
        self.speed = max_joint_speed
        self.accel = max_joint_accel

    def solve(self, estimate, now, joints, side='left', velocities=None):
        names = self.kin.arm_names(side)
        q0 = np.array([joints[n] for n in names])
        v0 = np.array([velocities.get(n,0.) for n in names]) if velocities is not None else np.zeros(7)
        candidates=list(np.linspace(.15,.75,13))
        current=estimate.predict(now)
        if current is None:
            return None
        palm,_=self.kin.palm(side,joints)
        state=current[0]
        roots=np.roots([-.5*9.81,state[5],state[2]-palm[2]-self.radius])
        candidates += [float(root.real)+delta for root in roots if abs(root.imag)<1e-8
                       for delta in [-.01,0,.01] if .15<=float(root.real)+delta<=.75]
        for tau in sorted(set(candidates)):
            prediction = estimate.predict(now + tau)
            if prediction is None:
                return None
            x, P = prediction
            p, velocity = x[:3], x[3:]
            # Commissioning workspace; catching below the ball apex with an upward palm.
            if not (.20 < p[0] < .55 and .72 < p[2] < 1.15 and abs(p[1]) < .40 and velocity[2] < .5):
                continue
            if abs(p[2]-palm[2]-self.radius)>.025:
                continue
            if np.sqrt(np.linalg.eigvalsh(P[:3,:3]).max()) > .025:
                continue
            # Face the incoming velocity so normal impact removes horizontal motion too.
            normal=-velocity/max(np.linalg.norm(velocity),1e-8)
            tilt=np.arctan2(np.linalg.norm(normal[:2]),max(normal[2],1e-8))
            tilt=min(.25,tilt)
            axis=np.cross([0,0,1.],normal)
            axis/=max(np.linalg.norm(axis),1e-8)
            rotation=Rotation.from_rotvec(axis*tilt).as_matrix()@self.rotation
            normal=rotation[:,0]
            palm = p - normal*self.radius
            solution, position_error, orientation_error = self.kin.solve_palm(side, palm, rotation, q0)
            if position_error > .005 or orientation_error > .08:
                continue
            q1 = np.array([solution[n] for n in names])
            J = self.kin.palm_jacobian(side, solution)
            # Match a bounded fraction of incoming velocity during the soft catch.
            v = velocity * .15
            v *= min(1., .35 / max(np.linalg.norm(v), 1e-8))
            qd = J.T @ np.linalg.solve(J@J.T + np.eye(6)*.002, np.r_[v,np.zeros(3)])
            speeds, accelerations = [], []
            for t in np.linspace(0,tau,21):
                q, s, a = quintic(t,tau,q0,v0,q1,qd)
                if any(q[i] < self.kin.limits[n][0] or q[i] > self.kin.limits[n][1] for i,n in enumerate(names)):
                    break
                speeds.append(np.abs(s));accelerations.append(np.abs(a))
            else:
                speed_limit = np.array([min(self.speed,self.kin.limits[n][2]) for n in names])
                if np.all(np.max(speeds,axis=0)<=speed_limit) and np.max(accelerations)<=self.accel:
                    return Interception(side,now+tau,p,velocity,solution,qd,float(np.max(speeds)))
        return None
