"""Time-stamped projectile estimation and physically interpretable toss references."""
import numpy as np

GRAVITY = np.array([0., 0., -9.81])


def release_velocity(start, target, flight_time, gravity=GRAVITY):
    """Required release velocity [m/s] for flight time [s] and positions [m]."""
    if not np.isfinite(flight_time) or flight_time <= 0:
        raise ValueError('flight_time must be positive and finite')
    return (np.asarray(target)-np.asarray(start)-.5*np.asarray(gravity)*flight_time**2)/flight_time


def projectile(position, velocity, dt, gravity=GRAVITY):
    return np.asarray(position)+np.asarray(velocity)*dt+.5*np.asarray(gravity)*dt**2


class BallisticFilter:
    """Position/velocity KF; measurements must carry acquisition time [s]."""
    def __init__(self, position_std=.003, acceleration_std=.15,gravity=GRAVITY):
        self.position_std = position_std
        self.acceleration_std = acceleration_std
        self.gravity=np.asarray(gravity,dtype=float).copy()
        self.reset()

    def reset(self):
        self.x = None
        self.P = None
        self.time = None
        self.first = None
        self.ready = False

    def observe(self, timestamp, position):
        p = np.asarray(position,dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(timestamp):
            return False
        if self.time is not None and timestamp <= self.time:
            return False
        if self.x is None:
            self.first = (timestamp,p.copy())
            self.x = np.r_[p,np.zeros(3)]
            self.P = np.diag([self.position_std**2]*3+[4.]*3)
            self.time = timestamp
            return True
        dt = timestamp-self.time
        if not self.ready:
            # Finite-difference position is average velocity; advance it to the second exposure.
            self.x = np.r_[p,(p-self.first[1])/dt+.5*self.gravity*dt]
            self.P = np.diag([self.position_std**2]*3+[2*self.position_std**2/dt**2]*3)
            self.time = timestamp
            self.ready = True
            return True
        F = np.eye(6); F[:3,3:] = np.eye(3)*dt
        G = np.vstack([np.eye(3)*.5*dt**2,np.eye(3)*dt])
        self.x = F@self.x + G@self.gravity
        self.P = F@self.P@F.T + G@G.T*self.acceleration_std**2
        H = np.c_[np.eye(3),np.zeros((3,3))]
        R = np.eye(3)*self.position_std**2
        K = np.linalg.solve(H@self.P@H.T+R,H@self.P).T
        self.x += K@(p-H@self.x)
        A=np.eye(6)-K@H
        self.P=A@self.P@A.T+K@R@K.T
        self.time=timestamp
        return True

    def predict(self, timestamp):
        if not self.ready:
            return None
        dt=timestamp-self.time
        if dt<0:
            raise ValueError('cannot predict before latest measurement')
        F=np.eye(6);F[:3,3:]=np.eye(3)*dt
        G=np.vstack([np.eye(3)*.5*dt**2,np.eye(3)*dt])
        x=F@self.x+G@self.gravity
        P=F@self.P@F.T+G@G.T*self.acceleration_std**2
        return x,P


def quintic(t, duration, q0, v0, q1, v1):
    """Position, velocity, acceleration for a trajectory with zero endpoint accelerations."""
    if duration<=0:
        raise ValueError('duration must be positive')
    T=duration
    q0,v0,q1,v1=map(np.asarray,[q0,v0,q1,v1])
    matrix=np.array([[T**3,T**4,T**5],[3*T**2,4*T**3,5*T**4],[6*T,12*T**2,20*T**3]])
    coeff=np.linalg.solve(matrix,np.stack([q1-q0-v0*T,v1-v0,np.zeros_like(q0)]))
    a,b,c=coeff
    t=np.clip(t,0,T)
    return (q0+v0*t+a*t**3+b*t**4+c*t**5,
            v0+3*a*t**2+4*b*t**3+5*c*t**4,
            6*a*t+12*b*t**2+20*c*t**3)
