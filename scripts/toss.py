"""Bounded normal-direction palm stroke for a physical toss; does not alter ball state."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


class NormalToss:
    def __init__(self,kinematics,side,joints,rotation_nominal,height=.16,duration=.17,tilt=.43):
        self.kin=kinematics;self.side=side;self.height=height;self.duration=duration
        self.names=kinematics.arm_names(side)
        p0,R0=kinematics.palm(side,joints)
        self.p0=p0
        angle=tilt if side=='left' else -tilt
        R1=Rotation.from_rotvec([angle,0,0]).as_matrix()@np.asarray(rotation_nominal)
        self.normal=R1[:,0]
        interpolate=Slerp([0.,duration],Rotation.from_matrix(np.stack([R0,R1])))
        self.times=np.linspace(0,duration,max(35,int(duration*200)+1))
        seed=np.array([joints[n] for n in self.names])
        poses=[];errors=[]
        for t in self.times:
            u=t/duration;s=10*u**3-15*u**4+6*u**5
            p=p0+self.normal*height*s
            R=interpolate([t]).as_matrix()[0]
            q,pe,re=kinematics.solve_palm(side,p,R,seed)
            if pe>.005 or re>.08:
                raise ValueError(f'toss pose unreachable at {t:.3f}: {pe}, {re}')
            seed=np.array([q[n] for n in self.names]);poses.append(seed.copy());errors.append(pe)
        self.positions=np.array(poses)
        self.velocities=np.gradient(self.positions,self.times,axis=0)
        self.max_speed=float(np.max(np.abs(self.velocities)))
        self.max_position_error=float(max(errors))
        if self.max_speed>8.:
            raise ValueError(f'toss joint speed {self.max_speed:.2f} exceeds 8 rad/s simulation limit')

    def sample(self,t):
        t=np.clip(t,0,self.duration)
        q=np.array([np.interp(t,self.times,self.positions[:,i]) for i in range(7)])
        qd=np.array([np.interp(t,self.times,self.velocities[:,i]) for i in range(7)])
        if t>=self.duration:qd[:]=0
        return dict(zip(self.names,q.tolist())),dict(zip(self.names,qd.tolist()))
