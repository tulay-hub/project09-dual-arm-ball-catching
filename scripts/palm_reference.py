"""Cached exact-IK palm tilt grid with fixed position and gravity feedforward."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from kinematics import ROOT

class PalmReference:
    def __init__(self,kin,initial,hand_rotation):
        self.x=np.linspace(-.18,.18,13);self.y=np.linspace(-.24,.18,15)
        signature=hashlib.sha256(kin.urdf.read_bytes()+Path(__file__).read_bytes()+
            (ROOT/'scripts/kinematics.py').read_bytes()+json.dumps(initial,sort_keys=True).encode()).hexdigest()[:16]
        path=ROOT/'assets/generated/training'/('palm_grid_'+signature+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            d=np.load(path);self.q=d['q'];self.effort=d['effort'];self.max_error=float(d['max_error']);self.validation=json.loads(str(d['validation']))
        else:
            self.q=np.zeros((2,len(self.x),len(self.y),7));self.effort=np.zeros_like(self.q);self.max_error=0.
            for s,side in enumerate(['left','right']):
                names=kin.arm_names(side);p,_=kin.palm(side,initial);seed=np.array([initial[n] for n in names])
                for ix,x in enumerate(self.x):
                    for iy,y in enumerate(self.y):
                        R=Rotation.from_rotvec([x,y,0]).as_matrix()@np.asarray(hand_rotation)
                        q,pe,re=kin.solve_palm(side,p,R,seed)
                        if pe>.002 or re>.03:raise RuntimeError(f'Palm grid unreachable: {side} {x} {y}: {pe}, {re}')
                        seed=np.array([q[n] for n in names]);self.q[s,ix,iy]=seed
                        full=dict(initial);full.update(q);torque=kin.gravity_torques(full)
                        self.effort[s,ix,iy]=[torque[n] for n in names];self.max_error=max(self.max_error,pe)
                print('[PALM_GRID]',side,'exact node max position error',self.max_error,flush=True)
            self.validation={}
            # Check interpolation between grid nodes at deterministic random orientations.
            rng=np.random.default_rng(7)
            for s,side in enumerate(['left','right']):
                p,_=kin.palm(side,initial);maxp=0.;maxr=0.
                for _ in range(40):
                    x=rng.uniform(self.x[0],self.x[-1]);y=rng.uniform(self.y[0],self.y[-1])
                    q,_=self.lookup(s,x,y);full=dict(initial);full.update(zip(kin.arm_names(side),q))
                    pp,rr=kin.palm(side,full);desired=Rotation.from_rotvec([x,y,0]).as_matrix()@hand_rotation
                    maxp=max(maxp,float(np.linalg.norm(pp-p)));maxr=max(maxr,float(Rotation.from_matrix(desired@rr.T).magnitude()))
                self.validation[side]={'interpolated_position_error_m':maxp,'interpolated_rotation_error_rad':maxr}
                if maxp>.003 or maxr>.03:raise RuntimeError('Interpolated palm reference accuracy failed')
            np.savez(path,q=self.q,effort=self.effort,max_error=self.max_error,validation=json.dumps(self.validation))
        self.path=path

    def lookup(self,side,x,y):
        x=np.clip(x,self.x[0],self.x[-1]);y=np.clip(y,self.y[0],self.y[-1])
        ix=min(len(self.x)-2,max(0,np.searchsorted(self.x,x)-1));iy=min(len(self.y)-2,max(0,np.searchsorted(self.y,y)-1))
        a=(x-self.x[ix])/(self.x[ix+1]-self.x[ix]);b=(y-self.y[iy])/(self.y[iy+1]-self.y[iy])
        def blend(v):return (1-a)*(1-b)*v[side,ix,iy]+a*(1-b)*v[side,ix+1,iy]+(1-a)*b*v[side,ix,iy+1]+a*b*v[side,ix+1,iy+1]
        return blend(self.q),blend(self.effort)
