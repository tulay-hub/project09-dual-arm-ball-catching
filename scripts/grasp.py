"""O6 sphere grasp targets from original URDF collision STL; no extra pads."""
import numpy as np
import trimesh
import hashlib,json
from pathlib import Path
from scipy.spatial.transform import Rotation


class SphereGraspPlanner:
    def __init__(self,kinematics,bounds,radius=.028):
        self.kin=kinematics;self.radius=radius
        self.signature=hashlib.sha256(self.kin.urdf.read_bytes()+Path(__file__).read_bytes()).hexdigest()[:16]
        self.meshes={};self.queries={}
        for link in self.kin.xml.findall('link'):
            name=link.get('name')
            if name not in bounds or 'metacarpals' in name:continue
            collision=link.find('collision');element=collision.find('geometry/mesh')
            mesh=trimesh.load_mesh((self.kin.urdf.parent/element.get('filename')).resolve(),process=False)
            mesh.apply_scale(np.fromstring(element.get('scale','1 1 1'),sep=' '))
            origin=collision.find('origin')
            if origin is not None:
                M=np.eye(4);M[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
                M[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');mesh.apply_transform(M)
            self.meshes[name]=mesh;self.queries[name]=trimesh.proximity.ProximityQuery(mesh)

    def contact(self,side,joints,link,center):
        prefix='lh_' if side=='left' else 'rh_'
        transforms=self.kin.forward(joints)
        hand=transforms[prefix+'hand_base_link']
        T=np.linalg.inv(hand)@transforms[link]
        point=T[:3,:3].T@(np.asarray(center)-T[:3,3])
        nearest,distances,_=self.queries[link].on_surface(point[None])
        closest=nearest[0]
        offset=point-closest
        distance=np.linalg.norm(offset)
        normal=T[:3,:3]@(offset/max(distance,1e-9))
        return distance-self.radius,normal,T[:3,3]+T[:3,:3]@closest

    def resting_center(self,side,initial,z=.078):
        link=('lh_' if side=='left' else 'rh_')+'hand_base_link'
        lower,upper=.02,.085
        for _ in range(24):
            x=(lower+upper)/2
            gap,_,_=self.contact(side,initial,link,[x,0,z])
            if gap>0:upper=x
            else:lower=x
        return np.array([(lower+upper)/2,0,z])

    def solve(self,side,center,initial=None):
        prefix='lh_' if side=='left' else 'rh_'
        center=np.asarray(center)
        key=hashlib.sha256(json.dumps([side,center.tolist(),initial,self.radius],sort_keys=True).encode()).hexdigest()[:16]
        cache=self.kin.urdf.parent/'../../generated/training'/f'mesh_grasp_{self.signature}_{key}.json'
        cache=cache.resolve();cache.parent.mkdir(parents=True,exist_ok=True)
        if cache.exists():return json.loads(cache.read_text())
        targets={} if initial is None else dict(initial)
        contacts={}
        for finger in ['index','middle','ring','pinky']:
            joint=prefix+finger+'_mcp_pitch';link=prefix+finger+'_distal'
            candidates=[]
            for angle in np.linspace(.60,1.55,100):
                q=dict(targets);q[joint]=angle
                gap,normal,point=self.contact(side,q,link,center)
                # Contact on the curled fingertip, opposing the palm rather than pushing out.
                if normal[0]<-.05:
                    cost=abs(gap+.001)+.002*abs(angle-1.)
                    candidates.append((cost,angle,gap,normal,point))
            if not candidates:return None
            _,angle,gap,normal,point=min(candidates,key=lambda x:x[0])
            targets[joint]=float(angle)
            contacts[finger]={'gap_m':float(gap),'normal_hand':normal.tolist(),'point_hand_m':point.tolist()}
        best=None
        for yaw in np.linspace(0,1.36,35):
            for pitch in np.linspace(0,.58,30):
                q=dict(targets);q[prefix+'thumb_cmc_yaw']=yaw;q[prefix+'thumb_cmc_pitch']=pitch
                gap,normal,point=self.contact(side,q,prefix+'thumb_distal',center)
                # Thumb opposes the fingers' component toward the wrist (-Z).
                cost=abs(gap+.001)+.015*max(0.,.25-normal[2])+.001*(yaw+pitch)
                if best is None or cost<best[0]:best=(cost,yaw,pitch,gap,normal,point)
        _,yaw,pitch,gap,normal,point=best
        targets[prefix+'thumb_cmc_yaw']=float(yaw);targets[prefix+'thumb_cmc_pitch']=float(pitch)
        contacts['thumb']={'gap_m':float(gap),'normal_hand':normal.tolist(),'point_hand_m':point.tolist()}
        usable=sum(abs(c['gap_m'])<.005 for c in contacts.values())
        result={'targets_rad':{n:float(v) for n,v in targets.items() if n.startswith(prefix) and n in self.kin.limits
                              and not any(j['name']==n and j['mimic'] for j in self.kin.joints)},
                'contacts':contacts,'usable_contacts':usable,'center_hand_m':center.tolist(),
                'geometry_feasible':usable>=4 and abs(contacts['thumb']['gap_m'])<.004
                    and contacts['thumb']['normal_hand'][2]>.2 and all(c['gap_m']>-.004 for c in contacts.values())}
        cache.write_text(json.dumps(result,indent=2));return result
