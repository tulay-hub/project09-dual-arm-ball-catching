"""URDF forward kinematics and bounded palm IK; SI, rotation matrices, XYZW."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


class RobotKinematics:
    def __init__(self, urdf=None, base_height=1.25):
        self.urdf = Path(urdf or ROOT / 'assets/lkls73_o2_dual_arm_o6_description/urdf/lkls73_o2_dual_arm_o6_description_portable.urdf')
        self.xml = ET.parse(self.urdf).getroot()
        self.base = np.eye(4)
        self.base[2, 3] = base_height
        self.joints = []
        self.limits = {}
        self.mass_com={}
        for link in self.xml.findall('link'):
            origin=link.find('inertial/origin')
            com=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            self.mass_com[link.get('name')]=(float(link.find('inertial/mass').get('value')),com)
        for j in self.xml.findall('joint'):
            o = j.find('origin')
            T = np.eye(4)
            if o is not None:
                T[:3, 3] = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
                T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ') if j.find('axis') is not None else np.zeros(3)
            m = j.find('mimic')
            self.joints.append(dict(name=j.get('name'), parent=j.find('parent').get('link'),
                                    child=j.find('child').get('link'), T=T, axis=axis,
                                    fixed=j.get('type') == 'fixed', mimic=m.attrib if m is not None else None))
            if j.find('limit') is not None:
                l = j.find('limit')
                self.limits[j.get('name')] = tuple(float(l.get(k)) for k in ['lower', 'upper', 'velocity', 'effort'])
        children={j['parent']:[] for j in self.joints}
        for j in self.joints:children[j['parent']].append(j['child'])
        def subtree(name):
            return [name]+[descendant for child in children.get(name,[]) for descendant in subtree(child)]
        self.descendants={j['name']:subtree(j['child']) for j in self.joints if not j['fixed']}

    def arm_names(self, side):
        return [j['name'] for j in self.joints if j['name'].startswith(side.title() + '_') and not j['fixed']]

    def forward(self, q):
        transforms = {'Base_Link': self.base.copy()}
        for j in self.joints:
            angle = q.get(j['name'], 0.0)
            if j['mimic']:
                m = j['mimic']
                angle = q.get(m['joint'], 0.0) * float(m.get('multiplier', 1)) + float(m.get('offset', 0))
            T = transforms[j['parent']] @ j['T']
            if not j['fixed']:
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(j['axis'] * angle).as_matrix()
                T = T @ motion
            transforms[j['child']] = T
        return transforms

    def palm(self, side, q, local=(0.025, 0.0, 0.07)):
        T = self.forward(q)[('lh_' if side == 'left' else 'rh_') + 'hand_base_link']
        return T[:3, 3] + T[:3, :3] @ np.asarray(local), T[:3, :3]

    def solve_palm(self, side, position, rotation, seed=None):
        names = self.arm_names(side)
        lower = np.array([self.limits[n][0] for n in names]) + 1e-5
        upper = np.array([self.limits[n][1] for n in names]) - 1e-5
        def residual(x):
            p, R = self.palm(side, dict(zip(names, x)))
            error=np.r_[p - position, Rotation.from_matrix(rotation @ R.T).as_rotvec() * 0.12]
            if seed is not None:
                error=np.r_[error,.002*(x-np.asarray(seed))]
            return error
        seeds = [seed] if seed is not None else [(lower + upper) / 2]
        rng = np.random.default_rng(9)
        seeds += [rng.uniform(lower, upper) for _ in range(6 if seed is None else 1)]
        best = None
        for x in seeds:
            fit = least_squares(residual, np.clip(x, lower, upper), bounds=(lower, upper), max_nfev=140,
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun):
                best = fit
            if np.linalg.norm(fit.fun) < 1e-5:
                break
        p, R = self.palm(side, dict(zip(names, best.x)))
        return dict(zip(names, best.x.tolist())), float(np.linalg.norm(p - position)), float(np.linalg.norm(Rotation.from_matrix(rotation @ R.T).as_rotvec()))

    def palm_jacobian(self, side, joints):
        """Geometric palm Jacobian: linear [m/rad], angular [rad/rad]."""
        names = self.arm_names(side)
        p, R = self.palm(side, joints)
        J = np.zeros((6, len(names)))
        epsilon = 1e-6
        for i, name in enumerate(names):
            shifted = dict(joints)
            shifted[name] = shifted.get(name, 0) + epsilon
            ps, Rs = self.palm(side, shifted)
            J[:3, i] = (ps - p) / epsilon
            J[3:, i] = Rotation.from_matrix(Rs @ R.T).as_rotvec() / epsilon
        return J

    def gravity_torques(self,q):
        """Gravity compensation [N m], including mimic virtual-work transfer."""
        T=self.forward(q)
        positions={name:T[name][:3,3]+T[name][:3,:3]@com for name,(_,com) in self.mass_com.items()}
        torque={}
        for j in self.joints:
            if j['fixed']:continue
            origin=T[j['parent']]@j['T'];axis=origin[:3,:3]@j['axis']
            moment=np.zeros(3)
            for name in self.descendants[j['name']]:
                force=np.array([0.,0.,-9.81*self.mass_com[name][0]])
                moment+=np.cross(positions[name]-origin[:3,3],force)
            torque[j['name']]=-float(axis@moment)
        for j in self.joints:
            if j['mimic']:
                torque[j['mimic']['joint']]+=float(j['mimic'].get('multiplier',1))*torque[j['name']]
                torque[j['name']]=0.
        return torque
