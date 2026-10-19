"""Off-center throws gated by bounded IK and the body camera's full-ball frustum."""
import hashlib
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from ballistics import projectile, release_velocity
from kinematics import ROOT


def sphere_in_view(points, radius, camera_position, camera_rotation, half_tan=.75, margin=.045):
    # OpenGL camera looks down -Z; full sphere must fit with an image-edge margin.
    local = (np.asarray(points) - camera_position) @ camera_rotation
    depth = -local[:, 2]
    allowed = half_tan - margin
    return bool(np.all((depth > radius + .03) & (depth < 3. - radius)
                      & (np.abs(local[:, 0]) + radius < (depth - radius) * allowed)
                      & (np.abs(local[:, 1]) + radius < (depth - radius) * allowed)))


def predict_landing(position, velocity, ball_height):
    p, v = np.asarray(position), np.asarray(velocity)
    discriminant = v[2] ** 2 + 2 * 9.81 * (p[2] - ball_height)
    if discriminant < 0:
        return None
    tau = (v[2] + np.sqrt(discriminant)) / 9.81
    if not .005 < tau < .9:
        return None
    return tau, projectile(p, v, tau)


class ThrowWorkspace:
    def __init__(self, kin, initial, cfg, contact_centers=None):
        self.radius = cfg['ball_radius_m']
        camera = cfg['overhead_camera']
        self.camera_position = kin.base[:3, 3] + np.array(camera['position_in_link_m'])
        self.camera_rotation = Rotation.from_quat(camera['quaternion_in_link_xyzw_opengl']).as_matrix()
        self.palms = []; self.rotations = []; self.goals = []; self.joints = []
        centers=contact_centers if contact_centers is not None else [[.025+self.radius,0,.07]]*2
        self.contact_offsets=[]
        signature = hashlib.sha256(kin.urdf.read_bytes() + Path(__file__).read_bytes()
            + json.dumps([initial,centers,cfg['ball_radius_m'],cfg['overhead_camera']], sort_keys=True).encode()).hexdigest()[:16]
        path = ROOT / 'assets/generated/training' / f'throw_workspace_{signature}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        cached = np.load(path) if path.exists() else None
        for s, side in enumerate(['left', 'right']):
            palm, rotation = kin.palm(side, initial)
            self.palms.append(palm); self.rotations.append(rotation)
            contact_offset=rotation@(np.array(centers[s])-np.array([.025,0,.07]))
            self.contact_offsets.append(contact_offset)
            seed = np.array([initial[n] for n in kin.arm_names(side)])
            goals = []; joints = []
            if cached is not None:
                goals = cached[f'goals{s}']; joints = cached[f'joints{s}']
            else:
                for radius in [.055, .08, .105]:
                    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                        target = palm + [radius * np.cos(angle), radius * np.sin(angle), 0.]
                        if not (.25 < target[0] < .49 and .10 < abs(target[1]) < .31):
                            continue
                        q, pe, re = kin.solve_palm(side, target, rotation, seed)
                        qv = np.array([q[n] for n in kin.arm_names(side)])
                        if pe < .003 and re < .035 and np.max(np.abs(qv - seed)) < .85:
                            goals.append(target + contact_offset); joints.append(qv)
            if len(goals) < 10:
                raise RuntimeError(f'Insufficient reachable off-center catch targets: {side}')
            self.goals.append(np.array(goals)); self.joints.append(np.array(joints))
        if cached is None:
            np.savez(path, **{f'goals{s}': self.goals[s] for s in range(2)},
                     **{f'joints{s}': self.joints[s] for s in range(2)})
        self.validation = {'target_counts': [len(g) for g in self.goals],
            'minimum_horizontal_offset_m': .055, 'maximum_horizontal_offset_m': .105,
            'ik_position_tolerance_m': .003, 'ik_rotation_tolerance_rad': .035,
            'camera_gate': 'full sphere, 31 samples from launch to landing, image margin',
            'throw_version':'front_side_v2','front_probability':.5,
            'side_probability':.5,'minimum_horizontal_flight_m':.26,
            'minimum_horizontal_speed_m_s':.75,
            'camera_position_m': self.camera_position.tolist(), 'cache': str(path)}

    def sample(self, rng, side):
        kind='front' if rng.random()<.5 else ('left_side' if side==0 else 'right_side')
        for _ in range(300):
            index = int(rng.integers(len(self.goals[side])))
            goal = self.goals[side][index]
            if kind=='front':
                start=np.array([rng.uniform(.74,.90),goal[1]+rng.uniform(-.06,.06),rng.uniform(.96,1.12)])
            else:
                start=np.array([goal[0]+rng.uniform(-.05,.05),rng.uniform(.52,.65)*(1 if side==0 else -1),rng.uniform(.96,1.12)])
            flight = rng.uniform(.34, .46)
            velocity = release_velocity(start, goal, flight)
            distance=np.linalg.norm((goal-start)[:2]);speed=np.linalg.norm(velocity[:2])
            impact_z=velocity[2]-9.81*flight
            if distance<(.30 if kind=='front' else .26) or speed<.75 or speed/max(abs(impact_z),1e-6)<.30:
                continue
            path = np.array([projectile(start, velocity, t) for t in np.linspace(0, flight, 31)])
            if sphere_in_view(path, self.radius, self.camera_position, self.camera_rotation):
                return start, velocity, goal, flight, index, kind
        raise RuntimeError(f'No reachable camera-visible {kind} throw found; refusing fallback to vertical drops')
