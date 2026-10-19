"""Derive ready poses from the local URDF and inspect hand mesh bounds."""
import json
from pathlib import Path
import numpy as np
import trimesh
from kinematics import RobotKinematics, ROOT


def main():
    kin = RobotKinematics()
    # Hand +X faces up, fingers (+Z) extend forward in world +X.
    R = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
    poses = {}
    q = {}
    for side, y in [('left', 0.18), ('right', -0.18)]:
        target = np.array([0.36, y, 0.90])
        joints, p_error, r_error = kin.solve_palm(side, target, R)
        if p_error > 0.002 or r_error > 0.03:
            raise RuntimeError(f'{side} ready pose unreachable: {p_error}, {r_error}')
        q.update(joints)
        poses[side] = {'palm_position_world_m': target.tolist(), 'palm_position_error_m': p_error,
                       'orientation_error_rad': r_error, 'joint_positions_rad': joints}
    bounds = {}
    for link in kin.xml.findall('link'):
        if link.get('name').startswith(('lh_', 'rh_')):
            m = link.find('visual/geometry/mesh')
            mesh = trimesh.load_mesh(kin.urdf.parent / m.get('filename'), process=False)
            bounds[link.get('name')] = mesh.bounds.tolist()
    cfg = {'status': 'offline_ik_verified_runtime_pending', 'base_height_m': 1.25,
           'palm_local_position_m': [0.025, 0, 0.07], 'palm_normal_local': [1, 0, 0],
           'hand_rotation_world': R.tolist(), 'ready': poses, 'arm_ready_joint_positions_rad': q,
           'hand_mesh_bounds_local_m': bounds, 'ball_radius_m': 0.028, 'ball_mass_kg': 0.045,
           'overhead_camera': {'mount_link': 'Base_Link', 'position_in_link_m': [0.36, 0, 0.65],
                               'quaternion_in_link_xyzw_opengl': [0, 0, 0, 1],
                               'resolution': [256, 256], 'rate_hz': 60},
           'physics_hz': 600, 'servo_hz': 200}
    path = ROOT / 'config/scene.json'
    path.write_text(json.dumps(cfg, indent=2) + '\n')
    print(json.dumps({'config': str(path), 'poses': poses,
                      'palm_mesh_bounds': {s: bounds[s + '_hand_base_link'] for s in ['lh', 'rh']}}, indent=2))


if __name__ == '__main__':
    main()
