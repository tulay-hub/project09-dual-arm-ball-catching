"""Isaac single-scene model/contact commissioning. No training or ball attachment."""
from __future__ import annotations
import argparse
import json
import shutil
import hashlib
from collections import deque
from datetime import datetime
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.physics import PhysicsCfg

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=['smoke', 'drop', 'release','catch'], default='smoke')
parser.add_argument('--side', choices=['left','right'], default='left')
parser.add_argument('--seconds', type=float)
parser.add_argument('--log_dir', type=Path)
parser.add_argument('--rebuild', action='store_true')
parser.add_argument('--camera_mount_probe', action='store_true', help='Diagnostic root translation/yaw; no catching or training')
parser.add_argument('--disable_self_collision', action='store_true', help='Diagnostic isolation only')
parser.add_argument('--hand_curl', type=float, default=.35, help='Finger pre-shape [rad], [0,1]')
parser.add_argument('--launch_x',type=float,default=.20,help='External throw start X [m]')
parser.add_argument('--launch_z',type=float,default=1.20,help='External throw start Z [m]')
parser.add_argument('--throw_time',type=float,default=.4,help='Nominal throw-to-goal flight time [s]')
parser.add_argument('--require_success',action='store_true',help='Fail if requested contact task is not achieved')
parser.add_argument('--catch_curl',type=float,help='Catch pre-shape finger angle [rad]')
parser.add_argument('--secure_curl',type=float,default=1.20,help='Contact-confirmed finger closure [rad]')
parser.add_argument('--secure_strategy',choices=['balance','grasp','geometry_grasp'],default='geometry_grasp')
parser.add_argument('--catch_goal_x',type=float,default=.370,help='Nominal throw target at palm middle [m]')
parser.add_argument('--grip_tilt',type=float,default=.8,help='Grasp verification palm tilt [rad]')
parser.add_argument('--ready_tilt',type=float,default=-.18,help='Initial catch palm pitch [rad], fixture prior')
parser.add_argument('--hand_dynamics',choices=['source','mirror_left'],default='mirror_left')
add_launcher_args(parser)
args = parser.parse_args()
if args.catch_curl is None:args.catch_curl=.75 if args.secure_strategy=='geometry_grasp' else .35
if args.seconds is None:args.seconds=8 if args.mode=='catch' and args.secure_strategy=='geometry_grasp' else 3
if not 0<=args.hand_curl<=1:
    parser.error('--hand_curl must be between 0 and 1 rad')
if not 0<=args.catch_curl<=args.secure_curl<=1.60:
    parser.error('finger angles must satisfy 0 <= catch_curl <= secure_curl <= 1.60 rad')
if args.throw_time<=0 or args.seconds<=.5:
    parser.error('throw_time must be positive; seconds must exceed 0.5 for camera commissioning')
if args.camera_mount_probe and args.mode != 'smoke':
    parser.error('--camera_mount_probe requires --mode smoke')
args.physics = 'isaacsim_physx'
args.enable_cameras = True


def main():
    output = args.log_dir or ROOT / 'logs' / (args.mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    snapshot=output/'source';snapshot.mkdir()
    source_hashes={}
    for source in ['scripts/run_scene.py','scripts/kinematics.py','scripts/ballistics.py','scripts/ball_tracker.py',
                   'scripts/interception.py','scripts/grasp.py','scripts/prepare_dynamics.py','scripts/run.sh','config/scene.json','config/contact_materials.json','config/task_proposal.json']:
        destination=snapshot/source;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/source,destination)
        source_hashes[source]=hashlib.sha256(destination.read_bytes()).hexdigest()
    (output/'source_sha256.json').write_text(json.dumps(source_hashes,indent=2))
    cfg = json.loads((ROOT / 'config/scene.json').read_text())
    task = json.loads((ROOT / 'config/task_proposal.json').read_text())
    materials = json.loads((ROOT / 'config/contact_materials.json').read_text())
    with launch_simulation(PhysicsCfg(), args) as physics:
        import numpy as np
        import torch
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
        import isaaclab.sim as su
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils import configclass
        from ball_tracker import BallTracker
        from ballistics import release_velocity,quintic
        from interception import InterceptionPlanner
        from kinematics import RobotKinematics
        from scipy.spatial.transform import Rotation
        from prepare_dynamics import prepare_dynamics
        from grasp import SphereGraspPlanner
        physics_urdf=prepare_dynamics(args.hand_dynamics)
        shutil.copy2(physics_urdf,output/'simulation_model.urdf')
        if args.hand_dynamics=='mirror_left':
            shutil.copy2(ROOT/'assets/generated/dynamics/dynamics_assumption.json',output/'dynamics_assumption.json')

        sim = su.SimulationContext(su.SimulationCfg(dt=1/cfg['physics_hz'], device=args.device, physics=physics))
        sim.set_camera_view(eye=(1.55, -1.5, 1.7), target=(0.2, 0, 0.85))
        converter = UrdfConverter(UrdfConverterCfg(
            asset_path=str(physics_urdf), usd_dir=str(ROOT / 'assets/generated/robot'/args.hand_dynamics),
            usd_file_name='robot.usda', fix_base=True, merge_fixed_joints=False,
            collision_type='Convex Decomposition', self_collision=True, force_usd_conversion=args.rebuild,
            joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0))))
        hand_independent = [name for name in task['joint_order'] if name.startswith(('lh_', 'rh_'))]
        init_joints = dict(cfg['arm_ready_joint_positions_rad'])
        kin=RobotKinematics(urdf=physics_urdf)
        if args.mode=='catch':
            ready_rotation=Rotation.from_rotvec([0,args.ready_tilt,0]).as_matrix()@np.array(cfg['hand_rotation_world'])
            names=kin.arm_names(args.side)
            ready_seed=np.array([init_joints[n] for n in names])
            ready,pe,re=kin.solve_palm(args.side,np.array(cfg['ready'][args.side]['palm_position_world_m']),ready_rotation,ready_seed)
            if pe>.005 or re>.08:
                raise RuntimeError('Pre-shaped catch palm orientation is not reachable')
            init_joints.update(ready)
        for name in hand_independent:
            init_joints[name] = 0.20 if 'thumb' in name else (args.catch_curl if args.mode=='catch' else .35)
            if args.mode=='catch' and args.secure_strategy=='geometry_grasp' and name.startswith('lh_' if args.side=='left' else 'rh_'):
                if 'thumb_cmc_yaw' in name:init_joints[name]=.75
                if 'thumb_cmc_pitch' in name:init_joints[name]=.06
        for m in task['mimic_joints']:
            init_joints[m['name']] = init_joints[m['joint']] * float(m['multiplier'])
        def mat(key):
            return su.RigidBodyMaterialCfg(**{k: materials[key][k] for k in
                ['static_friction','dynamic_friction','restitution','friction_combine_mode','restitution_combine_mode']})
        overview_eye=np.array([1.65,-1.60,1.65]);overview_target=np.array([.15,0.,.90])
        z_axis=overview_eye-overview_target;z_axis/=np.linalg.norm(z_axis)
        x_axis=np.cross([0,0,1.],z_axis);x_axis/=np.linalg.norm(x_axis)
        y_axis=np.cross(z_axis,x_axis)
        overview_quat=Rotation.from_matrix(np.stack([x_axis,y_axis,z_axis],axis=1)).as_quat()

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path='/World/Ground', spawn=su.GroundPlaneCfg())
            light = AssetBaseCfg(prim_path='/World/Light', spawn=su.DomeLightCfg(intensity=2200))
            robot = ArticulationCfg(
                prim_path='{ENV_REGEX_NS}/Robot',
                spawn=su.UsdFileCfg(usd_path=converter.usd_path, activate_contact_sensors=True,
                    articulation_props=su.ArticulationRootPropertiesCfg(enabled_self_collisions=not args.disable_self_collision,
                        solver_position_iteration_count=32, solver_velocity_iteration_count=8)),
                init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,cfg['base_height_m']), joint_pos=init_joints),
                actuators={
                    'arm': ImplicitActuatorCfg(joint_names_expr=['Left_.*_Joint', 'Right_.*_Joint'],
                        stiffness=180., damping=16., effort_limit_sim=36., velocity_limit_sim=3.),
                    'hand': ImplicitActuatorCfg(joint_names_expr=hand_independent,
                        stiffness=3., damping=.10, effort_limit_sim=.25, velocity_limit_sim=1.),
                    'passive_mimic': ImplicitActuatorCfg(joint_names_expr=[m['name'] for m in task['mimic_joints']],
                        stiffness=0., damping=0., effort_limit_sim=0., velocity_limit_sim=2.)})
            ball = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Ball',
                spawn=su.SphereCfg(radius=cfg['ball_radius_m'],
                    mass_props=su.MassPropertiesCfg(mass=cfg['ball_mass_kg']),
                    rigid_props=su.RigidBodyPropertiesCfg(solver_position_iteration_count=32,
                        solver_velocity_iteration_count=8, max_depenetration_velocity=.5),
                    collision_props=su.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0),
                    physics_material=mat('ball_initial_material'),
                    visual_material=su.PreviewSurfaceCfg(diffuse_color=(.95,.03,.02))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(args.launch_x,.18 if args.side=='left' else -.18,args.launch_z)
                    if args.mode=='catch' else (.36,.18,.98 if args.mode=='smoke' else 1.10)))
            camera_rig = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Robot/Geometry/'+cfg['overhead_camera']['mount_link']+'/CameraRig',
                spawn=su.CuboidCfg(size=(.060,.060,.025),visual_material=su.PreviewSurfaceCfg(diffuse_color=(.25,.25,.28))))
            camera = CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/Geometry/'+cfg['overhead_camera']['mount_link']+'/CameraRig/Overhead', height=256,width=256,
                update_period=0.0, update_latest_camera_pose=True, data_types=['rgb','distance_to_image_plane'],
                spawn=su.PinholeCameraCfg(focal_length=24, horizontal_aperture=36, vertical_aperture=36,
                    clipping_range=(.03,3.0)),
                offset=CameraCfg.OffsetCfg(pos=tuple(cfg['overhead_camera']['position_in_link_m']),
                    rot=tuple(cfg['overhead_camera']['quaternion_in_link_xyzw_opengl']),convention='opengl'))
            overview = CameraCfg(prim_path='{ENV_REGEX_NS}/Overview', height=512,width=512,
                update_period=1/60,data_types=['rgb'],spawn=su.PinholeCameraCfg(focal_length=24,horizontal_aperture=36,
                    vertical_aperture=36,clipping_range=(.03,5.)),
                offset=CameraCfg.OffsetCfg(pos=tuple(overview_eye),rot=tuple(overview_quat),convention='opengl'))
        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.5))
        robot, ball, camera = scene['robot'], scene['ball'], scene['camera']
        stage = sim.stage
        rubber_cfg = mat('o6_rubber_contact')
        rubber_cfg.func('/World/Materials/O6Rubber', rubber_cfg)
        rubber = UsdShade.Material.Get(stage, '/World/Materials/O6Rubber')
        body_paths = {}
        pad_paths = []
        mimic_apis = []
        filtered_pairs=[]
        # Traverse instance proxies because the imported visual/collision assets can be instanceable.
        for prim in list(Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot'), Usd.TraverseInstanceProxies())):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body_paths[prim.GetName()] = str(prim.GetPath())
            if any('Mimic' in api for api in prim.GetAppliedSchemas()):
                mimic_apis.append({'path':str(prim.GetPath()), 'schemas':prim.GetAppliedSchemas()})
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
        # Imported O6 geometry overlaps within each hand. Exclude intra-hand pairs only;
        # robot/arm/torso and left-right hand collision handling remains enabled.
        for prefix in ['lh_','rh_']:
            paths=[path for name,path in body_paths.items() if name.startswith(prefix)]
            for path in paths:
                others=[other for other in paths if other!=path]
                UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel().SetTargets(others)
                filtered_pairs.append({'body':path,'filtered':others})
        from hand_contact_material import bind_original_hand_collisions
        hand_bindings=bind_original_hand_collisions(stage,body_paths,rubber)
        pad_paths=[binding['path'] for binding in hand_bindings]
        # The imported Newton mimic metadata did not produce stable PhysX coupling in the smoke.
        # PhysX convention: q_child + gearing*q_parent + offset = 0 (angular offset in degrees).
        for m in task['mimic_joints']:
            joint=stage.GetPrimAtPath('/World/envs/env_0/Robot/Physics/'+m['name'])
            reference=stage.GetPrimAtPath('/World/envs/env_0/Robot/Physics/'+m['joint'])
            axis='rot'+str(joint.GetAttribute('physics:axis').Get() or 'X')
            ref_axis='rot'+str(reference.GetAttribute('physics:axis').Get() or 'X')
            mimic=PhysxSchema.PhysxMimicJointAPI.Apply(joint,axis)
            mimic.CreateReferenceJointRel().SetTargets([reference.GetPath()])
            mimic.CreateReferenceJointAxisAttr(ref_axis)
            mimic.CreateGearingAttr(-float(m['multiplier']))
            mimic.CreateOffsetAttr(-float(m['offset'])*180/np.pi)
        # Body-mounted visual rig: every part inherits the Base_Link pose.
        mount_path=body_paths[cfg['overhead_camera']['mount_link']]
        rig_path=mount_path+'/CameraRig'
        camera_local=np.array(cfg['overhead_camera']['position_in_link_m'])
        cx,cy,cz=camera_local
        support_material=su.PreviewSurfaceCfg(diffuse_color=(.25,.25,.28))
        pole=su.CuboidCfg(size=(.025,.025,cz-.11),visual_material=support_material)
        pole.func(rig_path+'/CameraStand',pole,translation=(0,0,(cz+.11)/2))
        bar=su.CuboidCfg(size=(cx+.025,.025,.025),visual_material=support_material)
        bar.func(rig_path+'/CameraCrossbar',bar,translation=(cx/2,cy,cz))
        housing=su.CuboidCfg(size=(.070,.060,.040),visual_material=su.PreviewSurfaceCfg(diffuse_color=(.10,.12,.15)))
        housing.func(rig_path+'/CameraHousing',housing,translation=tuple(camera_local+[0,0,.025]))
        lens=su.CylinderCfg(radius=.013,height=.006,visual_material=su.PreviewSurfaceCfg(diffuse_color=(.02,.02,.025)))
        lens.func(rig_path+'/CameraLens',lens,translation=tuple(camera_local+[0,0,.003]))
        (output/'import.json').write_text(json.dumps({'usd':converter.usd_path,'rigid_body_paths':body_paths,
            'mimic_apis':mimic_apis,'rubber_pad_paths':[],'original_hand_collision_bindings':hand_bindings,'intra_hand_filtered_pairs':filtered_pairs},indent=2))
        print('[IMPORT]',converter.usd_path,'bodies',len(body_paths),'mimic',len(mimic_apis),'original_rubber_collisions',len(pad_paths),flush=True)
        sim.reset()
        print('[JOINTS]',robot.joint_names,flush=True)
        robot.write_joint_state_to_sim_index(position=robot.data.default_joint_pos.torch.clone(),
                                             velocity=torch.zeros_like(robot.data.default_joint_vel.torch))
        robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_state.torch[:,:7].clone())
        ball.write_root_pose_to_sim_index(root_pose=ball.data.default_root_state.torch[:,:7].clone())
        ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(ball.data.default_root_state.torch[:,7:]))
        if args.mode=='catch':
            # Ground truth is used only to initialize a reproducible external throw.
            launch=np.array([args.launch_x,.18 if args.side=='left' else -.18,args.launch_z])
            goal=np.array([args.catch_goal_x,.18 if args.side=='left' else -.18,.93])
            velocity=release_velocity(launch,goal,args.throw_time)
            root_velocity=torch.zeros((1,6),device=args.device)
            root_velocity[0,:3]=torch.tensor(velocity,device=args.device)
            ball.write_root_velocity_to_sim_index(root_velocity=root_velocity)
        scene.reset()
        (output/'initial_state.json').write_text(json.dumps({'joint_names':robot.joint_names,
            'joint_positions':robot.data.joint_pos.torch.tolist(),
            'root_pose':robot.data.default_root_state.torch[:,:7].tolist(),
            'self_collision':not args.disable_self_collision},indent=2))
        base_id=robot.body_names.index(cfg['overhead_camera']['mount_link'])
        rig_xform=UsdGeom.Xformable(stage.GetPrimAtPath(rig_path))
        rig_original=rig_xform.GetLocalTransformation()
        rig_xform.ClearXformOpOrder()
        rig_xform.AddTransformOp().Set(rig_original)
        # Fabric does not update parent USD transforms; synchronize only this visual rig.
        def sync_body_camera():
            base_pos=robot.data.body_link_pos_w.torch[0,base_id].cpu().numpy()
            base_rot=Rotation.from_quat(robot.data.body_link_quat_w.torch[0,base_id].cpu().numpy())
            q=base_rot.as_quat()
            desired=Gf.Matrix4d(1.)
            desired.SetRotate(Gf.Quatd(float(q[3]),Gf.Vec3d(*map(float,q[:3]))))
            desired.SetTranslateOnly(Gf.Vec3d(*map(float,base_pos)))
            parent=UsdGeom.Xformable(stage.GetPrimAtPath(mount_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            rig_xform.GetOrderedXformOps()[0].Set(desired*parent.GetInverse())
            sim.render_context.reset_scene_state_cadence()

        if args.camera_mount_probe:
            # Diagnostic only: deliberately move the fixed root. Ball state is never moved.
            base_id=robot.body_names.index(cfg['overhead_camera']['mount_link'])
            probe=[]
            initial=robot.data.default_root_state.torch[:,:7].clone()
            for label,dx,dy,yaw in [('initial',0.,0.,0.),('translate',.20,-.12,0.),
                                    ('translate_yaw',.20,-.12,np.pi/6),('return',0.,0.,0.)]:
                pose=initial.clone()
                pose[0,0]+=dx;pose[0,1]+=dy
                pose[0,3:7]=torch.tensor(Rotation.from_euler('z',yaw).as_quat(),device=args.device)
                robot.write_root_pose_to_sim_index(root_pose=pose)
                robot.actuators.target_command.set_position_index(value=robot.data.default_joint_pos.torch.clone())
                sim.forward();scene.update(1/cfg['physics_hz'])
                sync_body_camera()
                # Flush images as well as transforms after discontinuous diagnostic poses.
                for _ in range(4):
                    sim.render_context.reset_scene_state_cadence()
                    sim.render();camera.update(0,force_recompute=True)
                scene['overview'].update(0,force_recompute=True)
                data=camera.data
                base_pos=robot.data.body_link_pos_w.torch[0,base_id].cpu().numpy()
                base_rot=Rotation.from_quat(robot.data.body_link_quat_w.torch[0,base_id].cpu().numpy())
                expected_pos=base_pos+base_rot.apply(camera_local)
                local_rot=Rotation.from_quat(cfg['overhead_camera']['quaternion_in_link_xyzw_opengl'])
                expected_ros=base_rot*local_rot*Rotation.from_euler('x',np.pi)
                actual_pos=data.pos_w.torch[0].cpu().numpy()
                actual_ros=Rotation.from_quat(data.quat_w_ros.torch[0].cpu().numpy())
                rgb=data.output['rgb'].torch[0,...,:3].cpu().numpy().astype('uint8')
                depth=data.output['distance_to_image_plane'].torch[0].cpu().numpy()
                tracker=BallTracker(cfg['ball_radius_m'])
                measured=tracker.observe(rgb,depth,data.intrinsic_matrices.torch[0].cpu().numpy(),
                    actual_pos,actual_ros.as_quat(),float(camera._timestamp_last_update.numpy()[0]))
                truth=ball.data.root_pos_w.torch[0].cpu().numpy()
                housing_world=np.array(UsdGeom.Xformable(stage.GetPrimAtPath(rig_path+'/CameraHousing')).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
                expected_housing=base_pos+base_rot.apply(camera_local+[0,0,.025])
                entry={'case':label,'requested_root_position_m':pose[0,:3].cpu().tolist(),
                    'base_position_world_m':base_pos.tolist(),'base_quaternion_xyzw':base_rot.as_quat().tolist(),
                    'camera_position_world_m':actual_pos.tolist(),
                    'camera_position_error_m':float(np.linalg.norm(actual_pos-expected_pos)),
                    'camera_rotation_error_rad':float((expected_ros.inv()*actual_ros).magnitude()),
                    'housing_position_error_m':float(np.linalg.norm(housing_world-expected_housing)),
                    'rgb_nonblack':bool(rgb.mean()>1),
                    'ball_center_error_m':None if measured is None else float(np.linalg.norm(measured-truth))}
                probe.append(entry)
                Image.fromarray(rgb).save(output/('camera_'+label+'.png'))
                Image.fromarray(scene['overview'].data.output['rgb'].torch[0,...,:3].cpu().numpy().astype('uint8')).save(output/('overview_'+label+'.png'))
                print('[CAMERA_MOUNT]',json.dumps(entry),flush=True)
            result={'mount_link':cfg['overhead_camera']['mount_link'],'local_extrinsics':cfg['overhead_camera'],
                'cases':probe,'validation':'fixed-root pose/render diagnostic translation and yaw without contact stepping; no mobile catching'}
            result['passed']=all(x['camera_position_error_m']<.001 and x['camera_rotation_error_rad']<.001
                and x['housing_position_error_m']<.001 and x['rgb_nonblack'] and x['ball_center_error_m'] is not None
                and x['ball_center_error_m']<.01 for x in probe)
            (output/'camera_mount_probe.json').write_text(json.dumps(result,indent=2))
            if not result['passed']:raise RuntimeError('Body camera mount probe failed; see camera_mount_probe.json')
            return
        mapping=[robot.joint_names.index(n) for n in task['joint_order']]
        hand_ids=[robot.joint_names.index(n) for n in hand_independent]
        target=robot.data.default_joint_pos.torch.clone()
        contact_names=[n for n in body_paths if n.startswith(('lh_','rh_'))]
        contact_paths=[body_paths[n] for n in contact_names]
        active_hand_indices=[i for i,n in enumerate(contact_names) if n.startswith('lh_' if args.side=='left' else 'rh_')]
        prefix='lh_' if args.side=='left' else 'rh_'
        hand_body_id=robot.body_names.index(prefix+'hand_base_link')
        thumb_indices=[i for i,n in enumerate(contact_names) if n.startswith(prefix+'thumb')]
        finger_indices={finger:[i for i,n in enumerate(contact_names) if n.startswith(prefix+finger)] for finger in ['index','middle','ring','pinky']}
        contacts=sim.physics_sim_view.create_rigid_contact_view(contact_paths,filter_patterns=[['/World/envs/env_0/Ball']]*len(contact_paths))
        records=[]
        camera_saved=False
        camera_health={}
        max_force=0.
        tracker=BallTracker(cfg['ball_radius_m'])
        vision_errors=[]
        vision_samples=0
        camera_frame_ids=[]
        last_camera_frame=None
        continuous_hold=0
        longest_hold=0
        kin=RobotKinematics(urdf=physics_urdf)
        planner=InterceptionPlanner(kin,cfg['hand_rotation_world'],ball_radius=cfg['ball_radius_m'])
        plan=None
        trajectory_start=None
        trajectory_velocity=None
        trajectory_time=None
        reference_names=kin.arm_names(args.side)
        reference_ids=[robot.joint_names.index(n) for n in reference_names]
        plan_count=0
        balance_updates=0
        balance_commands_applied=0
        grasp_planner=SphereGraspPlanner(kin,cfg['hand_mesh_bounds_local_m'],cfg['ball_radius_m'])
        grasp_plan=None
        grasp_start=None
        grasp_duration=None
        grasp_initial=None
        grasp_targets=None
        grasp_confirmed_time=None
        grasp_confirm_counter=0
        grasp_position_window=deque(maxlen=120)
        grip_longest=0;grip_continuous=0
        tilted_grip_longest=0;tilted_grip_continuous=0
        secure_time=None
        secure_arm_target=None
        secure_palm_position=None
        held_visual_last=None
        held_visual_time=None
        held_visual_velocity=np.zeros(3)
        palm_feedback=kin.palm(args.side,cfg['arm_ready_joint_positions_rad'])[0]
        hold_samples=0
        open_samples=0
        for step in range(int(args.seconds*cfg['physics_hz'])):
            t=step/cfg['physics_hz']
            for name,i in zip(hand_independent,hand_ids):
                if args.mode=='smoke':
                    value=(.2 if 'thumb' in name else .35)+.12*np.sin(2*np.pi*t)
                else:
                    initial=float(init_joints[name])
                    desired=initial if args.mode=='catch' else (.20 if 'thumb' in name else args.hand_curl)
                    value=initial+np.clip(desired-initial,-t*.5,t*.5)
                    if args.mode=='release' and t>=2:
                        value=max(0.,desired-(t-2)*.7)
                    active_prefix='lh_' if args.side=='left' else 'rh_'
                    if args.mode=='catch' and args.secure_strategy=='grasp' and secure_time is not None and name.startswith(active_prefix):
                        grip=.35 if 'thumb' in name else args.secure_curl
                        elapsed=max(0.,t-secure_time-(.5 if 'thumb' in name else 0.))
                        value=initial+min(grip-initial,elapsed*.7)
                    if args.mode=='catch' and args.secure_strategy=='geometry_grasp' and grasp_start is not None and name in grasp_targets:
                        fraction=np.clip((t-grasp_start)/grasp_duration,0,1)
                        fraction=.5*(1-np.cos(np.pi*fraction))
                        value=grasp_initial[name]+fraction*(grasp_targets[name]-grasp_initial[name])
                target[:,i]=value
            if args.mode=='release' and t>=2:
                # Commissioning gravity release: open the hand and tilt the upward palm.
                # A ballistic toss controller is a separate next-stage behavior.
                i=robot.joint_names.index('Left_Wrist_Roll_Joint')
                target[:,i]=cfg['arm_ready_joint_positions_rad']['Left_Wrist_Roll_Joint']+min(1.1,(t-2)*1.5)
            if args.mode=='catch' and plan is not None:
                q_end=np.array([plan.joints[n] for n in reference_names])
                q_ref,_,_=quintic(t-trajectory_time,plan.time-trajectory_time,trajectory_start,trajectory_velocity,q_end,plan.joint_velocity)
                target[:,reference_ids]=torch.tensor(q_ref,device=args.device,dtype=target.dtype)
            if args.mode=='catch' and secure_arm_target is not None:
                target[:,reference_ids]=secure_arm_target
                balance_commands_applied+=int(step%3==0)
            if step%3==0:
                robot.actuators.target_command.set_position_index(value=target)
                q_robot=robot.data.joint_pos.torch[0].detach().cpu().numpy()
                joint_feedback=dict(zip(robot.joint_names,q_robot.tolist()))
                compensation=kin.gravity_torques(joint_feedback)
                palm_feedback=kin.palm(args.side,joint_feedback)[0]
                efforts=torch.tensor([[compensation.get(n,0.) for n in robot.joint_names]],device=args.device,dtype=target.dtype)
                robot.actuators.target_command.set_effort_index(value=efforts)
            scene.write_data_to_sim();sim.step(render=False);scene.update(1/cfg['physics_hz'])
            if step%10==0:
                sync_body_camera();sim.render()
            force=contacts.get_contact_force_matrix(dt=1/cfg['physics_hz'])
            if hasattr(force, 'torch'):
                values=force.torch.detach().cpu().numpy()
            elif isinstance(force,torch.Tensor):
                values=force.detach().cpu().numpy()
            elif hasattr(force,'numpy'):
                values=force.numpy()
            else:
                values=np.asarray(force)
            f=float(np.linalg.norm(values.reshape(-1,3),axis=1).sum())
            per_body=np.linalg.norm(values.reshape(len(contact_paths),-1,3),axis=-1).sum(axis=1)
            f_active=float(per_body[active_hand_indices].sum())
            per_body_vector=values.reshape(len(contact_paths),-1,3).sum(axis=1)
            thumb_vector=per_body_vector[thumb_indices].sum(axis=0)
            finger_vectors={finger:per_body_vector[indices].sum(axis=0) for finger,indices in finger_indices.items()}
            finger_vector=sum(finger_vectors.values())
            thumb_force=float(np.linalg.norm(thumb_vector))
            finger_count=int(sum(np.linalg.norm(vector)>.02 for vector in finger_vectors.values()))
            opposition=float(-float(thumb_vector@finger_vector)/max(1e-8,np.linalg.norm(thumb_vector)*np.linalg.norm(finger_vector)))
            if not torch.isfinite(robot.data.joint_pos.torch).all():
                (output/'trace.json').write_text(json.dumps(records))
                (output/'failure.json').write_text(json.dumps({'step':step,'time':t,'reason':'nonfinite robot joint state',
                    'grasp_start':grasp_start,'grasp_confirmed':grasp_confirmed_time}))
                raise RuntimeError(f'Nonfinite robot dynamics at step {step}')
            hand_position=robot.data.body_link_pos_w.torch[0,hand_body_id].detach().cpu().numpy()
            hand_rotation=Rotation.from_quat(robot.data.body_link_quat_w.torch[0,hand_body_id].detach().cpu().numpy()).as_matrix()
            palm_feedback=hand_position+hand_rotation@np.array(cfg['palm_local_position_m'])
            if grasp_start is not None and grasp_confirmed_time is None and t>grasp_start+grasp_duration:
                q_grip=robot.data.joint_pos.torch[0].detach().cpu().numpy()
                grasp_position_window.append([float(q_grip[robot.joint_names.index(name)]) for name in grasp_targets])
                stopped=len(grasp_position_window)==120 and np.max(np.ptp(np.array(grasp_position_window),axis=0))<.05
                confirmed=thumb_force>.03 and finger_count>=2 and opposition>.1 and stopped
                grasp_confirm_counter=grasp_confirm_counter+1 if confirmed else 0
                if grasp_confirm_counter>=120:grasp_confirmed_time=t
            if step==0:
                q_contract=robot.data.joint_pos.torch[0].detach().cpu().numpy()
                fk_hand=kin.forward(dict(zip(robot.joint_names,q_contract.tolist())))[prefix+'hand_base_link']
                com_position=robot.data.body_com_pos_w.torch[0,hand_body_id].detach().cpu().numpy()
                (output/'pose_contract.json').write_text(json.dumps({'fk_hand_origin_world_m':fk_hand[:3,3].tolist(),
                    'runtime_hand_link_origin_world_m':hand_position.tolist(),'runtime_hand_com_world_m':com_position.tolist(),
                    'runtime_minus_fk_local_m':(hand_rotation.T@(hand_position-fk_hand[:3,3])).tolist(),
                    'rotation_error_rad':float(np.linalg.norm(Rotation.from_matrix(hand_rotation@fk_hand[:3,:3].T).as_rotvec()))},indent=2))
            max_force=max(max_force,f)
            if args.mode=='catch' and secure_time is None and t>.25 and f_active>.05 and tracker.last_measurement is not None:
                q_robot=robot.data.joint_pos.torch[0].detach().cpu().numpy()
                palm_now,_=kin.palm(args.side,dict(zip(robot.joint_names,q_robot.tolist())))
                if np.linalg.norm(tracker.last_measurement-palm_now)<.09:
                    secure_time=t
                    secure_arm_target=target[:,reference_ids].clone()
                    secure_q=dict(zip(reference_names,secure_arm_target[0].detach().cpu().numpy().tolist()))
                    secure_palm_position=kin.palm(args.side,secure_q)[0]
            p=ball.data.root_pos_w.torch[0].detach().cpu().numpy()
            v=ball.data.root_lin_vel_w.torch[0].detach().cpu().numpy()
            physical_grip=bool(thumb_force>.03 and finger_count>=2 and opposition>.1
                and np.linalg.norm(p-palm_feedback)<.075 and np.linalg.norm(v)<.25)
            if physical_grip:
                grip_continuous+=1;grip_longest=max(grip_longest,grip_continuous)
            else:grip_continuous=0
            if physical_grip and hand_rotation[2,0]<.8:
                tilted_grip_continuous+=1;tilted_grip_longest=max(tilted_grip_longest,tilted_grip_continuous)
            else:tilted_grip_continuous=0
            if np.linalg.norm(p-palm_feedback)<.075 and np.linalg.norm(v)<.20 and f_active>.01:
                hold_samples+=1
                continuous_hold+=1
                longest_hold=max(longest_hold,continuous_hold)
            else:
                continuous_hold=0
            if args.mode=='release' and t>2 and np.linalg.norm(v)>.3:
                open_samples+=1
            if step%10==0:
                d=camera.data
                frame_id=int(camera.frame.torch[0].item())
                capture_time=float(camera._timestamp_last_update.numpy()[0])
                fresh_frame=frame_id!=last_camera_frame
                camera_frame_ids.append(frame_id)
                last_camera_frame=frame_id
                rgb=d.output['rgb'].torch[0,...,:3].detach().cpu().numpy().astype('uint8')
                depth=d.output['distance_to_image_plane'].torch[0].detach().cpu().numpy()
                measured=tracker.observe(rgb,depth,d.intrinsic_matrices.torch[0].detach().cpu().numpy(),
                    d.pos_w.torch[0].detach().cpu().numpy(),d.quat_w_ros.torch[0].detach().cpu().numpy(),capture_time) if fresh_frame else None
                vision_samples+=int(fresh_frame)
                if measured is not None:
                    vision_errors.append(float(np.linalg.norm(measured-p)))
                q=robot.data.joint_pos.torch[0].detach().cpu().numpy()
                if args.mode=='catch' and secure_time is not None and args.secure_strategy in ['balance','geometry_grasp'] and grasp_confirmed_time is None and measured is not None:
                    center_hand=hand_rotation.T@(measured-hand_position)
                    if held_visual_last is not None:
                        elapsed=max(1e-4,capture_time-held_visual_time)
                        velocity_measured=np.clip((center_hand-held_visual_last)/elapsed,-1.,1.)
                        held_visual_velocity=.6*held_visual_velocity+.4*velocity_measured
                    held_visual_last=center_hand.copy();held_visual_time=capture_time
                    error=center_hand[1:]-np.array([0.,.078])
                    tangent_acceleration=-15*error-6*held_visual_velocity[1:]
                    acceleration=(hand_rotation[:,1]*tangent_acceleration[0]+hand_rotation[:,2]*tangent_acceleration[1])[:2]
                    tilt=np.clip(np.array([-acceleration[1],acceleration[0],0])/9.81,-.12,.12)
                    rotation=Rotation.from_rotvec(tilt).as_matrix()@np.array(cfg['hand_rotation_world'])
                    seed=q[reference_ids]
                    solution,pe,re=kin.solve_palm(args.side,secure_palm_position,rotation,seed)
                    if pe<.005 and re<.08:
                        secure_arm_target=torch.tensor([[solution[n] for n in reference_names]],device=args.device,dtype=target.dtype)
                        balance_updates+=1
                if args.mode=='catch' and args.secure_strategy=='geometry_grasp' and secure_time is not None and grasp_start is None and measured is not None:
                    center_hand=hand_rotation.T@(measured-hand_position)
                    if t-secure_time>.6 and .070<center_hand[2]<.084 and abs(center_hand[1])<.008 and np.linalg.norm(held_visual_velocity[1:])<.06:
                        proposal=grasp_planner.solve(args.side,center_hand,dict(zip(robot.joint_names,q.tolist())))
                        if proposal is not None and proposal['geometry_feasible']:
                            grasp_plan=proposal;grasp_start=t
                            grasp_initial={name:float(q[robot.joint_names.index(name)]) for name in proposal['targets_rad']}
                            grasp_targets=proposal['targets_rad']
                            delta=max(abs(grasp_targets[name]-grasp_initial[name]) for name in grasp_targets)
                            grasp_duration=max(.5,np.pi/2*delta/.7)
                            (output/'grasp_plan.json').write_text(json.dumps(proposal,indent=2))
                if grasp_confirmed_time is not None and t>grasp_confirmed_time+.3:
                    tilt=min(args.grip_tilt,max(0,t-grasp_confirmed_time-.3)*.8)
                    if args.side=='right':tilt=-tilt
                    rotation=Rotation.from_rotvec([tilt,0,0]).as_matrix()@np.array(cfg['hand_rotation_world'])
                    solution,pe,re=kin.solve_palm(args.side,secure_palm_position,rotation,q[reference_ids])
                    if pe<.005 and re<.08:
                        secure_arm_target=torch.tensor([[solution[name] for name in reference_names]],device=args.device,dtype=target.dtype)
                if args.mode=='catch' and secure_time is None and t<.35 and (plan is None or step%30==0):
                    # Controller uses only image-derived filter and robot joint state.
                    joint_state=dict(zip(robot.joint_names,q.tolist()))
                    qd_robot=robot.data.joint_vel.torch[0].detach().cpu().numpy()
                    velocities=dict(zip(robot.joint_names,qd_robot.tolist()))
                    controller_time=max(t+1/cfg['physics_hz'],capture_time)
                    candidate=planner.solve(tracker.filter,controller_time,joint_state,args.side,velocities)
                    if candidate is not None:
                        plan=candidate
                        trajectory_start=np.array([joint_state[n] for n in reference_names])
                        trajectory_velocity=np.array([velocities[n] for n in reference_names])
                        trajectory_time=controller_time
                        plan_count+=1
                records.append({'time':t,'ball_position':p.tolist(),'ball_velocity':v.tolist(),
                    'contact_force_sum_n':f,'joint_pos':q.tolist(),
                    'active_hand_contact_force_n':f_active,'palm_position':palm_feedback.tolist(),
                    'camera_frame_id':camera_frame_ids[-1],
                    'camera_capture_time_s':capture_time,
                    'thumb_contact_force_n':thumb_force,'finger_contact_count':finger_count,'grip_opposition':opposition,
                    'hand_palm_normal_world':hand_rotation[:,0].tolist(),'grip_started':grasp_start is not None,
                    'commanded_joint_positions':target[0].detach().cpu().numpy().tolist(),
                    'visual_ball_position':None if measured is None else measured.tolist()})
            if t>.5 and not camera_saved:
                rgb=camera.data.output['rgb'].torch[0,...,:3].detach().cpu().numpy().astype('uint8')
                depth=camera.data.output['distance_to_image_plane'].torch[0].detach().cpu().numpy()
                Image.fromarray(rgb).save(output/'overhead.png')
                overview_rgb=scene['overview'].data.output['rgb'].torch[0,...,:3].detach().cpu().numpy().astype('uint8')
                Image.fromarray(overview_rgb).save(output/'overview.png')
                np.save(output/'depth.npy',depth)
                camera_health={'nonblack':bool(rgb.max()>0),'valid_depth_fraction':float(np.isfinite(depth).mean())}
                camera_saved=True
            if not np.isfinite(p).all():
                raise RuntimeError('Nonfinite ball dynamics')
            if not torch.isfinite(robot.data.joint_pos.torch).all():
                (output/'trace.json').write_text(json.dumps(records))
                (output/'failure.json').write_text(json.dumps({'step':step,'time':t,'reason':'nonfinite robot joint state'}))
                raise RuntimeError(f'Nonfinite robot dynamics at step {step}')
        sync_body_camera();sim.render()
        for key in ['camera','overview']:
            scene[key].update(0.0,force_recompute=True)
            final_rgb=scene[key].data.output['rgb'].torch[0,...,:3].detach().cpu().numpy().astype('uint8')
            Image.fromarray(final_rgb).save(output/(key+'_final.png'))
        q=robot.data.joint_pos.torch[0].detach().cpu().numpy()
        mimic_errors={m['name']:float(abs(q[robot.joint_names.index(m['name'])]-q[robot.joint_names.index(m['joint'])]*float(m['multiplier'])-float(m['offset'])))
                      for m in task['mimic_joints'] if m['name'] in robot.joint_names}
        bindings=[]
        for path in pad_paths:
            material,_=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial('physics')
            bindings.append({'path':path,'material':str(material.GetPath())})
        result={'mode':args.mode,'physics_steps':step+1,'joint_names':robot.joint_names,'control_joint_indices':mapping,
                'self_collision':not args.disable_self_collision,
                'hand_dynamics_profile':args.hand_dynamics,
                'final_ball_position_m':p.tolist(),'final_ball_velocity_m_s':v.tolist(),
                'max_ball_contact_force_sum_n':max_force,'hold_total_seconds':hold_samples/cfg['physics_hz'],
                'longest_continuous_hold_seconds':longest_hold/cfg['physics_hz'],
                'vision_valid_fraction':len(vision_errors)/max(vision_samples,1),
                'vision_mean_ball_center_error_m':float(np.mean(vision_errors)) if vision_errors else None,
                'vision_interception_plans':plan_count,
                'visual_interception_executed':plan_count>0,
                'visual_balance_updates':balance_updates,
                'balance_commands_applied':balance_commands_applied,
                'grasp_started_s':grasp_start,'grasp_duration_s':grasp_duration,
                'grasp_contact_confirmed_s':grasp_confirmed_time,
                'longest_opposing_grip_seconds':grip_longest/cfg['physics_hz'],
                'longest_tilted_grip_seconds':tilted_grip_longest/cfg['physics_hz'],
                'grasp_sustained':bool(physical_grip and grip_longest/cfg['physics_hz']>=.5),
                'tilted_grip_verified':bool(physical_grip and tilted_grip_longest/cfg['physics_hz']>=.5),
                'grasp_joint_targets_rad':grasp_targets,
                'catch_side':args.side,
                'secure_contact_time_s':secure_time,
                'secure_strategy':args.secure_strategy,
                'ready_tilt_rad':args.ready_tilt,
                'catch_finger_angles_rad':{'preshape':args.catch_curl,'secure':args.secure_curl},
                'external_throw':{'start_position_m':launch.tolist(),'initial_velocity_m_s':velocity.tolist(),
                    'nominal_flight_time_s':args.throw_time} if args.mode=='catch' else None,
                'initial_capture_over_half_second':bool(args.mode=='catch' and longest_hold/cfg['physics_hz']>=.5),
                'catch_sustained':bool(args.mode=='catch' and longest_hold/cfg['physics_hz']>=.5
                    and p[2]>.7 and np.linalg.norm(p-palm_feedback)<.075 and f_active>.01 and np.linalg.norm(v)<.20),
                'post_open_moving_seconds':open_samples/cfg['physics_hz'],
                'mimic_max_error_rad':max(mimic_errors.values(),default=0),'mimic_errors_rad':mimic_errors,
                'camera_health':camera_health,'rubber_bindings':bindings,
                'camera_unique_frames':len(set(camera_frame_ids)),
                'rubber_usd_material':{'static_friction':UsdPhysics.MaterialAPI(rubber.GetPrim()).GetStaticFrictionAttr().Get(),
                    'dynamic_friction':UsdPhysics.MaterialAPI(rubber.GetPrim()).GetDynamicFrictionAttr().Get(),
                    'restitution':UsdPhysics.MaterialAPI(rubber.GetPrim()).GetRestitutionAttr().Get()},
                'ball_attachment_used':False,'validation':'single-scene diagnostic; task outcomes reported separately; no juggling or training'}
        (output/'trace.json').write_text(json.dumps(records))
        (output/'result.json').write_text(json.dumps(result,indent=2))
        print('[RESULT]',json.dumps(result),flush=True)
        if not camera_health.get('nonblack') or len(bindings)!=len(pad_paths) or not pad_paths:
            raise RuntimeError('Scene camera/material commissioning failed')
        if not np.isfinite(q).all() or max(mimic_errors.values(),default=0)>.10:
            raise RuntimeError('Joint/mimic commissioning failed')
        if args.require_success:
            success=((result['catch_sustained'] and (balance_commands_applied>0 or args.secure_strategy=='grasp')
                and (args.secure_strategy!='geometry_grasp' or (result['grasp_sustained'] and (args.grip_tilt==0 or result['tilted_grip_verified'])))) if args.mode=='catch' else
                     (result['longest_continuous_hold_seconds']>=.5 and (args.mode!='release' or result['post_open_moving_seconds']>.1)))
            if not success:
                raise RuntimeError('Task success criterion not achieved; see saved result.json')


if __name__=='__main__':
    main()
