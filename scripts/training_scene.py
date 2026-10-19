"""Project09 training scene, preserving commissioned contact and camera geometry."""
from pathlib import Path
from types import SimpleNamespace
import json
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
import isaaclab.sim as su
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from kinematics import RobotKinematics
from scipy.spatial.transform import Rotation
from prepare_dynamics import prepare_dynamics

def build_scene(sim,num_envs,output,camera_resolution=192,enable_overview=False):
    cfg=json.loads((ROOT/'config/scene.json').read_text())
    task=json.loads((ROOT/'config/task_proposal.json').read_text())
    materials=json.loads((ROOT/'config/contact_materials.json').read_text())
    args=SimpleNamespace(hand_dynamics='mirror_left',rebuild=False,mode='catch',side='left',ready_tilt=-.18,
                         catch_curl=.75,secure_strategy='geometry_grasp',launch_x=.80,launch_z=1.05,
                         disable_self_collision=False)
    physics_urdf=prepare_dynamics(args.hand_dynamics)
    sim.set_camera_view(eye=(1.55, -1.5, 1.7), target=(0.2, 0, 0.85))
    converter = UrdfConverter(UrdfConverterCfg(
        asset_path=str(physics_urdf), usd_dir=str(ROOT / 'assets/generated/robot'/args.hand_dynamics),
        usd_file_name='robot.usda', fix_base=True, merge_fixed_joints=False,
        collision_type='Convex Decomposition', self_collision=True, force_usd_conversion=args.rebuild,
        joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0))))
    hand_independent = [name for name in task['joint_order'] if name.startswith(('lh_', 'rh_'))]
    init_joints = dict(cfg['arm_ready_joint_positions_rad'])
    kin=RobotKinematics(urdf=physics_urdf)
    for side in ['left','right']:
        ready_rotation=Rotation.from_rotvec([0,args.ready_tilt,0]).as_matrix()@np.array(cfg['hand_rotation_world'])
        names=kin.arm_names(side)
        ready_seed=np.array([init_joints[n] for n in names])
        ready,pe,re=kin.solve_palm(side,np.array(cfg['ready'][side]['palm_position_world_m']),ready_rotation,ready_seed)
        if pe>.005 or re>.08:
            raise RuntimeError('Pre-shaped catch palm orientation is not reachable')
        init_joints.update(ready)
    for name in hand_independent:
        init_joints[name] = 0.20 if 'thumb' in name else (args.catch_curl if args.mode=='catch' else .35)
        if args.mode=='catch' and args.secure_strategy=='geometry_grasp':
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
        ground = AssetBaseCfg(prim_path='/World/Ground',
            spawn=su.CuboidCfg(size=(30.,30.,.02),collision_props=su.CollisionPropertiesCfg(),
                visual_material=su.PreviewSurfaceCfg(diffuse_color=(.55,.55,.55))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,-.01)))
        light = AssetBaseCfg(prim_path='/World/Light', spawn=su.DomeLightCfg(intensity=2200))
        robot = ArticulationCfg(
            prim_path='{ENV_REGEX_NS}/Robot',
            spawn=su.UsdFileCfg(usd_path=converter.usd_path, activate_contact_sensors=True,
                rigid_props=su.RigidBodyPropertiesCfg(max_depenetration_velocity=.5),
                articulation_props=su.ArticulationRootPropertiesCfg(enabled_self_collisions=not args.disable_self_collision,
                    solver_position_iteration_count=32, solver_velocity_iteration_count=8)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,cfg['base_height_m']), joint_pos=init_joints),
            actuators={
                'arm': ImplicitActuatorCfg(joint_names_expr=['Left_.*_Joint', 'Right_.*_Joint'],
                    stiffness=180., damping=16., armature=.02, effort_limit_sim=36., velocity_limit_sim=3.),
                'hand': ImplicitActuatorCfg(joint_names_expr=hand_independent,
                    stiffness=3., damping=.10, armature=.001, effort_limit_sim=.25, velocity_limit_sim=1.),
                'passive_mimic': ImplicitActuatorCfg(joint_names_expr=[m['name'] for m in task['mimic_joints']],
                    stiffness=0., damping=0., armature=.001, effort_limit_sim=0., velocity_limit_sim=2.)})
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
        camera = CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/Geometry/'+cfg['overhead_camera']['mount_link']+'/CameraRig/Overhead', height=camera_resolution,width=camera_resolution,
            update_period=0.0, update_latest_camera_pose=True, data_types=['rgb','distance_to_image_plane'],
            spawn=su.PinholeCameraCfg(focal_length=24, horizontal_aperture=36, vertical_aperture=36,
                clipping_range=(.03,3.0)),
            offset=CameraCfg.OffsetCfg(pos=tuple(cfg['overhead_camera']['position_in_link_m']),
                rot=tuple(cfg['overhead_camera']['quaternion_in_link_xyzw_opengl']),convention='opengl'))
        overview = CameraCfg(prim_path='{ENV_REGEX_NS}/PolicyOverview',height=480,width=640,
            update_period=0.0,update_latest_camera_pose=True,data_types=['rgb'],
            spawn=su.PinholeCameraCfg(focal_length=24,horizontal_aperture=36,vertical_aperture=27,clipping_range=(.03,10.)),
            offset=CameraCfg.OffsetCfg(pos=tuple(overview_eye),rot=tuple(overview_quat),convention='opengl')) if enable_overview else None
    scene = InteractiveScene(SceneCfg(num_envs=num_envs, env_spacing=2.5, replicate_physics=False))
    robot, ball, camera = scene['robot'], scene['ball'], scene['camera']
    stage = sim.stage
    rubber_cfg = mat('o6_rubber_contact')
    rubber_cfg.func('/World/Materials/O6Rubber', rubber_cfg)
    rubber = UsdShade.Material.Get(stage, '/World/Materials/O6Rubber')
    all_body_paths=[];all_hand_bindings=[]
    for env_id in range(num_envs):
        env_path=f'/World/envs/env_{env_id}'
        body_paths = {}
        pad_paths = []
        mimic_apis = []
        filtered_pairs=[]
        # Traverse instance proxies because the imported visual/collision assets can be instanceable.
        for prim in list(Usd.PrimRange(stage.GetPrimAtPath(env_path+'/Robot'), Usd.TraverseInstanceProxies())):
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
            joint=stage.GetPrimAtPath(env_path+'/Robot/Physics/'+m['name'])
            reference=stage.GetPrimAtPath(env_path+'/Robot/Physics/'+m['joint'])
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
        all_body_paths.append(body_paths);all_hand_bindings.extend(hand_bindings)
        PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(env_path+'/Ball')).CreateEnableSpeculativeCCDAttr(True)
    (output/'scene_import.json').write_text(json.dumps({'usd':converter.usd_path,
        'environments':num_envs,'rubber_pad_count':0,'original_hand_collision_bindings':all_hand_bindings,'friction_geometry':'original_mesh_no_added_shapes',
        'body_paths':all_body_paths,'dynamics_profile':'mirror_left',
        'physics_hz':cfg['physics_hz'],'ball_speculative_ccd':True},indent=2))
    return scene,cfg,task,kin,init_joints,all_body_paths
