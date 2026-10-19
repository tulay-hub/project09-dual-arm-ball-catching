"""Rendered RGB-D PPO residual catch and opposing grasp; no attachment or toss."""
from pathlib import Path
from collections import deque
import json,shutil,time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from pxr import Gf,Usd,UsdGeom
from isaaclab.utils.math import quat_apply,quat_apply_inverse
from training_scene import build_scene
from exposure_tracking import ExposureTrackingBatch,SynchronousSimulationClock
from grasp import SphereGraspPlanner
from throw_workspace import ThrowWorkspace,predict_landing
from grip_evidence import opposing_contacts,retention_contacts,stable_retention
from arm_tracking import joint_body_jacobian,joint_gravity_forces
from policy_contract import legacy_actions,visual_features,tracking_potential,V4_NAME,filter_residual_actions,XY_RESIDUAL_SCALE_M,XY_CONTRACT
from response_control import FAST_RESPONSE,arrival_joint_velocity
from predictive_tracking import position_priority_delta,early_arrival_velocity,REVISION as TRACKING_REVISION
from soft_catch import reception_target,reception_joint_velocity,SOFT_SETTINGS,REVISION as GRIP_REVISION
from grip_pose_finish import contact_rotation_target,settling_joint_velocity,REVISION as POSE_FINISH_REVISION
from training_profiles import (ROBUST_PROFILE,MIXED_PROFILE,CONSERVATIVE_PROFILES,ROBUST_PPO,ROBUST_DIRECTION_TARGET,
    sample_receivers,choose_throw_kind,sample_episode_levels,augment_measurement)
from fast_sampling import as_torch,contact_masks,reduce_contact_forces,sparse_red_pixels
from closure_trigger import contact_closure_step,anticipated_closure_fraction,incoming_hand_tilt

class BallCatchEnv(VecEnv):
    def __init__(self,sim,num_envs,log_dir,seed=42,camera_resolution=192,full_curriculum=False,enable_overview=False,gui_rebind_views=False,response_profile=None,policy_version='v3',training_mode=False,curriculum_level=0,tracking_profile='fast',grip_profile='fast',training_profile='legacy'):
        self.sim=sim;self.device=sim.device;self.num_envs=num_envs;self.log_dir=Path(log_dir)
        self.policy_version=policy_version;self.action_stride=7 if policy_version=='v4' else 4
        self.tracking_profile=tracking_profile
        self.grip_profile=grip_profile
        self.training_profile=training_profile
        if grip_profile=='pose_finish' and tracking_profile!='predictive':raise ValueError('pose_finish requires predictive visual tracking')
        self.training_mode=training_mode;self.curriculum_level=curriculum_level if training_mode and training_profile!=MIXED_PROFILE else 2
        self.sensor_rng=np.random.default_rng(seed+9047)
        self.dt=1/60;self.decimation=2;self.max_episode_length=300;self.num_actions=self.action_stride*2
        self.full_curriculum=full_curriculum;self.curriculum=1. if full_curriculum else 0.
        self.rng=np.random.default_rng(seed);torch.manual_seed(seed)
        self.scene,self.scene_cfg,self.task,self.kin,self.initial,self.paths=build_scene(sim,num_envs,self.log_dir,camera_resolution,enable_overview)
        self.response_profile=response_profile or self.task.get('controller_response',{}).get('profile','legacy')
        if self.policy_version=='v4' and self.response_profile!='fast':raise ValueError('v4 requires the verified fast controller')
        self.response_settings=dict(FAST_RESPONSE)
        if grip_profile in ('soft_catch','early_close'):
            if self.response_profile!='fast':raise ValueError('soft_catch requires fast physical hand drives')
            self.response_settings.update(SOFT_SETTINGS)
        self.physics_dt=1./self.scene_cfg['physics_hz'];self.decimation=round(self.dt/self.physics_dt)
        if self.decimation<1 or abs(self.decimation*self.physics_dt-self.dt)>1e-10:raise ValueError('Physics frequency must be a multiple of 60Hz')
        self.robot=self.scene['robot'];self.ball=self.scene['ball'];self.camera=self.scene['camera']
        sim.reset()
        if gui_rebind_views:
            print('[GUI_VIEW_BEFORE_REBIND]',sim.physics_sim_view.is_valid,flush=True)
            sim.stop()
            import omni.kit.app
            for _ in range(4):omni.kit.app.get_app().update()
            sim.reset()
            print('[GUI_VIEW_AFTER_REBIND]',sim.physics_sim_view.is_valid,flush=True)
        self.origins=self.scene.env_origins
        armature=self.robot.data.joint_armature.torch[0].cpu().tolist()
        settings=dict(zip(self.robot.joint_names,armature))
        (self.log_dir/'joint_armature_verified.json').write_text(json.dumps(settings,indent=2))
        for name,value in settings.items():
            expected=.001 if name.startswith(('lh_','rh_')) else .02
            if abs(value-expected)>1e-6:raise RuntimeError(f'Joint armature not applied: {name} {value}')
        self.jnames=self.robot.joint_names;self.control_ids=[self.jnames.index(n) for n in self.task['joint_order']]
        self.arm_ids=[[self.jnames.index(n) for n in self.kin.arm_names(s)] for s in ['left','right']]
        self.hand_ids=[[self.jnames.index(n) for n in self.task['joint_order'] if n.startswith(prefix)] for prefix in ['lh_','rh_']]
        self.hbody=[self.robot.body_names.index(p+'hand_base_link') for p in ['lh_','rh_']]
        self.base_id=self.robot.body_names.index('Base_Link')
        jacobian=self.robot.data.body_link_jacobian_w.torch
        (self.log_dir/'kinematic_contract.json').write_text(json.dumps({
            'joint_names':self.jnames,'body_names':self.robot.body_names,
            'jacobian_shape':list(jacobian.shape),'base_columns':jacobian.shape[-1]-len(self.jnames),
            'omitted_root_rows':len(self.robot.body_names)-jacobian.shape[1],
            'gravity_shape':list(self.robot.data.gravity_compensation_forces.torch.shape)},indent=2))
        self.rigs=[]
        for paths in self.paths:
            parent=paths['Base_Link'];rig=UsdGeom.Xformable(sim.stage.GetPrimAtPath(parent+'/CameraRig'))
            rig.ClearXformOpOrder();op=rig.AddTransformOp()
            self.rigs.append((parent,op))
        self.contacts=[];self.contact_names=[]
        contact_contract=[]
        self.total_contact=torch.zeros((num_envs,2),device=self.device)
        for e,paths in enumerate(self.paths):
            names=[n for n in paths if n.startswith(('lh_','rh_'))]
            view=sim.physics_sim_view.create_rigid_contact_view([paths[n] for n in names],
                filter_patterns=[[f'/World/envs/env_{e}/Ball']]*len(names),max_contact_data_count=512)
            actual_names=[]
            for sensor_path in view.sensor_paths:
                matches=[n for n in names if str(sensor_path)==paths[n] or str(sensor_path).startswith(paths[n]+'/')]
                if not matches:raise RuntimeError(f'Unmapped hand contact sensor: {sensor_path}')
                actual_names.append(max(matches,key=lambda n:len(paths[n])))
            self.contact_names.append(actual_names);self.contacts.append(view)
            contact_contract.append({'env':e,'sensor_count':view.sensor_count,'filter_count':view.filter_count,
                'sensor_paths':list(view.sensor_paths),'filter_paths':view.filter_paths,
                'mapped_names':actual_names,'force_shape':list(view.get_contact_force_matrix(dt=self.physics_dt).shape)})
        (self.log_dir/'contact_view_contract.json').write_text(json.dumps(contact_contract,indent=2))
        self.contact_side_masks,self.contact_finger_masks=contact_masks(self.contact_names,self.device)
        self.raw_contact_peak_n=0.
        self.exposure_batch=ExposureTrackingBatch(num_envs,self.scene_cfg['ball_radius_m'],filtered_state=tracking_profile=='predictive')
        self.exposure_clock=SynchronousSimulationClock(num_envs,self.scene_cfg['physics_hz'])
        self.vision=torch.zeros((num_envs,3),device=self.device);self.visual_velocity=torch.zeros_like(self.vision)
        self.visible=torch.zeros(num_envs,device=self.device,dtype=torch.bool);self.age=torch.ones(num_envs,device=self.device)
        self.side=torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.episode_level_host=np.full(num_envs,self.curriculum_level,dtype=np.int64)
        self.sampled_level_counts=np.zeros(3,dtype=np.int64)
        self.completed_level_counts=np.zeros(3,dtype=np.int64)
        self.level_success_counts=np.zeros(3,dtype=np.int64)
        self.episode_length_buf=torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.actions=torch.zeros((num_envs,self.num_actions),device=self.device);self.previous_actions=self.actions.clone()
        self.target=self.robot.data.default_joint_pos.torch.clone();self.limits=self.robot.data.joint_pos_limits.torch.clone()
        self.velocity_targets=torch.zeros_like(self.target)
        self.force_features=torch.zeros((num_envs,10),device=self.device)
        self.capture_time=torch.full((num_envs,),-1.,device=self.device)
        self.grasp_start=torch.full((num_envs,),-1.,device=self.device)
        self.grasp_curl_bias=torch.zeros(num_envs,device=self.device)
        self.preclose_fraction=torch.zeros(num_envs,device=self.device)
        self.prepare_time=torch.full((num_envs,),-1.,device=self.device)
        self.predicted_contact_time=torch.full((num_envs,),-1.,device=self.device)
        self.caught_tilt=torch.zeros((num_envs,3),device=self.device)
        self.reception_anchor=torch.zeros((num_envs,3),device=self.device)
        self.reception_feedforward=torch.zeros_like(self.reception_anchor)
        self.reception_shift=torch.zeros_like(self.reception_anchor)
        self.intercept_tilt=torch.zeros((num_envs,3),device=self.device);self.intercept_tilt[:,1]=-.18
        self.preparation_events=0;self.preparation_events_by_side=[0,0]
        self.contact_streak=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.grasp_origin_q=torch.zeros((num_envs,6),device=self.device)
        self.closure_events=0;self.closure_events_by_side=[0,0]
        self.closure_motion_max_rad=0.
        self.contact_probe=False
        self.hold=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.tilt_hold=torch.zeros_like(self.hold);self.best_hold=torch.zeros_like(self.hold)
        self.capture_rewarded=torch.zeros(num_envs,dtype=torch.bool,device=self.device)
        self.hold_center=torch.zeros((num_envs,3),device=self.device)
        self.reward_previous_center=torch.zeros_like(self.hold_center)
        self.catch_palm_target=torch.zeros_like(self.hold_center)
        self.palm_velocity=torch.zeros_like(self.hold_center);self.previous_palm=torch.zeros_like(self.hold_center)
        self.reward_potential=torch.zeros(num_envs,device=self.device)
        self.throw_goal=torch.zeros_like(self.hold_center);self.throw_start=torch.zeros_like(self.hold_center)
        self.throw_flight=torch.zeros(num_envs,device=self.device)
        self.throw_kind=['uninitialized']*num_envs
        self.throw_counts={'front':0,'left_side':0,'right_side':0}
        self.throw_launch_velocity=torch.zeros_like(self.throw_goal)
        self.predicted_intercept=torch.zeros_like(self.hold_center)
        self.contact_evidence=[{} for _ in range(num_envs)]
        self.returns=torch.zeros(num_envs,device=self.device)
        self.physics_failures=0;self.episodes=0;self.successes=0;self.total_steps=0;self.recent=deque(maxlen=200)
        self.diagnostic_callback=None
        self.vision_error_sum=0.;self.vision_count=0;self.vision_samples=0
        self.previous_local=torch.zeros((num_envs,3),device=self.device);self.local_velocity=torch.zeros_like(self.previous_local)
        self.nominal_p=[];self.nominal_r=[];self.inverse_j=[];self.nominal_grasp=[];contact_centers=[]
        planner=SphereGraspPlanner(self.kin,self.scene_cfg['hand_mesh_bounds_local_m'],self.scene_cfg['ball_radius_m'])
        for side in ['left','right']:
            p,R=self.kin.palm(side,self.initial);self.nominal_p.append(p);self.nominal_r.append(R)
            J=self.kin.palm_jacobian(side,self.initial)
            self.inverse_j.append(torch.tensor(J.T@np.linalg.inv(J@J.T+.002**2*np.eye(6)),device=self.device,dtype=torch.float32))
            center=planner.resting_center(side,self.initial);contact_centers.append(center.tolist())
            plan=planner.solve(side,center,self.initial)
            if plan is None or not plan['geometry_feasible']:raise RuntimeError('Nominal opposing grasp is infeasible')
            self.nominal_grasp.append(torch.tensor([plan['targets_rad'][self.jnames[i]] for i in self.hand_ids[len(self.nominal_grasp)]],device=self.device))
        self.nominal_grasp_cpu=[q.cpu().numpy() for q in self.nominal_grasp]
        self.hand_joint_ids=torch.tensor(self.hand_ids,device=self.device)
        self.non_thumb_masks=torch.tensor([['thumb' not in self.jnames[i] for i in ids] for ids in self.hand_ids],device=self.device)
        self.ready_hand_pose=self.robot.data.default_joint_pos.torch[0,self.hand_joint_ids].clone()
        if self.response_profile=='fast':
            self.ready_hand_pose+=self.response_settings['ready_finger_fraction']*(torch.stack(self.nominal_grasp)-self.ready_hand_pose)*self.non_thumb_masks
            self.ready_hand_pose[:,1]=self.response_settings['ready_thumb_pitch_rad']
        self.throw_workspace=ThrowWorkspace(self.kin,self.initial,self.scene_cfg,contact_centers)
        self.resting_centers_local=np.array(contact_centers)
        (self.log_dir/'mesh_resting_centers.json').write_text(json.dumps(contact_centers,indent=2))
        (self.log_dir/'throw_workspace_validation.json').write_text(json.dumps(self.throw_workspace.validation,indent=2))
        torque=self.kin.gravity_torques(self.initial)
        self.default_effort=torch.tensor([[torque.get(n,0.) for n in self.jnames]],device=self.device).repeat(num_envs,1)
        self.efforts=self.default_effort.clone()
        self.max_episode_length=360
        self.config={'name':'LKLS73CameraCatchRetentionResidual-v3','stage':'offcenter_mesh_catch_retention','num_envs':num_envs,
            'physics_hz':self.scene_cfg['physics_hz'],'policy_hz':60,'max_episode_seconds':6,'action_dim':8,'physics_replication':False,'action_slew_per_step':.1,
            'arm_target_tracking':'PhysX drive velocity limit 3rad/s; no additional baseline target slew',
            'closure_trigger':'any 120Hz hand-ball contact peak >0.03N starts closure next 60Hz tick; image visibility cannot block closure',
            'closure_command_duration_s':.45,'finger_target_slew_rad_s':.9,
            'closure_version':'visual_preclose_tactile_finish_v2',
            'preclose_rule':'RGB-D predicted arrival <=120ms, palm target error <30mm, image age <=100ms; at most 20% closure before tactile contact',
            'action_contract':'per side: world palm rx residual .06rad, ry .06rad, z .008m, closure .12rad',
            'exposure_tracking_version':'per_env_physics_clock_v2',
            'observation_source':'rendered RGB-D sphere fit/filter, joint state, contacts, side, phase, last action',
            'ground_truth_usage':'reset, reward, termination, diagnostics only','camera_resolution':camera_resolution,
            'dynamics_profile':'mirror_left','armature_kg_m2':{'arm':.02,'hand':.001,'mimic':.001},
            'robot_max_depenetration_velocity_m_s':.5,'ball_attachment_used':False,'continuous_toss_implemented':False,
            'throw_distribution':'50% front + 50% receiver-side throws, >=0.26m horizontal flight, >=0.75m/s horizontal velocity, 55-105mm landing offset, IK and full-ball camera gates',
            'throw_version':'front_side_v2',
            'friction_geometry':'original imported hand collision meshes, zero added shapes',
            'grasp_rule':'physical original hand-ball contact, no deep penetration, low relative speed and <8mm drift, continuous 1s including 0.3s under tilted palm; support accepted, force closure not required',
            'controller_response_profile':self.response_profile,'response_parameters':self.response_settings if self.response_profile!='legacy' else {},
            'reward_version':3,'legacy_checkpoint_compatible':False,'sampling_optimization':'batch_host_state_gpu_pixel_compaction_analytic_fit_v1',
            'baseline':'RGB-D ballistic landing prediction, live palm Jacobian translation/orientation servo, original STL opposing grasp'}
        if self.response_profile!='legacy':
            self.config['arm_target_tracking']='image-derived arrival-time joint velocity plus bounded live-Jacobian position correction, 3rad/s drive limit'
        if self.response_profile=='fast':
            self.config.update(closure_command_duration_s=self.response_settings['closure_duration_s'],
                finger_target_slew_rad_s=self.response_settings['finger_target_speed_rad_s'],closure_version='predictive_ready_tactile_linear_v3',
                preclose_rule='image-derived arrival <=200ms, palm alignment <40mm; up to45% of remaining closure; thumb entry clearance preserved',
                contact_arm_target='continue to the predicted intercept after first contact, instead of freezing at a glancing contact')
        if self.policy_version=='v4':
            self.config.update(name=V4_NAME,action_dim=14,reward_version=4,policy_version='v4',
                curriculum_level=self.curriculum_level,evaluation_full_throw_distribution=not self.training_mode,
                action_contract='per hand [rx .06rad, ry .06rad, z .008m, closure .12rad, x/y .020m per unit bounded at +/-.200m per axis, preclose time .040s]',
                **XY_CONTRACT,
                observation_contract='legacy82 prefix with mapped previous actions + target error3 + TTC1 + palm velocity3 + relative visual velocity3 + prediction valid1 + saturation1 + new previous actions6',
                reward_shaping='gamma*potential(next)-potential(previous), gamma=.99, terminal potential zero; no repeatable hold bonus')
        self.cfg=self.config
        self.config.update(tracking_profile=tracking_profile,tracking_revision=TRACKING_REVISION if tracking_profile=='predictive' else 'fast_legacy',
            xy_slew_m_per_step=.010 if tracking_profile=='predictive' else .002,
            visual_motion_model='ballistic_before_tactile_contact_constant_velocity_after_contact' if tracking_profile=='predictive' else 'ballistic',
            visual_state_pair='filtered_position_and_velocity_at_same_exposure' if tracking_profile=='predictive' else 'raw_position_filtered_velocity')
        self.config.update(grip_profile=grip_profile,grip_revision=GRIP_REVISION if grip_profile=='soft_catch' else ('early_fingers_static_intercept_v1' if grip_profile=='early_close' else 'fast_linear_closure'),
            reception_extent_m=.035 if grip_profile=='soft_catch' else 0.,reception_window_s=.18 if grip_profile=='soft_catch' else 0.)
        if grip_profile in ('soft_catch','early_close'):
            self.config.update(preclose_rule='240ms image arrival, alignment<50mm; fingers up to55% remaining closure, thumb capped at45%',
                closure_version=self.config['grip_revision'],contact_arm_target='bounded visual reception for180ms, then retain final catch target' if grip_profile=='soft_catch' else 'continue to predicted intercept; no added reception motion')
        if grip_profile=='pose_finish':
            self.config.update(grip_revision=POSE_FINISH_REVISION,contact_arm_target='finish the last valid predicted receiving rotation, then hold; no added Cartesian reception motion',
                contact_pose_correction_max_rad=.35,contact_settling_window_s=.12)
        self.config.update(training_profile=training_profile,
            training_sampling_target=ROBUST_DIRECTION_TARGET if training_profile in CONSERVATIVE_PROFILES else {'front':.5,'left_side':.25,'right_side':.25},
            receiving_right_hand_probability=.60 if training_profile in CONSERVATIVE_PROFILES else .50,
            evaluation_sampling='original_uniform_receiver_50pct_front_50pct_receiver_side',
            perception_augmentation={'measurement_std_m':[.0005,.0005,.001],'drop_probability':.02} if self.training_mode and self.curriculum_level>=2 else {'measurement_std_m':[0.,0.,0.],'drop_probability':0.})
        self.config.update(curriculum_mode='mixed_all' if training_mode and training_profile==MIXED_PROFILE else 'single_level',
            curriculum_level_probabilities={str(i):1/3 for i in range(3)} if training_mode and training_profile==MIXED_PROFILE else {str(self.curriculum_level):1.},
            perception_augmentation_scope='training_level_2_episodes_only')
        if training_profile in CONSERVATIVE_PROFILES:self.config['ppo_continuation_parameters']=dict(ROBUST_PPO)
        self.reset(torch.arange(num_envs,device=self.device))
        self.render_and_track()
        obs=self.get_observations();self.num_obs=obs['policy'].shape[-1];self.config['observation_dim']=self.num_obs

    def _sync_camera(self):
        pos=self.robot.data.body_link_pos_w.torch[:,self.base_id].cpu().numpy()
        quat=self.robot.data.body_link_quat_w.torch[:,self.base_id].cpu().numpy()
        for e,(parent,op) in enumerate(self.rigs):
            q=quat[e];M=Gf.Matrix4d(1.);M.SetRotate(Gf.Quatd(float(q[3]),Gf.Vec3d(*map(float,q[:3]))));M.SetTranslateOnly(Gf.Vec3d(*map(float,pos[e])))
            P=UsdGeom.Xformable(self.sim.stage.GetPrimAtPath(parent)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            op.Set(M*P.GetInverse())
        self.sim.render_context.reset_scene_state_cadence()

    def render_and_track(self):
        self._sync_camera();self.sim.render();self.camera.update(0,force_recompute=True)
        self.exposure_clock.record_capture(self.sim.get_physics_step_count())
        self._track()

    def _track(self):
        d=self.camera.data
        if self.tracking_profile=='predictive':self.exposure_batch.set_contact_mode((self.capture_time>=0).cpu().numpy())
        timestamps=self.exposure_clock.timestamps
        def fetch_measurements(ids):
            selected=torch.as_tensor(ids,device=self.device,dtype=torch.long)
            pixels=sparse_red_pixels(d.output['rgb'].torch[selected,...,:3],
                d.output['distance_to_image_plane'].torch[selected]).cpu().numpy()
            K=d.intrinsic_matrices.torch[selected].cpu().numpy()
            positions=d.pos_w.torch[selected].cpu().numpy();quats=d.quat_w_ros.torch[selected].cpu().numpy()
            rows=pixels[:,0].astype(int);bounds=np.searchsorted(rows,np.arange(len(ids)+1))
            measurements=[]
            for row,e in enumerate(ids):
                candidates=pixels[bounds[row]:bounds[row+1]]
                measurements.append(self.exposure_batch.trackers[e].measure_sparse(candidates[:,1],candidates[:,2],
                    candidates[:,3],K[row],positions[row],quats[row]))
            if self.policy_version=='v4' and self.training_mode:
                for row,m in enumerate(measurements):
                    measurements[row]=augment_measurement(m,self.sensor_rng,self.episode_level_host[int(ids[row])])
            return measurements
        ids=self.exposure_batch.update(timestamps,control_time=self.exposure_clock.control_time(self.sim.get_physics_step_count()),
            fetch_measurements=fetch_measurements)
        if len(ids):
            selected=torch.as_tensor(ids,device=self.device,dtype=torch.long)
            self.vision[selected]=torch.tensor(self.exposure_batch.position[ids],device=self.device,dtype=self.vision.dtype)
            self.visual_velocity[selected]=torch.tensor(self.exposure_batch.velocity[ids],device=self.device,dtype=self.visual_velocity.dtype)
            accepted=ids[self.exposure_batch.visible[ids]]
            if len(accepted):
                # Object truth is used only for error reporting, after image processing.
                selected_valid=torch.as_tensor(accepted,device=self.device,dtype=torch.long)
                truth=self.ball.data.root_pos_w.torch[selected_valid].cpu().numpy()
                self.vision_error_sum+=float(np.linalg.norm(self.exposure_batch.position[accepted]-truth,axis=1).sum())
                self.vision_count+=len(accepted)
        self.visible.copy_(torch.as_tensor(self.exposure_batch.visible,device=self.device))
        self.age.copy_(torch.as_tensor(self.exposure_batch.age,device=self.device,dtype=self.age.dtype))
        self.vision_samples=self.exposure_batch.samples

    def _force(self,accumulate=False):
        vectors=torch.stack([as_torch(view.get_contact_force_matrix(dt=self.physics_dt)).reshape(len(names),-1,3).sum(1)
                             for view,names in zip(self.contacts,self.contact_names)])
        totals,features,peak=reduce_contact_forces(vectors,self.contact_side_masks,self.contact_finger_masks)
        self._last_contact_vectors=vectors
        self.raw_contact_peak_n=max(self.raw_contact_peak_n,float(peak))
        self.total_contact.copy_(torch.maximum(self.total_contact,totals) if accumulate else totals)
        self.force_features.copy_(torch.maximum(self.force_features,features) if accumulate else features)

    def hands(self):
        p=self.robot.data.body_link_pos_w.torch[:,self.hbody];q=self.robot.data.body_link_quat_w.torch[:,self.hbody]
        return p,q

    def _sample_training_throw(self,side,level=None):
        profile=getattr(self,'training_profile','legacy')
        level=self.curriculum_level if level is None else level
        if self.policy_version!='v4' or not self.training_mode or (level>=2 and profile not in CONSERVATIVE_PROFILES):
            return self.throw_workspace.sample(self.rng,side)
        kind=choose_throw_kind(self.rng,side,profile)
        radius=.080001 if level==0 else .105001
        minimum_flight=.40 if level==0 else (.37 if level==1 else .34)
        nominal=self.throw_workspace.palms[side]+self.throw_workspace.contact_offsets[side]
        for _ in range(500):
            sample=self.throw_workspace.sample(self.rng,side)
            if sample[5]==kind and sample[3]>=minimum_flight and np.linalg.norm(sample[2][:2]-nominal[:2])<=radius:return sample
        raise RuntimeError('No curriculum throw found; refusing a changed throw direction or vertical fallback')

    def reset(self,ids):
        if not len(ids):return
        # Both fixed bases stay at their initial pose; only robot joints and ball are reset.
        self.side[ids]=sample_receivers(len(ids),self.device,self.training_profile if self.training_mode else 'legacy')
        levels=sample_episode_levels(len(ids),self.training_profile if self.training_mode else 'legacy',self.curriculum_level)
        self.episode_level_host[ids.cpu().numpy()]=levels
        self.sampled_level_counts+=np.bincount(levels,minlength=3)
        q=self.robot.data.default_joint_pos.torch[ids].clone()
        if self.response_profile=='fast':
            local=torch.arange(len(ids),device=self.device);joints=self.hand_joint_ids[self.side[ids]]
            q[local[:,None],joints]=self.ready_hand_pose[self.side[ids]]
            for mimic in self.task['mimic_joints']:
                q[:,self.jnames.index(mimic['name'])]=q[:,self.jnames.index(mimic['joint'])]*float(mimic['multiplier'])+float(mimic.get('offset',0))
        self.robot.write_joint_state_to_sim_index(position=q,velocity=torch.zeros_like(q),env_ids=ids)
        self.velocity_targets[ids]=0
        self.robot.actuators.target_command.set_velocity_index(value=self.velocity_targets[ids],env_ids=ids)
        self.efforts[ids]=self.default_effort[ids]
        self.target[ids]=q;self.robot.actuators.target_command.set_position_index(value=q,env_ids=ids)
        self.robot.actuators.target_command.set_effort_index(value=self.efforts[ids],env_ids=ids)
        pose=self.ball.data.default_root_state.torch[ids,:7].clone()
        samples=[self._sample_training_throw(int(s),int(level)) for s,level in zip(self.side[ids].cpu().tolist(),levels)]
        local=torch.tensor(np.array([s[0] for s in samples]),device=self.device,dtype=torch.float32)
        pose[:,:3]=local+self.origins[ids]
        velocity=torch.zeros((len(ids),6),device=self.device)
        velocity[:,:3]=torch.tensor(np.array([s[1] for s in samples]),device=self.device,dtype=torch.float32)
        self.throw_start[ids]=local;self.throw_goal[ids]=torch.tensor(np.array([s[2] for s in samples]),device=self.device,dtype=torch.float32)
        self.throw_flight[ids]=torch.tensor([s[3] for s in samples],device=self.device)
        self.throw_launch_velocity[ids]=velocity[:,:3]
        for row,e in enumerate(ids.cpu().tolist()):
            self.throw_kind[e]=samples[row][5];self.throw_counts[samples[row][5]]+=1
            s=int(self.side[e]);self.catch_palm_target[e]=torch.tensor(self.nominal_p[s],device=self.device)+self.origins[e]
            self.predicted_intercept[e]=self.catch_palm_target[e]
        self.ball.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        self.ball.write_root_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
        self.ball.permanent_wrench_composer.reset(env_ids=ids)
        self.episode_length_buf[ids]=0;self.actions[ids]=0;self.previous_actions[ids]=0
        self.capture_time[ids]=-1;self.grasp_start[ids]=-1;self.hold[ids]=0;self.returns[ids]=0;self.age[ids]=1;self.visible[ids]=False
        self.grasp_curl_bias[ids]=0
        self.preclose_fraction[ids]=0;self.prepare_time[ids]=-1;self.predicted_contact_time[ids]=-1;self.caught_tilt[ids]=0
        self.contact_streak[ids]=0;self.grasp_origin_q[ids]=0
        self.tilt_hold[ids]=0;self.best_hold[ids]=0;self.capture_rewarded[ids]=False
        self.force_features[ids]=0;self.total_contact[ids]=0
        self.hold_center[ids]=0;self.reward_previous_center[ids]=0
        self.local_velocity[ids]=0;self.previous_local[ids]=0
        self.palm_velocity[ids]=0;self.reward_potential[ids]=0
        self.reception_anchor[ids]=0;self.reception_feedforward[ids]=0;self.reception_shift[ids]=0
        self.intercept_tilt[ids]=torch.tensor([0.,-.18,0.],device=self.device)
        self.previous_palm[ids]=self.origins[ids]+torch.tensor(np.array(self.nominal_p),device=self.device,dtype=torch.float32)[self.side[ids]]
        self.exposure_batch.reset(ids.cpu().numpy())
        self.vision[ids]=0.;self.visual_velocity[ids]=0.
        self.sim.forward()

    def get_observations(self):
        hp,hq=self.hands();idx=torch.arange(self.num_envs,device=self.device)
        hand=hp[idx,self.side];palm=hand+quat_apply(hq[idx,self.side],torch.tensor([.025,0,.07],device=self.device).expand(self.num_envs,3))
        visual=torch.where((self.age<.15)[:,None],self.vision,palm)
        q=self.robot.data.joint_pos.torch[:,self.control_ids];qd=self.robot.data.joint_vel.torch[:,self.control_ids]
        qref=self.robot.data.default_joint_pos.torch[:,self.control_ids]
        phase=torch.stack((self.capture_time>=0,self.grasp_start>=0),-1).float()
        obs=torch.cat((q-qref,qd*.1,(visual-palm)*5,self.visual_velocity*.5,
            self.visible[:,None].float(),self.age.clamp(0,1)[:,None],torch.nn.functional.one_hot(self.side,2).float(),
            phase,self.force_features.clamp(0,3),legacy_actions(self.actions)),-1)
        if self.policy_version=='v4':
            valid=(self.predicted_contact_time>0)&(self.age<=.10)&(self.capture_time<0)
            error=torch.where(valid[:,None],self.catch_palm_target-palm,0.)
            arm=torch.tensor(self.arm_ids,device=self.device)[self.side]
            saturation=(self.robot.data.joint_vel.torch[idx[:,None],arm].abs()>2.85).float().mean(-1)
            extra=visual_features(error,torch.where(valid,self.predicted_contact_time,0.),self.palm_velocity,
                self.visual_velocity-self.palm_velocity,valid,saturation,self.actions)
            obs=torch.cat((obs,extra),dim=-1)
        if not torch.isfinite(obs).all():raise RuntimeError('Nonfinite actor observation; refusing corrupted rollout')
        return TensorDict({'policy':obs.clamp(-10,10)},batch_size=[self.num_envs])

    def _command(self,actions):
        hp,hq=self.hands();qactual=self.robot.data.joint_pos.torch
        fast_arm=getattr(self,'response_profile','legacy')!='legacy'
        fast_grip=getattr(self,'response_profile','legacy')=='fast'
        stride=getattr(self,'action_stride',4)
        predictive=getattr(self,'tracking_profile','fast')=='predictive'
        soft=getattr(self,'grip_profile','fast')=='soft_catch'
        finish_pose=getattr(self,'grip_profile','fast')=='pose_finish'
        early_grip=getattr(self,'grip_profile','fast') in ('soft_catch','early_close')
        if hasattr(self,'velocity_targets'):self.velocity_targets.zero_()
        hp_cpu=hp.cpu().numpy();hq_cpu=hq.cpu().numpy();vision=self.vision.cpu().numpy();velocity=self.visual_velocity.cpu().numpy()
        origin=self.origins.cpu().numpy();q_cpu=qactual.cpu().numpy();actions_cpu=actions.cpu().numpy()
        # One transfer for scalar state, instead of synchronizing for each environment.
        fields=['side','episode_length_buf','visible','age','contact_streak','capture_time','grasp_start',
                'grasp_curl_bias','prepare_time','preclose_fraction','predicted_contact_time']
        host=torch.stack([getattr(self,k).float() for k in fields],dim=1).cpu().numpy().astype(float)
        state={k:host[:,i] for i,k in enumerate(fields)}
        totals=self.total_contact.cpu().numpy();caught_tilts=self.caught_tilt.cpu().numpy()
        catch_targets=self.catch_palm_target.cpu().numpy();intercepts=self.predicted_intercept.cpu().numpy()
        previous_tilts=self.intercept_tilt.cpu().numpy() if predictive else None
        anchors=self.reception_anchor.cpu().numpy() if soft else None
        receive_velocity=np.zeros((self.num_envs,3)) if soft else None
        default=self.robot.data.default_joint_pos.torch
        reference_rotation=np.array(self.scene_cfg['hand_rotation_world'])
        side=state['side'].astype(int);elapsed=state['episode_length_buf']*self.dt
        rotations=Rotation.from_quat(hq_cpu[np.arange(self.num_envs),side]).as_matrix()
        capture_palms=(hp[torch.arange(self.num_envs,device=self.device),self.side]
                       +quat_apply(hq[torch.arange(self.num_envs,device=self.device),self.side],
                           torch.tensor([.025,0,.07],device=self.device).expand(self.num_envs,3))).cpu().numpy()
        desired_rotations=[];desired_positions=[];newly_captured=[];events=[]
        for e in range(self.num_envs):
            s=side[e];t=elapsed[e];R=rotations[e]
            a=actions_cpu[e,s*stride:(s+1)*stride]
            center=R.T@(vision[e]-hp_cpu[e,s]);fresh=bool(state['visible'][e])
            streak,close_now=contact_closure_step(int(state['contact_streak'][e]),float(totals[e,s]),t)
            state['contact_streak'][e]=streak
            if close_now and state['grasp_start'][e]<0:
                state['capture_time'][e]=np.float32(t);state['grasp_start'][e]=np.float32(t)
                if not fast_grip:catch_targets[e]=capture_palms[e]
                if soft:anchors[e]=catch_targets[e]
                caught_tilts[e]=Rotation.from_matrix(R@reference_rotation.T).as_rotvec()
                if finish_pose:caught_tilts[e]=contact_rotation_target(R,reference_rotation,previous_tilts[e],state['predicted_contact_time'][e],state['age'][e])
                state['grasp_curl_bias'][e]=np.float32(np.clip((.078-center[2])*6.,-.12,.16)) if fresh and abs(center[1])<.04 and .03<center[2]<.12 else 0.
                newly_captured.append((e,s,center.copy()))
                self.closure_events+=1;self.closure_events_by_side[s]+=1
                if self.closure_events<=50:
                    events.append({'event':'physical_contact_started_closure','env':e,'side':int(s),'episode_time_s':float(t),
                        'hand_contact_n':float(totals[e,s]),'camera_visible':fresh,
                        'preclose_fraction':float(state['preclose_fraction'][e]),
                        'preparation_lead_s':float(t-state['prepare_time'][e]) if state['prepare_time'][e]>=0 else None,
                        'joint_names':[self.jnames[j] for j in self.hand_ids[s]],
                        'initial_joint_rad':q_cpu[e,self.hand_ids[s]].tolist(),
                        'grasp_target_rad':self.nominal_grasp_cpu[s].tolist()})
            captured=state['capture_time'][e]>=0
            tilt=previous_tilts[e].copy() if predictive else np.array([0.,-.18,0.]);pos=np.zeros(3)
            if predictive and not captured:
                state['predicted_contact_time'][e]=max(-1.,state['predicted_contact_time'][e]-self.dt)
            age=state['age'][e];palm_now=hp_cpu[e,s]+R@np.array([.025,0,.07]);prep=0.
            if age<=.10 and not captured and not self.contact_probe:
                height=self.nominal_p[s][2]+self.throw_workspace.contact_offsets[s][2]
                visual_position=vision[e]-origin[e]+velocity[e]*age+np.array([0,0,-.5*9.81*age*age])
                visual_velocity=velocity[e]+np.array([0,0,-9.81*age])
                prediction=predict_landing(visual_position,visual_velocity,height)
                if prediction is not None:
                    tau,point=prediction;impact_velocity=visual_velocity+np.array([0,0,-9.81*tau])
                    tilt=incoming_hand_tilt(impact_velocity)
                    if predictive:previous_tilts[e]=tilt
                    incoming_rotation=Rotation.from_rotvec(tilt).as_matrix()@reference_rotation
                    offset=incoming_rotation@(self.resting_centers_local[s]-np.array([.025,0,.07]))
                    target=point-offset
                    if .25<target[0]<.49 and .10<abs(target[1])<.31:
                        catch_targets[e]=target+origin[e];intercepts[e]=point+origin[e]
                        state['predicted_contact_time'][e]=np.float32(tau)
                        options=({'window_s':self.response_settings['preclose_window_s'],'alignment_m':self.response_settings['preclose_alignment_m'],
                            'maximum_fraction':self.response_settings['preclose_maximum_fraction']} if fast_grip else {})
                        if stride==7:
                            options['window_s']+=float(a[6])*.04
                            options['maximum_fraction']=float(np.clip(options['maximum_fraction']+a[6]*.10,.35,.55))
                        correction=np.r_[a[4:6]*XY_RESIDUAL_SCALE_M,0.] if stride==7 else np.zeros(3)
                        prep=anticipated_closure_fraction(tau,np.linalg.norm(target+origin[e]+correction-palm_now),age,**options)
            if not captured:
                if prep>0 and state['prepare_time'][e]<0:
                    state['prepare_time'][e]=np.float32(t);self.preparation_events+=1;self.preparation_events_by_side[s]+=1
                if prep==0 and state['prepare_time'][e]>=0 and t-state['prepare_time'][e]<.25:prep=state['preclose_fraction'][e]
                state['preclose_fraction'][e]=np.float32(prep)
            base_rotation=Rotation.from_rotvec(tilt).as_matrix()@reference_rotation
            if captured:
                caught=Rotation.from_rotvec(caught_tilts[e]).as_matrix()@reference_rotation
                probe=Rotation.from_rotvec([0,.85,0]).as_matrix()@reference_rotation
                blend=np.clip((t-state['grasp_start'][e]-1.25)/1.4,0,1)
                turn=Rotation.from_matrix(probe@caught.T).as_rotvec()*blend
                base_rotation=Rotation.from_rotvec(turn).as_matrix()@caught
            if soft and captured:
                offset=base_rotation@(self.resting_centers_local[s]-np.array([.025,0,.07]))
                catch_targets[e],receive_velocity[e]=reception_target(anchors[e],catch_targets[e],vision[e],velocity[e],age,
                    t-state['capture_time'][e],offset,self.dt)
            residual_tilt=np.r_[a[:2]*.06,0.];pos[2]=a[2]*.008
            if stride==7:pos[:2]=a[4:6]*XY_RESIDUAL_SCALE_M
            desired_rotations.append(Rotation.from_rotvec(residual_tilt).as_matrix()@base_rotation)
            desired_positions.append(catch_targets[e]+np.asarray(pos,dtype=np.float32))
        for key in fields[4:]:
            getattr(self,key).copy_(torch.as_tensor(state[key],device=self.device,dtype=getattr(self,key).dtype))
        self.caught_tilt.copy_(torch.as_tensor(caught_tilts,device=self.device));self.catch_palm_target.copy_(torch.as_tensor(catch_targets,device=self.device))
        self.predicted_intercept.copy_(torch.as_tensor(intercepts,device=self.device))
        if predictive:self.intercept_tilt.copy_(torch.as_tensor(previous_tilts,device=self.device))
        if soft:
            self.reception_anchor.copy_(torch.as_tensor(anchors,device=self.device,dtype=torch.float32))
            self.reception_feedforward.copy_(torch.as_tensor(receive_velocity,device=self.device,dtype=torch.float32))
            shifts=np.where((state['capture_time']>=0)[:,None],catch_targets-anchors,0.)
            self.reception_shift.copy_(torch.as_tensor(shifts,device=self.device,dtype=torch.float32))
        if events:
            with (self.log_dir/'closure_events.jsonl').open('a') as file:
                file.writelines(json.dumps(event)+'\n' for event in events)
        if newly_captured:
            rows=torch.tensor([x[0] for x in newly_captured],device=self.device)
            joints=self.hand_joint_ids[self.side[rows]]
            self.previous_local[rows]=torch.as_tensor(np.array([x[2] for x in newly_captured]),device=self.device,dtype=torch.float32)
            self.grasp_origin_q[rows]=qactual[rows[:,None],joints]
        errors=Rotation.from_matrix(np.asarray(desired_rotations)@rotations.transpose(0,2,1)).as_rotvec()
        error_tensor=torch.tensor(errors,device=self.device,dtype=torch.float32)
        position_tensor=torch.tensor(np.array(desired_positions),device=self.device,dtype=torch.float32)
        # Finger interpolation and writes are now batched for each hand.
        duration=self.response_settings['closure_duration_s'] if fast_grip else .45
        fractions=np.clip((elapsed-state['grasp_start'])/duration,0,1)
        if not fast_grip:fractions=.5*(1-np.cos(np.pi*fractions))
        fraction_tensor=torch.tensor(fractions,device=self.device,dtype=torch.float32)
        Jall=self.robot.data.body_link_jacobian_w.torch
        for s in range(2):
            selected=(self.side==s).nonzero(as_tuple=False).flatten()
            if not len(selected):continue
            hids=self.hand_joint_ids[s];rows=selected[:,None];fraction=fraction_tensor[selected,None]
            start=self.ready_hand_pose[s].expand(len(selected),-1) if fast_grip else default[rows,hids[None,:]]
            nominal=self.nominal_grasp[s]
            latched=self.grasp_origin_q[selected]
            closed=latched+fraction*(nominal-latched)
            closed+=fraction*self.grasp_curl_bias[selected,None]*self.non_thumb_masks[s]
            closed+=fraction*actions[selected,s*stride+3,None]*.12
            preparation=self.preclose_fraction[selected,None]
            if early_grip:preparation=torch.where(self.non_thumb_masks[s][None,:],preparation,preparation.clamp(max=.45))
            prepared=start+preparation*(nominal-start)
            hand=torch.where((self.grasp_start[selected]>=0)[:,None],closed,prepared)
            finger_speed=self.response_settings['finger_target_speed_rad_s'] if fast_grip else .9
            previous=self.target[rows,hids[None,:]].clone()
            self.target[rows,hids[None,:]]+=(hand-previous).clamp(-finger_speed*self.dt,finger_speed*self.dt)
            if fast_grip:self.velocity_targets[rows,hids[None,:]]=((self.target[rows,hids[None,:]]-previous)/self.dt).clamp(-1.,1.)
            arm=self.arm_ids[s]
            J=joint_body_jacobian(Jall[selected],self.hbody[s],len(self.robot.body_names),len(self.jnames),arm).clone()
            offset=quat_apply(hq[selected,s],torch.tensor([.025,0,.07],device=self.device).expand(len(selected),3))
            J[:,:3]+=torch.linalg.cross(J[:,3:].transpose(1,2),offset[:,None,:]).transpose(1,2)
            palm=hp[selected,s]+offset
            unweighted=J.clone()
            residual=torch.cat((position_tensor[selected]-palm,error_tensor[selected]*.12),dim=1)
            J[:,3:]*=.12
            delta=J.transpose(1,2)@torch.linalg.solve(J@J.transpose(1,2)+.012**2*torch.eye(6,device=self.device),residual[:,:,None])
            delta=delta.squeeze(-1)
            arm_index=torch.tensor(arm,device=self.device);qarm=qactual[selected][:,arm]
            arm_limits=self.limits[selected[:,None],arm_index[None,:]]
            if predictive:
                approach=position_priority_delta(unweighted,position_tensor[selected]-palm,error_tensor[selected],qarm,arm_limits)
                delta=torch.where((self.capture_time[selected]<0)[:,None],approach,delta)
            self.target[selected[:,None],arm_index[None,:]]=qarm+delta.clamp(-.045,.045)
            if fast_arm:
                velocity=arrival_joint_velocity(delta,self.predicted_contact_time[selected],self.capture_time[selected]>=0)
                if predictive:
                    early=early_arrival_velocity(delta,self.predicted_contact_time[selected],qarm,arm_limits)
                    velocity=torch.where((self.capture_time[selected]<0)[:,None],early,velocity)
                if soft:velocity=reception_joint_velocity(velocity,unweighted[:,:3],self.reception_feedforward[selected],qarm,arm_limits)
                if finish_pose:
                    since_contact=self.episode_length_buf[selected]*self.dt-self.capture_time[selected]
                    velocity=settling_joint_velocity(velocity,delta,since_contact,self.capture_time[selected]>=0,qarm,arm_limits)
                self.velocity_targets[selected[:,None],torch.tensor(arm,device=self.device)[None,:]]=velocity
        self.efforts.copy_(joint_gravity_forces(self.robot.data.gravity_compensation_forces.torch,len(self.jnames)))
        self.target=torch.maximum(torch.minimum(self.target,self.limits[:,:,1]-.0001),self.limits[:,:,0]+.0001)
        self.robot.actuators.target_command.set_position_index(value=self.target)
        if hasattr(self,'velocity_targets'):self.robot.actuators.target_command.set_velocity_index(value=self.velocity_targets)
        self.robot.actuators.target_command.set_effort_index(value=self.efforts)

    def step(self,actions):
        step_begin=time.monotonic()
        self.previous_actions=self.actions.clone()
        requested,self.actions=filter_residual_actions(actions,self.previous_actions,.5 if self.tracking_profile=='predictive' else .1)
        self._command(self.actions)
        command_end=time.monotonic()
        # Configured physical substeps, sixty Hz rendered perception and residual control.
        self.scene.write_data_to_sim()
        self.total_contact.zero_();self.force_features.zero_()
        for k in range(self.decimation):
            self.sim.step(render=False);self.scene.update(self.physics_dt)
            self._force(accumulate=True)
            if k!=self.decimation-1:continue
            q=self.robot.data.joint_pos.torch;qd=self.robot.data.joint_vel.torch
            bp=self.robot.data.body_link_pos_w.torch[:,self.base_id]
            expected=self.origins+torch.tensor([0,0,1.25],device=self.device)
            invalid=~torch.isfinite(q).all(-1)|~torch.isfinite(qd).all(-1)|~torch.isfinite(bp).all(-1)|((bp-expected).norm(dim=-1)>.02)
            if invalid.any():
                ids=invalid.nonzero(as_tuple=False).flatten()
                self.physics_failures+=len(ids)
                failure={'step':self.total_steps,'env_ids':ids.cpu().tolist(),'base_world_m':bp[ids].cpu().tolist(),
                    'joint_positions':q[ids].cpu().tolist(),'applied_actions':self.actions[ids].cpu().tolist()}
                (self.log_dir/f'physics_failure_{self.physics_failures}.json').write_text(json.dumps(failure,indent=2))
                if self.physics_failures>=3:
                    raise RuntimeError('Repeated PhysX instability; refusing continued training on reset failures')
                self.episodes+=len(ids);self.recent.extend([False]*len(ids))
                root=self.robot.data.default_root_state.torch[ids,:7].clone();root[:,:3]+=self.origins[ids]
                self.robot.write_root_pose_to_sim_index(root_pose=root,env_ids=ids)
                self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(ids),6),device=self.device),env_ids=ids)
                self.reset(ids)
                self.render_and_track()
                self.total_steps+=self.num_envs
                reward=torch.zeros(self.num_envs,device=self.device);reward[ids]=-15.
                return self.get_observations(),reward,invalid.long(),{'time_outs':torch.zeros_like(invalid),
                    'log':{'Physics/invalid_episodes':float(len(ids)),'Episode/success_rate':0.}}
        physics_end=time.monotonic()
        self.render_and_track();vision_end=time.monotonic()
        # Kit publishes contact reports during the app/render event pump.
        # Also read after that pump; substep reads alone can be empty.
        self._force(accumulate=True)
        force_end=time.monotonic()
        self.episode_length_buf+=1
        hp,hq=self.hands();idx=torch.arange(self.num_envs,device=self.device)
        for s in range(2):
            closing=(self.side==s)&(self.grasp_start>=0)
            if closing.any():
                motion=(self.robot.data.joint_pos.torch[closing][:,self.hand_ids[s]]-self.grasp_origin_q[closing]).abs().max()
                self.closure_motion_max_rad=max(self.closure_motion_max_rad,float(motion))
        palm=hp[idx,self.side]+quat_apply(hq[idx,self.side],torch.tensor([.025,0,.07],device=self.device).expand(self.num_envs,3))
        self.palm_velocity.copy_((palm-self.previous_palm)/self.dt);self.previous_palm.copy_(palm)
        truth=self.ball.data.root_pos_w.torch;v=self.ball.data.root_lin_vel_w.torch
        dist=(truth-palm).norm(dim=-1);speed=v.norm(dim=-1)
        forces=self.force_features.reshape(self.num_envs,2,5)[idx,self.side]
        local_center=quat_apply_inverse(hq[idx,self.side],truth-hp[idx,self.side])
        relative_speed=(local_center-self.reward_previous_center).norm(dim=-1)/self.dt
        self.reward_previous_center.copy_(local_center)
        thumb=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device);fingers=thumb.clone()
        opposition=torch.zeros(self.num_envs,device=self.device);grip=thumb.clone()
        # Gather contact buffers once per field, not once per environment/scalar.
        vectors_batch=self._last_contact_vectors.cpu().numpy()
        contact_data=[view.get_contact_data(dt=self.physics_dt) for view in self.contacts]
        arrays=[torch.stack([as_torch(data[i]) for data in contact_data]).cpu().numpy() for i in [1,3,4,5]]
        points_batch,separations_batch,counts_batch,starts_batch=arrays
        side_cpu=self.side.cpu().numpy();truth_cpu=truth.cpu().numpy();center_cpu=local_center.cpu().numpy()
        scalar=torch.stack((relative_speed,(local_center-self.hold_center).norm(dim=-1),dist,self.hold.float()),dim=1).cpu().numpy()
        valid=np.zeros(self.num_envs,dtype=bool);thumb_cpu=valid.copy();fingers_cpu=valid.copy();opposition_cpu=np.zeros(self.num_envs)
        self.hold_center.copy_(torch.where((self.hold==0)[:,None],local_center,self.hold_center))
        for e,names in enumerate(self.contact_names):
            prefix='lh_' if side_cpu[e]==0 else 'rh_'
            arguments=(names,vectors_batch[e],points_batch[e],counts_batch[e],starts_batch[e],separations_batch[e],
                       truth_cpu[e],self.scene_cfg['ball_radius_m'],prefix)
            evidence=opposing_contacts(*arguments);retention=retention_contacts(*arguments)
            evidence['retention']=retention;self.contact_evidence[e]=evidence
            thumb_cpu[e]=evidence['thumb'];fingers_cpu[e]=evidence['fingers'];opposition_cpu[e]=evidence['opposition']
            valid[e]=stable_retention(retention['valid'],float(scalar[e,0]),0. if scalar[e,3]==0 else float(scalar[e,1]),float(scalar[e,2]))
        grip=torch.as_tensor(valid,device=self.device);thumb=torch.as_tensor(thumb_cpu,device=self.device)
        fingers=torch.as_tensor(fingers_cpu,device=self.device);opposition=torch.as_tensor(opposition_cpu,device=self.device,dtype=torch.float32)
        self.hold=torch.where(grip,self.hold+1,torch.zeros_like(self.hold))
        normal=quat_apply(hq[idx,self.side],torch.tensor([1.,0,0],device=self.device).expand(self.num_envs,3))
        tilted=normal[:,2]<.75
        self.tilt_hold=torch.where(grip&tilted,self.tilt_hold+1,torch.zeros_like(self.tilt_hold))
        success=(self.hold>=60)&(self.tilt_hold>=18)
        capture=(dist<.065)&(speed<.6)&(self.total_contact[idx,self.side]>.03)
        capture_bonus=(capture&~self.capture_rewarded).float()*5.
        self.capture_rewarded|=capture
        new_best=torch.maximum(self.best_hold,self.hold.clamp(max=60))
        progress=(new_best-self.best_hold).float()*(10./60)
        self.best_hold=new_best
        failed=(truth[:,2]-self.origins[:,2]<.65)|(dist>.7)
        reward=(.02*torch.exp(-40*dist)+capture_bonus+progress+50.*success.float()
            -.02*self.actions.square().mean(-1)-.01*(self.actions-self.previous_actions).square().mean(-1)
            -.01-10.*failed.float())
        timeout=self.episode_length_buf>=self.max_episode_length
        done=success|failed|timeout
        shaping=torch.zeros_like(reward)
        if self.policy_version=='v4':
            expected=hp[idx,self.side]+quat_apply(hq[idx,self.side],torch.tensor(self.resting_centers_local,device=self.device,dtype=torch.float32)[self.side])
            error_xy=(expected-self.throw_goal-self.origins)[:,:2].norm(dim=-1)
            potential=tracking_potential(error_xy,relative_speed,self.capture_time>=0,done)
            shaping=.99*potential-self.reward_potential
            self.reward_potential.copy_(potential);reward+=shaping
        self.returns+=reward
        if self.diagnostic_callback is not None:
            # Read-only pre-reset telemetry; object truth is diagnostic-only.
            self.diagnostic_callback({'grip':grip,'thumb':thumb,'fingers':fingers,
                'opposition':opposition,'distance':dist,'speed':relative_speed,'hold':self.hold,
                'success':success,'failed':failed,'timeout':timeout,'done':done,
                'side':self.side,'visible':self.visible,'capture_time':self.capture_time,
                'grasp_start':self.grasp_start,'reward':reward,'actions':self.actions,
                'previous_actions':self.previous_actions,'requested_actions':requested,
                'forces':forces,'ball_position':truth-self.origins,
                'episode_time_s':self.episode_length_buf*self.dt,
                'palm_position':palm,'ball_velocity':v,'throw_goal':self.throw_goal+self.origins,
                'ball_expected_at_hand':hp[idx,self.side]+quat_apply(hq[idx,self.side],
                    torch.tensor(self.resting_centers_local,device=self.device,dtype=torch.float32)[self.side]),
                'total_contact':self.total_contact[idx,self.side],
                'hand_q':self.robot.data.joint_pos.torch[idx[:,None],self.hand_joint_ids[self.side]],
                'hand_q_nominal':torch.stack(self.nominal_grasp)[self.side],
                'hand_q_default':self.robot.data.default_joint_pos.torch[idx[:,None],self.hand_joint_ids[self.side]],
                'arm_qd':self.robot.data.joint_vel.torch[idx[:,None],torch.tensor(self.arm_ids,device=self.device)[self.side]],
                'arm_target_error':self.target[idx[:,None],torch.tensor(self.arm_ids,device=self.device)[self.side]]-
                    self.robot.data.joint_pos.torch[idx[:,None],torch.tensor(self.arm_ids,device=self.device)[self.side]],
                'reward_components':torch.stack((.02*torch.exp(-40*dist)+shaping,capture_bonus,progress,50.*success.float(),
                    -.02*self.actions.square().mean(-1),-.01*(self.actions-self.previous_actions).square().mean(-1)-.01-10.*failed.float()),dim=-1)})
        extras={'time_outs':timeout,'log':{'Catch/grip_fraction':grip.float().mean(),'Catch/palm_distance_m':dist.mean(),
            'Tracking/valid_fraction':self.visible.float().mean(),'Curriculum/landing_strength':
                float(self.episode_level_host.mean()/2) if self.training_mode and self.training_profile==MIXED_PROFILE else self.curriculum}}
        ids=done.nonzero(as_tuple=False).flatten()
        if len(ids):
            finished_levels=self.episode_level_host[ids.cpu().numpy()]
            self.completed_level_counts+=np.bincount(finished_levels,minlength=3)
            self.level_success_counts+=np.bincount(finished_levels,weights=success[ids].cpu().numpy(),minlength=3).astype(np.int64)
            self.episodes+=len(ids);self.successes+=int(success[ids].sum());self.recent.extend(success[ids].cpu().tolist())
            extras['log'].update({'Episode/success_rate':success[ids].float().mean(),'Episode/return':self.returns[ids].mean()})
            if self.policy_version=='v3' and not self.full_curriculum and len(self.recent)>=50 and np.mean(self.recent)>.6:self.curriculum=min(1.,self.curriculum+.01)
            self.reset(ids)
            # Synchronize reset images before returning the next episode observation.
            self.render_and_track()
        self.total_steps+=self.num_envs
        if self.total_steps%(self.num_envs*20)==0:
            status={'num_envs':self.num_envs,'diagnostic_sample_envs':min(8,self.num_envs),'physics_invalid_episodes':self.physics_failures,'transitions':self.total_steps,'episodes':self.episodes,'successes':self.successes,
                'success_rate':self.successes/max(self.episodes,1),'recent_success_rate':float(np.mean(self.recent)) if self.recent else None,
                'curriculum':self.curriculum,'vision_valid_fraction':self.vision_count/max(self.vision_samples,1),
                'unique_exposure_samples':self.exposure_batch.samples,'duplicate_exposures_skipped':self.exposure_batch.skipped,
                'observation_age_max_s':float(self.age.max()),
                'camera_clock_drift_s':self.exposure_clock.control_time(self.sim.get_physics_step_count())-float(self.camera._timestamp_last_update.numpy()[0]),
                'exposure_tracking_version':'per_env_physics_clock_v2',
                'vision_mean_error_m':self.vision_error_sum/max(self.vision_count,1),'grip_fraction':float(grip.float().mean()),
                'stage':self.config['stage'],'continuous_toss_implemented':False,'task_version':4 if self.policy_version=='v4' else 3,'policy_version':self.policy_version,'curriculum_level':self.curriculum_level,
                'curriculum_mode':self.config['curriculum_mode'],
                'active_episode_level_counts':np.bincount(self.episode_level_host,minlength=3).tolist(),
                'sampled_episode_level_counts':self.sampled_level_counts.tolist(),
                'completed_episode_level_counts':self.completed_level_counts.tolist(),
                'successes_by_level':self.level_success_counts.tolist(),
                'hold_frames':self.hold[:8].cpu().tolist(),'tilted_hold_frames':self.tilt_hold[:8].cpu().tolist(),
                'grip_evidence':self.contact_evidence[:8],
                'throw_start_m':self.throw_start[:8].cpu().tolist(),'throw_goal_m':self.throw_goal[:8].cpu().tolist(),
                'predicted_intercept_m':(self.predicted_intercept[:8]-self.origins[:8]).cpu().tolist(),
                'side':self.side[:8].cpu().tolist(),'capture_time_s':self.capture_time[:8].cpu().tolist(),
                'grasp_start_s':self.grasp_start[:8].cpu().tolist(),
                'throw_kind':self.throw_kind[:8],'throw_counts':self.throw_counts,
                'throw_launch_velocity_m_s':self.throw_launch_velocity[:8].cpu().tolist(),
                'preparation_events':self.preparation_events,'preparation_events_by_side':self.preparation_events_by_side,
                'preclose_fraction':self.preclose_fraction[:8].cpu().tolist(),
                'predicted_time_to_contact_s':self.predicted_contact_time[:8].cpu().tolist(),
                'closure_events':self.closure_events,'closure_events_by_side':self.closure_events_by_side,
                'closure_motion_max_rad':self.closure_motion_max_rad,
                'raw_contact_peak_n':self.raw_contact_peak_n,'active_hand_contact_peak_n':self.total_contact[idx,self.side][:8].cpu().tolist(),
                'active_hand_joint_rad':[self.robot.data.joint_pos.torch[e,self.hand_ids[int(self.side[e])]].cpu().tolist() for e in range(min(8,self.num_envs))],
                'active_hand_target_rad':[self.target[e,self.hand_ids[int(self.side[e])]].cpu().tolist() for e in range(min(8,self.num_envs))],
                'base_local_m':(self.robot.data.body_link_pos_w.torch[:8,self.base_id]-self.origins[:8]).cpu().tolist(),
                'palm_local_m':(palm[:8]-self.origins[:8]).cpu().tolist(),'ball_local_m':(truth[:8]-self.origins[:8]).cpu().tolist(),
                'hand_local_center_m':quat_apply_inverse(hq[idx[:8],self.side[:8]],self.vision[:8]-hp[idx[:8],self.side[:8]]).cpu().tolist(),
                'contact_forces_n':forces[:8].cpu().tolist()}
            tmp=self.log_dir/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(self.log_dir/'status.json')
            print('[CATCH_STATUS] '+json.dumps(status),flush=True)
        finish=time.monotonic()
        self.step_profile={'command_ms':(command_end-step_begin)*1000,
            'physics_ms':(physics_end-command_end)*1000,'render_tracking_ms':(vision_end-physics_end)*1000,
            'force_read_ms':(force_end-vision_end)*1000,'grasp_reset_observation_ms':(finish-force_end)*1000,
            'total_ms':(finish-step_begin)*1000}
        extras['log'].update({'Timing/'+k:v for k,v in self.step_profile.items()})
        return self.get_observations(),reward,done.long(),extras

    def save_camera(self,path):
        from PIL import Image
        Image.fromarray(self.camera.data.output['rgb'].torch[0,...,:3].cpu().numpy().astype('uint8')).save(path)
