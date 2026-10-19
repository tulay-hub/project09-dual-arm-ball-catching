"""MuJoCo adapter for the original image-based catch policy and controller."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast,hashlib,json,time
import mujoco
import numpy as np
import torch
from tensordict import TensorDict
from scipy.spatial.transform import Rotation
from build_mujoco_catch import build_model
from kinematics import ROOT,RobotKinematics
from grasp import SphereGraspPlanner
from throw_workspace import ThrowWorkspace,predict_landing
from exposure_tracking import ExposureTrackingBatch
from grip_evidence import retention_contacts,stable_retention
from arm_tracking import joint_body_jacobian,joint_gravity_forces
from policy_contract import legacy_actions,visual_features,filter_residual_actions,XY_RESIDUAL_SCALE_M
from response_control import FAST_RESPONSE,arrival_joint_velocity
from closure_trigger import contact_closure_step,anticipated_closure_fraction,incoming_hand_tilt
from predictive_tracking import position_priority_delta,early_arrival_velocity
from grip_pose_finish import contact_rotation_target,settling_joint_velocity
from soft_catch import reception_target,reception_joint_velocity
from fast_sampling import contact_masks,reduce_contact_forces


def quat_apply(q,v):
    t=2*torch.linalg.cross(q[...,:3],v,dim=-1)
    return v+q[...,3:]*t+torch.linalg.cross(q[...,:3],t,dim=-1)


def drive_command(position,target,velocity_target,gravity,kp,kd,velocity_limit):
    """Velocity-saturated PD drive; never edits generalized positions/velocities."""
    speed=np.clip(velocity_target+kp/kd*(target-position),-velocity_limit,velocity_limit)
    return position+kd/kp*speed+gravity/kp


def shared_method(name):
    """Load only pure control/observation functions, without importing Isaac/Kit."""
    source=ROOT/'scripts/ball_catch_env.py';tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BallCatchEnv')
    if name not in ('_command','get_observations'):raise ValueError(name)
    function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace=globals().copy();exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    return namespace[name]


class FrozenActor(torch.nn.Module):
    def __init__(self,saved):
        super().__init__();state=saved['actor_state_dict']
        if saved['infos']['task']['observation_dim']!=100 or saved['infos']['task']['action_dim']!=14:raise ValueError('Expected v4 100x14')
        from rsl_rl.modules import EmpiricalNormalization
        self.epsilon=float(EmpiricalNormalization(100).eps)
        self.register_buffer('mean',state['obs_normalizer._mean'].clone())
        self.register_buffer('std',state['obs_normalizer._std'].clone())
        layers=[]
        for index in [0,2,4,6]:
            weight=state[f'mlp.{index}.weight'];layer=torch.nn.Linear(weight.shape[1],weight.shape[0])
            layer.weight.data.copy_(weight);layer.bias.data.copy_(state[f'mlp.{index}.bias']);layers.append(layer)
            if index<6:layers.append(torch.nn.ELU())
        self.mlp=torch.nn.Sequential(*layers);self.eval();self.requires_grad_(False)

    def forward(self,observation):return self.mlp((observation-self.mean)/(self.std+self.epsilon))


class MujocoCatch:
    _command=shared_method('_command')
    get_observations=shared_method('get_observations')

    def __init__(self,checkpoint,log_dir,seed=42,resolution=192):
        self.checkpoint=Path(checkpoint).resolve();self.log_dir=Path(log_dir);self.log_dir.mkdir(parents=True,exist_ok=True)
        path,report=build_model(self.checkpoint,self.log_dir/'ball_catch.xml');self.model=mujoco.MjModel.from_xml_path(str(path));self.data=mujoco.MjData(self.model)
        self.gravity_data=mujoco.MjData(self.model);self.scene_cfg=report['scene'];self.task=report['task'];self.initial=report['initial']
        self.kin=RobotKinematics(report['urdf']);self.config=report;self.device='cpu';self.num_envs=1;self.dt=1/60
        self.policy_version='v4';self.action_stride=7;self.response_profile='fast';self.tracking_profile='predictive';self.grip_profile='pose_finish'
        self.response_settings=dict(FAST_RESPONSE);self.contact_probe=False;self.physics_steps=0;self.rng=np.random.default_rng(seed);torch.manual_seed(seed)
        contract=json.loads((self.checkpoint.parent/'kinematic_contract.json').read_text());self.jnames=contract['joint_names'];self.body_names=contract['body_names']
        self.joint_ids=np.array([self.model.joint(n).id for n in self.jnames]);self.qadr=self.model.jnt_qposadr[self.joint_ids];self.vadr=self.model.jnt_dofadr[self.joint_ids]
        self.body_ids=np.array([self.model.body(n).id for n in self.body_names]);self.hbody=[self.body_names.index(p+'hand_base_link') for p in ['lh_','rh_']]
        self.arm_ids=[[self.jnames.index(n) for n in self.kin.arm_names(s)] for s in ['left','right']]
        self.control_ids=[self.jnames.index(n) for n in self.task['joint_order']]
        self.hand_ids=[[self.jnames.index(n) for n in self.task['joint_order'] if n.startswith(p)] for p in ['lh_','rh_']]
        self.hand_joint_ids=torch.tensor(self.hand_ids);self.non_thumb_masks=torch.tensor([['thumb' not in self.jnames[i] for i in ids] for ids in self.hand_ids])
        tensor=lambda x:torch.tensor(x,dtype=torch.float32)
        default=tensor([[self.initial[n] for n in self.jnames]])
        self.limits=tensor([[self.kin.limits[n][:2] for n in self.jnames]])
        self.target=default.clone();self.velocity_targets=torch.zeros_like(default);self.efforts=torch.zeros_like(default)
        wrapped=lambda value:NS(torch=value)
        self.robot=NS(body_names=self.body_names,data=NS(joint_pos=wrapped(default.clone()),joint_vel=wrapped(torch.zeros_like(default)),
            default_joint_pos=wrapped(default),body_link_jacobian_w=wrapped(torch.zeros(1,len(self.body_names),6,len(self.jnames))),
            gravity_compensation_forces=wrapped(torch.zeros_like(default))),actuators=NS(target_command=NS(
                set_position_index=lambda **kw:None,set_velocity_index=lambda **kw:None,set_effort_index=lambda **kw:None)))
        self.nominal_p=[];self.nominal_grasp=[];centers=[]
        planner=SphereGraspPlanner(self.kin,self.scene_cfg['hand_mesh_bounds_local_m'],.02)
        for side in ['left','right']:
            p,_=self.kin.palm(side,self.initial);self.nominal_p.append(p)
            center=planner.resting_center(side,self.initial);centers.append(center)
            plan=planner.solve(side,center,self.initial)
            if not plan or not plan['geometry_feasible']:raise ValueError('Original grasp unavailable')
            self.nominal_grasp.append(tensor([plan['targets_rad'][self.jnames[i]] for i in self.hand_ids[len(self.nominal_grasp)]]))
        self.nominal_grasp_cpu=[q.numpy() for q in self.nominal_grasp];self.resting_centers_local=np.array(centers)
        self.ready_hand_pose=default[0,self.hand_joint_ids].clone()
        self.ready_hand_pose+=.25*(torch.stack(self.nominal_grasp)-self.ready_hand_pose)*self.non_thumb_masks;self.ready_hand_pose[:,1]=.02
        self.throw_workspace=ThrowWorkspace(self.kin,self.initial,self.scene_cfg,[c.tolist() for c in centers])
        self.origins=torch.zeros(1,3);self.side=torch.zeros(1,dtype=torch.long);self.episode_length_buf=torch.zeros(1,dtype=torch.long)
        for name in ['actions','previous_actions']:setattr(self,name,torch.zeros(1,14))
        for name in ['vision','visual_velocity','palm_velocity','previous_palm','caught_tilt','catch_palm_target','predicted_intercept',
                     'previous_local','reception_anchor','reception_feedforward','reception_shift','intercept_tilt']:
            setattr(self,name,torch.zeros(1,3))
        self.intercept_tilt[:,1]=-.18;self.grasp_origin_q=torch.zeros(1,6)
        for name in ['age','capture_time','grasp_start','grasp_curl_bias','prepare_time','preclose_fraction','predicted_contact_time']:setattr(self,name,torch.zeros(1))
        self.visible=torch.zeros(1,dtype=torch.bool);self.contact_streak=torch.zeros(1,dtype=torch.long)
        self.total_contact=torch.zeros(1,2);self.force_features=torch.zeros(1,10)
        self.closure_events=0;self.closure_events_by_side=[0,0];self.preparation_events=0;self.preparation_events_by_side=[0,0]
        self.contact_names=[n for n in self.body_names if n.startswith(('lh_','rh_'))]
        self.side_masks,self.finger_masks=contact_masks([self.contact_names],'cpu')
        self.exposure_batch=ExposureTrackingBatch(1,.02,filtered_state=True)
        self.renderer=mujoco.Renderer(self.model,height=resolution,width=resolution);self.resolution=resolution
        self.camera_id=self.model.camera('policy_camera').id;f=resolution/1.5;self.K=np.array([[f,0,resolution/2],[0,f,resolution/2],[0,0,1.]])
        self.camera_option=mujoco.MjvOption();self.camera_option.geomgroup[3]=0
        self.actuator_ids=np.array([self.model.actuator(n+'_drive').id for n in self.task['joint_order']])
        self.kp=np.array([3. if n.startswith(('lh_','rh_')) else 180. for n in self.task['joint_order']])
        self.kd=np.array([.1 if n.startswith(('lh_','rh_')) else 16. for n in self.task['joint_order']])
        self.drive_velocity_limits=np.array([1. if n.startswith(('lh_','rh_')) else 3. for n in self.task['joint_order']])
        self.ball_body=self.model.body('Ball').id;self.ball_geom=self.model.geom('ball').id;self.ball_joint=self.model.joint('ball_free').id
        self.episodes=0;self.successes=0;self.physics_failures=0;self.steps=0;self.vision_error_sum=0.;self.vision_count=0
        self.reset()
        (self.log_dir/'plant_contract.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

    def hands(self):return self.hp,self.hq

    def sync(self):
        mujoco.mj_forward(self.model,self.data)
        self.robot.data.joint_pos.torch[:]=torch.as_tensor(self.data.qpos[self.qadr].copy(),dtype=torch.float32)
        self.robot.data.joint_vel.torch[:]=torch.as_tensor(self.data.qvel[self.vadr].copy(),dtype=torch.float32)
        ids=self.body_ids[self.hbody];self.hp=torch.tensor(self.data.xpos[ids][None],dtype=torch.float32)
        self.hq=torch.tensor(self.data.xquat[ids][:,[1,2,3,0]][None],dtype=torch.float32)
        jac=self.robot.data.body_link_jacobian_w.torch
        for row,body in zip(self.hbody,ids):
            jp=np.zeros((3,self.model.nv));jr=jp.copy();mujoco.mj_jacBody(self.model,self.data,jp,jr,int(body))
            jac[0,row]=torch.tensor(np.r_[jp[:,self.vadr],jr[:,self.vadr]],dtype=torch.float32)
        self.gravity_data.qpos[:]=self.data.qpos;self.gravity_data.qvel[:]=0;mujoco.mj_forward(self.model,self.gravity_data)
        self.robot.data.gravity_compensation_forces.torch[:]=torch.tensor(self.gravity_data.qfrc_bias[self.vadr].copy(),dtype=torch.float32)

    def track(self):
        self.renderer.disable_depth_rendering();self.renderer.update_scene(self.data,camera='policy_camera',scene_option=self.camera_option)
        rgb=self.renderer.render().copy();self.renderer.enable_depth_rendering();depth=self.renderer.render().copy();self.renderer.disable_depth_rendering()
        self.last_rgb=rgb;self.last_depth=depth
        rotation=self.data.cam_xmat[self.camera_id].reshape(3,3)@np.diag([1.,-1.,-1.])
        quaternion=Rotation.from_matrix(rotation).as_quat();position=self.data.cam_xpos[self.camera_id].copy();now=self.physics_steps/120
        self.exposure_batch.set_contact_mode(np.array([bool(self.capture_time[0]>=0)]))
        self.exposure_batch.update(now,control_time=now,fetch_frames=lambda ids:(rgb[None],depth[None],self.K[None],position[None],quaternion[None]))
        for name,array in [('vision',self.exposure_batch.position),('visual_velocity',self.exposure_batch.velocity),('age',self.exposure_batch.age),('visible',self.exposure_batch.visible)]:
            target=getattr(self,name);target.copy_(torch.as_tensor(array,dtype=target.dtype))
        if self.visible[0]:
            self.vision_error_sum+=float(np.linalg.norm(self.exposure_batch.position[0]-self.data.xpos[self.ball_body]));self.vision_count+=1

    def force(self,accumulate=True):
        vectors=np.zeros((len(self.contact_names),3));points=[[] for _ in self.contact_names];distances=[[] for _ in self.contact_names]
        for contact_id in range(self.data.ncon):
            c=self.data.contact[contact_id]
            if self.ball_geom not in (c.geom1,c.geom2):continue
            geom=c.geom2 if c.geom1==self.ball_geom else c.geom1;name=self.model.body(int(self.model.geom_bodyid[geom])).name
            if name not in self.contact_names:continue
            i=self.contact_names.index(name);force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,contact_id,force)
            world=c.frame.reshape(3,3).T@force[:3]
            vectors[i]+=world if geom==c.geom2 else -world;points[i].append(c.pos.copy());distances[i].append(float(c.dist))
        totals,features,_=reduce_contact_forces(torch.tensor(vectors[None],dtype=torch.float32),self.side_masks,self.finger_masks)
        self.total_contact.copy_(torch.maximum(self.total_contact,totals) if accumulate else totals)
        self.force_features.copy_(torch.maximum(self.force_features,features) if accumulate else features)
        counts=np.array([len(p) for p in points]);starts=np.r_[0,np.cumsum(counts)[:-1]]
        self.contact_data=(vectors,np.array([p for group in points for p in group]).reshape(-1,3),counts,starts,np.array([d for group in distances for d in group]))

    def reset(self):
        mujoco.mj_resetData(self.model,self.data);self.data.qpos[self.qadr]=[self.initial[n] for n in self.jnames]
        self.side[:]=torch.randint(0,2,(1,));side=int(self.side[0]);self.data.qpos[self.qadr[self.hand_ids[side]]]=self.ready_hand_pose[side].numpy()
        for m in self.task['mimic_joints']:
            self.data.qpos[self.model.joint(m['name']).qposadr[0]]=self.data.qpos[self.model.joint(m['joint']).qposadr[0]]*float(m['multiplier'])+float(m.get('offset',0))
        start,velocity,goal,flight,index,kind=self.throw_workspace.sample(self.rng,side)
        qadr=self.model.jnt_qposadr[self.ball_joint];vadr=self.model.jnt_dofadr[self.ball_joint]
        self.data.qpos[qadr:qadr+3]=start;self.data.qpos[qadr+3:qadr+7]=[1,0,0,0];self.data.qvel[vadr:vadr+3]=velocity
        self.episode_length_buf.zero_();self.actions.zero_();self.previous_actions.zero_();self.velocity_targets.zero_()
        for name in ['capture_time','grasp_start','prepare_time','predicted_contact_time']:getattr(self,name).fill_(-1)
        for name in ['grasp_curl_bias','preclose_fraction','contact_streak','total_contact','force_features']:getattr(self,name).zero_()
        self.intercept_tilt[:]=torch.tensor([0.,-.18,0.]);self.target.copy_(torch.tensor(self.data.qpos[self.qadr][None],dtype=torch.float32))
        self.hold=0;self.tilt_hold=0;self.best_hold=0;self.hold_center=np.zeros(3);self.previous_center=np.zeros(3)
        self.exposure_batch.reset([0]);self.sync();self.data.ctrl[self.actuator_ids]=self.target[0,self.control_ids].numpy()
        self.catch_palm_target[0]=torch.tensor(self.nominal_p[side],dtype=torch.float32);self.predicted_intercept.copy_(self.catch_palm_target)
        self.previous_palm.copy_(self.hp[:,side]+quat_apply(self.hq[:,side],torch.tensor([[.025,0,.07]])))
        self.throw={'start':start.tolist(),'velocity':velocity.tolist(),'goal':goal.tolist(),'flight':float(flight),'kind':kind,'side':side}
        self.track()

    def step(self,actions):
        self.previous_actions=self.actions.clone();_,self.actions=filter_residual_actions(actions,self.previous_actions,.5)
        self._command(self.actions);ids=self.control_ids
        self.total_contact.zero_();self.force_features.zero_()
        for _ in range(2):
            self.data.ctrl[self.actuator_ids]=drive_command(self.data.qpos[self.qadr[ids]],self.target[0,ids].numpy(),
                self.velocity_targets[0,ids].numpy(),self.efforts[0,ids].numpy(),self.kp,self.kd,self.drive_velocity_limits)
            mujoco.mj_step(self.model,self.data);self.physics_steps+=1;self.force()
        self.sync();self.force();self.track();self.episode_length_buf+=1;self.steps+=1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or np.max(np.abs(self.data.qvel))>500:
            self.physics_failures+=1;raise RuntimeError('MuJoCo physics instability')
        side=int(self.side[0]);rotation=Rotation.from_quat(self.hq[0,side].numpy()).as_matrix();hand=self.hp[0,side].numpy()
        palm=hand+rotation@np.array([.025,0,.07]);self.palm_velocity.copy_((torch.tensor(palm[None],dtype=torch.float32)-self.previous_palm)/self.dt);self.previous_palm[:]=torch.tensor(palm)
        truth=self.data.xpos[self.ball_body].copy();local=rotation.T@(truth-hand);relative_speed=np.linalg.norm(local-self.previous_center)/self.dt;self.previous_center=local.copy()
        if self.hold==0:self.hold_center=local.copy()
        vectors,points,counts,starts,distances=self.contact_data
        evidence=retention_contacts(self.contact_names,vectors,points,counts,starts,distances,truth,.02,'lh_' if side==0 else 'rh_')
        grip=stable_retention(evidence['valid'],relative_speed,np.linalg.norm(local-self.hold_center),np.linalg.norm(truth-palm))
        self.hold=self.hold+1 if grip else 0;self.best_hold=max(self.best_hold,self.hold)
        self.tilt_hold=self.tilt_hold+1 if grip and rotation[2,0]<.75 else 0
        success=self.hold>=60 and self.tilt_hold>=18;failed=truth[2]<.65 or np.linalg.norm(truth-palm)>.7;timeout=int(self.episode_length_buf[0])>=360
        if success or failed or timeout:
            self.episodes+=1;self.successes+=int(success)
            with (self.log_dir/'episodes.jsonl').open('a') as f:f.write(json.dumps({**self.throw,'success':bool(success),'failed':bool(failed),'timeout':bool(timeout),'best_hold_steps':self.best_hold})+'\n')
            self.reset()
        return self.get_observations()['policy']
