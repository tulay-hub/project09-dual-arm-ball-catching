import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))


@unittest.skipUnless(importlib.util.find_spec('mujoco'),'Optional MuJoCo dependency')
class MujocoTransferTests(unittest.TestCase):
    def test_drive_matches_pd_below_limit_and_saturates_requested_velocity(self):
        from mujoco_catch import drive_command
        position=np.array([.1,.2]);target=np.array([.11,.205]);velocity=np.array([.2,.1]);gravity=np.array([.3,.01])
        kp=np.array([180.,3.]);kd=np.array([16.,.1]);limits=np.array([3.,1.])
        actual=drive_command(position,target,velocity,gravity,kp,kd,limits)
        np.testing.assert_allclose(actual,target+kd/kp*velocity+gravity/kp)
        saturated=drive_command(position,target+100,velocity,gravity,kp,kd,limits)
        np.testing.assert_allclose(kp/kd*(saturated-position-gravity/kp),limits)

    def test_frozen_actor_matches_reference_normalization_and_mlp(self):
        from mujoco_catch import FrozenActor
        from policy_migration import source_mean
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict
        reference=MLPModel(TensorDict({'policy':torch.zeros(1,100)},batch_size=[1]),{'actor':['policy']},'actor',14,
            hidden_dims=[128,128,64],activation='elu',obs_normalization=True)
        reference.obs_normalizer._mean.normal_();reference.obs_normalizer._std.uniform_(.2,2.)
        state=reference.state_dict();saved={'actor_state_dict':state,'infos':{'task':{'observation_dim':100,'action_dim':14}}}
        actor=FrozenActor(saved);observation=torch.randn(256,100);before={k:v.clone() for k,v in actor.state_dict().items()}
        with torch.inference_mode():
            expected=reference(TensorDict({'policy':observation},batch_size=[256]));actual=actor(observation)
            exported=torch.jit.script(actor)(observation)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6);torch.testing.assert_close(exported,expected)
        self.assertFalse(any(p.requires_grad for p in actor.parameters()))
        self.assertTrue(all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items()))

    def test_shared_controller_preserves_target_and_velocity_commands(self):
        from mujoco_catch import shared_method
        from validate_fast_sampling import fake_env,command_from
        from response_control import FAST_RESPONSE,arrival_joint_velocity
        with tempfile.TemporaryDirectory() as directory:
            source,_=fake_env('cpu',directory)
            source.response_profile='fast';source.tracking_profile='predictive';source.grip_profile='pose_finish';source.action_stride=7
            source.response_settings=dict(FAST_RESPONSE);source.velocity_targets=torch.zeros_like(source.target)
            source.ready_hand_pose=source.robot.data.default_joint_pos.torch[0,source.hand_joint_ids].clone()
            source.robot.actuators.target_command.set_velocity_index=lambda **kw:None
            source.intercept_tilt=torch.zeros(source.num_envs,3);source.intercept_tilt[:,1]=-.18
            migrated=copy.deepcopy(source);actions=torch.randn(source.num_envs,14)*.2
            original=command_from(ROOT/'scripts/ball_catch_env.py');original.__globals__['arrival_joint_velocity']=arrival_joint_velocity
            original(source,actions);shared_method('_command')(migrated,actions)
            for name in ['target','velocity_targets','capture_time','grasp_start','preclose_fraction','catch_palm_target']:
                torch.testing.assert_close(getattr(source,name),getattr(migrated,name),atol=1e-6,rtol=1e-6)

    def test_model_mapping_mass_and_palm_jacobians(self):
        import mujoco
        from build_mujoco_catch import build_model
        from kinematics import RobotKinematics
        import xml.etree.ElementTree as ET
        checkpoint=ROOT/'checkpoints/model_1445.pt'
        if not checkpoint.exists():checkpoint=ROOT/'logs/ppo_catch/v2/mixed_all_from1345_20261019_134853/segment_0001/model_1445.pt'
        if not checkpoint.exists():self.skipTest('Reference checkpoint is local')
        with tempfile.TemporaryDirectory() as directory:
            path,report=build_model(checkpoint,Path(directory)/'robot.xml');model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
            for name,value in report['initial'].items():data.qpos[model.joint(name).qposadr[0]]=value
            mujoco.mj_forward(model,data);kin=RobotKinematics(report['urdf'])
            mass=sum(float(m.get('value')) for m in ET.parse(report['urdf']).getroot().findall('./link/inertial/mass'))
            self.assertAlmostEqual(float(model.body_mass.sum()),mass+.016399,places=5)
            self.assertEqual(model.nu,26);self.assertEqual(model.neq,10)
            self.assertFalse(any(model.equality(i).name and 'Ball' in model.equality(i).name for i in range(model.neq)))
            for side,prefix in [('left','lh_'),('right','rh_')]:
                body=model.body(prefix+'hand_base_link').id;point=data.xpos[body]+data.xmat[body].reshape(3,3)@np.array([.025,0,.07])
                jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jac(model,data,jp,jr,point,body)
                columns=[model.joint(n).dofadr[0] for n in kin.arm_names(side)]
                np.testing.assert_allclose(np.r_[jp[:,columns],jr[:,columns]],kin.palm_jacobian(side,report['initial']),atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
