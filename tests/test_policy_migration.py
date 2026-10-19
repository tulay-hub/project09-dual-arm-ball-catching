import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from policy_migration import migrate_state,source_mean
from policy_contract import LEGACY_ACTION_INDEX,NEW_ACTION_INDEX,legacy_actions,visual_features,tracking_potential
from validate_fast_sampling import fake_env,command_from
from response_control import FAST_RESPONSE,arrival_joint_velocity


def states(obs,actions):
    s={'obs_normalizer._mean':torch.randn(1,obs),'obs_normalizer._std':torch.rand(1,obs)+.2,
       'obs_normalizer._var':torch.ones(1,obs),'obs_normalizer.count':torch.tensor(9000000)}
    for layer,shape in [(0,(128,obs)),(2,(128,128)),(4,(64,128)),(6,(actions,64))]:
        s[f'mlp.{layer}.weight']=torch.randn(*shape)*.04;s[f'mlp.{layer}.bias']=torch.randn(shape[0])*.1
    s['distribution.log_std_param']=torch.full((actions,),-2.)
    return s


class PolicyMigrationTests(unittest.TestCase):
    def test_soft_reception_command_is_finite_and_keeps_finger_slew(self):
        import tempfile
        from soft_catch import SOFT_SETTINGS
        with tempfile.TemporaryDirectory() as directory:
            command=command_from(Path(__file__).resolve().parents[1]/'scripts/ball_catch_env.py')
            command.__globals__['arrival_joint_velocity']=arrival_joint_velocity
            env,_=fake_env('cpu',directory);env.response_profile='fast';env.action_stride=7;env.grip_profile='soft_catch'
            env.response_settings={**FAST_RESPONSE,**SOFT_SETTINGS};env.velocity_targets=torch.zeros_like(env.target)
            env.ready_hand_pose=env.robot.data.default_joint_pos.torch[0,env.hand_joint_ids].clone()
            env.robot.actuators.target_command.set_velocity_index=lambda **kwargs:None
            env.reception_anchor=env.catch_palm_target.clone();env.reception_feedforward=torch.zeros(256,3);env.reception_shift=torch.zeros(256,3)
            previous=env.target.clone();command(env,torch.zeros(256,14))
            self.assertTrue(bool(torch.isfinite(env.target).all()))
            self.assertLessEqual(float(env.reception_shift.norm(dim=-1).max()),.035001)
            self.assertLessEqual(float(env.velocity_targets.abs().max()),3.)
            hand=env.hand_joint_ids.flatten()
            self.assertLessEqual(float((env.target[:,hand]-previous[:,hand]).abs().max()),.95/60+1e-6)

    def test_predictive_controller_expires_arrival_and_respects_drives(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            command=command_from(Path(__file__).resolve().parents[1]/'scripts/ball_catch_env.py')
            command.__globals__['arrival_joint_velocity']=arrival_joint_velocity
            env,_=fake_env('cpu',directory);env.response_profile='fast';env.action_stride=7
            env.tracking_profile='predictive';env.response_settings=dict(FAST_RESPONSE)
            env.velocity_targets=torch.zeros_like(env.target)
            env.ready_hand_pose=env.robot.data.default_joint_pos.torch[0,env.hand_joint_ids].clone()
            env.robot.actuators.target_command.set_velocity_index=lambda **kwargs:None
            env.capture_time[:]=-1;env.grasp_start[:]=-1;env.total_contact.zero_();env.age[:]=.12
            env.predicted_contact_time[:]=.02;env.intercept_tilt=torch.zeros(256,3);env.intercept_tilt[:,1]=-.25
            command(env,torch.zeros(256,14));self.assertTrue(bool((env.predicted_contact_time>0).all()))
            command(env,torch.zeros(256,14));self.assertTrue(bool((env.predicted_contact_time<0).all()))
            self.assertTrue(bool(torch.isfinite(env.target).all()))
            self.assertLessEqual(float(env.velocity_targets.abs().max()),3.)
            self.assertTrue(bool((env.target>=env.limits[:,:,0]).all()))
            self.assertTrue(bool((env.target<=env.limits[:,:,1]).all()))
            torch.testing.assert_close(env.intercept_tilt[:,1],torch.full((256,),-.25))

    def test_xy_and_preclose_actions_reach_controller(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            command=command_from(Path(__file__).resolve().parents[1]/'scripts/ball_catch_env.py')
            command.__globals__['arrival_joint_velocity']=arrival_joint_velocity
            def make():
                env,_=fake_env('cpu',directory);env.response_profile='fast';env.action_stride=7
                env.response_settings=dict(FAST_RESPONSE);env.velocity_targets=torch.zeros_like(env.target)
                env.ready_hand_pose=env.robot.data.default_joint_pos.torch[0,env.hand_joint_ids].clone()
                env.robot.actuators.target_command.set_velocity_index=lambda **kwargs:None
                env.capture_time[:]=-1;env.grasp_start[:]=-1;env.total_contact.zero_()
                env.age.zero_();env.prepare_time[:]=-1;env.preclose_fraction.zero_()
                return env
            baseline=make();changed=make();timing=make();zero=torch.zeros(256,14)
            xy=zero.clone();xy[:,[4,5,11,12]]=torch.tensor([1.,-1.,1.,-1.])
            early=zero.clone();early[:,[6,13]]=1.
            command(baseline,zero);command(changed,xy);command(timing,early)
            self.assertGreater(float((changed.target-baseline.target).abs().max()),1e-4)
            self.assertLessEqual(float(changed.velocity_targets.abs().max()),3.)
            self.assertGreater(float((timing.preclose_fraction-baseline.preclose_fraction).max()),.01)

    def test_old_actions_preserved_and_new_outputs_ignore_new_observations(self):
        torch.manual_seed(43);old=states(82,8);new=migrate_state(old,states(100,14),True)
        observation=torch.randn(70,100);expected=source_mean(old,observation[:,:82],.01);actual=source_mean(new,observation,.01)
        torch.testing.assert_close(actual[:,LEGACY_ACTION_INDEX],expected,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(actual[:,NEW_ACTION_INDEX],torch.zeros(70,6))
        self.assertEqual(new['obs_normalizer.count'].item(),16384)
        torch.testing.assert_close(new['mlp.0.weight'][:,:82],old['mlp.0.weight'])

    def test_mapping_previous_actions_has_the_old_eight_order(self):
        action=torch.arange(14)[None];torch.testing.assert_close(legacy_actions(action),action[:,LEGACY_ACTION_INDEX])
        self.assertEqual(visual_features(torch.zeros(1,3),torch.zeros(1),torch.zeros(1,3),torch.zeros(1,3),torch.ones(1),torch.zeros(1),action).shape,(1,18))

    def test_discounted_potential_cannot_generate_a_repeatable_hold_bonus(self):
        phi=tracking_potential(torch.tensor([.10,.04,0.]),torch.zeros(3),torch.zeros(3,dtype=torch.bool),torch.tensor([False,False,True]))
        self.assertEqual(phi[-1].item(),0.)
        previous=torch.tensor(0.);total=0.
        for t,current in enumerate(phi):total+=.99**t*(.99*current-previous);previous=current
        self.assertAlmostEqual(float(total),0.,places=6)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA migration regression')
    def test_cpu_checkpoint_into_cuda_model(self):
        source=states(82,8);target={k:v.cuda() for k,v in states(100,14).items()}
        result=migrate_state(source,target,True)
        self.assertTrue(all(v.is_cuda for v in result.values()))
        torch.testing.assert_close(result['mlp.6.weight'][LEGACY_ACTION_INDEX].cpu(),source['mlp.6.weight'])

    def test_unknown_shape_is_rejected(self):
        old=states(82,8);target=states(100,14);target['mlp.2.weight']=torch.zeros(64,128)
        with self.assertRaises(ValueError):migrate_state(old,target,True)

if __name__=='__main__':unittest.main()
