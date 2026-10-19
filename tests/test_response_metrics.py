import sys,tempfile,unittest,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from response_metrics import failure_stage,crossing_errors,failed_retention_criteria
from response_diagnostics import run_response_diagnostics


class ResponseMetricsTests(unittest.TestCase):
    def test_failure_stages_do_not_call_every_contact_a_grasp(self):
        self.assertEqual(failure_stage('failed',False,0),'no_contact')
        self.assertEqual(failure_stage('failed',True,0),'contact_without_stable_hold')
        self.assertEqual(failure_stage('failed',True,59),'held_less_than_1s')
        self.assertEqual(failure_stage('timeout',True,60),'held_1s_but_not_full_success')

    def test_ball_deflection_is_distinct_from_hand_target_error(self):
        errors=crossing_errors([.001,0,1],[.08,0,1],[0,0,1])
        self.assertAlmostEqual(errors['hand_to_goal_xy_m'],.001)
        self.assertAlmostEqual(errors['ball_to_goal_xy_m'],.08)
        reasons=failed_retention_criteria({'hand_contact_force_n':0,'surface_contact_points':0,'max_penetration_m':0},.2,.01,.02)
        self.assertIn('contact_force_lost',reasons);self.assertIn('relative_speed_over_015',reasons)
        self.assertNotIn('palm_distance_over_65mm',reasons)

    def test_new_schema_collects_a_complete_frozen_rollout(self):
        class Policy(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            def forward(self,obs):return torch.zeros(2,14)
        policy=Policy();runner=NS(get_inference_policy=lambda:policy)
        env=NS(num_actions=14,num_envs=2,device='cpu',origins=torch.zeros(2,3),side=torch.tensor([0,1]),
            throw_kind=['front','right_side'],throw_start=torch.tensor([[0.,0,1.1],[0.,0,1.1]]),
            throw_launch_velocity=torch.zeros(2,3),throw_goal=torch.tensor([[0.,0,1.],[0.,0,1.]]),
            ball=NS(data=NS(root_pos_w=NS(torch=torch.tensor([[0.,0,1.1],[0.,0,1.1]])))),
            tilt_hold=torch.zeros(2),catch_palm_target=torch.zeros(2,3),predicted_contact_time=torch.ones(2)*.1,
            preclose_fraction=torch.ones(2)*.2,reward_previous_center=torch.zeros(2,3),hold_center=torch.zeros(2,3),
            contact_evidence=[{'retention':{}} for _ in range(2)],physics_failures=0,max_episode_length=3)
        env.arm_ids=[list(range(7)),list(range(7,14))];env.jnames=[f'joint_{i}' for i in range(14)]
        env.robot=NS(data=NS(joint_pos=NS(torch=torch.zeros(2,14))))
        env.velocity_targets=torch.zeros(2,14);env.limits=torch.tensor([[[-1.,1.]]*14]*2)
        env.get_observations=lambda:{'policy':torch.zeros(2,100)};counter=[0]
        def step(actions):
            first=counter[0]==0;counter[0]+=1
            data={'side':env.side,'episode_time_s':torch.ones(2)*(.2 if first else .3),'ball_position':torch.tensor([[0.,0,.95],[0.,0,.95]]),
                'throw_goal':env.throw_goal,'ball_velocity':torch.tensor([[0.,0,-1.],[0.,0,-1.]]),
                'ball_expected_at_hand':torch.tensor([[.01,0,1.],[.01,0,1.]]),'total_contact':torch.tensor([1. if first else 0.,0.]),
                'grasp_start':torch.tensor([.2,-1.]),'hand_q_nominal':torch.ones(2,6),'hand_q_default':torch.zeros(2,6),
                'hand_q':torch.ones(2,6)*.5,'arm_qd':torch.zeros(2,7),'hold':torch.tensor([1 if first else 0,0]),
                'grip':torch.tensor([first,False]),'palm_position':torch.zeros(2,3),'arm_target_error':torch.zeros(2,7),
                'actions':actions,'visible':torch.ones(2,dtype=torch.bool),'speed':torch.ones(2)*.2,'distance':torch.ones(2)*.02,
                'done':torch.tensor([not first,True]),'success':torch.zeros(2,dtype=torch.bool),
                'failed':torch.tensor([not first,True])}
            env.diagnostic_callback(data)
        env.step=step
        with tempfile.TemporaryDirectory() as d:
            run_response_diagnostics(env,runner,NS(),Path(d))
            result=json.loads((Path(d)/'response_summary.json').read_text())
            self.assertEqual(result['diagnostic_schema_version'],5)
            self.assertEqual(result['failure_stage_counts'],{'held_less_than_1s':1,'no_contact':1})
            self.assertEqual(result['retention_break_criteria_counts']['contact_force_lost'],1)
            self.assertTrue(result['policy_state_unchanged'])

if __name__=='__main__':unittest.main()
