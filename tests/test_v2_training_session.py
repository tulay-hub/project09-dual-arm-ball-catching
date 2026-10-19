import sys,tempfile,unittest,json
from unittest.mock import patch
import torch
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_policy_v2 import TrainingSession,aggregate


class V2TrainingSessionTests(unittest.TestCase):
    def test_mixed_continuation_starts_from_1346_and_preserves_all_levels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'reports').mkdir();(root/'reports/active_training.json').write_text('{}')
            source=root/'model_1345.pt'
            torch.save({'infos':{'task':{'name':'v4','observation_dim':100,'action_dim':14,'curriculum_level':2,
                'training_profile':'robust_l2'},'completed_updates':1346}},source)
            args=SimpleNamespace(output=root/'mixed',initial_checkpoint=source,baseline_checkpoint=source,
                total_updates=10000,validation_seeds=[11011,12012,13013,14014],training_profile='mixed_all',
                initial_curriculum_level=0,grip_profile='pose_finish',tracking_profile='predictive')
            with patch('train_policy_v2.ROOT',root),patch('train_policy_v2.signal.signal'):
                session=TrainingSession(args)
            self.assertEqual(session.completed,1346);self.assertEqual(session.level,2)
            self.assertEqual(session.state['first_validation_after_updates'],1446)
            self.assertEqual(session.state['curriculum_mode'],'mixed_all')
            self.assertEqual(session.state['curriculum_level_probabilities'],{'0':1/3,'1':1/3,'2':1/3})
            self.assertEqual(torch.load(session.checkpoint,weights_only=False)['infos']['completed_updates'],1346)

    def test_user_selected_987_continues_at_L2_with_new_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'reports').mkdir();(root/'reports/active_training.json').write_text('{}')
            source=root/'model_986.pt'
            torch.save({'infos':{'task':{'name':'v4','observation_dim':100,'action_dim':14,'curriculum_level':1},
                'completed_updates':987}},source)
            args=SimpleNamespace(output=root/'from987',initial_checkpoint=source,baseline_checkpoint=source,
                total_updates=1087,validation_seeds=[11011,12012,13013,14014],final_test_seeds=[21021,22022,23023,24024],
                initial_curriculum_level=2,training_profile='robust_l2',plateau_patience=8,min_new_updates_before_plateau=2000,
                tracking_profile='predictive',grip_profile='pose_finish')
            with patch('train_policy_v2.ROOT',root),patch('train_policy_v2.signal.signal'):
                session=TrainingSession(args)
            self.assertEqual(session.completed,987);self.assertEqual(session.state['first_validation_after_updates'],1087)
            self.assertEqual(session.level,2);self.assertEqual(session.state['continuation_start_completed_updates'],987)
            commands=[];evaluations=[];session.publish=lambda:None
            def worker(command,run,phase):
                commands.append(command);run.mkdir();(run/'training_complete.json').write_text('{"completed_updates":1087}')
                checkpoint=run/'model_1086.pt';checkpoint.touch();(run/'latest.pt').symlink_to(checkpoint)
            def evaluate(checkpoint,tag,seeds):
                evaluations.append((tag,seeds));return {'success_rate':.65,'throw_hashes':['same']*4}
            session.worker=worker;session.evaluate=evaluate;session.run()
            self.assertEqual(commands[0][commands[0].index('--total_iterations')+1],'1087')
            self.assertEqual(commands[0][commands[0].index('--training_profile')+1],'robust_l2')
            self.assertEqual(commands[0][commands[0].index('--curriculum_level')+1],'2')
            self.assertEqual(evaluations[-1],('final_test',args.final_test_seeds))
            self.assertEqual(evaluations[0],('baseline',args.validation_seeds))

    def test_explicit_harder_curriculum_and_checkpoint_curriculum_survive_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'reports').mkdir();(root/'reports/active_training.json').write_text('{}')
            for i,(saved_level,requested) in enumerate([(0,1),(1,None)]):
                source=root/f'source{i}.pt'
                torch.save({'infos':{'task':{'name':'v4','observation_dim':100,'action_dim':14,'curriculum_level':saved_level},'completed_updates':487}},source)
                args=SimpleNamespace(output=root/f'session{i}',initial_checkpoint=source,baseline_checkpoint=None,
                    total_updates=10000,validation_seeds=[3003,4004,5005,6006],initial_curriculum_level=requested,grip_profile='soft_catch',tracking_profile='predictive')
                with patch('train_policy_v2.ROOT',root),patch('train_policy_v2.signal.signal'):
                    session=TrainingSession(args)
                self.assertEqual(session.level,1);self.assertEqual(session.state['curriculum_level'],1)
                self.assertEqual(session.state['grip_profile'],'soft_catch')
                self.assertEqual(session.state['first_validation_after_updates'],587)

    def test_continuation_preserves_progress_and_keeps_original_baseline_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'reports').mkdir();(root/'reports/active_training.json').write_text('{}')
            task={'name':'v4','observation_dim':100,'action_dim':14}
            source=root/'source.pt';baseline=root/'baseline.pt'
            for path,updates in [(source,303),(baseline,0)]:
                torch.save({'infos':{'task':task,'completed_updates':updates}},path)
            args=SimpleNamespace(output=root/'continuation',initial_checkpoint=source,baseline_checkpoint=baseline,
                total_updates=10000,validation_seeds=[3003,4004,5005,6006])
            with patch('train_policy_v2.ROOT',root),patch('train_policy_v2.signal.signal'):
                session=TrainingSession(args)
            self.assertEqual(session.completed,303)
            self.assertNotEqual(session.checkpoint,session.baseline)
            self.assertEqual(torch.load(session.baseline,weights_only=False)['infos']['completed_updates'],0)
            self.assertEqual(session.state['first_validation_after_updates'],403)
            self.assertTrue(session.state['xy_action_hard_clip'])
            self.assertEqual(session.state['xy_residual_hard_limit_m'],.2)
            self.assertIsNone(session.state['latest_validation'])

    def test_aggregation_rejects_invalid_physics_or_updated_policy(self):
        valid={'episodes':256,'successes':180,'physics_invalid_episodes':0,'policy_state_unchanged':True,'throw_sha256':'a'}
        self.assertEqual(aggregate([valid]*4)['episodes'],1024)
        with self.assertRaises(ValueError):aggregate([{**valid,'physics_invalid_episodes':1}])
        with self.assertRaises(ValueError):aggregate([{**valid,'policy_state_unchanged':False}])

    def test_segment_counters_curriculum_and_final_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            session=TrainingSession.__new__(TrainingSession);session.root=Path(d);session.baseline=Path(d)/'initial.pt'
            session.checkpoint=session.baseline;session.completed=0;session.level=0;session.state={};session.publish=lambda:None
            session.args=SimpleNamespace(total_updates=301,validation_seeds=[3003,4004,5005,6006]);commands=[];evaluations=[]
            def worker(command,run,phase):
                commands.append(command);run.mkdir();target=int(command[command.index('--total_iterations')+1])
                (run/'training_complete.json').write_text(json.dumps({'completed_updates':target}))
                cp=run/f'model_{target-1}.pt';cp.touch();(run/'latest.pt').symlink_to(cp)
            def evaluate(cp,tag,seeds):
                evaluations.append((tag,seeds));score=.63 if tag=='baseline' else {100:.86,300:.91,301:.92}[session.completed]
                return {'success_rate':score,'throw_hashes':['same']*4}
            session.worker=worker;session.evaluate=evaluate;session.run()
            self.assertEqual([c[c.index('--total_iterations')+1] for c in commands],['100','300','301'])
            self.assertEqual([c[c.index('--curriculum_level')+1] for c in commands],['0','1','2'])
            self.assertTrue(all(c[c.index('--num_envs')+1]=='256' for c in commands))
            self.assertEqual(evaluations[-1],('final_test',[7007,8008,9009,10010]))
            self.assertEqual(session.state['status'],'completed')

if __name__=='__main__':unittest.main()
