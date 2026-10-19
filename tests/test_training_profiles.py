import ast,copy,random,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from training_profiles import (ROBUST_PROFILE,MIXED_PROFILE,CONSERVATIVE_PROFILES,configure_optimizer,resolve_training_profile,
    sample_receivers,choose_throw_kind,validation_selection,plateau_reached,sample_episode_levels,augment_measurement)


class TrainingProfilesTests(unittest.TestCase):
    def test_adam_moments_preserved_and_loaded_lr_explicitly_replaced(self):
        model=torch.nn.Linear(3,2);optimizer=torch.optim.Adam(model.parameters(),lr=.00011390625)
        model(torch.ones(4,3)).sum().backward();optimizer.step()
        before=copy.deepcopy(optimizer.state_dict()['state'])
        algorithm=SimpleNamespace(optimizer=optimizer,learning_rate=.00011390625,schedule='adaptive')
        result=configure_optimizer(algorithm,ROBUST_PROFILE)
        self.assertEqual(algorithm.learning_rate,5e-5);self.assertEqual(algorithm.schedule,'fixed')
        self.assertTrue(all(g['lr']==5e-5 for g in optimizer.param_groups))
        for parameter,state in optimizer.state_dict()['state'].items():
            for key,value in state.items():self.assertTrue(torch.equal(value,before[parameter][key]))
        self.assertFalse(result['optimizer_moments_reset'])
        self.assertIsNone(configure_optimizer(algorithm,'legacy'))

    def test_training_receiver_and_direction_probabilities(self):
        torch.manual_seed(987);rng=random.Random(987)
        sides=sample_receivers(100000,'cpu',ROBUST_PROFILE).tolist()
        kinds=[choose_throw_kind(rng,side,ROBUST_PROFILE) for side in sides]
        for kind,expected in [('front',.35),('left_side',.25),('right_side',.40)]:
            self.assertAlmostEqual(kinds.count(kind)/len(kinds),expected,delta=.005)
        self.assertAlmostEqual(sum(sides)/len(sides),.60,delta=.005)

    def test_unbiased_evaluation_receiver_distribution(self):
        torch.manual_seed(987)
        self.assertAlmostEqual(float(sample_receivers(100000,'cpu','legacy').float().mean()),.5,delta=.005)

    def test_evaluation_sampler_bypasses_training_filters(self):
        source=(Path(__file__).resolve().parents[1]/'scripts/ball_catch_env.py').read_text()
        cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='BallCatchEnv')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_sample_training_throw')
        namespace={'CONSERVATIVE_PROFILES':CONSERVATIVE_PROFILES,'choose_throw_kind':choose_throw_kind,'np':np}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<sampler>','exec'),namespace)
        sentinel=object();calls=[]
        env=SimpleNamespace(training_profile=ROBUST_PROFILE,training_mode=False,policy_version='v4',
            curriculum_level=2,rng=random.Random(987),
            throw_workspace=SimpleNamespace(sample=lambda rng,side:(calls.append(side) or sentinel)))
        self.assertIs(namespace['_sample_training_throw'](env,1),sentinel)
        self.assertEqual(calls,[1])

    def test_all_levels_are_resampled_with_equal_probability(self):
        torch.manual_seed(1346)
        first=sample_episode_levels(90000,MIXED_PROFILE,2)
        second=sample_episode_levels(90000,MIXED_PROFILE,2)
        for level in range(3):self.assertAlmostEqual(float((first==level).mean()),1/3,delta=.006)
        self.assertGreater(float((first!=second).mean()),.6)
        self.assertTrue(bool((sample_episode_levels(256,'legacy',2)==2).all()))

    def test_mixed_profile_keeps_optimizer_and_direction_settings(self):
        optimizer=torch.optim.Adam(torch.nn.Linear(1,1).parameters(),lr=5e-5)
        alg=SimpleNamespace(optimizer=optimizer,schedule='adaptive',learning_rate=5e-5)
        self.assertIsNotNone(configure_optimizer(alg,MIXED_PROFILE));self.assertEqual(alg.schedule,'fixed')
        torch.manual_seed(42);sides=sample_receivers(100000,'cpu',MIXED_PROFILE)
        self.assertAlmostEqual(float(sides.float().mean()),.6,delta=.006)

    def test_sensor_augmentation_is_limited_to_level2_and_training(self):
        point=np.array([.1,.2,.3])
        class NoDraw:
            def random(self):raise AssertionError('Unwanted random draw')
        for level in [0,1]:self.assertIs(augment_measurement(point,NoDraw(),level),point)
        self.assertIs(augment_measurement(point,NoDraw(),2,training=False),point)
        self.assertIsNone(augment_measurement(None,NoDraw(),2))
        rng=SimpleNamespace(random=lambda:0.)
        self.assertIsNone(augment_measurement(point,rng,2))
        rng=SimpleNamespace(random=lambda:.5,normal=lambda *args:np.array([.001,.002,.003]))
        np.testing.assert_allclose(augment_measurement(point,rng,2),point+[.001,.002,.003])

    def test_mixed_sampler_enforces_each_episode_difficulty(self):
        source=(Path(__file__).resolve().parents[1]/'scripts/ball_catch_env.py').read_text()
        cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='BallCatchEnv')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_sample_training_throw')
        namespace={'CONSERVATIVE_PROFILES':CONSERVATIVE_PROFILES,'choose_throw_kind':lambda *args:'front','np':np}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<sampler>','exec'),namespace)
        for level,offset,flight in [(0,.08,.40),(1,.105,.37),(2,.105,.34)]:
            candidates=iter([(None,None,np.array([.105,0,0]),.34,0,'front'),
                (None,None,np.array([offset,0,0]),flight,0,'front')])
            env=SimpleNamespace(training_profile=MIXED_PROFILE,training_mode=True,policy_version='v4',curriculum_level=2,
                rng=object(),throw_workspace=SimpleNamespace(palms=[np.zeros(3)],contact_offsets=[np.zeros(3)],sample=lambda *args:next(candidates)))
            result=namespace['_sample_training_throw'](env,0,level)
            self.assertGreaterEqual(result[3],flight);self.assertLessEqual(np.linalg.norm(result[2]),offset+1e-6)

    def test_profile_inheritance_and_explicit_override(self):
        self.assertEqual(resolve_training_profile(None,{}),'legacy')
        self.assertEqual(resolve_training_profile(None,{'training_profile':ROBUST_PROFILE}),ROBUST_PROFILE)
        self.assertEqual(resolve_training_profile(ROBUST_PROFILE,{}),ROBUST_PROFILE)
        self.assertEqual(resolve_training_profile('legacy',{'training_profile':ROBUST_PROFILE}),'legacy')

    def test_987_highest_score_is_kept_below_plateau_threshold(self):
        selection=validation_selection(676/1024,676/1024,0,681/1024)
        self.assertTrue(selection['replace_best']);self.assertEqual(selection['best'],681/1024)
        self.assertEqual(selection['reference'],676/1024);self.assertEqual(selection['stale'],1)
        next_selection=validation_selection(selection['best'],selection['reference'],1,683/1024)
        self.assertEqual(next_selection['stale'],0);self.assertEqual(next_selection['reference'],683/1024)
        worse=validation_selection(681/1024,676/1024,1,663/1024)
        self.assertFalse(worse['replace_best']);self.assertEqual(worse['best'],681/1024)

    def test_harder_curriculum_has_minimum_new_learning_window(self):
        self.assertFalse(plateau_reached(2587,987,8,8,2000))
        self.assertFalse(plateau_reached(3087,987,7,8,2000))
        self.assertTrue(plateau_reached(3087,987,8,8,2000))


if __name__=='__main__':unittest.main()
