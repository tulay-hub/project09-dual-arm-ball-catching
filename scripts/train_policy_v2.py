"""Sequential v2 training/held-out validation; GPU workers never overlap."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,json,os,re,shutil,signal,subprocess,time
import torch
from policy_contract import XY_CONTRACT
from training_profiles import MIXED_PROFILE,resolve_training_profile,validation_selection,plateau_reached

ROOT=Path(__file__).resolve().parents[1]


def atomic_json(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.supervisor.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def aggregate(summaries):
    if not summaries or any(s['physics_invalid_episodes'] or not s['policy_state_unchanged'] for s in summaries):
        raise ValueError('Invalid frozen evaluation')
    episodes=sum(s['episodes'] for s in summaries);successes=sum(s['successes'] for s in summaries)
    return {'episodes':episodes,'successes':successes,'success_rate':successes/episodes,'physics_invalid_episodes':0,
            'policy_state_unchanged':True,'throw_hashes':[s['throw_sha256'] for s in summaries]}


class TrainingSession:
    def __init__(self,args):
        self.args=args;self.root=args.output.resolve();self.root.mkdir(parents=True,exist_ok=False)
        self.tracking_profile=getattr(args,'tracking_profile','fast')
        self.grip_profile=getattr(args,'grip_profile','fast')
        self.stop=False;self.child=None;self.phase=None;self.cached_checkpoint=None
        self.checkpoint=self.root/'model_initial.pt';shutil.copy2(args.initial_checkpoint,self.checkpoint)
        initial=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        task=initial['infos']['task']
        if task.get('observation_dim')!=100 or task.get('action_dim')!=14:raise ValueError('Expected a migrated v4 checkpoint')
        self.baseline=self.checkpoint
        if args.baseline_checkpoint:
            self.baseline=self.root/'model_baseline.pt';shutil.copy2(args.baseline_checkpoint,self.baseline)
            baseline=torch.load(self.baseline,map_location='cpu',weights_only=False)['infos']['task']
            if baseline.get('name')!=task.get('name') or baseline.get('action_dim')!=14 or baseline.get('observation_dim')!=100:raise ValueError('Incompatible validation baseline')
        self.completed=initial['infos']['completed_updates']
        self.training_profile=resolve_training_profile(getattr(args,'training_profile',None),task)
        self.session_start_updates=self.completed
        self.final_test_seeds=getattr(args,'final_test_seeds',None) or [7007,8008,9009,10010]
        self.plateau_patience=getattr(args,'plateau_patience',5)
        self.min_new_updates_before_plateau=getattr(args,'min_new_updates_before_plateau',0)
        if self.plateau_patience<1 or self.min_new_updates_before_plateau<0:raise ValueError('Invalid plateau settings')
        if set(self.final_test_seeds)&set(args.validation_seeds):raise ValueError('Final test and validation seeds must be disjoint')
        requested_level=getattr(args,'initial_curriculum_level',None)
        self.level=int(task.get('curriculum_level',0)) if requested_level is None else requested_level
        if self.training_profile==MIXED_PROFILE:self.level=2
        old=json.loads((ROOT/'reports/active_training.json').read_text());atomic_json(self.root/'previous_training_state.json',old)
        self.state={'name':'v2','status':'starting','phase':'training','pid':None,'supervisor_pid':os.getpid(),
            'session_dir':str(self.root),'run_dir':None,'stdout_log':None,'resume_checkpoint':str(self.checkpoint),
            'last_verified_checkpoint':str(self.checkpoint),'completed_updates':self.completed,
            'last_observed_completed_updates':self.completed,'planned_iterations':args.total_updates,'num_envs':256,
            'steps_per_env':64,'camera_resolution':192,'physics_hz':120,'policy_hz':60,'ball_diameter_mm':40,
            'policy_version':'v4','observation_dim':100,'action_dim':14,'controller_response_profile':'fast',
            'stage':'offcenter_mesh_catch_retention','lineage':initial['infos'].get('lineage'),
            'parent_checkpoint':str(args.initial_checkpoint.resolve()),'from_scratch':False,'optimizer_reset_on_migration':True,
            'start_time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'auto_resume':False,
            'validation_every_updates':200,'first_validation_after_updates':100,'validation_seeds':args.validation_seeds,
            'validation_episodes':1024,'evaluation_kind':'heldout_validation_reused_for_selection',
            'stop_hint':'SIGINT supervisor_pid forwards a save-and-stop signal to the active training worker'}
        self.state.update(**XY_CONTRACT,baseline_checkpoint=str(self.baseline),
            optimizer_restored_on_continuation=self.completed>0,
            continuation_start_completed_updates=self.completed,
            first_validation_after_updates=self.completed+100,
            validation_protocol=f'v4_xy200_{self.tracking_profile}_{self.grip_profile}_{self.training_profile}_full_distribution_v3',
            training_profile=self.training_profile,final_test_seeds=self.final_test_seeds,
            curriculum_mode='mixed_all' if self.training_profile==MIXED_PROFILE else 'single_level',
            curriculum_level_probabilities={str(i):1/3 for i in range(3)} if self.training_profile==MIXED_PROFILE else {str(self.level):1.},
            plateau_patience=self.plateau_patience,min_new_updates_before_plateau=self.min_new_updates_before_plateau,
            plateau_material_improvement_threshold=.005,best_selection='highest_observed_validation_success_rate',
            grip_profile=self.grip_profile,curriculum_level=self.level,
            initial_curriculum_level=self.level,curriculum_user_override=requested_level is not None,
            tracking_profile=self.tracking_profile,
            latest_validation=None,baseline_validation=None)
        atomic_json(self.root/'run_plan.json',dict(self.state));self.publish()
        for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,self.request_stop)

    def publish(self):
        self.state['updated_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        atomic_json(ROOT/'reports/active_training.json',self.state);atomic_json(self.root/'session_state.json',self.state)

    def request_stop(self,*_):
        self.stop=True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGINT if self.phase=='training' else signal.SIGTERM)

    def refresh(self,run,log):
        if not log.exists():return
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-250000));text=f.read().decode(errors='replace')
        indices=[int(x)+1 for x in re.findall(r'Learning iteration\s+(\d+)',text,re.I)]
        if indices:self.state.update(status='running_verified',last_observed_completed_updates=max(indices))
        timing=[float(x) for x in re.findall(r'Collection time:\s*([\d.]+)s',text)]
        if timing:self.state['recent_collection_seconds']=timing[-10:]
        for name,key in [('camera_health.json','camera_health'),('status.json','last_verified_rollout_status')]:
            p=run/name
            if p.exists():
                value=json.loads(p.read_text())
                if name=='status.json':value={k:value.get(k) for k in ['episodes','successes','success_rate','recent_success_rate',
                    'physics_invalid_episodes','vision_valid_fraction','vision_mean_error_m','closure_events','preparation_events','curriculum_level',
                    'curriculum_mode','sampled_episode_level_counts','completed_episode_level_counts','successes_by_level']}
                self.state[key]=value
        latest=run/'latest.pt'
        if latest.exists() and str(latest.resolve())!=self.cached_checkpoint:
            path=latest.resolve();saved=torch.load(path,map_location='cpu',weights_only=False)
            finite=all(bool(torch.isfinite(v).all()) for role in ['actor_state_dict','critic_state_dict'] for v in saved[role].values())
            if not finite:raise RuntimeError('Nonfinite model checkpoint')
            self.cached_checkpoint=str(path);self.state.update(last_verified_checkpoint=str(path),completed_updates=saved['infos']['completed_updates'],model_state_finite=True)
        self.publish()

    def worker(self,command,run,phase):
        if self.stop:raise InterruptedError('User stopped the session')
        env=os.environ.copy();env.update({'OMNI_KIT_ACCEPT_EULA':'YES','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                                         'MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'})
        log=run.parent/(run.name+'.stdout.log');log.parent.mkdir(parents=True,exist_ok=True)
        self.phase=phase
        with log.open('xb') as output:
            self.child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            self.state.update(pid=self.child.pid,phase=phase,status='starting' if phase=='training' else 'evaluating_frozen_checkpoint')
            if phase=='training':self.state.update(run_dir=str(run),stdout_log=str(log),training_pid=self.child.pid)
            else:self.state.update(evaluation_run_dir=str(run),evaluation_stdout_log=str(log),training_pid=None)
            self.publish();last=0
            while self.child.poll() is None:
                if phase=='training' and time.monotonic()-last>=10:self.refresh(run,log);last=time.monotonic()
                time.sleep(.5)
            code=self.child.returncode
        if phase=='training':self.refresh(run,log)
        self.child=None
        if self.stop or code!=0:
            stopped=run/'training_stopped.json'
            if stopped.exists() and json.loads(stopped.read_text()).get('exception_type')=='KeyboardInterrupt':self.stop=True
            if self.stop:raise InterruptedError('User stopped the worker')
            raise RuntimeError(f'{phase} worker failed ({code}); inspect {log}')

    def evaluate(self,checkpoint,tag,seeds):
        summaries=[]
        for seed in seeds:
            run=self.root/'evaluations'/f'{tag}_seed{seed}'
            cmd=['bash','scripts/run.sh','train','--response_probe','--policy_version','v4','--response_profile','fast',
                 '--grip_profile',getattr(self,'grip_profile','fast'),'--tracking_profile',getattr(self,'tracking_profile','fast'),
                 '--training_profile',getattr(self,'training_profile','legacy'),
                 '--checkpoint',str(checkpoint),'--device','cuda:0','--num_envs','256','--camera_resolution','192',
                 '--seed',str(seed),'--log_dir',str(run)]
            self.worker(cmd,run,'evaluation');summaries.append(json.loads((run/'response_summary.json').read_text()))
        result=aggregate(summaries);atomic_json(self.root/'evaluations'/f'{tag}_summary.json',result);return result

    def run(self):
        baseline_score=None;best=-1.;reference=-1.;best_checkpoint=self.baseline;stale=0;segment=0;plateau=False
        while self.completed<self.args.total_updates:
            segment+=1;size=100 if segment==1 else 200;target=min(self.completed+size,self.args.total_updates)
            run=self.root/f'segment_{segment:04d}'
            self.state.update(segment_target_updates=target,curriculum_level=self.level,completed_updates_before_resume=self.completed,
                              resume_checkpoint=str(self.checkpoint),status='starting',phase='training')
            command=['bash','scripts/run.sh','train','--policy_version','v4','--response_profile','fast','--grip_profile',getattr(self,'grip_profile','fast'),'--tracking_profile',getattr(self,'tracking_profile','fast'),'--checkpoint',str(self.checkpoint),
                     '--device','cuda:0','--num_envs','256','--steps_per_env','64','--camera_resolution','192','--seed',str(42+segment-1),
                     '--training_profile',getattr(self,'training_profile','legacy'),
                     '--curriculum_level',str(self.level),'--total_iterations',str(target),'--log_dir',str(run)]
            self.state['command']=command;self.worker(command,run,'training')
            if not (run/'training_complete.json').exists():raise RuntimeError('Missing completed training segment')
            self.checkpoint=(run/'latest.pt').resolve();self.completed=json.loads((run/'training_complete.json').read_text())['completed_updates']
            if self.completed!=target:raise RuntimeError('Unexpected completed update count')
            if baseline_score is None:
                baseline_score=self.evaluate(self.baseline,'baseline',self.args.validation_seeds)
                best=baseline_score['success_rate'];reference=best;self.state['baseline_validation']=baseline_score
            result=self.evaluate(self.checkpoint,f'update_{self.completed:05d}',self.args.validation_seeds)
            if result['throw_hashes']!=baseline_score['throw_hashes']:raise RuntimeError('Validation initial throws do not match baseline')
            self.state['latest_validation']=result
            selection=validation_selection(best,reference,stale,result['success_rate'])
            if selection['replace_best']:best_checkpoint=self.checkpoint
            best=selection['best'];reference=selection['reference'];stale=selection['stale']
            link=self.root/'best.tmp';link.symlink_to(best_checkpoint);link.replace(self.root/'best.pt')
            self.state.update(best_validation_success_rate=best,best_checkpoint=str(best_checkpoint),
                              validation_rounds_without_improvement=stale,plateau_reference_success_rate=reference)
            if self.level==0 and result['success_rate']>=.85:self.level=1
            elif self.level==1 and result['success_rate']>=.90:self.level=2
            self.publish()
            if plateau_reached(self.completed,getattr(self,'session_start_updates',0),stale,
                               getattr(self,'plateau_patience',5),getattr(self,'min_new_updates_before_plateau',0)):
                plateau=True;break
        final=self.evaluate(best_checkpoint,'final_test',getattr(self,'final_test_seeds',[7007,8008,9009,10010]))
        self.state.update(status='stopped_on_validation_plateau' if plateau else 'completed',phase='finished',pid=None,
                          final_independent_test=final,last_verified_checkpoint=str(self.checkpoint),best_checkpoint=str(best_checkpoint),supervisor_pid=None)
        self.publish();atomic_json(self.root/'session_complete.json',self.state)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--initial_checkpoint',type=Path,required=True)
    p.add_argument('--initial_curriculum_level',type=int,choices=[0,1,2])
    p.add_argument('--grip_profile',choices=['fast','pose_finish','early_close','soft_catch'],default='fast')
    p.add_argument('--tracking_profile',choices=['fast','predictive'],default='fast')
    p.add_argument('--training_profile',choices=['legacy','robust_l2','mixed_all'])
    p.add_argument('--plateau_patience',type=int,default=5)
    p.add_argument('--min_new_updates_before_plateau',type=int,default=0)
    p.add_argument('--final_test_seeds',type=int,nargs=4)
    p.add_argument('--baseline_checkpoint',type=Path,help='Re-evaluate the original baseline under this session action contract')
    p.add_argument('--output',type=Path,default=ROOT/'logs/ppo_catch/v2');p.add_argument('--total_updates',type=int,default=10000)
    p.add_argument('--validation_seeds',type=int,nargs=4,default=[3003,4004,5005,6006]);args=p.parse_args()
    session=TrainingSession(args)
    try:session.run()
    except BaseException as exc:
        if session.child is not None and session.child.poll() is None:
            session.child.send_signal(signal.SIGINT if session.phase=='training' else signal.SIGTERM)
            session.child.wait(timeout=45)
        session.state['resume_checkpoint']=session.state.get('last_verified_checkpoint',str(session.checkpoint))
        session.state.update(status='paused_by_user' if isinstance(exc,InterruptedError) else 'failed',phase='stopped',pid=None,
                             supervisor_pid=None,error=f'{type(exc).__name__}: {exc}')
        session.publish()
        if not isinstance(exc,InterruptedError):raise

if __name__=='__main__':main()
