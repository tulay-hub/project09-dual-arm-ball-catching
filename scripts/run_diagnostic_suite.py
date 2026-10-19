"""Fresh-simulator paired evaluation orchestration. No training is launched here."""
from pathlib import Path
import json,os,subprocess,datetime,shutil,time
ROOT=Path(__file__).resolve().parents[1]
plan_path=ROOT/'reports/reward_noise_diagnostic_plan_20261014.json'
plan=json.loads(plan_path.read_text());suite=ROOT/'logs/evaluation/diagnose_20261014_fresh_suite_v1';suite.mkdir(exist_ok=False,parents=True)
plan.update(diagnostic_suite_pid=os.getpid(),diagnostic_run_dir=str(suite),diagnostic_status='fresh_suite_running')
plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
env=os.environ.copy();env['OMNI_KIT_ACCEPT_EULA']='YES';runs=[]
for mode in ['mean','noise_learned','noise_005']:
    run=suite/mode;log=ROOT/f'reports/diagnose_20261014_fresh_{mode}_v1.log'
    args=['bash','scripts/run.sh','train','--diagnose','--diagnostic_mode',mode,'--num_envs','128','--camera_resolution','192','--checkpoint',plan['checkpoint'],'--diagnostic_rounds','1','--diagnostic_seed','1001','--seed','1001','--log_dir',str(run)]
    with log.open('x') as output:
        child=subprocess.Popen(args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
        plan.update(diagnostic_pid=child.pid,diagnostic_current_mode=mode,diagnostic_log=str(log),diagnostic_command=args)
        plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
        code=child.wait()
    if code!=0 or not (run/'diagnostic_results.json').exists():
        (suite/'suite_failed.json').write_text(json.dumps({'mode':mode,'exit_code':code,'log':str(log)},indent=2));raise SystemExit(1)
    runs.append(run)
base=None;maximum_difference={};hashes=[]
for run in runs:
    import numpy as np
    initial=np.load(run/f'initial_{run.name}_0.npz')
    if base is None:base={k:initial[k].copy() for k in initial.files}
    assert np.array_equal(base['physical_state'],initial['physical_state']),'Initial physical state mismatch across fresh processes'
    difference=float(np.max(np.abs(base['policy_observation']-initial['policy_observation'])))
    maximum_difference[run.name]=difference
    if difference>.005:raise RuntimeError('Fresh-process policy observation mismatch')
results=[json.loads((run/'diagnostic_results.json').read_text()) for run in runs]
manifest=results[0]['manifest'];manifest.update(modes=['mean','noise_learned','noise_005'],fresh_process_per_condition=True,
    paired_physical_states_exactly_equal=True,initial_observation_max_difference=maximum_difference,source_runs=[str(run) for run in runs])
assert all(r['manifest']['actor_state_unchanged'] for r in results)
manifest['checkpoint_hashes_equal']=len(set(r['manifest']['checkpoint_sha256'] for r in results))==1
manifest['physics_invalid_episodes']=sum(r['manifest']['physics_invalid_episodes'] for r in results)
manifest['vision_valid_fraction_per_mode']={run.name:r['manifest']['vision_valid_fraction'] for run,r in zip(runs,results)}
combined={'manifest':manifest,'summary':{mode:v for r in results for mode,v in r['summary'].items()}}
(suite/'diagnostic_results.json').write_text(json.dumps(combined,indent=2))
for filename in ['diagnostic_episodes.jsonl','diagnostic_trajectories.jsonl']:
    with (suite/filename).open('w') as target:
        for run in runs:target.write((run/filename).read_text())
(suite/'suite_complete.json').write_text(json.dumps({'time':datetime.datetime.now().isoformat(),'initial_observation_max_difference':maximum_difference,'runs':[str(run) for run in runs]},indent=2))
plan.update(diagnostic_status='completed_fresh_suite');plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
