"""Reproducible paired reward/noise diagnostic analysis; reads evaluation files only."""
from pathlib import Path
import argparse,json,datetime
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
from diagnostic_metrics import summarize

def analyze(run):
    result=json.loads((run/'diagnostic_results.json').read_text())
    records=[json.loads(line) for line in (run/'diagnostic_episodes.jsonl').read_text().splitlines()]
    groups=defaultdict(dict)
    for r in records:groups[r['mode']][(r['round'],r['env_id'])]=r
    comparisons={};rng=np.random.default_rng(1401)
    for first,second in [('mean','noise_learned'),('noise_005','noise_learned'),('mean','noise_005')]:
        assert groups[first].keys()==groups[second].keys()
        keys=sorted(groups[first]);x=np.array([groups[first][k]['outcome']=='success' for k in keys]);y=np.array([groups[second][k]['outcome']=='success' for k in keys]);delta=x.astype(int)-y.astype(int)
        b=int(np.sum(x&~y));c=int(np.sum(~x&y));samples=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(-1)
        comparisons[f'{first}_minus_{second}']={'paired_trials':len(keys),'success_rate_difference':float(delta.mean()),
            'paired_bootstrap_95':np.quantile(samples,[.025,.975]).tolist(),'first_only_success':b,'second_only_success':c,
            'mcnemar_exact_two_sided_p':float(binomtest(b,b+c,.5).pvalue) if b+c else 1.}
    trajectory=defaultdict(list)
    with (run/'diagnostic_trajectories.jsonl').open() as f:
        for line in f:
            r=json.loads(line);trajectory[(r['mode'],r['round'],r['env_id'])].append(r)
    interrupted={};case_details=[];action_variation={}
    for mode in groups:
        lengths=[];safe=[];case_count=0;changes={'grip_break':[],'continuous_grip':[]}
        for key,trace in trajectory.items():
            if key[0]!=mode:continue
            case_count+=1;start=None;gaps=[]
            side=groups[mode][(key[1],key[2])]['side']
            for i,r in enumerate(trace):
                if i and trace[i-1]['grip']:
                    delta=np.abs(np.asarray(r['actions']).reshape(2,4)[side]-np.asarray(trace[i-1]['actions']).reshape(2,4)[side])
                    changes['continuous_grip' if r['grip'] else 'grip_break'].append(delta)
                if i and trace[i-1]['grip'] and not r['grip']:start=i
                if start is not None and r['grip']:
                    gap=trace[start:i];lengths.append(len(gap));ok=all(t['distance']<.075 and t['speed']<.25 for t in gap);safe.append(ok);gaps.append(len(gap));start=None
            record=groups[mode][(key[1],key[2])]
            case_details.append({'mode':mode,'round':key[1],'env_id':key[2],'outcome':record['outcome'],'max_hold_steps':record['max_hold_steps'],'return':record['return'],'completed_grip_gap_lengths':gaps})
        interrupted[mode]={'sampled_trajectories':case_count,'completed_grip_interruptions':len(lengths),
            'one_or_two_control_step_interruptions':sum(x<=2 for x in lengths),
            'ball_position_and_speed_good_through_gap':sum(safe),'median_gap_steps':float(np.median(lengths)) if lengths else None,
            'sampling_note':'First four environments per round; trailing gaps excluded; not all episodes.'}
        action_variation[mode]={name:{'control_steps':len(values),'mean_normalized_delta_per_active_action':np.mean(values,axis=0).tolist() if values else None} for name,values in changes.items()}
    per_round={mode:{str(i):{'episodes':sum(r['round']==i for r in values.values()),'successes':sum(r['round']==i and r['outcome']=='success' for r in values.values())} for i in sorted(set(r['round'] for r in values.values()))} for mode,values in groups.items()}
    report={'checked_at':datetime.datetime.now().astimezone().isoformat(),'run_dir':str(run.resolve()),'manifest':result['manifest'],
        'episodes':len(records),'summary':summarize(records),'paired_comparisons':comparisons,'per_round':per_round,
        'sampled_trajectory_interruptions':interrupted,'sampled_cases':case_details,'sampled_action_variation':action_variation,
        'interpretation_limits':['Fixed checkpoint/difficulty and two seed families; not a full curriculum evaluation.',
            'Reward/outcome comparisons are observational and duration-confounded; constructed offline audit establishes a reward-design risk.',
            'Brief contact-condition gaps with low ball speed do not establish full physical grasp stability.',
            'No reward/noise training change is made by this script.']}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=analyze(args.run);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'episodes':report['episodes'],'summary':{k:{f:v[f] for f in ['episodes','successes','success_rate','outcomes','reward_by_outcome']} for k,v in report['summary'].items()},'paired_comparisons':report['paired_comparisons']},ensure_ascii=False,indent=2))
