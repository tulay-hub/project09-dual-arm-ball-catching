"""Read-only episode metrics, independent of Isaac and policy implementation."""
import numpy as np

COMPONENTS=['proximity','capture','grip','terminal_success','action_penalty','slew_penalty']

def wilson(successes,total):
    if total==0:return [0.,1.]
    z=1.96;p=successes/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d
    radius=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [float(center-radius),float(center+radius)]

class EpisodeMetrics:
    def __init__(self,num_envs,mode,round_id,gamma=.99):
        self.num_envs=num_envs;self.mode=mode;self.round_id=round_id;self.gamma=gamma
        self.active=np.ones(num_envs,dtype=bool);self.previous_grip=np.zeros(num_envs,dtype=bool)
        self.steps=np.zeros(num_envs,dtype=int);self.max_hold=np.zeros(num_envs,dtype=int)
        self.grip_steps=np.zeros(num_envs,dtype=int);self.breaks=np.zeros(num_envs,dtype=int)
        self.components=np.zeros((num_envs,6));self.discounted=np.zeros(num_envs)
        self.criteria=np.zeros((num_envs,5),dtype=int);self.action_delta=np.zeros(num_envs)
        self.action_saturation=np.zeros(num_envs);self.request_saturation=np.zeros(num_envs)
        self.visibility=np.zeros(num_envs);self.records=[]

    def consume(self,d):
        active=self.active.copy();ids=np.flatnonzero(active)
        if not len(ids):return []
        checks=np.stack((d['thumb'],d['fingers'],d['opposition']>.1,d['distance']<.075,d['speed']<.25),axis=-1)
        broken=active & self.previous_grip & ~d['grip']
        self.breaks+=broken;self.criteria[broken]+=~checks[broken]
        self.max_hold[ids]=np.maximum(self.max_hold[ids],d['hold'][ids])
        self.grip_steps[ids]+=d['grip'][ids]
        self.components[ids]+=d['reward_components'][ids]
        self.discounted[ids]+=self.gamma**self.steps[ids]*d['reward'][ids]
        assert np.allclose(d['reward_components'][ids].sum(-1),d['reward'][ids],atol=1e-5)
        self.steps[ids]+=1;self.visibility[ids]+=d['visible'][ids]
        side=d['side'][ids].astype(int);applied=d['actions'].reshape(self.num_envs,2,4)[ids,side]
        previous=d['previous_actions'].reshape(self.num_envs,2,4)[ids,side]
        requested=d['requested_actions'].reshape(self.num_envs,2,4)[ids,side]
        self.action_delta[ids]+=np.abs(applied-previous).mean(-1)
        self.action_saturation[ids]+=(np.abs(applied)>.99).mean(-1)
        self.request_saturation[ids]+=(np.abs(requested)>.99).mean(-1)
        ended=[]
        for i in np.flatnonzero(active&d['done']):
            outcome='success' if d['success'][i] else ('failed' if d['failed'][i] else 'timeout')
            record={'mode':self.mode,'round':self.round_id,'env_id':int(i),'side':int(d['side'][i]),
                'outcome':outcome,'steps':int(self.steps[i]),'max_hold_steps':int(self.max_hold[i]),
                'grip_steps':int(self.grip_steps[i]),'hold_interruptions':int(self.breaks[i]),
                'break_criteria_failures':self.criteria[i].tolist(),
                'reward_components':dict(zip(COMPONENTS,self.components[i].tolist())),
                'return':float(self.components[i].sum()),'discounted_return':float(self.discounted[i]),
                'mean_active_action_delta':float(self.action_delta[i]/self.steps[i]),
                'applied_saturation_fraction':float(self.action_saturation[i]/self.steps[i]),
                'requested_saturation_fraction':float(self.request_saturation[i]/self.steps[i]),
                'visible_fraction':float(self.visibility[i]/self.steps[i]),
                'capture_started':bool(d['capture_time'][i]>=0),'closure_started':bool(d['grasp_start'][i]>=0)}
            self.records.append(record);ended.append(record);self.active[i]=False
        self.previous_grip[ids]=d['grip'][ids]
        return ended

def summarize(records):
    result={}
    for mode in sorted(set(r['mode'] for r in records)):
        group=[r for r in records if r['mode']==mode];n=len(group);success=sum(r['outcome']=='success' for r in group)
        summary={'episodes':n,'successes':success,'success_rate':success/n,'success_rate_wilson_95':wilson(success,n),
            'outcomes':{s:sum(r['outcome']==s for r in group) for s in ['success','failed','timeout']}}
        for field in ['steps','max_hold_steps','grip_steps','hold_interruptions','return','discounted_return','mean_active_action_delta','applied_saturation_fraction','requested_saturation_fraction','visible_fraction']:
            values=np.array([r[field] for r in group]);summary[field]={'mean':float(values.mean()),'median':float(np.median(values))}
        summary['break_criteria_failures']=dict(zip(['thumb','two_fingers','opposing_force','palm_distance','ball_speed'],np.sum([r['break_criteria_failures'] for r in group],axis=0).tolist()))
        summary['reward_by_outcome']={}
        for outcome in ['success','failed','timeout']:
            subset=[r for r in group if r['outcome']==outcome]
            if subset:
                summary['reward_by_outcome'][outcome]={'episodes':len(subset),'return_mean':float(np.mean([r['return'] for r in subset])),
                    'discounted_return_mean':float(np.mean([r['discounted_return'] for r in subset])),
                    'components_mean':{c:float(np.mean([r['reward_components'][c] for r in subset])) for c in COMPONENTS}}
        summary['left_right_success_rate']={str(side):float(np.mean([r['outcome']=='success' for r in group if r['side']==side])) if any(r['side']==side for r in group) else None for side in [0,1]}
        result[mode]=summary
    return result
