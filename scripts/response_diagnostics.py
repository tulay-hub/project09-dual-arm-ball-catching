"""Read-only first-episode tracking/closure metrics with frozen policy weights."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from collections import Counter
from policy_contract import XY_CONTRACT,XY_RESIDUAL_SCALE_M,XY_ACTION_LIMIT
from response_metrics import failure_stage,crossing_errors,failed_retention_criteria


def distribution(values):
    a=np.asarray([v for v in values if v is not None],dtype=float)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.quantile(a,.9))} if len(a) else {'n':0}


def run_response_diagnostics(env,runner,args,run):
    policy=runner.get_inference_policy();before={k:v.detach().clone() for k,v in policy.state_dict().items()}
    active=np.ones(env.num_envs,dtype=bool);previous_z=env.ball.data.root_pos_w.torch[:,2].cpu().numpy().copy()
    origins=env.origins.cpu().numpy().copy();previous_grip=np.zeros(env.num_envs,dtype=bool)
    indices=torch.arange(env.num_envs,device=env.device)[:,None]
    arm_ids=torch.tensor(env.arm_ids,device=env.device)[env.side]
    limits=env.limits[indices,arm_ids].cpu().numpy()
    arm_names=[[env.jnames[j] for j in env.arm_ids[int(side)]] for side in env.side.cpu().tolist()]
    contact_time=np.full(env.num_envs,np.nan);records=[{'env':e,'side':int(env.side[e]),'throw_kind':env.throw_kind[e],
        'contact_detected':False,'closure_latch_latency_s':None,'contact_to_70pct_s':None,'landing_xy_error_m':None,
        'max_reception_shift_mm':0.,'postcontact_samples_120ms':0,'postcontact_relative_speed_sum':0.,'first_contact_loss_s':None,'max_hold_steps':0,'arm_peak_speed_rad_s':0.,'max_tilted_hold_steps':0,'grip_breaks':0,
        'grip_break_criteria':{},'first_contact_time_s':None,'hand_to_goal_xy_m':None,'ball_to_goal_xy_m':None,
        'hand_to_goal_xy_at_contact_m':None,'preclose_fraction_at_contact':None,'precontact_steps':0,
        'prediction_xy_error_sum_m':0.,'prediction_frames':0,'precontact_xy_cap_saturated_sum':0.,'precontact_xy_abs_sum_mm':0.,'precontact_xy_beyond_20mm_sum':0.,'precontact_timing_sum_ms':0.,
        'precontact_arm_saturated_sum':0.,'minimum_joint_limit_margin_rad':None,'precontact_limit_push_frames':0,'precontact_limit_push_joints':{}} for e in range(env.num_envs)]
    throws=torch.cat((env.side[:,None],env.throw_start,env.throw_launch_velocity,env.throw_goal),dim=1).cpu().numpy()
    digest=hashlib.sha256(throws.tobytes()).hexdigest();traces=[];start=time.monotonic()
    def collect(tensors):
        nonlocal previous_z
        data={k:v.detach().cpu().numpy() for k,v in tensors.items()}
        arm_q=env.robot.data.joint_pos.torch[indices,arm_ids].cpu().numpy()
        velocity_command=env.velocity_targets[indices,arm_ids].cpu().numpy()
        margins=np.minimum(arm_q-limits[:,:,0],limits[:,:,1]-arm_q)
        pushing=((arm_q-limits[:,:,0]<.015)&(velocity_command<-.1))|((limits[:,:,1]-arm_q<.015)&(velocity_command>.1))
        tilted=env.tilt_hold.cpu().numpy();target=env.catch_palm_target.cpu().numpy()
        prediction=getattr(env,'predicted_intercept',env.catch_palm_target).cpu().numpy()
        predicted_ttc=env.predicted_contact_time.cpu().numpy();preclose=env.preclose_fraction.cpu().numpy()
        receive_shift=env.reception_shift.cpu().numpy() if hasattr(env,'reception_shift') else np.zeros((env.num_envs,3))
        drift=(env.reward_previous_center-env.hold_center).norm(dim=-1).cpu().numpy()
        for e in np.flatnonzero(active):
            rec=records[e];t=float(data['episode_time_s'][e]);z=data['ball_position'][e,2]+origins[e,2]
            if previous_z[e]>data['throw_goal'][e,2]>=z and data['ball_velocity'][e,2]<0 and rec['landing_xy_error_m'] is None:
                ball_world=data['ball_position'][e]+origins[e]
                rec.update(crossing_errors(data['ball_expected_at_hand'][e],ball_world,data['throw_goal'][e]))
                rec['contact_preceded_plane_crossing']=bool(rec['contact_detected'])
                rec['landing_xy_error_m']=float(np.linalg.norm(data['ball_expected_at_hand'][e,:2]-(data['ball_position'][e,:2]+origins[e,:2])))
            previous_z[e]=z
            if t>.15 and data['total_contact'][e]>.03 and not rec['contact_detected']:
                rec['contact_detected']=True;contact_time[e]=t;rec['first_contact_time_s']=t
                rec['hand_to_goal_xy_at_contact_m']=float(np.linalg.norm((data['ball_expected_at_hand'][e]-data['throw_goal'][e])[:2]))
                rec['preclose_fraction_at_contact']=float(preclose[e])
            rec['max_reception_shift_mm']=max(rec['max_reception_shift_mm'],float(np.linalg.norm(receive_shift[e])*1000))
            if rec['contact_detected']:
                if 0<=t-contact_time[e]<=.12:
                    rec['postcontact_samples_120ms']+=1
                    rec['postcontact_relative_speed_sum']+=float(data['speed'][e])
                if data['total_contact'][e]<=.03 and rec['first_contact_loss_s'] is None:rec['first_contact_loss_s']=t-contact_time[e]
                if data['grasp_start'][e]>=0 and rec['closure_latch_latency_s'] is None:
                    rec['closure_latch_latency_s']=float(data['grasp_start'][e]-contact_time[e])
                denominator=data['hand_q_nominal'][e]-data['hand_q_default'][e]
                fraction=(data['hand_q'][e]-data['hand_q_default'][e])/np.where(abs(denominator)>.05,denominator,1.)
                if np.min(fraction[[0,2,3]])>=.7 and rec['contact_to_70pct_s'] is None:rec['contact_to_70pct_s']=t-contact_time[e]
            if not rec['contact_detected']:
                rec['precontact_steps']+=1
                if predicted_ttc[e]>0:
                    rec['prediction_frames']+=1
                    rec['prediction_xy_error_sum_m']+=float(np.linalg.norm((prediction[e]-data['throw_goal'][e])[:2]))
                if pushing[e].any():
                    rec['precontact_limit_push_frames']+=1
                    for j in np.flatnonzero(pushing[e]):
                        name=arm_names[e][j];rec['precontact_limit_push_joints'][name]=rec['precontact_limit_push_joints'].get(name,0)+1
                rec['precontact_arm_saturated_sum']+=float(np.mean(abs(data['arm_qd'][e])>2.85))
                if data['actions'].shape[-1]==14:
                    action=data['actions'][e,int(data['side'][e])*7:int(data['side'][e])*7+7]
                    rec['precontact_xy_abs_sum_mm']+=float(np.abs(action[4:6]).mean()*XY_RESIDUAL_SCALE_M*1000)
                    rec['precontact_xy_cap_saturated_sum']+=float(np.mean(np.abs(action[4:6])>XY_ACTION_LIMIT*.95))
                    rec['precontact_xy_beyond_20mm_sum']+=float(np.mean(np.abs(action[4:6])>1.0))
                    rec['precontact_timing_sum_ms']+=float(action[6]*40)
            margin=float(margins[e].min())
            rec['minimum_joint_limit_margin_rad']=margin if rec['minimum_joint_limit_margin_rad'] is None else min(margin,rec['minimum_joint_limit_margin_rad'])
            rec['max_tilted_hold_steps']=max(rec['max_tilted_hold_steps'],int(tilted[e]))
            if previous_grip[e] and not data['grip'][e]:
                rec['grip_breaks']+=1
                retention=env.contact_evidence[e].get('retention',{})
                for reason in failed_retention_criteria(retention,float(data['speed'][e]),float(drift[e]),float(data['distance'][e])):
                    rec['grip_break_criteria'][reason]=rec['grip_break_criteria'].get(reason,0)+1
            previous_grip[e]=bool(data['grip'][e])
            rec['max_hold_steps']=max(rec['max_hold_steps'],int(data['hold'][e]));rec['arm_peak_speed_rad_s']=max(rec['arm_peak_speed_rad_s'],float(abs(data['arm_qd'][e]).max()))
            if e<8:traces.append({'env':int(e),'time_s':t,'contact_n':float(data['total_contact'][e]),'palm':data['palm_position'][e].tolist(),
                'goal':data['throw_goal'][e].tolist(),'arm_qd':data['arm_qd'][e].tolist(),'arm_target_error':data['arm_target_error'][e].tolist(),
                'hand_q':data['hand_q'][e].tolist(),'grip':bool(data['grip'][e]),'hold':int(data['hold'][e]),
                'ball_world':(data['ball_position'][e]+origins[e]).tolist(),'ball_velocity':data['ball_velocity'][e].tolist(),
                'catch_palm_target':target[e].tolist(),'predicted_intercept':prediction[e].tolist(),'predicted_ttc_at_command_s':float(predicted_ttc[e]),
                'actions':data['actions'][e].tolist(),'visible':bool(data['visible'][e]),
                'arm_q':arm_q[e].tolist(),'arm_velocity_command':velocity_command[e].tolist(),'arm_limit_margin':margins[e].tolist(),
                'relative_speed':float(data['speed'][e]),'relative_drift':float(drift[e]),'tilted_hold':int(tilted[e])})
            if data['done'][e]:
                rec['outcome']='success' if data['success'][e] else ('failed' if data['failed'][e] else 'timeout');rec['duration_s']=t;active[e]=False
                rec['failure_stage']=failure_stage(rec['outcome'],rec['contact_detected'],rec['max_hold_steps'])
                rec['mean_postcontact_relative_speed_120ms']=rec['postcontact_relative_speed_sum']/max(1,rec['postcontact_samples_120ms'])
                count=max(1,rec['precontact_steps'])
                rec['mean_prediction_xy_error_m']=rec['prediction_xy_error_sum_m']/rec['prediction_frames'] if rec['prediction_frames'] else None
                rec['precontact_xy_cap_saturation_fraction']=rec['precontact_xy_cap_saturated_sum']/count
                rec['mean_precontact_xy_abs_mm']=rec['precontact_xy_abs_sum_mm']/count
                rec['mean_precontact_timing_shift_ms']=rec['precontact_timing_sum_ms']/count
                rec['precontact_xy_beyond_20mm_fraction']=rec['precontact_xy_beyond_20mm_sum']/count
                rec['precontact_arm_saturation_fraction']=rec['precontact_arm_saturated_sum']/count
    env.diagnostic_callback=collect
    for step in range(env.max_episode_length+1):
        with torch.no_grad():env.step(policy(env.get_observations()))
        if env.physics_failures:raise RuntimeError('Physics failure invalidates response probe')
        if not active.any():break
    env.diagnostic_callback=None
    assert not active.any(),'Censored first episode';assert all(torch.equal(v,policy.state_dict()[k]) for k,v in before.items())
    summary={'episodes':len(records),'successes':sum(x['outcome']=='success' for x in records),'contacts':sum(x['contact_detected'] for x in records),
        'outcomes':{key:sum(x['outcome']==key for x in records) for key in ['success','failed','timeout']},
        'diagnostic_schema_version':5,'grip_profile':getattr(env,'grip_profile','fast'),'tracking_profile':getattr(env,'tracking_profile','fast'),'action_contract':dict(XY_CONTRACT) if env.num_actions==14 else {'policy_version':'v3'},'failure_stage_counts':dict(Counter(x['failure_stage'] for x in records)),
        'precontact_limit_push_joint_counts':dict(sum((Counter(x['precontact_limit_push_joints']) for x in records),Counter())),
        'retention_break_criteria_counts':dict(sum((Counter(x['grip_break_criteria']) for x in records),Counter())),
        'first_episode_only':True,'policy_state_unchanged':True,'throw_sha256':digest,'physics_invalid_episodes':env.physics_failures,'elapsed_seconds':time.monotonic()-start,
        'metrics':{key:distribution([x[key] for x in records]) for key in ['max_reception_shift_mm','mean_postcontact_relative_speed_120ms','first_contact_loss_s','landing_xy_error_m','closure_latch_latency_s','contact_to_70pct_s','max_hold_steps','arm_peak_speed_rad_s','hand_to_goal_xy_m','ball_to_goal_xy_m','hand_to_goal_xy_at_contact_m',
            'mean_prediction_xy_error_m','precontact_xy_cap_saturation_fraction','mean_precontact_xy_abs_mm','mean_precontact_timing_shift_ms','precontact_xy_beyond_20mm_fraction','precontact_arm_saturation_fraction','minimum_joint_limit_margin_rad','precontact_limit_push_frames']}}
    (Path(run)/'response_episodes.json').write_text(json.dumps(records,indent=2));(Path(run)/'response_traces.json').write_text(json.dumps(traces,indent=2));(Path(run)/'response_summary.json').write_text(json.dumps(summary,indent=2));print('[RESPONSE_PROBE]',json.dumps(summary),flush=True)
