"""Paired frozen-policy evaluation; never updates policy, optimizer or reward."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from diagnostic_metrics import EpisodeMetrics,summarize

def run_paired_diagnostics(env,runner,args,run):
    actor=runner.get_inference_policy();actor.eval()
    initial_model={k:v.detach().clone() for k,v in actor.state_dict().items()}
    actor.distribution.update(actor(env.get_observations()))
    std=actor.output_std[0].detach().clone()
    raw_std=actor.distribution.log_std_param.detach().exp().cpu().tolist()
    curriculum=env.curriculum;env.full_curriculum=True  # Freeze the loaded difficulty, not force 1.0.
    metadata={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'num_envs':env.num_envs,'camera_resolution':args.camera_resolution,'curriculum':curriculum,'rounds':args.diagnostic_rounds,
        'modes':['mean','noise_learned','noise_005'],'effective_std':std.cpu().tolist(),'raw_std':raw_std,
        'seeds':[args.diagnostic_seed+(i//2)*1000+(i%2)*101 for i in range(args.diagnostic_rounds)],
        'policy_and_normalization_frozen':True,'success_criteria_unchanged':True,'matched_initial_throw_hashes':{},
        'abandoned_auto_reset_episodes_excluded':True,'maximum_steps_per_trial':300}
    (run/'diagnostic_manifest.json').write_text(json.dumps(metadata,indent=2))
    metadata['paired_reset_version']='robot_ball_and_contact_observation_v2'
    if args.diagnostic_mode!='all':metadata['modes']=[args.diagnostic_mode]
    metadata['initial_observation_max_difference']={}
    all_records=[];start=time.monotonic();hashes={};initial_observations={};progress=run/'diagnostic_progress.json'
    trajectory=(run/'diagnostic_trajectories.jsonl').open('w')
    try:
        for mode in metadata['modes']:
            for r,seed in enumerate(metadata['seeds']):
                torch.manual_seed(seed)
                env.curriculum=curriculum
                ids=torch.arange(env.num_envs,device=env.device)
                root=env.robot.data.default_root_state.torch[:,:7].clone();root[:,:3]+=env.origins
                env.robot.write_root_pose_to_sim_index(root_pose=root)
                env.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((env.num_envs,6),device=env.device))
                env.reset(ids)
                # A trial must not inherit contact features from the preceding
                # episode; this correction is evaluation-only.
                env.force_features.zero_();env.total_contact.zero_()
                # Flush renderer buffers after a discontinuous paired-trial reset;
                # fit only the final exposure of this new episode generation.
                for _ in range(3):
                    env._sync_camera();env.sim.render();env.camera.update(0,force_recompute=True)
                env.exposure_batch.reset(ids.cpu().numpy());env.render_and_track()
                initial=np.concatenate((env.side[:,None].cpu().numpy(),(env.ball.data.root_pos_w.torch-env.origins).cpu().numpy(),
                    env.ball.data.root_lin_vel_w.torch.cpu().numpy(),env.robot.data.joint_pos.torch.cpu().numpy(),
                    env.robot.data.joint_vel.torch.cpu().numpy(),env.force_features.cpu().numpy(),env.actions.cpu().numpy()),axis=1)
                digest=hashlib.sha256(initial.tobytes()).hexdigest();metadata['matched_initial_throw_hashes'][f'{mode}:{r}']=digest
                if r in hashes and hashes[r]!=digest:raise RuntimeError('Unmatched initial conditions across modes')
                hashes[r]=digest
                observation=env.get_observations()['policy'].cpu().numpy().copy()
                np.savez_compressed(run/f'initial_{mode}_{r}.npz',physical_state=initial,policy_observation=observation)
                difference=float(np.max(np.abs(observation-initial_observations[r]))) if r in initial_observations else 0.
                metadata['initial_observation_max_difference'][f'{mode}:{r}']=difference
                if difference>.005:
                    (run/'initial_observation_mismatch.json').write_text(json.dumps({'mode':mode,'round':r,'max_difference':difference,
                        'max_difference_per_feature':np.max(np.abs(observation-initial_observations[r]),axis=0).tolist()},indent=2))
                    raise RuntimeError('Unmatched initial policy observations across modes')
                initial_observations.setdefault(r,observation)
                collector=EpisodeMetrics(env.num_envs,mode,r)
                noise_generator=torch.Generator(device=env.device);noise_generator.manual_seed(seed+17000)
                for steps in range(300):
                    def collect(tensors):
                        data={k:v.detach().cpu().numpy().copy() for k,v in tensors.items()}
                        for e in range(min(4,env.num_envs)):
                            if collector.active[e]:
                                row={'mode':mode,'round':r,'env_id':e,'step':steps+1}
                                for key in ['grip','hold','opposition','distance','speed','thumb','fingers','actions','requested_actions','forces','ball_position','capture_time','grasp_start']:
                                    value=data[key][e];row[key]=value.tolist() if hasattr(value,'tolist') else value
                                trajectory.write(json.dumps(row)+'\n')
                        collector.consume(data)
                    env.diagnostic_callback=collect
                    # Environment state is reset outside this block. no_grad
                    # keeps writable tensors, unlike inference_mode tensors.
                    with torch.no_grad():
                        mean=actor(env.get_observations())
                        noise=torch.randn(mean.shape,device=env.device,generator=noise_generator)
                        actions=mean if mode=='mean' else mean+noise*(std if mode=='noise_learned' else .05)
                        env.step(actions)
                    if env.physics_failures:raise RuntimeError('Physics failure invalidates diagnostic comparison')
                    if not collector.active.any():break
                    if steps%50==0:
                        progress.write_text(json.dumps({'mode':mode,'round':r,'control_steps':steps+1,'completed_trials':len(all_records)+len(collector.records),'elapsed_seconds':time.monotonic()-start},indent=2))
                env.diagnostic_callback=None
                if collector.active.any():raise RuntimeError('Censored episodes at 300 steps; do not count as failures')
                all_records.extend(collector.records)
                with (run/'diagnostic_episodes.jsonl').open('a') as f:
                    for record in collector.records:f.write(json.dumps(record)+'\n')
                partial={'manifest':metadata,'elapsed_seconds':time.monotonic()-start,'summary':summarize(all_records)}
                (run/'diagnostic_partial.json').write_text(json.dumps(partial,indent=2))
                print('[DIAGNOSTIC_ROUND]',mode,r,json.dumps(partial['summary'][mode]),flush=True)
        assert all(torch.equal(initial_model[k],v) for k,v in actor.state_dict().items()),'Actor/normalization changed during evaluation'
        metadata['actor_state_unchanged']=True
        metadata['matched_initial_throw_hashes_passed']=True
        metadata['physics_invalid_episodes']=env.physics_failures
        metadata['vision_valid_fraction']=env.vision_count/max(env.vision_samples,1)
        metadata['vision_mean_error_m']=env.vision_error_sum/max(env.vision_count,1)
        result={'manifest':metadata,'elapsed_seconds':time.monotonic()-start,'summary':summarize(all_records)}
        (run/'diagnostic_results.json').write_text(json.dumps(result,indent=2));env.save_camera(run/'camera_final.png')
        print('[DIAGNOSTIC_COMPLETE]',json.dumps(result),flush=True)
    finally:
        env.diagnostic_callback=None;trajectory.close()
