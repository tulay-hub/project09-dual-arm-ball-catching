"""Project09 camera-based PPO catch/grasp training, evaluation and smoke entry."""
from pathlib import Path
from datetime import datetime
import argparse,json,shutil,hashlib,os,time,traceback
from isaaclab.app import add_launcher_args,launch_simulation
from isaaclab.physics import PhysicsCfg
from training_progress import completed_updates_from_checkpoint
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--num_envs',type=int,default=8)
p.add_argument('--iterations',type=int,default=10000)
p.add_argument('--total_iterations',type=int,help='Total completed-update target, including checkpoint progress')
p.add_argument('--steps_per_env',type=int,default=64)
p.add_argument('--camera_resolution',type=int,default=192,choices=[128,160,192,256])
p.add_argument('--seed',type=int,default=42)
p.add_argument('--log_dir',type=Path)
p.add_argument('--checkpoint',type=Path)
p.add_argument('--migrate_checkpoint',type=Path,help='Explicit v3 to v4 model migration; optimizer resets')
p.add_argument('--policy_version',choices=['v3','v4'],default='v3')
p.add_argument('--training_profile',choices=['legacy','robust_l2','mixed_all'],default=None)
p.add_argument('--curriculum_level',type=int,choices=[0,1,2],default=0)
p.add_argument('--play',action='store_true')
p.add_argument('--baseline',action='store_true')
p.add_argument('--smoke_steps',type=int,default=0)
p.add_argument('--smoke_xy_action',type=float,default=0.,help='Diagnostic-only fixed XY command; requires --smoke_steps and v4')
p.add_argument('--eval_steps',type=int,default=600)
p.add_argument('--full_curriculum',action='store_true')
p.add_argument('--grip_profile',choices=['fast','pose_finish','early_close','soft_catch'],default=None,help='Defaults to saved checkpoint grip controller')
p.add_argument('--tracking_profile',choices=['fast','predictive'],default=None,help='Defaults to the checkpoint controller when resuming or playing')
p.add_argument('--response_profile',choices=['legacy','fast_arm','fast'],default=None,help='Controller response profile; defaults to task configuration')
p.add_argument('--response_probe',action='store_true',help='Frozen-policy catch tracking and closure latency diagnostics')
p.add_argument('--profile_eval',action='store_true',help='Playback-only step timing and rendered frame export for numerical regression')
p.add_argument('--diagnose',action='store_true',help='Read-only paired mean/training-noise/low-noise evaluation')
p.add_argument('--diagnostic_rounds',type=int,default=4)
p.add_argument('--diagnostic_seed',type=int,default=1001)
p.add_argument('--diagnostic_mode',choices=['all','mean','noise_learned','noise_005'],default='all')
p.add_argument('--live_view',action='store_true',help='Show a live third-person camera via local MJPEG page')
p.add_argument('--live_port',type=int,default=6020)
p.add_argument('--play_rebind_views',action='store_true',help='Rebind physics views after first GUI startup')
p.add_argument('--full_ui',action='store_true',help='Keep full Kit editor panels during native playback')
p.add_argument('--capture_viewport',action='store_true',help='Export one native viewport frame for playback validation')
p.add_argument('--closure_probe',action='store_true',help='Playback-only controlled gravity-drop check of tactile closure')
add_launcher_args(p);args=p.parse_args();args.physics='isaacsim_physx';args.enable_cameras=True
kit_requested=('kit' in args.visualizer) if isinstance(args.visualizer,(list,tuple)) else args.visualizer=='kit'
if args.diagnose:
    if not args.checkpoint:p.error('--diagnose requires a frozen checkpoint')
    if args.diagnostic_rounds<1:p.error('--diagnostic_rounds must be positive')
    args.play=True
if args.checkpoint and args.migrate_checkpoint:p.error('Choose resume or migrate, not both')
if args.migrate_checkpoint and args.policy_version!='v4':p.error('Migration requires --policy_version v4')
if args.diagnose and args.policy_version=='v4':p.error('Use --response_probe for v4 first-episode evaluation')
if args.play and not (args.checkpoint or args.migrate_checkpoint or args.baseline):p.error('--play requires --checkpoint or --baseline')
if args.response_probe:
    if not (args.checkpoint or args.migrate_checkpoint):p.error('--response_probe requires a frozen checkpoint')
    args.play=True
if args.smoke_xy_action and (not args.smoke_steps or args.policy_version!='v4'):p.error('--smoke_xy_action requires --smoke_steps and v4')
if args.profile_eval and not args.play:p.error('--profile_eval is playback-only')
if args.baseline and not args.play:p.error('--baseline is evaluation-only')
if args.live_view and (not args.play or args.diagnose):p.error('--live_view is policy playback only')
if args.play_rebind_views and not args.play:p.error('--play_rebind_views is playback-only')
if args.closure_probe and (not args.play or not args.baseline or args.checkpoint):p.error('--closure_probe requires --play --baseline and cannot train')
if args.device=='cpu' and not args.closure_probe:p.error('This task requires CUDA hand-ball contact tensors on the validated Isaac 6.1 stack; use --device cuda:0. CPU is allowed only for --closure_probe diagnostics.')
if args.total_iterations is not None and args.total_iterations<1:p.error('--total_iterations must be positive')
if args.num_envs<1 or args.num_envs>256 or args.iterations<1:p.error('num_envs must be 1..256; iterations must be positive')

def main():
    if args.tracking_profile is None or args.grip_profile is None or args.training_profile is None:
        import torch
        from policy_contract import resolve_tracking_profile,resolve_grip_profile
        from training_profiles import resolve_training_profile
        checkpoint_task=(torch.load(args.checkpoint,map_location='cpu',weights_only=False).get('infos') or {}).get('task',{}) if args.checkpoint else None
        proposal=json.loads((ROOT/'config/task_proposal.json').read_text())
        default=proposal.get('controller_response',{}).get('tracking_profile','fast') if args.policy_version=='v4' else 'fast'
        args.tracking_profile=resolve_tracking_profile(args.tracking_profile,checkpoint_task,default)
        grip_default=proposal.get('controller_response',{}).get('grip_profile','fast') if args.policy_version=='v4' else 'fast'
        args.grip_profile=resolve_grip_profile(args.grip_profile,checkpoint_task,grip_default)
        training_default=proposal.get('training',{}).get('training_profile','legacy') if args.policy_version=='v4' else 'legacy'
        args.training_profile=resolve_training_profile(args.training_profile,checkpoint_task,training_default)
    run=(args.log_dir or ROOT/'logs/ppo_catch'/datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    run.mkdir(parents=True,exist_ok=False)
    (run/'launch.json').write_text(json.dumps({'pid':os.getpid(),'start_time':datetime.now().isoformat(),
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'stage':'offcenter_mesh_catch_retention'},indent=2,default=str))
    snapshot=run/'source';snapshot.mkdir()
    hashes={}
    for pattern in ['scripts/*.py','scripts/run.sh','config/*.json']:
        for f in ROOT.glob(pattern):
            destination=snapshot/f.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(f,destination);hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    (run/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    with launch_simulation(PhysicsCfg(),args) as physics:
        import torch
        import isaaclab.sim as su
        from ball_catch_env import BallCatchEnv
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlMLPModelCfg,RslRlOnPolicyRunnerCfg,RslRlPpoAlgorithmCfg
        scene_parameters=json.loads((ROOT/'config/scene.json').read_text())
        physics_hz=scene_parameters['physics_hz']
        sim=su.SimulationContext(su.SimulationCfg(dt=1./physics_hz,render_interval=physics_hz//60,device=args.device,physics=physics))
        env=BallCatchEnv(sim,args.num_envs,run,args.seed,args.camera_resolution,args.full_curriculum,args.live_view,args.play_rebind_views,args.response_profile,args.policy_version,not args.play,args.curriculum_level,args.tracking_profile,args.grip_profile,args.training_profile)
        if args.play and kit_requested:
            origin=env.origins[0].cpu().tolist()
            sim.set_camera_view(eye=tuple(origin[i]+v for i,v in enumerate((1.55,-1.5,1.7))),
                target=tuple(origin[i]+v for i,v in enumerate((.2,0,.85))))
            if not args.full_ui:
                from compact_playback import configure_compact_playback
                configure_compact_playback(run,physics_hz)
                for viz in sim.visualizers:
                    if viz.cfg.visualizer_type=='kit':viz.cfg.window_width=960;viz.cfg.window_height=540
        (run/'task_config.json').write_text(json.dumps(env.config,indent=2))
        rgb=env.camera.data.output['rgb'].torch[...,:3]
        health={'rgb_nonblack_fraction':float((rgb.amax(dim=(1,2,3))>0).float().mean()),
            'ball_visible_fraction':float(env.visible.float().mean()),'observation_shape':list(env.get_observations()['policy'].shape),
            'action_dim':env.num_actions,'rubber_pad_count':0,
            'friction_geometry':'original_hand_collision_meshes'}
        (run/'camera_health.json').write_text(json.dumps(health,indent=2));print('[CAMERA_HEALTH]',json.dumps(health),flush=True)
        if health['rgb_nonblack_fraction']<.99 or health['ball_visible_fraction']<.75:
            import numpy as np
            from PIL import Image,ImageDraw
            frames=rgb.cpu().numpy().astype('uint8');vis=env.visible.cpu().numpy()
            depth=env.camera.data.output['distance_to_image_plane'].torch.cpu().numpy()
            from ball_tracker import red_ball_mask
            red=np.stack([red_ball_mask(frames[e],depth[e]) for e in range(env.num_envs)])
            counts=red.sum(axis=(1,2));records=[]
            for e in range(env.num_envs):
                records.append({'env':e,'kind':env.throw_kind[e],'visible':bool(vis[e]),'red_pixels':int(counts[e]),
                    'ball_world':env.ball.data.root_pos_w.torch[e].cpu().tolist(),
                    'camera_world':env.camera.data.pos_w.torch[e].cpu().tolist(),
                    'camera_quat_ros':env.camera.data.quat_w_ros.torch[e].cpu().tolist(),
                    'intrinsics':env.camera.data.intrinsic_matrices.torch[e].cpu().tolist()})
            selected=[]
            for kind in ['front','left_side','right_side']:
                for flag in [True,False]:selected.extend([r['env'] for r in records if r['kind']==kind and r['visible']==flag][:3])
            montage=Image.new('RGB',(4*192,((len(selected)+3)//4)*216),(40,40,40));draw=ImageDraw.Draw(montage)
            for n,e in enumerate(selected):
                x=(n%4)*192;y=(n//4)*216;montage.paste(Image.fromarray(frames[e]),(x,y));draw.text((x,y+193),f'{e} {env.throw_kind[e]} red={counts[e]} ok={vis[e]}',fill='white')
            montage.save(run/'camera_commissioning_montage.png')
            (run/'camera_commissioning_details.json').write_text(json.dumps(records,indent=2))
            raise RuntimeError('Camera commissioning failed; refusing training')
        env.save_camera(run/'camera_initial.png')
        if args.smoke_steps:
            begin=time.monotonic()
            from policy_contract import XY_ACTION_INDEX,XY_RESIDUAL_SCALE_M
            command=torch.zeros((args.num_envs,env.num_actions),device=args.device)
            if args.smoke_xy_action:command[:,XY_ACTION_INDEX]=torch.tensor([1.,-1.,-1.,1.],device=args.device)*args.smoke_xy_action
            max_xy=torch.zeros(4,device=args.device)
            for _ in range(args.smoke_steps):
                env.step(command)
                if env.num_actions==14:max_xy=torch.maximum(max_xy,env.actions[:,XY_ACTION_INDEX].abs().amax(dim=0))
            env.save_camera(run/'camera_final.png')
            (run/'smoke.json').write_text(json.dumps({'steps':args.smoke_steps,'elapsed_seconds':time.monotonic()-begin,
                'requested_xy_action':args.smoke_xy_action,'max_applied_xy_mm':(max_xy*XY_RESIDUAL_SCALE_M*1000).cpu().tolist(),
                'physics_invalid_episodes':env.physics_failures,
                'transitions':env.total_steps,'episodes':env.episodes,'successes':env.successes,
                'finite_observations':bool(torch.isfinite(env.get_observations()['policy']).all()),
                'vision_mean_error_m':env.vision_error_sum/max(env.vision_count,1)},indent=2));return
        cfg=RslRlOnPolicyRunnerCfg(seed=args.seed,device=args.device,num_steps_per_env=args.steps_per_env,
            max_iterations=args.total_iterations or args.iterations,save_interval=5,experiment_name='lkls73_camera_catch_grasp',
            obs_groups={'actor':['policy'],'critic':['policy']},
            actor=RslRlMLPModelCfg(hidden_dims=[128,128,64],activation='elu',obs_normalization=True,
                distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=.15,std_type='log')),
            critic=RslRlMLPModelCfg(hidden_dims=[128,128,64],activation='elu',obs_normalization=True),
            algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,
                entropy_coef=.001,num_learning_epochs=5,num_mini_batches=4,learning_rate=1e-4 if args.policy_version=='v4' else 3e-4,
                schedule='adaptive',gamma=.99,lam=.95,desired_kl=.015,max_grad_norm=1.)).to_dict()
        for model in ['actor','critic']:
            for key in ['stochastic','init_noise_std','noise_std_type','state_dependent_std']:cfg[model].pop(key,None)
        cfg['actor']['distribution_cfg']['std_range']=[.05,.25]
        from training_profiles import CONSERVATIVE_PROFILES,ROBUST_PPO,configure_optimizer
        if args.training_profile in CONSERVATIVE_PROFILES:cfg['algorithm'].update(ROBUST_PPO)
        (run/'runner_config.json').write_text(json.dumps(cfg,indent=2))
        class CatchRunner(OnPolicyRunner):
            def save(self,path,infos=None):
                metadata={'task':env.config,'transitions':env.total_steps,'curriculum':env.curriculum,
                    'completed_updates':self.completed_updates_at_start+self.logger.tot_timesteps//(env.num_envs*args.steps_per_env*self.gpu_world_size),
                    'lineage':getattr(self,'lineage',None)}
                if infos:metadata.update(infos)
                super().save(path,infos=metadata)
                tmp=run/'latest.tmp';tmp.symlink_to(Path(path).resolve());tmp.replace(run/'latest.pt')
        runner=CatchRunner(env,cfg,str(run),device=args.device)
        runner.completed_updates_at_start=0
        runner.lineage=None
        if args.checkpoint:
            saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            contract=(saved.get('infos') or {}).get('task',{})
            if contract.get('name')!=env.config['name'] or contract.get('observation_dim')!=env.num_obs or contract.get('action_dim')!=env.num_actions:raise ValueError('Incompatible checkpoint task/observation contract')
            info=runner.load(str(args.checkpoint)) or {}
            runner.lineage=info.get('lineage')
            completed=completed_updates_from_checkpoint(saved,args.checkpoint.name)
            runner.completed_updates_at_start=completed
            if not args.play:runner.current_learning_iteration=completed
            (run/'resume.json').write_text(json.dumps({'checkpoint':str(args.checkpoint),'completed_updates':completed,
                'next_iteration_index':runner.current_learning_iteration,'optimizer_restored':True},indent=2))
            env.total_steps=int(info.get('transitions',0));env.curriculum=1. if args.full_curriculum else float(info.get('curriculum',0.))
            if args.policy_version=='v4':env.curriculum=env.curriculum_level/2
            env.reset(torch.arange(env.num_envs,device=args.device));env.render_and_track()
        if args.migrate_checkpoint:
            from policy_migration import migrate_state,source_mean,validate_source
            from policy_contract import LEGACY_ACTION_INDEX,NEW_ACTION_INDEX
            saved=torch.load(args.migrate_checkpoint,map_location='cpu',weights_only=False);validate_source(saved,env.config)
            runner.alg.actor.load_state_dict(migrate_state(saved['actor_state_dict'],runner.alg.actor.state_dict(),True))
            runner.alg.critic.load_state_dict(migrate_state(saved['critic_state_dict'],runner.alg.critic.state_dict(),False))
            parent_updates=completed_updates_from_checkpoint(saved,args.migrate_checkpoint.name)
            runner.lineage={'parent_checkpoint':str(args.migrate_checkpoint.resolve()),'parent_completed_updates':parent_updates,
                'migration':'v3_82x8_to_v4_100x14','optimizer_reset':True,'normalizer_pseudo_count':16384}
            # Use the same reset generation as ordinary checkpoint playback.
            env.reset(torch.arange(env.num_envs,device=args.device));env.render_and_track()
            policy=runner.get_inference_policy();observation=env.get_observations()
            with torch.no_grad():
                actual=policy(observation)
                source={k:v.to(args.device) for k,v in saved['actor_state_dict'].items()}
                expected=source_mean(source,observation['policy'][:,:82],runner.alg.actor.obs_normalizer.eps)
                difference=float((actual[:,LEGACY_ACTION_INDEX]-expected).abs().max())
                new_max=float(actual[:,NEW_ACTION_INDEX].abs().max())
            if difference>2e-5 or new_max>1e-8:raise RuntimeError('Migrated policy does not preserve legacy mean actions')
            manifest={**runner.lineage,'new_completed_updates':0,'legacy_action_max_difference':difference,
                'new_action_max_absolute_value':new_max,'optimizer_state_entries':len(runner.alg.optimizer.state)}
            (run/'migration.json').write_text(json.dumps(manifest,indent=2));print('[POLICY_MIGRATION]',json.dumps(manifest),flush=True)
            runner.save(str(run/'model_migrated_initial.pt'))
        if not args.play:
            adjustment=configure_optimizer(runner.alg,args.training_profile)
            if adjustment is not None:
                (run/'optimizer_profile.json').write_text(json.dumps(adjustment,indent=2))
                resume_file=run/'resume.json'
                if resume_file.exists():
                    resume=json.loads(resume_file.read_text());resume['optimizer_profile']=adjustment
                    resume_file.write_text(json.dumps(resume,indent=2))
        # Diagnostics only: verify the actual sampled curriculum, never an actor input.
        centers=torch.tensor([(env.nominal_p[i]+env.throw_workspace.contact_offsets[i]).tolist() for i in range(2)],device=args.device,dtype=torch.float32)
        offsets=(env.throw_goal-centers[env.side])[:,:2].norm(dim=-1)
        flights=env.throw_flight
        throw_distribution={'curriculum_level':env.curriculum_level,'training_mode':not args.play,
            'training_profile':args.training_profile,'receiving_hand_counts':{'left':int((env.side==0).sum()),'right':int((env.side==1).sum())},
            'perception_augmentation':env.config['perception_augmentation'],
            'grip_profile':args.grip_profile,'episodes':env.num_envs,
            'flight_time_min_s':float(flights.min()),'flight_time_max_s':float(flights.max()),
            'landing_offset_min_mm':float(offsets.min()*1000),'landing_offset_max_mm':float(offsets.max()*1000),
            'fraction_flight_below_040s':float((flights<.4).float().mean()),
            'fraction_landing_beyond_80mm':float((offsets>.08001).float().mean()),
            'direction_counts':{kind:env.throw_kind.count(kind) for kind in ['front','left_side','right_side']},
            'usage':'diagnostic_only_not_actor_input'}
        throw_distribution['curriculum_mode']=env.config['curriculum_mode']
        throw_distribution['level_distribution']={}
        for level in range(3):
            mask=torch.as_tensor(env.episode_level_host==level,device=args.device)
            count=int(mask.sum())
            if count:
                throw_distribution['level_distribution'][str(level)]={'episodes':count,
                    'flight_time_min_s':float(flights[mask].min()),'flight_time_max_s':float(flights[mask].max()),
                    'landing_offset_max_mm':float(offsets[mask].max()*1000)}
        (run/'initial_throw_distribution.json').write_text(json.dumps(throw_distribution,indent=2))
        print('[THROW_DISTRIBUTION]',json.dumps(throw_distribution),flush=True)
        if args.play:
            if args.response_probe:
                from response_diagnostics import run_response_diagnostics
                run_response_diagnostics(env,runner,args,run)
                return
            if args.diagnose:
                from catch_diagnostics import run_paired_diagnostics
                run_paired_diagnostics(env,runner,args,run)
                return
            policy=runner.get_inference_policy() if (args.checkpoint or args.migrate_checkpoint) else None
            if args.closure_probe:
                from isaaclab.utils.math import quat_apply
                env.contact_probe=True
                env.side[:]=torch.arange(env.num_envs,device=env.device)%2
                hp,hq=env.hands();idx=torch.arange(env.num_envs,device=env.device)
                env.catch_palm_target.copy_(hp[idx,env.side]+quat_apply(hq[idx,env.side],torch.tensor([.025,0,.07],device=env.device).expand(env.num_envs,3)))
                local=torch.tensor(json.loads((run/'mesh_resting_centers.json').read_text()),device=env.device)
                goal=hp[idx,env.side]+quat_apply(hq[idx,env.side],local[env.side])
                pose=env.ball.data.default_root_state.torch[:,:7].clone()
                pose[:,:3]=goal+torch.tensor([0,0,.16],device=env.device)
                env.ball.write_root_pose_to_sim_index(root_pose=pose)
                env.ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros((env.num_envs,6),device=env.device))
                env.exposure_batch.reset(idx.cpu().numpy());env.sim.forward();env.render_and_track()
                (run/'closure_probe.json').write_text(json.dumps({'training':False,'mode':'initial gravity drop only; no ball attachment',
                    'ball_initial_positions':pose[:,:3].cpu().tolist(),'side':env.side.cpu().tolist()},indent=2))
            viewer=None
            if args.live_view:
                from live_policy_view import LivePolicyView
                viewer=LivePolicyView(args.live_port,args.checkpoint.name if args.checkpoint else 'baseline_v3',run)
                print('[LIVE_VIEW]',viewer.url,flush=True)
            try:
                begin=time.monotonic();last_wall=begin;last_step=0;viewport_captured=False;profiles=[]
                if args.profile_eval:
                    import numpy as np
                    d=env.camera.data
                    np.savez_compressed(run/'tracking_frames.npz',rgb=d.output['rgb'].torch[...,:3].cpu().numpy(),
                        depth=d.output['distance_to_image_plane'].torch.cpu().numpy(),K=d.intrinsic_matrices.torch.cpu().numpy(),
                        position=d.pos_w.torch.cpu().numpy(),quaternion=d.quat_w_ros.torch.cpu().numpy())
                for step in range(args.eval_steps):
                    with torch.inference_mode():
                        actions=torch.zeros((env.num_envs,env.num_actions),device=args.device) if args.baseline else policy(env.get_observations())
                        env.step(actions)
                        if args.profile_eval:profiles.append(dict(env.step_profile))
                        if viewer is not None and step%2==0:
                            camera=env.scene['overview'];camera.update(0,force_recompute=True)
                            rgb=camera.data.output['rgb'].torch[0,...,:3].cpu().numpy().astype('uint8')
                            viewer.publish(rgb,{'control_steps':step+1,'simulated_seconds':round((step+1)/60,2),
                                'episodes':env.episodes,'successes':env.successes,'side':'左手' if int(env.side[0])==0 else '右手',
                                'phase':'收手抓握' if env.grasp_start[0]>=0 else ('掌面捕获' if env.capture_time[0]>=0 else '等待来球'),
                                'physics_invalid_episodes':env.physics_failures})
                    if args.capture_viewport and kit_requested and not viewport_captured and step>=20:
                        from PIL import Image
                        for viz in sim.visualizers:
                            if viz.cfg.visualizer_type=='kit':
                                frame=viz.render_rgb_array()
                                if frame.max()>0:
                                    Image.fromarray(frame).save(run/'isaac_viewport.png');viewport_captured=True
                    if (step+1)%60==0:
                        now=time.monotonic();hz=(step+1-last_step)/max(now-last_wall,1e-9)
                        performance={'control_steps':step+1,'wall_seconds':now-begin,
                            'control_frames_per_wall_second':hz,'realtime_factor':hz/60,
                            'physics_hz':physics_hz,'control_hz':60,'policy_camera_resolution':args.camera_resolution,
                            'environment_step_profile':getattr(env,'step_profile',{})}
                        (run/'playback_performance.json').write_text(json.dumps(performance,indent=2))
                        print('[PLAYBACK_PERFORMANCE]',json.dumps(performance),flush=True)
                        last_wall=now;last_step=step+1
            finally:
                if args.profile_eval and profiles:
                    selected=profiles[10:] or profiles
                    (run/'step_profile.json').write_text(json.dumps({'steps':len(profiles),
                        'mean_ms':{k:float(np.mean([x[k] for x in selected])) for k in selected[0]},'samples':profiles},indent=2))
                if viewer is not None:viewer.close()
            result={'checkpoint':str(args.checkpoint) if args.checkpoint else None,'baseline':args.baseline,
                'seed':args.seed,'num_envs':env.num_envs,'steps':args.eval_steps,'episodes':env.episodes,'successes':env.successes,
                'success_rate':env.successes/max(env.episodes,1),'curriculum':env.curriculum,'stage':env.config['stage']}
            (run/'evaluation.json').write_text(json.dumps(result,indent=2));env.save_camera(run/'camera_final.png');print('[EVALUATION]',json.dumps(result),flush=True)
        else:
            before={n:v.detach().clone() for n,v in runner.alg.actor.named_parameters()}
            runner.save(str(run/'model_initial.pt'))
            iterations_to_run=(args.total_iterations-runner.completed_updates_at_start) if args.total_iterations is not None else args.iterations
            if iterations_to_run<=0:
                print('[TARGET_ALREADY_COMPLETE]',runner.completed_updates_at_start,flush=True);return
            try:runner.learn(iterations_to_run,init_at_random_ep_len=False)
            except BaseException as exc:
                failure={'exception_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
                    'last_completed_iteration_index':runner.current_learning_iteration,'transitions':env.total_steps,
                    'physics_invalid_episodes':env.physics_failures,'rollout_incomplete':True}
                finite=all(bool(torch.isfinite(v).all()) for model in [runner.alg.actor,runner.alg.critic] for v in model.state_dict().values())
                failure['model_state_finite']=finite
                if finite:
                    path=run/f'model_stopped_{runner.current_learning_iteration}.pt'
                    runner.save(str(path),infos={'stopped':failure})
                    failure['saved_checkpoint']=str(path)
                (run/'training_stopped.json').write_text(json.dumps(failure,indent=2))
                raise
            changed=sum(float((v.detach()-before[n]).abs().sum()) for n,v in runner.alg.actor.named_parameters())
            summary={'iterations_this_run':iterations_to_run,'completed_updates':runner.completed_updates_at_start+iterations_to_run,'actor_parameter_abs_change':changed,'transitions':env.total_steps,
                'episodes':env.episodes,'successes':env.successes,'stage':env.config['stage']}
            (run/'training_complete.json').write_text(json.dumps(summary,indent=2));print('[TRAIN_COMPLETE]',json.dumps(summary),flush=True)

if __name__=='__main__':main()
