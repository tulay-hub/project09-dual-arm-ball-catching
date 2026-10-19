"""Play the Project09 frozen 100x14 residual policy with real MuJoCo RGB-D/contact."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,copy,hashlib,json,os,time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--log_dir',type=Path)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--steps',type=int,default=36000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--camera_resolution',type=int,default=192)
    args=parser.parse_args()
    os.environ.setdefault('MUJOCO_GL','egl' if args.headless else 'glfw')
    import mujoco
    import torch
    from PIL import Image
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from kinematics import ROOT
    from mujoco_catch import MujocoCatch,FrozenActor
    torch.set_num_threads(1)
    run=(args.log_dir or ROOT/'logs/playback'/('mujoco_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))).resolve()
    run.mkdir(parents=True,exist_ok=False);checkpoint=args.checkpoint.resolve();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();actor=FrozenActor(saved)
    config=json.loads((checkpoint.parent/'runner_config.json').read_text());actor_config=copy.deepcopy(config['actor']);actor_config.pop('class_name',None)
    reference=MLPModel(TensorDict({'policy':torch.zeros(1,100)},batch_size=[1]),config['obs_groups'],'actor',14,**actor_config)
    reference.load_state_dict(saved['actor_state_dict']);reference.eval()
    probe=torch.randn(512,100).clamp(-10,10)
    with torch.inference_mode():
        expected=reference(TensorDict({'policy':probe},batch_size=[len(probe)]));actual=actor(probe)
    error=float((actual-expected).abs().max())
    if error>1e-6:raise RuntimeError(f'Actor export mismatch: {error}')
    script=torch.jit.script(actor);script.save(str(run/'actor.pt'))
    manifest={'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'completed_updates':saved['infos']['completed_updates'],
        'observation_dim':100,'action_dim':14,'actor_reference_max_error':error,'normalization_epsilon':actor.epsilon,
        'policy_frozen':True,'ball_attachment_used':False,'mujoco_version':mujoco.__version__,'source_controller':'ball_catch_env._command, get_observations',
        'source_sha256':hashlib.sha256((ROOT/'scripts/ball_catch_env.py').read_bytes()).hexdigest(),'perception':'rendered RGB-D sphere fit; object truth only for resets, outcome and diagnostic error'}
    manifest['adapter_sha256']={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in
        ['build_mujoco_catch.py','mujoco_catch.py','play_mujoco_policy.py','policy_contract.py','ball_tracker.py']}
    (run/'migration.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    env=MujocoCatch(checkpoint,run,args.seed,args.camera_resolution);viewer=None
    if not args.headless:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(env.model,env.data,show_left_ui=False,show_right_ui=False)
        viewer.cam.lookat[:]=[.25,0,1.05];viewer.cam.distance=2.35;viewer.cam.azimuth=135;viewer.cam.elevation=-18;viewer.opt.geomgroup[3]=0
    observation=env.get_observations()['policy'];start=time.monotonic();clock=start
    try:
        for step in range(args.steps):
            if viewer is not None and not viewer.is_running():break
            with torch.inference_mode():actions=actor(observation);observation=env.step(actions)
            if not bool(torch.isfinite(observation).all() and torch.isfinite(actions).all()):raise RuntimeError('Nonfinite inference')
            if viewer is not None:
                viewer.sync();clock+=1/60;time.sleep(max(0,clock-time.monotonic()))
            if step==0:
                Image.fromarray(env.last_rgb).save(run/'policy_camera.png')
                # Independent scene overview capture, with the same model/data.
                camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera);camera.lookat[:]=[.25,0,1.05];camera.distance=2.35;camera.azimuth=135;camera.elevation=-18
                with mujoco.Renderer(env.model,height=540,width=960) as overview:
                    overview.update_scene(env.data,camera=camera,scene_option=env.camera_option);Image.fromarray(overview.render()).save(run/'overview.png')
            if step%60==59 or step==args.steps-1:
                elapsed=time.monotonic()-start
                status={'pid':os.getpid(),'control_steps':step+1,'wall_seconds':elapsed,'control_frames_per_wall_second':(step+1)/elapsed,
                    'realtime_factor':(step+1)/60/elapsed,'episodes':env.episodes,'successes':env.successes,
                    'success_rate':env.successes/max(1,env.episodes),'physics_invalid_episodes':env.physics_failures,
                    'vision_accepted':env.vision_count,'vision_mean_error_m':env.vision_error_sum/max(1,env.vision_count),
                    'vision_current_valid':bool(env.visible[0]),'closure_events':env.closure_events,'preparation_events':env.preparation_events,
                    'max_current_hold_steps':env.best_hold,'finite_observations':True,'finite_actions':True,'headless':args.headless,
                    'policy_frozen':True,'physics_hz':120,'policy_hz':60,'camera_resolution':args.camera_resolution,
                    'max_joint_speed_rad_s':float(env.robot.data.joint_vel.torch.abs().max()),'max_mimic_error_rad':max(abs(env.data.qpos[env.model.joint(m['name']).qposadr[0]]-env.data.qpos[env.model.joint(m['joint']).qposadr[0]]*float(m['multiplier'])) for m in env.task['mimic_joints'])}
                (run/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');print('[MUJOCO_PLAYBACK]',json.dumps(status),flush=True)
    finally:
        if viewer is not None:viewer.close()
        env.renderer.close()
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=digest:raise RuntimeError('Checkpoint was modified')
    (run/'complete.json').write_text(json.dumps({'steps':env.steps,'episodes':env.episodes,'successes':env.successes,'checkpoint_unchanged':True},indent=2))


if __name__=='__main__':main()
