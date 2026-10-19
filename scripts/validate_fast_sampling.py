"""Compare sampling optimization against a preserved source and real RGB-D frames."""
from pathlib import Path
from types import SimpleNamespace as NS
from tempfile import TemporaryDirectory
import ast,copy,importlib.util,json,sys,time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from arm_tracking import joint_body_jacobian,joint_gravity_forces
from closure_trigger import contact_closure_step,anticipated_closure_fraction,incoming_hand_tilt
from throw_workspace import predict_landing
from ball_tracker import BallTracker
from fast_sampling import sparse_red_pixels
from soft_catch import reception_target,reception_joint_velocity
from grip_pose_finish import contact_rotation_target,settling_joint_velocity
from policy_contract import XY_RESIDUAL_SCALE_M
from predictive_tracking import position_priority_delta,early_arrival_velocity


def quat_apply(q,v):
    t=2*torch.linalg.cross(q[...,:3],v,dim=-1)
    return v+q[...,3:]*t+torch.linalg.cross(q[...,:3],t,dim=-1)


def command_from(path):
    tree=ast.parse(Path(path).read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='BallCatchEnv')
    f=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_command')
    namespace=globals().copy();exec(compile(ast.Module(body=[f],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['_command']


def fake_env(device,directory):
    rng=np.random.default_rng(617);n=256;nj=36
    tensor=lambda x,dtype=torch.float32:torch.tensor(x,device=device,dtype=dtype)
    q=tensor(rng.uniform(.05,.9,(n,nj)));defaults=q.clone();target=q.clone()
    reference=np.array([[0,0,1],[0,-1,0],[1,0,0.]])
    rotations=Rotation.from_rotvec(rng.normal(0,.03,(n*2,3))).as_matrix()@reference
    hq=tensor(Rotation.from_matrix(rotations).as_quat().reshape(n,2,4))
    hp=tensor(np.tile([[.29,.18,.875],[.29,-.18,.875]],(n,1,1)))
    jac=tensor(rng.normal(0,.2,(n,3,6,nj+6)))
    jnames=['arm'+str(i) for i in range(14)]+[p+f for p in ['lh_','rh_'] for f in ['thumb_yaw','thumb_pitch','index','middle','ring','pinky']]+['mimic'+str(i) for i in range(10)]
    hand_ids=[list(range(14,20)),list(range(20,26))]
    e=NS(num_envs=n,device=device,dt=1/60,jnames=jnames,log_dir=Path(directory),arm_ids=[list(range(7)),list(range(7,14))],hand_ids=hand_ids,hbody=[2,3],
        robot=NS(body_names=['root','torso','lh','rh'],data=NS(joint_pos=NS(torch=q),default_joint_pos=NS(torch=defaults),
            body_link_jacobian_w=NS(torch=jac),gravity_compensation_forces=NS(torch=torch.zeros(n,nj+6,device=device))),
            actuators=NS(target_command=NS(set_position_index=lambda **kw:None,set_effort_index=lambda **kw:None))),
        target=target,limits=tensor(np.tile([-3,3],(n,nj,1))),efforts=torch.zeros(n,nj,device=device),scene_cfg={'hand_rotation_world':reference.tolist()},
        nominal_p=[np.array([.36,.18,.9]),np.array([.36,-.18,.9])],
        throw_workspace=NS(contact_offsets=[np.array([0,0,.008]),np.array([0,0,.008])]),resting_centers_local=np.tile([.0324,0,.078],(2,1)),
        contact_probe=False,closure_events=0,closure_events_by_side=[0,0],preparation_events=0,preparation_events_by_side=[0,0])
    e.hands=lambda:(hp,hq);e.side=tensor(np.arange(n)%2,torch.long);e.episode_length_buf=tensor(rng.choice([6,9,10,24,80,120,240],n),torch.long)
    e.visible=tensor(rng.uniform(size=n)>.2,torch.bool);e.age=tensor(rng.choice([0,.05,.1,.11,.2],n))
    e.contact_streak=tensor(rng.integers(0,3,n),torch.long);e.capture_time=tensor(np.full(n,-1.));e.grasp_start=tensor(np.full(n,-1.))
    caught=(np.arange(n)%4==0);e.grasp_start[caught]=.2;e.capture_time[caught]=.2
    e.grasp_curl_bias=tensor(rng.uniform(-.1,.1,n));e.prepare_time=tensor(np.full(n,-1.));e.prepare_time[1::7]=.1
    e.preclose_fraction=tensor(rng.uniform(0,.2,n));e.predicted_contact_time=tensor(np.full(n,-1.))
    e.total_contact=tensor(rng.choice([0,.02,.03,.04,1.],(n,2)));e.caught_tilt=tensor(rng.normal(0,.2,(n,3)))
    e.origins=tensor(np.zeros((n,3)));e.catch_palm_target=tensor(np.tile([.36,.18,.9],(n,1)));e.catch_palm_target[1::2,1]=-.18
    e.predicted_intercept=torch.zeros(n,3,device=device);e.previous_local=torch.zeros(n,3,device=device)
    e.grasp_origin_q=q[:,14:20].clone();e.nominal_grasp=[tensor([.9,.5,1.3,1.2,1.1,1.]),tensor([.9,.5,1.3,1.2,1.1,1.])]
    e.nominal_grasp_cpu=[x.cpu().numpy() for x in e.nominal_grasp];e.hand_joint_ids=tensor(hand_ids,torch.long)
    e.non_thumb_masks=tensor([['thumb' not in jnames[j] for j in ids] for ids in hand_ids],torch.bool)
    e.vision=tensor(np.tile([.39,.18,1.],(n,1)));e.vision[1::2,1]=-.18;e.visual_velocity=tensor(np.tile([-.5,0,-.2],(n,1)))
    return e,tensor(rng.uniform(-1,1,(n,8)))


def validate(reference,frames,destination):
    ref_command=command_from(Path(reference)/'ball_catch_env.py');new_command=command_from(Path(__file__).with_name('ball_catch_env.py'))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    with TemporaryDirectory() as a,TemporaryDirectory() as b:
        old,actions=fake_env(device,a);new,_=fake_env(device,b)
        ref_command(old,actions);new_command(new,actions)
        names=['target','capture_time','grasp_start','grasp_curl_bias','contact_streak','prepare_time','preclose_fraction',
               'predicted_contact_time','caught_tilt','catch_palm_target','predicted_intercept','previous_local','grasp_origin_q']
        errors={}
        for k in names:
            left=getattr(old,k).cpu().numpy();right=getattr(new,k).cpu().numpy()
            errors[k]=float(np.abs(left-right).max());np.testing.assert_allclose(left,right,atol=2e-6,rtol=1e-6,err_msg=k)
        for k in ['closure_events','closure_events_by_side','preparation_events','preparation_events_by_side']:
            assert getattr(old,k)==getattr(new,k),(k,getattr(old,k),getattr(new,k))
    spec=importlib.util.spec_from_file_location('reference_tracker',Path(reference)/'ball_tracker.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    d=np.load(frames);rgb=d['rgb'];depth=d['depth'];K=d['K'];pos=d['position'];quat=d['quaternion']
    pixels=sparse_red_pixels(torch.tensor(rgb,device=device),torch.tensor(depth,device=device)).cpu().numpy()
    rows=pixels[:,0].astype(int);bounds=np.searchsorted(rows,np.arange(len(rgb)+1));differences=[];accepted=0;old_time=0.;new_time=0.
    for e in range(len(rgb)):
        old=module.BallTracker(radius=.02);new=BallTracker(radius=.02);start=time.perf_counter()
        m=old.observe(rgb[e],depth[e],K[e],pos[e],quat[e],0.);old_time+=time.perf_counter()-start
        p=pixels[bounds[e]:bounds[e+1]];start=time.perf_counter()
        v=new.measure_sparse(p[:,1],p[:,2],p[:,3],K[e],pos[e],quat[e]);new_time+=time.perf_counter()-start
        assert (m is None)==(v is None),(e,m,v)
        if m is not None:
            accepted+=1;differences.append(float(np.linalg.norm(m-v)));np.testing.assert_allclose(m,v,atol=1e-4,rtol=0)
    result={'control_max_absolute_difference':errors,'control_event_counts_identical':True,'frames':len(rgb),'accepted_identically':accepted,
            'maximum_center_difference_m':max(differences,default=0),'mean_center_difference_m':float(np.mean(differences)),
            'reference_cpu_vision_seconds':old_time,'optimized_sparse_cpu_fit_seconds':new_time,'device':device}
    Path(destination).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':validate(*sys.argv[1:])
