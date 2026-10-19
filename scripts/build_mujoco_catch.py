"""Independent MuJoCo plant from the same simulation URDF; no ball attachment."""
from pathlib import Path
import hashlib,json,tempfile,xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from kinematics import ROOT,RobotKinematics


def initial_pose(cfg,task,kin):
    initial=dict(cfg['arm_ready_joint_positions_rad'])
    for side in ['left','right']:
        rotation=Rotation.from_rotvec([0,-.18,0]).as_matrix()@np.array(cfg['hand_rotation_world'])
        names=kin.arm_names(side)
        pose,pe,re=kin.solve_palm(side,np.array(cfg['ready'][side]['palm_position_world_m']),rotation,[initial[n] for n in names])
        if pe>.005 or re>.08:raise ValueError('Unreachable original ready pose')
        initial.update(pose)
    for name in task['joint_order']:
        if name.startswith(('lh_','rh_')):initial[name]=.75 if 'thumb_cmc_pitch' not in name else .06
    for mimic in task['mimic_joints']:
        initial[mimic['name']]=initial[mimic['joint']]*float(mimic['multiplier'])+float(mimic.get('offset',0))
    return initial


def build_model(checkpoint,output=None):
    checkpoint=Path(checkpoint).resolve();run=checkpoint.parent
    cfg=json.loads((run/'source/config/scene.json').read_text())
    task=json.loads((run/'source/config/task_proposal.json').read_text())
    source=ROOT/'assets/generated/dynamics/robot_mirror_left.urdf'
    robot=ET.parse(source).getroot()
    for mesh in robot.findall('.//mesh'):
        mesh.set('filename',str((source.parent/mesh.get('filename')).resolve()))
    extension=ET.SubElement(robot,'mujoco')
    ET.SubElement(extension,'compiler',fusestatic='false',discardvisual='false')
    model=mujoco.MjModel.from_xml_string(ET.tostring(robot,encoding='unicode'))
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'robot.xml';mujoco.mj_saveLastXML(str(path),model);xml=ET.parse(path)
    root=xml.getroot();root.set('model','Project09_current_policy')
    option=ET.SubElement(root,'option',timestep=str(1/cfg['physics_hz']),integrator='implicitfast',iterations='64',gravity='0 0 -9.81')
    ET.SubElement(root,'visual');visual=root.find('visual')
    ET.SubElement(visual,'global',offwidth='960',offheight='540')
    ET.SubElement(visual,'headlight',diffuse='.65 .65 .65',ambient='.4 .4 .4',specular='.1 .1 .1')
    world=root.find('worldbody');base=world.find("body[@name='Base_Link']");base.set('pos',f"0 0 {cfg['base_height_m']}")
    ET.SubElement(world,'light',pos='1 -1 3',dir='-.3 .3 -1',diffuse='.7 .7 .7')
    ET.SubElement(world,'geom',name='floor',type='plane',size='4 4 .1',rgba='.26 .29 .33 1',friction='1 .005 .0001')
    camera=cfg['overhead_camera'];q=camera['quaternion_in_link_xyzw_opengl']
    ET.SubElement(base,'camera',name='policy_camera',pos=' '.join(map(str,camera['position_in_link_m'])),quat=' '.join(map(str,[q[3],*q[:3]])),fovy=str(np.degrees(2*np.arctan(.75))))
    ball=ET.SubElement(world,'body',name='Ball',pos='.8 .18 1.05')
    ET.SubElement(ball,'freejoint',name='ball_free')
    ET.SubElement(ball,'geom',name='ball',type='sphere',size=str(cfg['ball_radius_m']),mass=str(cfg['ball_mass_kg']),rgba='.95 .025 .015 1',friction='1 .005 .0001',condim='4')
    contact=ET.SubElement(root,'contact');equality=ET.SubElement(root,'equality');actuator=ET.SubElement(root,'actuator')
    for prefix in ['lh_','rh_']:
        names=[b.get('name') for b in root.findall('.//body') if b.get('name','').startswith(prefix)]
        for i,name in enumerate(names):
            for other in names[i+1:]:ET.SubElement(contact,'exclude',body1=name,body2=other)
    for mimic in task['mimic_joints']:
        ET.SubElement(equality,'joint',name='mimic_'+mimic['name'],joint1=mimic['name'],joint2=mimic['joint'],
            polycoef=f"{mimic.get('offset',0)} {mimic['multiplier']} 0 0 0",solref='.004 1')
    for body in root.findall('.//body'):
        name=body.get('name','')
        # mj_saveLastXML rounds imported inertials. Preserve the URDF precision.
        link=robot.find(f"link[@name='{name}']")
        if link is not None and link.find('inertial') is not None:
            original=link.find('inertial');origin=original.find('origin');inertia=original.find('inertia')
            position=origin.get('xyz','0 0 0') if origin is not None else '0 0 0'
            rotation=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)).as_matrix()
            xx,yy,zz,xy,xz,yz=[float(inertia.get(k)) for k in ['ixx','iyy','izz','ixy','ixz','iyz']]
            tensor=rotation@np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])@rotation.T
            inertial=body.find('inertial')
            if inertial is None:inertial=ET.SubElement(body,'inertial')
            inertial.attrib.clear();inertial.set('pos',position);inertial.set('mass',original.find('mass').get('value'))
            inertial.set('fullinertia',' '.join(f'{v:.17g}' for v in [tensor[0,0],tensor[1,1],tensor[2,2],tensor[0,1],tensor[0,2],tensor[1,2]]))
        for geom in body.findall('geom'):
            if name=='Ball':continue
            visual_only=geom.get('contype')=='0'
            geom.set('group','2' if visual_only else '3')
            if visual_only:geom.set('rgba','.72 .75 .8 1')
            else:
                geom.set('rgba','.55 .6 .66 1');geom.set('friction','1 .005 .0001' if name.startswith(('lh_','rh_')) else '.6 .005 .0001')
                geom.set('condim','4')
    for name in task['joint_order']:
        hand=name.startswith(('lh_','rh_'));kp,kd,effort=(3.,.1,.25) if hand else (180.,16.,36.)
        ET.SubElement(actuator,'position',name=name+'_drive',joint=name,kp=str(kp),kv=str(kd),forcelimited='true',forcerange=f'{-effort} {effort}')
    for joint in root.findall('.//body/joint'):
        if joint.get('type')=='free':continue
        joint.set('armature','.001' if joint.get('name','').startswith(('lh_','rh_')) else '.02')
        joint.set('damping','0');joint.set('frictionloss','0')
    output=Path(output or ROOT/'assets/generated/mujoco/ball_catch.xml');output.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(root);xml.write(output,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(output));data=mujoco.MjData(model)
    expected_mass=sum(float(m.get('value')) for m in robot.findall('./link/inertial/mass'))+cfg['ball_mass_kg']
    if abs(float(model.body_mass.sum())-expected_mass)>1e-9:raise ValueError('URDF mass changed during import')
    kin=RobotKinematics(source,base_height=cfg['base_height_m']);initial=initial_pose(cfg,task,kin)
    for name,value in initial.items():data.qpos[model.joint(name).qposadr[0]]=value
    mujoco.mj_forward(model,data);errors={}
    for side,prefix in [('left','lh_'),('right','rh_')]:
        body=model.body(prefix+'hand_base_link').id;p=data.xpos[body]+data.xmat[body].reshape(3,3)@np.array([.025,0,.07])
        expected,_=kin.palm(side,initial);errors[side]=float(np.linalg.norm(p-expected))
    if max(errors.values())>1e-6:raise ValueError(f'FK mismatch: {errors}')
    report={'model':str(output),'urdf':str(source),'urdf_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'model_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'initial':initial,'scene':cfg,'task':task,
        'palm_fk_error_m':errors,'nq':model.nq,'nv':model.nv,'nu':model.nu,'ball_attachment_used':False,
        'contact_limitation':'Imported original collision meshes use MuJoCo convex hulls; PhysX uses convex decomposition. Sliding friction=1.0, torsional=.005, rolling=.0001, not a direct static/dynamic friction conversion.',
        'physics_hz':cfg['physics_hz'],'policy_hz':60,'checkpoint':str(checkpoint)}
    report.update(total_mass_kg=float(model.body_mass.sum()),urdf_and_ball_mass_kg=expected_mass,
        velocity_limit_mapping='Velocity-saturated PD approximates source drives (arm3rad/s, fingers1rad/s); external impulses are not clamped. This is not an identical PhysX absolute velocity constraint.')
    output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return output,report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True);args=parser.parse_args()
    path,report=build_model(args.checkpoint);print(json.dumps({k:report[k] for k in ['model','nq','nv','nu','palm_fk_error_m']},indent=2))
