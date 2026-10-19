"""Build a separate simulation URDF; never modify the source asset snapshot."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]


def prepare_dynamics(profile='mirror_left'):
    cfg=json.loads((ROOT/'config/task_proposal.json').read_text())
    source=ROOT/cfg['robot_urdf']
    if profile=='source':return source
    robot=ET.parse(source).getroot()
    links={link.get('name'):link for link in robot.findall('link')}
    changes=[]
    mirror=np.diag([1.,-1.,1.])
    for name,left in links.items():
        if not name.startswith('lh_'):continue
        right=links['rh_'+name[3:]]
        old_mass=float(right.find('inertial/mass').get('value'))
        inertia=copy.deepcopy(left.find('inertial'))
        origin=inertia.find('origin')
        com=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        R=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        I=inertia.find('inertia');a={k:float(I.get(k)) for k in ['ixx','iyy','izz','ixy','ixz','iyz']}
        tensor=np.array([[a['ixx'],a['ixy'],a['ixz']],[a['ixy'],a['iyy'],a['iyz']],[a['ixz'],a['iyz'],a['izz']]])
        transformed=mirror@R@tensor@R.T@mirror
        origin.set('xyz',' '.join(f'{v:.12g}' for v in mirror@com));origin.set('rpy','0 0 0')
        for key,i,j in [('ixx',0,0),('iyy',1,1),('izz',2,2),('ixy',0,1),('ixz',0,2),('iyz',1,2)]:
            I.set(key,f'{transformed[i,j]:.12g}')
        right.remove(right.find('inertial'));right.insert(0,inertia)
        changes.append({'link':right.get('name'),'source_mass_kg':old_mass,'simulation_mass_kg':float(inertia.find('mass').get('value'))})
    for mesh in robot.findall('.//mesh'):
        mesh.set('filename',str((source.parent/mesh.get('filename')).resolve()))
    directory=ROOT/'assets/generated/dynamics';directory.mkdir(parents=True,exist_ok=True)
    output=directory/'robot_mirror_left.urdf'
    ET.indent(robot)
    ET.ElementTree(robot).write(output,encoding='utf-8',xml_declaration=True)
    report={'profile':'mirror_left','source_urdf':str(source),'simulation_urdf':str(output),
        'assumption':'Simulation diagnostic only: same-model O6 hands use left inertial parameters reflected across local Y. Not hardware calibration.',
        'original_source_unchanged':True,'changes':changes}
    (directory/'dynamics_assumption.json').write_text(json.dumps(report,indent=2)+'\n')
    return output


if __name__=='__main__':print(prepare_dynamics())
