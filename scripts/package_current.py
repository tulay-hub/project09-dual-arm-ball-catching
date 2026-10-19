"""Prepare a portable current-policy release. Never delete data or publish remotely."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,shutil,tarfile,xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,str) and value.startswith(str(ROOT)+'/'):return value[len(str(ROOT))+1:]
    if isinstance(value,str) and value.startswith('/home/'):return 'local_source/'+Path(value).name
    return value


def copy_checkpoint(source,destination):
    destination.mkdir(parents=True,exist_ok=False);shutil.copy2(source,destination/source.name)
    for name in ['runner_config.json','kinematic_contract.json','mesh_resting_centers.json','joint_armature_verified.json','task_config.json','training_complete.json']:
        path=source.parent/name
        if path.exists():(destination/name).write_text(json.dumps(clean(json.loads(path.read_text())),ensure_ascii=False,indent=2)+'\n')
    config=destination/'source/config';config.mkdir(parents=True)
    for name in ['scene.json','task_proposal.json']:
        value=clean(json.loads((source.parent/'source/config'/name).read_text()))
        if name=='task_proposal.json':value['training'].update(status='paused_by_user',pid=None,resume_checkpoint='checkpoints/'+source.name,run_dir='logs/ppo_catch',last_verified_checkpoint='checkpoints/'+source.name)
        (config/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def inventory(destination,source):
    files={str(p.relative_to(destination)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(destination.rglob('*'))
        if p.is_file() and '.git' not in p.parts and p.name not in ['PACKAGE_MANIFEST.json','SHA256SUMS']
        and '__pycache__' not in p.parts and not p.relative_to(destination).parts[0]=='logs'}
    manifest={'project':'Project09 dual-arm RGB-D ball catch','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'checkpoint':'checkpoints/model_1445.pt','checkpoint_completed_updates':1446,'checkpoint_original_sha256':sha(source),
        'historical_training_data_included':False,'ball_attachment_used':False,'files':files}
    (destination/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    sums={**{name:info['sha256'] for name,info in files.items()},'PACKAGE_MANIFEST.json':sha(destination/'PACKAGE_MANIFEST.json')}
    (destination/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(sums.items())))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);p.add_argument('--refresh_manifest',action='store_true');args=p.parse_args()
    source=Path(json.loads((ROOT/'reports/active_training.json').read_text())['last_verified_checkpoint'])
    if not source.is_absolute():source=ROOT/source
    if source.name!='model_1445.pt':raise ValueError('Expected user-selected current checkpoint')
    destination=args.destination.resolve()
    if destination==ROOT or ROOT in destination.parents:raise ValueError('Release must be outside source project')
    if args.refresh_manifest:
        manifest=inventory(destination,source);print(json.dumps({'files':len(manifest['files']),'bytes':sum(v['bytes'] for v in manifest['files'].values())}));return
    destination.mkdir(parents=True,exist_ok=False)
    for directory in ['scripts','tests','config']:
        target=destination/directory;target.mkdir()
        for file in (ROOT/directory).iterdir():
            if file.is_file() and file.suffix in ['.py','.sh','.json']:
                if file.suffix=='.json':(target/file.name).write_text(json.dumps(clean(json.loads(file.read_text())),ensure_ascii=False,indent=2)+'\n')
                else:shutil.copy2(file,target/file.name)
    shutil.copytree(ROOT/'assets/lkls73_o2_dual_arm_o6_description',destination/'assets/lkls73_o2_dual_arm_o6_description')
    dynamics=destination/'assets/generated/dynamics';dynamics.mkdir(parents=True)
    xml=ET.parse(ROOT/'assets/generated/dynamics/robot_mirror_left.urdf')
    for mesh in xml.getroot().findall('.//mesh'):
        rel=(ROOT/'assets/generated/dynamics'/mesh.get('filename')).resolve().relative_to(ROOT)
        mesh.set('filename',os.path.relpath(destination/rel,dynamics))
    ET.indent(xml);xml.write(dynamics/'robot_mirror_left.urdf',encoding='utf-8',xml_declaration=True)
    copy_checkpoint(source,destination/'checkpoints')
    from importlib.metadata import version
    packages=['mujoco','numpy','scipy','trimesh','rtree','tensordict','rsl-rl-lib','pillow','glfw']
    requirements=['# Verified with Python 3.12; select PyTorch CPU/CUDA wheels for your platform.','torch==2.11.0']+[f'{name}=={version(name)}' for name in packages]
    (destination/'requirements-mujoco.txt').write_text('\n'.join(requirements)+'\n')
    docs=destination/'docs';docs.mkdir()
    text=(ROOT/'docs/mujoco_policy_playback.md').read_text().replace(str(ROOT),'.').replace('logs/ppo_catch/v2/mixed_all_from1345_20261019_134853/segment_0001/model_1445.pt','checkpoints/model_1445.pt')
    (docs/'mujoco_policy_playback.md').write_text(text)
    shutil.copy2(ROOT/'docs/release_readme.md',destination/'README.md')
    shutil.copy2(ROOT/'docs/release_readme.md',docs/'release_readme.md')
    (destination/'.gitignore').write_text('__pycache__/\n*.py[cod]\n.git/\n.venv/\n.env\nlogs/\nMUJOCO_LOG.TXT\nassets/generated/mujoco/\n')
    # Respect user-selected interpreter; portable fallback instead of a creator-specific Conda path.
    launcher=destination/'scripts/run.sh';launcher.write_text(launcher.read_text().replace('/home/tulay/miniconda3/envs/isaaclab61/bin/python','python3'))
    reports=destination/'reports';reports.mkdir()
    for name in ['mujoco_migration_1446_20261019.json']:
        value=clean(json.loads((ROOT/'reports'/name).read_text()));(reports/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    state=json.loads((ROOT/'reports/active_training.json').read_text())
    state.update(pid=None,supervisor_pid=None,training_pid=None,status='paused_by_user',phase='stopped',
        session_dir='logs/ppo_catch',run_dir='logs/ppo_catch',last_verified_checkpoint='checkpoints/model_1445.pt',resume_checkpoint='checkpoints/model_1445.pt')
    (reports/'active_training.json').write_text(json.dumps(clean(state),ensure_ascii=False,indent=2)+'\n')
    asset_note='机器人资产保留源package.xml中的许可标记：TODO: Confirm upstream model licenses。本包未将第三方URDF/STL重新许可。\n'
    (destination/'NOTICE.md').write_text('# 来源与许可边界\n\n'+asset_note+'\nMuJoCo、PyTorch、RSL-RL等依赖由包管理器安装，分别遵守其上游许可；不包含Isaac/Omniverse二进制或登录凭证。\n')
    manifest=inventory(destination,source)
    print(json.dumps({'destination':str(destination),'files':len(manifest['files']),'bytes':sum(v['bytes'] for v in manifest['files'].values()),'checkpoint_sha256':sha(source)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
