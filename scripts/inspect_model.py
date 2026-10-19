"""Inspect the Project 09 URDF snapshot; does not launch a simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def determinant(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def normalized(a):
    scale = max(abs(x) for row in a for x in row)
    return [[x / scale for x in row] for row in a] if scale else a


def inertia_issues(node):
    if node is None:
        return ['missing inertia']
    values = {k: float(node.get(k, 'nan')) for k in ['ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz']}
    if not all(math.isfinite(x) for x in values.values()):
        return ['nonfinite inertia']
    xx, yy, zz, xy, xz, yz = (values[k] for k in ['ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz'])
    matrix = normalized([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    # Sylvester criterion for positive definiteness.
    issues = []
    if not (matrix[0][0] > 0 and matrix[0][0] * matrix[1][1] - matrix[0][1] ** 2 > 0
            and determinant(matrix) > 0):
        issues.append('inertia is not positive definite')
    # Physical second moment S = trace(I)/2 * identity - I must be PSD.
    trace_half = sum(matrix[i][i] for i in range(3)) / 2
    second = [[(trace_half if i == j else 0) - matrix[i][j] for j in range(3)] for i in range(3)]
    minors = [second[i][i] for i in range(3)]
    minors += [second[i][i] * second[j][j] - second[i][j] ** 2
               for i in range(3) for j in range(i + 1, 3)]
    minors.append(determinant(second))
    if min(minors) < -1e-10:
        issues.append('inertia principal moments violate physical triangle condition')
    return issues


def inspect():
    cfg = json.loads((ROOT / 'config/task_proposal.json').read_text())
    urdf = ROOT / cfg['robot_urdf']
    robot = ET.parse(urdf).getroot()
    links = {x.get('name'): x for x in robot.findall('link')}
    joints = robot.findall('joint')
    errors, warnings = [], []
    children = set()
    adjacency = {name: [] for name in links}
    independent, mimics = [], []
    for joint in joints:
        parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
        if parent not in links or child not in links:
            errors.append(f"invalid link reference: {joint.get('name')}")
            continue
        if child in children:
            errors.append(f'multiple parents: {child}')
        children.add(child)
        adjacency[parent].append(child)
        if joint.get('type') != 'fixed':
            if joint.find('mimic') is None:
                independent.append(joint.get('name'))
            else:
                mimics.append({'name': joint.get('name'), **joint.find('mimic').attrib})
            limit = joint.find('limit')
            if limit is None:
                errors.append(f"missing limit: {joint.get('name')}")
            elif not (float(limit.get('lower')) < float(limit.get('upper'))
                      and float(limit.get('velocity')) > 0 and float(limit.get('effort')) > 0):
                errors.append(f"invalid limit: {joint.get('name')}")
    roots = sorted(set(links) - children)
    visited, active = set(), set()

    def visit(name):
        if name in active:
            errors.append(f'kinematic cycle: {name}')
            return
        if name in visited:
            return
        active.add(name)
        for child in adjacency[name]:
            visit(child)
        active.remove(name)
        visited.add(name)

    for name in roots:
        visit(name)
    if len(roots) != 1 or visited != set(links):
        errors.append('kinematic tree is not a single connected rooted tree')
    if len(set(cfg['joint_order'])) != len(cfg['joint_order']) or set(cfg['joint_order']) != set(independent):
        errors.append('configured joint order does not match independent joints')
    by_joint = {j.get('name'): j for j in joints}
    for mimic in mimics:
        source = by_joint.get(mimic['joint'])
        if source is None or source.find('limit') is None:
            errors.append(f"invalid mimic source: {mimic['name']}")
            continue
        a, b = float(mimic.get('multiplier', 1)), float(mimic.get('offset', 0))
        lim, dest = source.find('limit'), by_joint[mimic['name']].find('limit')
        mapped = sorted([a * float(lim.get('lower')) + b, a * float(lim.get('upper')) + b])
        if mapped[0] < float(dest.get('lower')) - 1e-8 or mapped[1] > float(dest.get('upper')) + 1e-8:
            warnings.append(f"mimic mapping exceeds child limit: {mimic['name']}")
    meshes = sorted({x.get('filename') for x in robot.findall('.//mesh')})
    missing = [name for name in meshes if not (urdf.parent / name).is_file()]
    errors.extend(f'missing mesh: {name}' for name in missing)
    hand_mass = {prefix: sum(float(link.find('inertial/mass').get('value'))
                 for name, link in links.items() if name.startswith(prefix)) for prefix in ['lh_', 'rh_']}
    for name, link in links.items():
        mass = link.find('inertial/mass')
        if mass is None or not math.isfinite(float(mass.get('value'))) or float(mass.get('value')) <= 0:
            errors.append(f'invalid mass: {name}')
        warnings.extend(f'{name}: {issue}' for issue in inertia_issues(link.find('inertial/inertia')))
    provenance = json.loads((ROOT / 'config/asset_provenance.json').read_text())
    for item in provenance['files']:
        path = ROOT / provenance['snapshot_directory'] / item['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            errors.append(f"snapshot checksum mismatch: {item['path']}")
    materials = json.loads((ROOT / 'config/contact_materials.json').read_text())
    for key in ['o6_rubber_contact', 'ball_initial_material']:
        material = materials[key]
        if not (material['static_friction'] >= material['dynamic_friction'] >= 0
                and 0 <= material['restitution'] <= 1):
            errors.append(f'invalid contact material: {key}')
    warnings.extend([
        'Left/right hand masses differ; measure mass and inertia before dynamics conclusions.',
        'All collision geometries are meshes; author separate palm/finger rubber patches and verify convex decomposition.',
        'Wrist-to-hand installation origins are inferred in the source model, not measured.',
        'URDF limits are model metadata, not verified hardware speed/torque capability.',
        'Simulator import, mimic behavior, palm frames, self-collision, contact material binding and catch feasibility are unverified.'
    ])
    return {'validation_scope': 'static URDF, mesh paths, control order, inertia math, snapshot checksums, material values only',
            'robot': robot.get('name'), 'root_links': roots, 'links': len(links), 'joints': len(joints),
            'independent_joints': len(independent), 'mimic_joints': len(mimics), 'hand_mass_kg': hand_mass,
            'collision_geometry_types': sorted({g[0].tag for g in robot.findall('.//collision/geometry')}),
            'unique_meshes': len(meshes), 'snapshot_files_verified': len(provenance['files']),
            'joint_order': cfg['joint_order'], 'mimic': mimics,
            'rubber_material': materials['o6_rubber_contact'],
            'errors': errors, 'warnings': warnings}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON report path')
    args = parser.parse_args()
    report = inspect()
    output = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end='')
    raise SystemExit(bool(report['errors']))
