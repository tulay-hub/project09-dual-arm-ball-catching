"""Contact evidence for a grasp, independent of PPO rewards and force magnitude."""
import numpy as np


def opposing_contacts(names, vectors, points, counts, starts, separations,
                      ball_center, radius, prefix):
    """Require thumb + two fingers on opposite sides of the actual sphere.

    Vectors are actual per-body contact forces. Points/separations are actual
    PhysX contacts; no hand attachment or synthetic contacts are accepted.
    """
    vectors = np.asarray(vectors).reshape(-1, 3)
    points = np.asarray(points).reshape(-1, 3)
    counts = np.asarray(counts).reshape(-1)
    starts = np.asarray(starts).reshape(-1)
    separations = np.asarray(separations).reshape(-1)
    center = np.asarray(ball_center)
    groups = {f: [] for f in ['thumb', 'index', 'middle', 'ring', 'pinky']}
    forces = {f: np.zeros(3) for f in groups}
    max_penetration = 0.
    for i, name in enumerate(names):
        finger = next((f for f in groups if name.startswith(prefix + f)), None)
        if finger is None:
            continue
        forces[finger] += vectors[i]
        start, count = int(starts[i]), int(counts[i])
        for k in range(start, start + count):
            max_penetration = max(max_penetration, max(0., -float(separations[k])))
            radial = points[k] - center
            length = np.linalg.norm(radial)
            if abs(length - radius) <= .006 and separations[k] >= -.003 and length > 1e-6:
                groups[finger].append(radial / length)
    thumb_force = np.linalg.norm(forces['thumb'])
    finger_count = sum(np.linalg.norm(forces[f]) > .02 and bool(groups[f]) for f in groups if f != 'thumb')
    finger_force = sum((forces[f] for f in groups if f != 'thumb'), start=np.zeros(3))
    opposition = -float(forces['thumb'] @ finger_force) / max(
        thumb_force * np.linalg.norm(finger_force), 1e-9)
    point_opposition = 0.
    for thumb in groups['thumb']:
        for f in groups:
            if f != 'thumb':
                for point in groups[f]:
                    point_opposition = max(point_opposition, -float(thumb @ point))
    valid = (thumb_force > .03 and finger_count >= 2 and opposition > .25
             and point_opposition > .25 and max_penetration <= .003)
    return {'valid': bool(valid), 'thumb': bool(thumb_force > .03),
            'fingers': bool(finger_count >= 2), 'finger_count': int(finger_count),
            'opposition': float(opposition), 'point_opposition': float(point_opposition),
            'max_penetration_m': float(max_penetration)}


def stable_grasp(contact_valid, closing_complete, relative_speed, relative_drift,
                 finger_speed, palm_distance):
    return bool(contact_valid and closing_complete and relative_speed < .15
                and relative_drift < .008 and finger_speed < .35 and palm_distance < .065)


def retention_contacts(names, vectors, points, counts, starts, separations,
                       ball_center, radius, prefix):
    """Support on original palm/fingers is sufficient; enclosing is not required."""
    vectors=np.asarray(vectors).reshape(-1,3);points=np.asarray(points).reshape(-1,3)
    counts=np.asarray(counts).reshape(-1);starts=np.asarray(starts).reshape(-1)
    separations=np.asarray(separations).reshape(-1);center=np.asarray(ball_center)
    force=0.;valid_points=0;penetration=0.
    for i,name in enumerate(names):
        if not name.startswith(prefix):continue
        force+=float(np.linalg.norm(vectors[i]))
        for k in range(int(starts[i]),int(starts[i])+int(counts[i])):
            penetration=max(penetration,max(0.,-float(separations[k])))
            if abs(np.linalg.norm(points[k]-center)-radius)<.006 and separations[k]>=-.003:
                valid_points+=1
    return {'valid':bool(force>.03 and valid_points>0 and penetration<=.003),
            'hand_contact_force_n':force,'surface_contact_points':valid_points,
            'max_penetration_m':penetration}


def stable_retention(contact_valid, relative_speed, relative_drift, palm_distance):
    return bool(contact_valid and relative_speed<.15 and relative_drift<.008 and palm_distance<.065)
