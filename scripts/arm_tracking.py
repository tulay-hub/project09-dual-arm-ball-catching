"""Public Isaac Jacobian indexing, with explicit base/body axis contracts."""


def joint_body_jacobian(jacobians,body_index,num_bodies,num_joints,joint_ids):
    base_columns=jacobians.shape[-1]-num_joints
    omitted_bodies=num_bodies-jacobians.shape[1]
    if base_columns not in (0,6) or omitted_bodies not in (0,1):
        raise ValueError(f'Unsupported articulation Jacobian shape: {jacobians.shape}')
    return jacobians[:,body_index-omitted_bodies][:,:,[j+base_columns for j in joint_ids]]


def joint_gravity_forces(generalized_forces,num_joints):
    base_columns=generalized_forces.shape[-1]-num_joints
    if base_columns not in (0,6):raise ValueError('Unsupported gravity force shape')
    return generalized_forces[:,base_columns:]
