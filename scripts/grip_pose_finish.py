"""A first touch starts closure; it must not freeze an unfinished receiving pose."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

REVISION='finish_planned_receiving_pose_v1'


def contact_rotation_target(current,reference,planned_tilt,predicted_time,age,max_correction=.35):
    if not (predicted_time>0 and 0<=age<=.10):return Rotation.from_matrix(current@reference.T).as_rotvec()
    planned=Rotation.from_rotvec(planned_tilt).as_matrix()@reference
    error=Rotation.from_matrix(planned@current.T).as_rotvec();size=float(np.linalg.norm(error))
    if size>max_correction:error*=max_correction/size
    target=Rotation.from_rotvec(error).as_matrix()@current
    return Rotation.from_matrix(target@reference.T).as_rotvec()


def settling_joint_velocity(base,delta,elapsed,captured,q,limits):
    progress=(elapsed/.12).clamp(0,1)
    horizon=.075+.075*progress
    requested=(delta/horizon[:,None]).clamp(-3.,3.)
    lower=((limits[:,:,0]+.005-q)/.04).clamp(-3.,3.)
    upper=((limits[:,:,1]-.005-q)/.04).clamp(-3.,3.)
    requested=torch.maximum(torch.minimum(requested,upper),lower)
    active=captured&(elapsed>=-1e-5)&(elapsed<.12)
    return torch.where(active[:,None],requested,base)
