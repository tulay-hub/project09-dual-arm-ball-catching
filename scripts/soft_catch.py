"""Visual reception while fingers close; all motion remains physical."""
import numpy as np
import torch

REVISION='early_fingers_bounded_visual_reception_v1'
SOFT_SETTINGS={'preclose_window_s':.24,'preclose_alignment_m':.05,
    'preclose_maximum_fraction':.55,'closure_duration_s':.18}


def clip_norm(vector,maximum):
    vector=np.asarray(vector,dtype=float)
    return vector*min(1.,maximum/max(float(np.linalg.norm(vector)),1e-12))


def reception_target(anchor,previous,ball_position,ball_velocity,age,elapsed,palm_offset,dt,
                     window=.18,extent=.035,speed=.35):
    """Follow a recent image centre briefly, bounded around the original intercept."""
    if not (0<=elapsed<=window and 0<=age<=.06):return np.asarray(previous).copy(),np.zeros(3)
    center=np.asarray(ball_position)+np.asarray(ball_velocity)*age
    desired=np.asarray(anchor)+clip_norm(center-np.asarray(palm_offset)-anchor,extent)
    target=np.asarray(previous)+clip_norm(desired-previous,speed*dt)
    target=np.asarray(anchor)+clip_norm(target-anchor,extent)
    feedforward=clip_norm(np.asarray(ball_velocity)*.35,speed)*(1-elapsed/window)
    return target,feedforward


def reception_joint_velocity(base,jacobian,cartesian_velocity,q,limits):
    """Add bounded image-velocity feedforward and brake before physical joint limits."""
    eye=torch.eye(3,device=q.device,dtype=q.dtype)
    feed=(jacobian.transpose(1,2)@torch.linalg.solve(
        jacobian@jacobian.transpose(1,2)+.012**2*eye,cartesian_velocity[:,:,None])).squeeze(-1).clamp(-1.5,1.5)
    requested=(base+feed).clamp(-3.,3.)
    lower=((limits[:,:,0]+.005-q)/.04).clamp(-3.,3.)
    upper=((limits[:,:,1]-.005-q)/.04).clamp(-3.,3.)
    return torch.maximum(torch.minimum(requested,upper),lower)
