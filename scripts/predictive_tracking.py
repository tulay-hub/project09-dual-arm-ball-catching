"""Position-priority interception using measured joints and image-derived arrival."""
import torch

REVISION='position_priority_early_arrival_contact_filter_v2'


def position_priority_delta(jacobian,position_error,rotation_error,q,limits):
    """Bounded local IK: position first, orientation in its remaining null space.

    Active joints are fixed at their bounds and the remaining joints re-solve
    the position task. Physical joint limits are never expanded.
    """
    n,joints=q.shape;eye3=torch.eye(3,device=q.device,dtype=q.dtype)
    free=torch.ones_like(q);fixed=torch.zeros_like(q)
    lower=limits[:,:,0]+.005-q;upper=limits[:,:,1]-.005-q
    jp=jacobian[:,:3];jr=jacobian[:,3:]
    for _ in range(joints+1):
        a=jp*free[:,None,:]
        inverse=a.transpose(1,2)@torch.linalg.solve(a@a.transpose(1,2)+.008**2*eye3,eye3.expand(n,-1,-1))
        residual=position_error-(jp@fixed[:,:,None]).squeeze(-1)
        primary=fixed+(inverse@residual[:,:,None]).squeeze(-1)
        null=torch.diag_embed(free)-inverse@a
        secondary=jr@null
        remaining=rotation_error-(jr@primary[:,:,None]).squeeze(-1)
        turn=(null@secondary.transpose(1,2)@torch.linalg.solve(
            secondary@secondary.transpose(1,2)+.08**2*eye3,remaining[:,:,None])).squeeze(-1)
        solution=primary+.45*turn
        violated=(solution<lower)|(solution>upper)
        fixed=torch.where(violated,torch.maximum(torch.minimum(solution,upper),lower),fixed)
        free=free*(~violated).to(q.dtype)
    return torch.maximum(torch.minimum(solution,upper),lower)


def early_arrival_velocity(delta,time_to_contact,q,limits,limit=3.):
    """Spend at most 40% of remaining flight on positioning, with limit braking."""
    horizon=torch.where(time_to_contact>=0,((time_to_contact-.04)*.4).clamp(.045,.12),.10)
    velocity=(delta/horizon[:,None]).clamp(-limit,limit)
    lower=((limits[:,:,0]+.005-q)/.04).clamp(-limit,limit)
    upper=((limits[:,:,1]-.005-q)/.04).clamp(-limit,limit)
    return torch.maximum(torch.minimum(velocity,upper),lower)
