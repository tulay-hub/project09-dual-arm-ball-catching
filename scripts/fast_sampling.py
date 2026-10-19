"""Batch transfer and image compaction without changing the catch task rules."""
import numpy as np
import torch


def as_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)
    if hasattr(value, 'torch'):
        return value.torch
    import warp as wp
    return wp.to_torch(value)


def contact_masks(names_by_env, device):
    """Cache masks in the actual PhysX sensor order, including original palms."""
    sides=[]; fingers=[]
    for names in names_by_env:
        sides.append([[n.startswith(p) for n in names] for p in ['lh_', 'rh_']])
        fingers.append([[n.startswith(p+f) for n in names]
                        for p in ['lh_', 'rh_'] for f in ['thumb','index','middle','ring','pinky']])
    return (torch.tensor(sides,device=device,dtype=torch.bool),
            torch.tensor(fingers,device=device,dtype=torch.bool))


def reduce_contact_forces(vectors, side_masks, finger_masks):
    norms=torch.linalg.vector_norm(vectors,dim=-1)
    totals=(norms[:,None,:]*side_masks).sum(-1)
    features=torch.where(finger_masks,norms[:,None,:],0.).amax(-1)
    return totals,features,norms.amax()


def sparse_red_pixels(rgb, depth):
    """Return [env,u,v,depth] in row-major order, using the original mask."""
    z=depth[...,0] if depth.ndim==4 else depth
    r=rgb[...,0].float();g=rgb[...,1].float();b=rgb[...,2].float()
    mask=((r>40)&(r-g>25)&(r-b>25)&(r>1.25*g)&(r>1.25*b)
          &torch.isfinite(z)&(z>.03)&(z<3.))
    e,v,u=mask.nonzero(as_tuple=True)
    return torch.stack((e.float(),u.float(),v.float(),z[e,v,u]),dim=1)
