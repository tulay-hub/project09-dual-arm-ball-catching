"""Versioned observation/action mapping for local XY and preclosure learning."""
import torch

LEGACY_ACTION_INDEX=[0,1,2,3,7,8,9,10]
NEW_ACTION_INDEX=[4,5,6,11,12,13]
V4_NAME='LKLS73CameraCatchXYTimingResidual-v4'
V4_OBS=100
V4_ACTIONS=14
XY_ACTION_INDEX=[4,5,11,12]
XY_RESIDUAL_SCALE_M=.020
XY_ACTION_LIMIT=10.
XY_CONTRACT={'action_contract_revision':'v4_xy200_predictive_v1',
    'xy_action_hard_clip':True,'xy_residual_hard_limit_m':.200,
    'xy_residual_scale_m_per_unit':XY_RESIDUAL_SCALE_M}


def resolve_tracking_profile(explicit,checkpoint_task=None,default='fast'):
    """Replay new checkpoints with their saved controller; old checkpoints stay fast."""
    if explicit is not None:return explicit
    if checkpoint_task is not None:return checkpoint_task.get('tracking_profile','fast')
    return default


def resolve_grip_profile(explicit,checkpoint_task=None,default='fast'):
    if explicit is not None:return explicit
    if checkpoint_task is not None:return checkpoint_task.get('grip_profile','fast')
    return default


def filter_residual_actions(actions,previous,xy_slew_per_step=.1):
    """v4 XY is bounded at 200mm per axis; preserve original physical units."""
    raw=actions.detach()
    if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite policy actions')
    requested=raw.clamp(-1,1)
    slew=torch.full_like(raw,.1)
    if raw.shape[-1]==V4_ACTIONS:
        requested[...,XY_ACTION_INDEX]=raw[...,XY_ACTION_INDEX].clamp(-XY_ACTION_LIMIT,XY_ACTION_LIMIT)
        slew[...,XY_ACTION_INDEX]=xy_slew_per_step
    difference=requested-previous
    return requested,previous+torch.maximum(torch.minimum(difference,slew),-slew)


def legacy_actions(actions):
    return actions if actions.shape[-1]==8 else actions[...,LEGACY_ACTION_INDEX]


def visual_features(target_error,ttc,palm_velocity,relative_velocity,valid,saturation,actions):
    return torch.cat(((target_error*5).clamp(-2,2),ttc.clamp(0,1)[:,None],
        (palm_velocity*.5).clamp(-3,3),(relative_velocity*.5).clamp(-3,3),
        valid.float()[:,None],saturation[:,None],actions[:,NEW_ACTION_INDEX]),dim=-1)


def tracking_potential(error_xy,relative_speed,captured,done):
    alignment=2*torch.exp(-(error_xy/.05).square())
    retention=2*torch.exp(-(relative_speed/.20).square())
    return torch.where(done,0.,torch.where(captured,retention,alignment))
