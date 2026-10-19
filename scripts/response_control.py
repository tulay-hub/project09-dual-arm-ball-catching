"""Arrival-time drive targets within the commissioned simulation drive limits."""
import torch


FAST_RESPONSE={
    'ready_finger_fraction':.25,
    'ready_thumb_pitch_rad':.02,
    'preclose_window_s':.20,
    'preclose_alignment_m':.04,
    'preclose_maximum_fraction':.45,
    'closure_duration_s':.25,
    'finger_target_speed_rad_s':.95,
    'arm_velocity_limit_rad_s':3.,
    'arrival_margin_s':.06,
}


def arrival_joint_velocity(delta, time_to_contact, captured, limit=3.):
    """Use only image-derived arrival times and measured joint/palm errors."""
    valid=(time_to_contact>=0)&~captured
    horizon=torch.where(valid,(time_to_contact-.06).clamp(.06,.35),.15)
    return (delta/horizon[:,None]).clamp(-limit,limit)
