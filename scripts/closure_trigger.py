"""Tactile catch-to-close latch; visual occlusion cannot block finger closure."""
import numpy as np


def contact_closure_step(streak, force_n, elapsed_s):
    # A catch impact may last only one physical substep. Peak tactile load is
    # accumulated over both 120Hz substeps, and closure starts next control tick.
    streak = streak + 1 if elapsed_s > .15 and force_n > .03 else 0
    return streak, streak >= 1


def anticipated_closure_fraction(time_to_contact, palm_error_m, observation_age_s,
                                 window_s=.12,alignment_m=.03,maximum_fraction=.20):
    """Keep the configured closure budget for contact; use only visual estimates."""
    if not (0<=time_to_contact<=window_s and palm_error_m<alignment_m and observation_age_s<=.10):
        return 0.
    return float(maximum_fraction*(1-time_to_contact/window_s))


def incoming_hand_tilt(impact_velocity,max_angle=.55):
    velocity=np.asarray(impact_velocity,dtype=float)
    normal=-velocity/max(np.linalg.norm(velocity),1e-9)
    horizontal=np.linalg.norm(normal[:2])
    if horizontal<1e-9 or normal[2]<=0:return np.zeros(3)
    angle=min(max_angle,np.arctan2(horizontal,normal[2]))
    return np.array([-normal[1],normal[0],0.])*angle/horizontal
