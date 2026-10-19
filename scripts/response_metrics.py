"""Diagnostic classifications; these never change rewards or control decisions."""
import numpy as np


def failure_stage(outcome,contact,hold):
    if outcome=='success':return 'success'
    if not contact:return 'no_contact'
    if hold==0:return 'contact_without_stable_hold'
    if hold<60:return 'held_less_than_1s'
    return 'held_1s_but_not_full_success'


def crossing_errors(hand_center,ball_center,goal):
    hand,ball,goal=map(np.asarray,(hand_center,ball_center,goal))
    return {'hand_to_goal_xy_m':float(np.linalg.norm((hand-goal)[:2])),
            'ball_to_goal_xy_m':float(np.linalg.norm((ball-goal)[:2]))}


def failed_retention_criteria(retention,speed,drift,distance):
    reasons=[]
    if retention.get('hand_contact_force_n',0)<=.03:reasons.append('contact_force_lost')
    if retention.get('surface_contact_points',0)<=0:reasons.append('no_valid_surface_contact')
    if retention.get('max_penetration_m',0)>.003:reasons.append('penetration_over_3mm')
    if speed>=.15:reasons.append('relative_speed_over_015')
    if drift>=.008:reasons.append('drift_over_8mm')
    if distance>=.065:reasons.append('palm_distance_over_65mm')
    return reasons
