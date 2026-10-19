"""Per-environment exposure tracking; duplicate renders preserve valid feedback."""
import numpy as np
from ball_tracker import BallTracker
from ballistics import GRAVITY


class ExposureTrackingBatch:
    """Track unique image exposures, allowing a reset to start a new generation.

    fetch_frames(ids) returns RGB, metric depth, K, camera positions and XYZW ROS
    quaternions for those ids only. No simulator object state is accepted here.
    Visibility expires by control time, so a frozen sensor cannot stay fresh.
    """
    def __init__(self, num_envs, radius=.028, max_age=.15,filtered_state=False):
        self.num_envs=num_envs;self.max_age=max_age;self.filtered_state=filtered_state
        self.trackers=[BallTracker(radius) for _ in range(num_envs)]
        self.contact_mode=np.zeros(num_envs,dtype=bool)
        self.last_processed=np.full(num_envs,np.nan)
        self.last_measurement=np.full(num_envs,np.nan)
        self.position=np.zeros((num_envs,3));self.velocity=np.zeros((num_envs,3))
        self.visible=np.zeros(num_envs,dtype=bool);self.age=np.ones(num_envs)
        self.samples=0;self.accepted=0;self.skipped=0

    def reset(self, ids):
        ids=np.asarray(ids,dtype=int).reshape(-1)
        for i in ids:
            self.trackers[i].filter.gravity=GRAVITY.copy()
            self.trackers[i].filter.acceleration_std=.15
            self.trackers[i].reset()
        self.contact_mode[ids]=False
        self.last_processed[ids]=np.nan;self.last_measurement[ids]=np.nan
        self.position[ids]=0.;self.velocity[ids]=0.;self.visible[ids]=False;self.age[ids]=1.

    def set_contact_mode(self,contact_mask):
        """Tactile contact changes the estimator model, never simulator physics."""
        contact=np.asarray(contact_mask,dtype=bool)
        if contact.shape!=(self.num_envs,):raise ValueError('One tactile mode per environment required')
        for e in np.flatnonzero(contact!=self.contact_mode):
            self.trackers[e].filter.gravity=np.zeros(3) if contact[e] else GRAVITY.copy()
            self.trackers[e].filter.acceleration_std=3. if contact[e] else .15
            self.trackers[e].reset()
        self.contact_mode[:]=contact

    def update(self, timestamps, fetch_frames=None, control_time=None, fetch_measurements=None):
        times=np.asarray(timestamps,dtype=float)
        if times.ndim==0:times=np.full(self.num_envs,float(times))
        if times.shape!=(self.num_envs,) or not np.isfinite(times).all():
            raise ValueError('Exposure timestamps must be finite and have one value per environment')
        now=float(np.max(times) if control_time is None else control_time)
        if not np.isfinite(now):raise ValueError('Control time must be finite')
        ids=np.flatnonzero(np.isnan(self.last_processed)|(times>self.last_processed))
        self.skipped+=self.num_envs-len(ids)
        if len(ids):
            if fetch_measurements is None:
                rgb,depth,K,pos,quat=fetch_frames(ids)
            else:
                measurements=fetch_measurements(ids)
            self.last_processed[ids]=times[ids]
            self.visible[ids]=False
            self.samples+=len(ids)
            for row,e in enumerate(ids):
                measured=(self.trackers[e].observe(rgb[row],depth[row],K[row],pos[row],quat[row],times[e])
                    if fetch_measurements is None else self.trackers[e].observe_world(measurements[row],times[e]))
                if measured is not None:
                    self.position[e]=measured
                    state=self.trackers[e].state(times[e])
                    if self.filtered_state and state is not None:self.position[e]=state[0][:3]
                    self.velocity[e]=state[0][3:] if state is not None else 0.
                    self.last_measurement[e]=times[e];self.visible[e]=True;self.accepted+=1
        valid=np.isfinite(self.last_measurement)
        self.age[valid]=np.maximum(0.,now-self.last_measurement[valid])
        self.age[~valid]=1.
        self.visible[self.age>self.max_age]=False
        return ids


class SynchronousSimulationClock:
    """Exact capture/control clock for a synchronously rendered simulator.

    Record only after rendering/sensor update completed. Polling or stepping
    physics alone cannot advance capture time, so frozen-camera expiry still
    works. Hardware cameras must use their own calibrated clock conversion.
    """
    def __init__(self, num_envs, physics_hz):
        if num_envs<1 or not np.isfinite(physics_hz) or physics_hz<=0:
            raise ValueError('Positive environment count and physics frequency required')
        self.physics_hz=float(physics_hz)
        self.capture_steps=np.full(num_envs,-1,dtype=np.int64)

    def control_time(self, step_count):
        if not isinstance(step_count,(int,np.integer)) or step_count<0:
            raise ValueError('Physics step count must be a nonnegative integer')
        return int(step_count)/self.physics_hz

    def record_capture(self, step_count, ids=None):
        self.control_time(step_count)
        selected=slice(None) if ids is None else np.asarray(ids,dtype=int)
        if np.any(self.capture_steps[selected]>step_count):
            raise ValueError('Simulation capture step cannot go backwards')
        self.capture_steps[selected]=step_count

    @property
    def timestamps(self):
        if np.any(self.capture_steps<0):raise RuntimeError('Capture must be recorded before tracking')
        return self.capture_steps.astype(np.float64)/self.physics_hz
