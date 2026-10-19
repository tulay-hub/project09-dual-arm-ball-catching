"""Training-only sampling and conservative PPO continuation; evaluation stays uniform."""
import torch

ROBUST_PROFILE='robust_l2'
MIXED_PROFILE='mixed_all'
CONSERVATIVE_PROFILES=(ROBUST_PROFILE,MIXED_PROFILE)
ROBUST_PPO={'learning_rate':5e-5,'schedule':'fixed','num_learning_epochs':3}
ROBUST_DIRECTION_TARGET={'front':.35,'left_side':.25,'right_side':.40}


def resolve_training_profile(explicit,checkpoint_task=None,default='legacy'):
    if explicit is not None:return explicit
    if checkpoint_task is not None:return checkpoint_task.get('training_profile','legacy')
    return default


def sample_receivers(count,device,profile='legacy'):
    if profile in CONSERVATIVE_PROFILES:return (torch.rand(count,device=device)<.60).long()
    return torch.randint(0,2,(count,),device=device)


def choose_throw_kind(rng,side,profile='legacy'):
    front_probability=(.375 if side==0 else 1/3) if profile in CONSERVATIVE_PROFILES else .5
    return 'front' if rng.random()<front_probability else ('left_side' if side==0 else 'right_side')


def configure_optimizer(algorithm,profile):
    """Preserve Adam moments while explicitly applying the new LR after load()."""
    if profile not in CONSERVATIVE_PROFILES:return None
    before=[float(group['lr']) for group in algorithm.optimizer.param_groups]
    algorithm.learning_rate=ROBUST_PPO['learning_rate'];algorithm.schedule='fixed'
    for group in algorithm.optimizer.param_groups:group['lr']=ROBUST_PPO['learning_rate']
    return {'profile':profile,'source_learning_rates':before,'learning_rates':[group['lr'] for group in algorithm.optimizer.param_groups],
        'schedule':'fixed','optimizer_moments_reset':False}


def sample_episode_levels(count,profile,level):
    """Resample all supported difficulties independently at every episode reset."""
    if profile==MIXED_PROFILE:return torch.randint(0,3,(count,)).numpy()
    return torch.full((count,),level,dtype=torch.long).numpy()


def augment_measurement(measurement,rng,level,training=True):
    if measurement is None or not training or level<2:return measurement
    return None if rng.random()<.02 else measurement+rng.normal(0,[.0005,.0005,.001])


def validation_selection(best,reference,stale,candidate,threshold=.005):
    """Retain every highest score; plateau needs a separate material improvement."""
    replace=candidate>best
    material=candidate>reference+threshold
    return {'replace_best':replace,'best':max(best,candidate),
        'reference':candidate if material else reference,'stale':0 if material else stale+1}


def plateau_reached(completed,start,stale,patience=5,min_new_updates=0):
    return completed>=1100 and completed-start>=min_new_updates and stale>=patience
