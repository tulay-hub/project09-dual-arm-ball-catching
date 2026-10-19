"""Explicit v3->v4 weight migration; incompatible optimizer tensors are not loaded."""
import math
import torch
from policy_contract import LEGACY_ACTION_INDEX,V4_NAME


def migrate_state(source,target,actor):
    result={k:v.clone() for k,v in target.items()}
    if set(source)!=set(target):raise ValueError('Unexpected source/target model keys')
    for key,value in source.items():
        out=result[key]
        value=value.to(device=out.device,dtype=out.dtype)
        if key=='obs_normalizer.count':
            out.fill_(16384)
        elif out.shape==value.shape:
            out.copy_(value)
        elif key=='mlp.0.weight' and value.shape[1]==82 and out.shape[1]==100:
            out.zero_();out[:,:82]=value
        elif key.startswith('obs_normalizer.') and value.shape==(1,82) and out.shape==(1,100):
            out.fill_(0 if key.endswith('_mean') else 1);out[:,:82]=value
        elif actor and value.shape[0]==8 and out.shape[0]==14 and value.shape[1:]==out.shape[1:]:
            out.fill_(math.log(.08) if key=='distribution.log_std_param' else 0)
            out[LEGACY_ACTION_INDEX]=value
        else:raise ValueError(f'Unsupported migration tensor {key}: {value.shape}->{out.shape}')
    return result


def source_mean(state,observation,epsilon):
    x=(observation-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+epsilon)
    for layer in [0,2,4]:
        x=torch.nn.functional.elu(torch.nn.functional.linear(x,state[f'mlp.{layer}.weight'],state[f'mlp.{layer}.bias']))
    return torch.nn.functional.linear(x,state['mlp.6.weight'],state['mlp.6.bias'])


def validate_source(saved,target_config):
    task=saved.get('infos',{}).get('task',{})
    if task.get('name')!='LKLS73CameraCatchRetentionResidual-v3' or task.get('observation_dim')!=82 or task.get('action_dim')!=8:
        raise ValueError('Migration accepts only verified v3 82-observation/8-action checkpoints')
    if target_config['name']!=V4_NAME or target_config['observation_dim']!=100 or target_config['action_dim']!=14:
        raise ValueError('Migration target must be the v4 100-observation/14-action contract')
