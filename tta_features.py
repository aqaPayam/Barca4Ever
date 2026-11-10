"""Label-free symmetry transformations on the original11-channel input."""
import numpy as np


def team_swap(raw):
    if raw.ndim!=3 or raw.shape[1:]!=(36,11):raise ValueError('Expected original fixed36-object/11-channel input')
    order=np.r_[0,1,np.arange(19,36),np.arange(2,19)]
    result=raw[:,order].copy()
    result[:,:,7:9]=result[:,:,7:9][:,:,::-1]
    return result


def reflect_pitch(raw):
    result=raw.copy()
    seen=raw[:,:,10]==0
    velocity_seen=seen&np.r_[seen[:1],seen[:-1]]
    result[:,:,:2][seen]*=-1
    result[:,:,2:4][velocity_seen]*=-1
    return result


def restore_team_probabilities(probabilities):
    return probabilities[:,[1,0,2]]


def average_predictions(predictions,candidate):
    keys={'swap':['original','swap'],'reflection':['original','reflection'],
          'both':['original','swap','reflection']}[candidate]
    return np.stack([predictions[key] for key in keys]).mean(0)
