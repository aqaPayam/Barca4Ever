"""Load a contact checkpoint with function-preserving zero-padded inputs."""
from paper2023_model import PaperPossession
from torch_runtime import torch


def expand(parent):
    assert parent['architecture']['channels']==17
    architecture=dict(parent['architecture'],channels=20)
    model=PaperPossession(**architecture)
    state={k:v.clone() for k,v in parent['state_dict'].items()}
    for key in ['encoder.stem.weight','instant.0.weight']:
        value=state[key]
        shape=list(value.shape);shape[1]=20
        padded=torch.zeros(shape,dtype=value.dtype)
        padded[:,:17]=value
        state[key]=padded
    model.load_state_dict(state)
    return model,architecture


def load(path,device):
    artifact=torch.load(path,map_location='cpu',weights_only=True)
    model=PaperPossession(**artifact['architecture'])
    model.load_state_dict(artifact['state_dict'])
    return model.to(device).eval(),artifact
