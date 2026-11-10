"""Protect every old tensor while training only new missing-observation columns."""
import hashlib
import numpy as np
from hz25_model import expand
from paper2023_model import PaperPossession
from torch_runtime import torch

INPUT_WEIGHTS=('encoder.stem.weight','instant.0.weight')


def initialize(parent,device):
    model,architecture=expand(parent)
    model=model.to(device).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    protected={};handles=[]
    parameters=dict(model.named_parameters())
    for name in INPUT_WEIGHTS:
        parameter=parameters[name];parameter.requires_grad_(True)
        protected[name]=parameter[:,:17].detach().clone()
        mask=torch.ones_like(parameter);mask[:,:17]=0
        handles.append(parameter.register_hook(lambda grad,mask=mask:grad*mask))
    return model,architecture,protected,handles


def restore_protected_columns(model,protected):
    parameters=dict(model.named_parameters())
    with torch.no_grad():
        for name,value in protected.items():parameters[name][:,:17].copy_(value)


def protected_proof(model,parent):
    current=model.state_dict();before=hashlib.sha256();after=hashlib.sha256();failed=[]
    for name,old in parent['state_dict'].items():
        now=current[name]
        if name in INPUT_WEIGHTS:now=now[:,:17]
        old_array=old.detach().cpu().contiguous().numpy();now_array=now.detach().cpu().contiguous().numpy()
        prefix=(name+str(old_array.dtype)+str(old_array.shape)).encode()
        before.update(prefix);before.update(old_array.tobytes())
        after.update(prefix);after.update(now_array.tobytes())
        if old_array.dtype!=now_array.dtype or old_array.shape!=now_array.shape or old_array.tobytes()!=now_array.tobytes():failed.append(name)
    if failed:raise RuntimeError('Protected legacy tensors changed: '+','.join(failed))
    if any(module.training for module in model.modules()):raise RuntimeError('Residual adaptation must remain in eval mode')
    return dict(all_old_tensors_bit_exact=True,tensors_checked=len(current),before_sha256=before.hexdigest(),
                after_sha256=after.hexdigest(),model_eval=True)


def extra_norm(model):
    parameters=dict(model.named_parameters())
    return float(torch.sqrt(sum((parameters[name][:,17:]**2).sum() for name in INPUT_WEIGHTS)).detach().cpu())


def load(path,device):
    artifact=torch.load(path,map_location='cpu',weights_only=True)
    assert artifact['input_mode']=='missing20' and artifact['rate']==16
    model=PaperPossession(**artifact['architecture']).to(device).eval()
    model.load_state_dict(artifact['state_dict'])
    return model,artifact
