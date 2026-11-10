"""External validation stress check for unavailable goalkeeper metadata."""
import json
from pathlib import Path
import numpy as np
from train_paper2023 import load_model
from paper2023_data import get,relative_features
from paper2023_model import outputs,upsample,joint_probabilities
from research_data import validation
from temporal_decoders import boundary_features
from fuse_paper2023 import predict
from develop_v2 import score
from torch_runtime import torch
import joblib

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    target=ROOT/'external_reports/contact_missing_keeper_validation.json'
    if target.exists():raise RuntimeError('Preserve original feature-absence assessment')
    torch.set_num_threads(2)
    model,_=load_model(ROOT/'models/paper2023/relative.pt','cpu')
    choice=json.loads((ROOT/'models/paper2023/fusion.validation.json').read_text())['selected']
    base=joblib.load(ROOT/'models/external_v3.joblib')
    state=joblib.load(ROOT/'models/state_boundary.joblib')
    rows=[]
    for h,noisy,d in validation():
        obj=get('J03WR9',h,noisy,interpolation='short')
        obj['x'][:,:,6]=0  # metadata unavailable, never invent a role from labels
        marginal=upsample(outputs(model,relative_features(obj['x']),'cpu'),obj['samples'],len(d['y']))
        p=(1-choice['neural_weight'])*base['model'].predict_proba(d['x'])+choice['neural_weight']*joint_probabilities(marginal)
        bx=boundary_features(d['x'])
        b=(base['boundary_model'].predict_proba(bx)[:,1:2] if choice['boundary_kind']=='binary' else
           state['boundary_model'].predict_proba(bx)[:,1:])
        result=dict(half=h+1,noisy=noisy,**score(d['y'],predict(p,b,choice['boundary_kind'])))
        rows.append(result);print('Keeper-absence external check',h+1,noisy,result['spell_f1'],flush=True)
    target.write_text(json.dumps(dict(status='External feature-absence stress assessment; CPU inference versus recorded GPU validation reference',
        rows=rows,reference_rows=choice['rows']),indent=2))
