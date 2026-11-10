"""External-only role-fallback validation; no competition data access."""
import json
from pathlib import Path
import numpy as np
from external_idsse import load,SPLITS
from paper2023_data import descriptors,object_features,relative_features
from contact_input import infer_goalkeepers
from train_paper2023 import load_model
from paper2023_model import outputs,upsample,joint_probabilities
from learned_possession import corrupt
from possession_v2 import features
from temporal_decoders import boundary_features
from fuse_paper2023 import predict
from develop_v2 import score
from torch_runtime import torch
import joblib

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    target=ROOT/'external_reports/contact_geometry_keeper_validation.json'
    if target.exists():raise RuntimeError('Preserve role-fallback evidence')
    torch.set_num_threads(2);model,_=load_model(ROOT/'models/paper2023/relative.pt','cpu')
    choice=json.loads((ROOT/'models/paper2023/fusion.validation.json').read_text())['selected']
    old=joblib.load(ROOT/'models/external_v3.joblib');state=joblib.load(ROOT/'models/state_boundary.joblib')
    counts=dict(tp=0,fp=0,fn=0);rows=[]
    for match in SPLITS['train']+SPLITS['validation']:
        desc=descriptors(match)
        for h,half in enumerate(load(match)):
            truth=np.array([v[1] for v in desc[h]])
            guess=infer_goalkeepers(half['players'],half['teams'],half['pitch'])
            counts['tp']+=int((truth&guess).sum());counts['fp']+=int((~truth&guess).sum());counts['fn']+=int((truth&~guess).sum())
            if match!='J03WR9':continue
            for noisy in [False,True]:
                p,b=half['players'],half['ball']
                if noisy:p,b=corrupt(p,b,8400+h)
                kk=infer_goalkeepers(p,half['teams'],half['pitch'])
                obj,sample=object_features(p,half['teams'],b,half['pitch'],kk,interpolation='short')
                marginal=upsample(outputs(model,relative_features(obj),'cpu'),sample,len(half['labels']))
                x=features(p,half['teams'],b,half['pitch']);bx=boundary_features(x)
                probability=(1-choice['neural_weight'])*old['model'].predict_proba(x)+choice['neural_weight']*joint_probabilities(marginal)
                boundary=(old['boundary_model'].predict_proba(bx)[:,1:2] if choice['boundary_kind']=='binary'
                          else state['boundary_model'].predict_proba(bx)[:,1:])
                r=dict(half=h+1,noisy=noisy,**score(half['labels'],predict(probability,boundary,choice['boundary_kind'])))
                rows.append(r);print('Geometry role fallback',h+1,noisy,r['spell_f1'],flush=True)
    target.write_text(json.dumps(dict(status='External geometry-role assessment, no fitting; CPU inference',role_counts=counts,rows=rows),indent=2))
