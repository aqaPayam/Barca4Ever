"""Frozen state-boundary candidate, compared on reused external regression games."""
import json
from pathlib import Path
from external_idsse import load,SPLITS
from learned_possession import corrupt
from possession_v2 import features
from temporal_decoders import boundary_features,adaptive_decode
from research_state_boundaries import decode
from develop_v2 import score
import joblib

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    target=ROOT/'external_reports/state_boundary_regression.json'
    if target.exists():raise RuntimeError('Preserve original regression results')
    old=joblib.load(ROOT/'models/external_v3.joblib')
    new=joblib.load(ROOT/'models/state_boundary.joblib')
    rows=[]
    for match in SPLITS['test']:
        for h,half in enumerate(load(match)):
            for noisy in [False,True]:
                p,b=half['players'],half['ball']
                if noisy:p,b=corrupt(p,b,9500+h)
                x=features(p,half['teams'],b,half['pitch']);bx=boundary_features(x)
                prob=old['model'].predict_proba(x)
                binary=old['boundary_model'].predict_proba(bx)[:,1]
                state=new['boundary_model'].predict_proba(bx)[:,1:]
                for name,pred in [('v3',adaptive_decode(prob,binary,old['penalty'],old['boundary_scale'])),
                                  ('state_boundary',decode(prob,state,new['penalty'],new['boundary_scale']))]:
                    r=dict(match=match,half=h+1,noisy=noisy,model=name,**score(half['labels'],pred))
                    rows.append(r);print(json.dumps({k:v for k,v in r.items() if k!='recall_by_duration'}),flush=True)
    target.write_text(json.dumps(dict(status='Regression on previously inspected external games, not fresh blind testing',rows=rows),indent=2))
