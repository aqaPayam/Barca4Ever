"""Frozen boundary candidate vs V2 on previously inspected external matches."""
import json
from pathlib import Path
from external_idsse import load,SPLITS
from learned_possession import corrupt,viterbi
from possession_v2 import features
from temporal_decoders import boundary_features,adaptive_decode
from develop_v2 import score
import joblib

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    target=ROOT/'external_reports/research_v3/boundary_regression.json'
    if target.exists():raise RuntimeError('Preserve original regression results')
    candidate=joblib.load(ROOT/'models/research_v3/learned_boundary.joblib')
    rows=[]
    for match in SPLITS['test']:
        for h,half in enumerate(load(match)):
            for noisy in [False,True]:
                p,b=half['players'],half['ball']
                if noisy:p,b=corrupt(p,b,9500+h)
                x=features(p,half['teams'],b,half['pitch'])
                prob=candidate['model'].predict_proba(x)
                boundary=candidate['boundary_model'].predict_proba(boundary_features(x))[:,1]
                for name,pred in [('v2',viterbi(prob,8)),('learned_boundary',adaptive_decode(prob,boundary,
                    candidate['penalty'],candidate['boundary_scale']))]:
                    r=dict(match=match,half=h+1,noisy=noisy,model=name,**score(half['labels'],pred))
                    rows.append(r)
                    print(json.dumps({k:v for k,v in r.items() if k!='recall_by_duration'}),flush=True)
    target.write_text(json.dumps(dict(status='Regression on previously inspected external test matches; not fresh blind evidence',rows=rows),indent=2))
