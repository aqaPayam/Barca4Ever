"""Frozen destination-boundary settings, evaluated on prior external CV folds."""
import json
from pathlib import Path
import numpy as np
from external_idsse import load,SPLITS
from learned_possession import corrupt
from possession_v2 import features
from research_trees import boundary_features
from research_state_boundaries import fit,decode
from develop_v2 import score
import joblib

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    target=ROOT/'external_reports/state_boundary_crosscheck.json'
    if target.exists():raise RuntimeError('Preserve original comparison results')
    choice=json.loads((ROOT/'external_reports/state_boundary_validation.json').read_text())['selected']
    original=json.loads((ROOT/'external_reports/research_v3/boundary_crosscheck.json').read_text())['rows']
    rows=[dict(r,model='v3') for r in original if r['model']=='learned_boundary']
    folds=[(SPLITS['train'][:2],SPLITS['train'][2:]),(SPLITS['train'][2:],SPLITS['train'][:2])]
    for fold,(train,test) in enumerate(folds):
        base=joblib.load(ROOT/f'external_cache/research_fusion_cv/base_fold{fold+1}.joblib')
        boundary=fit(train)
        assert boundary.classes_.tolist()==[0,1,2,3]
        for m,match in enumerate(test):
            for h,half in enumerate(load(match)):
                for noisy in [False,True]:
                    p,b=half['players'],half['ball']
                    if noisy:p,b=corrupt(p,b,11200+100*fold+10*m+h)
                    x=features(p,half['teams'],b,half['pitch'])
                    prob=base.predict_proba(x);onset=boundary.predict_proba(boundary_features(x))[:,1:]
                    pred=decode(prob,onset,choice['penalty'],choice['boundary_scale'])
                    r=dict(fold=fold+1,match=match,half=h+1,noisy=noisy,model='state_boundary',**score(half['labels'],pred))
                    rows.append(r);print(json.dumps({k:v for k,v in r.items() if k!='recall_by_duration'}),flush=True)
    target.write_text(json.dumps(dict(status='Development cross-validation, not a fresh blind test',selection=choice,rows=rows),indent=2))
