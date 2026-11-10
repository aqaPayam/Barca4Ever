"""External controlled two-fold check of the learned-boundary design vs V2."""
import json
from pathlib import Path
import numpy as np
from external_idsse import load,SPLITS
from learned_possession import corrupt,viterbi
from possession_v2 import features
from develop_v2 import fit_matches,score
from research_trees import fit_boundary,boundary_features,adaptive_decode

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    target=ROOT/'external_reports/research_v3/boundary_crosscheck.json'
    if target.exists():raise RuntimeError('Preserve previous cross-check results')
    selection=json.loads((ROOT/'external_reports/research_v3/learned_boundary.json').read_text())['selected']
    folds=[(SPLITS['train'][:2],SPLITS['train'][2:]),(SPLITS['train'][2:],SPLITS['train'][:2])]
    rows=[]
    for fold,(train,test) in enumerate(folds):
        print('Boundary cross-check fold '+str(fold+1),flush=True)
        base=fit_matches(train);boundary=fit_boundary(train)
        for m,match in enumerate(test):
            for h,half in enumerate(load(match)):
                for noisy in [False,True]:
                    p,b=half['players'],half['ball']
                    if noisy:p,b=corrupt(p,b,11200+100*fold+10*m+h)
                    x=features(p,half['teams'],b,half['pitch'])
                    prob=base.predict_proba(x);bc=boundary.predict_proba(boundary_features(x))[:,1]
                    for name,pred in [('v2',viterbi(prob,8)),('learned_boundary',adaptive_decode(
                        prob,bc,selection['penalty'],selection['boundary_scale']))]:
                        r=dict(fold=fold+1,match=match,half=h+1,noisy=noisy,model=name,**score(half['labels'],pred))
                        rows.append(r)
                        print(json.dumps({k:v for k,v in r.items() if k!='recall_by_duration'}),flush=True)
    target.write_text(json.dumps(dict(status='Controlled cross-validation on previously used development data; no blind test',
                       selection=selection,rows=rows),indent=2))
