"""External-only interpolation of generic and destination-specific boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np
from research_data import validation
from research_trees import boundary_features,fit_boundary
from research_state_boundaries import decode,fit
from external_idsse import load,SPLITS
from learned_possession import corrupt
from possession_v2 import features
from develop_v2 import score
import joblib

ROOT=Path(__file__).resolve().parent


def predictions(base,binary,state,x,weight):
    bx=boundary_features(x)
    b=binary.predict_proba(bx)[:,1,None]
    s=state.predict_proba(bx)[:,1:]
    return decode(base.predict_proba(x),(1-weight)*b+weight*s,16,4)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['validation','crosscheck','regression'],default='validation')
    args=parser.parse_args()
    target=ROOT/f'external_reports/blend_boundary_{args.stage}.json'
    if target.exists():raise RuntimeError('Preserve original blend evidence')
    if args.stage=='validation':
        old=joblib.load(ROOT/'models/external_v3.joblib');new=joblib.load(ROOT/'models/state_boundary.joblib')
        ready=[]
        for h,n,d in validation():
            x=d['x'];bx=boundary_features(x)
            ready.append((h,n,d['y'],old['model'].predict_proba(x),
                old['boundary_model'].predict_proba(bx)[:,1,None],new['boundary_model'].predict_proba(bx)[:,1:]))
        trials=[]
        for weight in [.25,.5,.75]:
            rows=[dict(half=h+1,noisy=n,**score(y,decode(p,(1-weight)*b+weight*s,16,4))) for h,n,y,p,b,s in ready]
            trial=dict(weight=weight,rows=rows,spell_f1=float(np.mean([r['spell_f1'] for r in rows])),
                       macro_f1=float(np.mean([r['macro_f1'] for r in rows])))
            trials.append(trial);print('Boundary blend',weight,trial['spell_f1'],flush=True)
        best=max(trials,key=lambda r:(r['spell_f1'],r['macro_f1']))
        target.write_text(json.dumps(dict(status='External validation selection only',trials=trials,selected=best),indent=2))
        joblib.dump(dict(model=old['model'],binary_model=old['boundary_model'],state_model=new['boundary_model'],
                         weight=best['weight'],kind='blended_boundary',feature_version=2,penalty=16,boundary_scale=4),ROOT/'models/blended_boundary.joblib')
    else:
        weight=json.loads((ROOT/'external_reports/blend_boundary_validation.json').read_text())['selected']['weight']
        rows=[]
        if args.stage=='crosscheck':
            groups=[(SPLITS['train'][:2],SPLITS['train'][2:]),(SPLITS['train'][2:],SPLITS['train'][:2])]
            cache=ROOT/'external_cache/blended_boundaries';cache.mkdir(exist_ok=True)
        else:groups=[(None,SPLITS['test'])]
        for fold,(train,test) in enumerate(groups):
            if train:
                base=joblib.load(ROOT/f'external_cache/research_fusion_cv/base_fold{fold+1}.joblib')
                binary=fit_boundary(train);state=fit(train)
                joblib.dump(dict(binary=binary,state=state),cache/f'fold{fold+1}.joblib')
            else:
                artifact=joblib.load(ROOT/'models/blended_boundary.joblib')
                base,binary,state=artifact['model'],artifact['binary_model'],artifact['state_model']
            for m,match in enumerate(test):
                for h,half in enumerate(load(match)):
                    for noisy in [False,True]:
                        seed=11200+100*fold+10*m+h if train else 9500+h
                        p,b=half['players'],half['ball']
                        if noisy:p,b=corrupt(p,b,seed)
                        x=features(p,half['teams'],b,half['pitch'])
                        pred=predictions(base,binary,state,x,weight)
                        row=dict(match=match,half=h+1,fold=fold+1,noisy=noisy,model='boundary_blend',**score(half['labels'],pred))
                        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='recall_by_duration'}),flush=True)
        target.write_text(json.dumps(dict(status='Development cross-validation' if train else 'Reused external regression, not fresh blind testing',weight=weight,rows=rows),indent=2))
