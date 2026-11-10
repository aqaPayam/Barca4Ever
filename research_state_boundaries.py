"""External-only extension: predict which state begins at a boundary.

V3 detects any change. This extension supplies destination-specific change scores.
No competition inputs or labels are accessed.
"""
import json
from pathlib import Path
import numpy as np
from research_data import get,validation
from research_trees import boundary_features
from external_idsse import SPLITS
from develop_v2 import score
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
from temporal_state import decode

ROOT=Path(__file__).resolve().parent


def onset_targets(y):
    target=np.zeros(len(y),np.int8)
    changes=np.flatnonzero((y[1:]!=y[:-1])&(y[1:]<3)&(y[:-1]<3))+1
    # Process wider offsets first: nearest boundary wins, exact changes last.
    for distance in [3,2,1,0]:
        for shift in ([-distance,distance] if distance else [0]):
            ix=changes+shift;valid=(ix>=0)&(ix<len(y))
            target[ix[valid]]=y[changes[valid]]+1
    return target


def fit(matches):
    xx,yy=[],[]
    for match in matches:
        for h in [0,1]:
            for noisy in [False,True]:
                for swap in [False,True]:
                    d=get(match,h,noisy,swap);take=np.arange(0,len(d['y']),5)
                    xx.append(boundary_features(d['x'])[take]);yy.append(onset_targets(d['y'])[take])
    x,y=np.concatenate(xx),np.concatenate(yy)
    print('Fitting external state-boundary classifier',x.shape,flush=True)
    model=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,learning_rate=.08,
        l2_regularization=3,min_samples_leaf=60,early_stopping=False,random_state=20261107)
    model.fit(x,y,sample_weight=np.where(y,10.,1.))
    return model


if __name__=='__main__':
    target=ROOT/'external_reports/state_boundary_validation.json'
    if target.exists():raise RuntimeError('Preserve existing external results')
    model=fit(SPLITS['train']);base=joblib.load(ROOT/'models/external_v2.joblib')['model']
    results=[]
    for h,noisy,d in validation():
        x=d['x'];results.append((h,noisy,d,base.predict_proba(x),model.predict_proba(boundary_features(x))[:,1:]))
    trials=[]
    for penalty in [8,16]:
        for scale in [4,16]:
            rows=[dict(half=h+1,noisy=n,**score(d['y'],decode(p,b,penalty,scale))) for h,n,d,p,b in results]
            trial=dict(penalty=penalty,boundary_scale=scale,rows=rows,
                       spell_f1=float(np.mean([r['spell_f1'] for r in rows])))
            trials.append(trial);print('State boundary trial',penalty,scale,trial['spell_f1'],flush=True)
    best=max(trials,key=lambda r:r['spell_f1'])
    target.write_text(json.dumps(dict(status='External validation selection only',trials=trials,selected=best),indent=2))
    joblib.dump(dict(model=base,boundary_model=model,penalty=best['penalty'],boundary_scale=best['boundary_scale'],
                     feature_version=2,kind='state_boundary'),ROOT/'models/state_boundary.joblib')
