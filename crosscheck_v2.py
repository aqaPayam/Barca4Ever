"""Whole-match cross-validation check, not a fresh blind test.

Uses only the four original training matches, with each fold's evaluated matches
excluded from both fitted classifiers. Penalties are selected separately on WR9.
The previously inspected WMX/WN1 test matches and competition match are untouched.
"""
import json
from pathlib import Path
import numpy as np
from learned_possession import features, corrupt, predict
from possession_v2 import features as new_features
from external_idsse import load, SPLITS
from develop_v2 import fit_matches, score
from sklearn.ensemble import HistGradientBoostingClassifier


def fit_reference(matches):
    xs, ys = [], []
    for m,match in enumerate(matches):
        for h,half in enumerate(load(match)):
            y=half['labels']; take=(np.arange(len(y))%5==0)&(y<3)
            for noisy in [False,True]:
                p,b=half['players'],half['ball']
                if noisy:p,b=corrupt(p,b,7300+10*m+h)
                for swap in [False,True]:
                    x=features(p,1-half['teams'] if swap else half['teams'],b,half['pitch'])
                    xs.append(x[take]);ys.append(np.where(y[take]<2,1-y[take],y[take]) if swap else y[take])
    model=HistGradientBoostingClassifier(max_iter=130,max_leaf_nodes=15,learning_rate=.08,
        l2_regularization=2,min_samples_leaf=60,early_stopping=False,random_state=20261104)
    model.fit(np.concatenate(xs),np.concatenate(ys))
    return model


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    target=root/'external_reports/v2_crosscheck.json'
    if target.exists():raise RuntimeError('Cross-validation check already exists; preserve original evidence')
    selected=json.loads((root/'external_reports/v2_validation.json').read_text())['selected']['penalty']
    folds=[(SPLITS['train'][:2],SPLITS['train'][2:]),(SPLITS['train'][2:],SPLITS['train'][:2])]
    rows=[]
    for fold,(train,test) in enumerate(folds):
        assert not set(train)&set(test)
        print('FOLD {} train={} evaluate={}'.format(fold+1,train,test),flush=True)
        old=fit_reference(train)
        new=fit_matches(train)
        for m,match in enumerate(test):
            for h,half in enumerate(load(match)):
                for noisy in [False,True]:
                    p,b=half['players'],half['ball']
                    if noisy:p,b=corrupt(p,b,11200+100*fold+10*m+h)
                    for name,model,fn,penalty in [('v1',old,features,16),('v2',new,new_features,selected)]:
                        x=fn(p,half['teams'],b,half['pitch'])
                        r=dict(fold=fold+1,match=match,half=h+1,noisy=noisy,model=name,
                               **score(half['labels'],predict(model,x,penalty)))
                        rows.append(r)
                        print(json.dumps({k:v for k,v in r.items() if k!='recall_by_duration'}),flush=True)
    target.write_text(json.dumps(dict(status='Cross-validation on previously used development matches; no fresh blind test',
                      folds=folds,penalties=dict(v1=16,v2=selected),rows=rows),indent=2))
