"""External-only short-possession experiment; never overwrites candidate v1."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from learned_possession import corrupt, predict, features as v1_features
from possession_v2 import features, spell_weights
from external_idsse import load, SPLITS
from evaluation import possession_metrics as pm
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

ROOT = Path(__file__).resolve().parent


def score(y, p):
    r = pm.evaluate(y, p, bootstrap_n=0)
    runs = pm.valid_runs(y)
    gs = pm.possession_spells(y, runs)
    ps = pm.possession_spells(p, runs)
    overlaps = pm.spell_overlaps(gs, ps)
    flags = pm.matched_gt_flags(overlaps, len(gs), 0.5)
    bins = {}
    for name, low, high in [('under_2s', 0, 50), ('2_to_5s', 50, 125), ('over_5s', 125, 10**9)]:
        take = np.array([low <= b-a < high for a,b,k in gs])
        bins[name] = dict(count=int(take.sum()), recall_iou50=float(flags[take].mean()) if take.any() else None)
    return dict(spell_f1=r['spell_f1_mean'], macro_f1=r['macro_f1'],
                gt_spells=r['n_gt_spells'], predicted_spells=r['n_pred_spells'], recall_by_duration=bins)


def fit_matches(matches):
    xs, ys, ws = [], [], []
    for m, match in enumerate(matches):
        print('Preparing v2 training ' + match, flush=True)
        for h, half in enumerate(load(match)):
            y = half['labels']
            take = (np.arange(len(y)) % 5 == 0) & (y < 3)
            w = spell_weights(y)
            for noisy in [False, True]:
                p, b = half['players'], half['ball']
                if noisy:
                    p, b = corrupt(p,b,7300+10*m+h)
                for swapped in [False,True]:
                    teams = 1-half['teams'] if swapped else half['teams']
                    x = features(p, teams, b, half['pitch'])
                    xs.append(x[take]); ws.append(w[take])
                    ys.append(np.where(y[take]<2,1-y[take],y[take]) if swapped else y[take])
    model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=23,learning_rate=.07,
              l2_regularization=3,min_samples_leaf=60,early_stopping=False,random_state=20261105)
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)
    print('Fitting v2 '+str(x.shape),flush=True)
    model.fit(x,y,sample_weight=w)
    return model


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['fit','regression'],default='fit')
    args=parser.parse_args()
    artifact_path=ROOT/'models/external_v2_experimental.joblib'
    report_path=ROOT/'external_reports/v2_validation.json'
    if args.stage=='fit':
        model=fit_matches(SPLITS['train'])
        baseline=joblib.load(ROOT/'models/external_v1.joblib')
        validation=[]
        for h,half in enumerate(load('J03WR9')):
            for noisy in [False,True]:
                p,b=half['players'],half['ball']
                if noisy: p,b=corrupt(p,b,8400+h)
                validation.append((half['labels'],features(p,half['teams'],b,half['pitch']),h,noisy))
        trials=[]
        for penalty in [2,4,8,16,32]:
            rows=[dict(half=h+1,noisy=noisy,**score(y,predict(model,x,penalty))) for y,x,h,noisy in validation]
            row=dict(penalty=penalty,spell_f1=float(np.mean([r['spell_f1'] for r in rows])),
                     macro_f1=float(np.mean([r['macro_f1'] for r in rows])),rows=rows)
            trials.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)
        best=max(trials,key=lambda r:(r['spell_f1'],r['macro_f1']))
        artifact=dict(model=model,penalty=best['penalty'],feature_version=2,splits=SPLITS)
        joblib.dump(artifact,artifact_path)
        result=dict(status='External validation selection only, not an independent test',trials=trials,
                    selected=best,artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest())
        report_path.write_text(json.dumps(result,indent=2))
    else:
        target=ROOT/'external_reports/v2_regression.json'
        if target.exists(): raise RuntimeError('Regression report already exists')
        v2=joblib.load(artifact_path); v1=joblib.load(ROOT/'models/external_v1.joblib')
        rows=[]
        for match in SPLITS['test']:
            for h,half in enumerate(load(match)):
                for noisy in [False,True]:
                    p,b=half['players'],half['ball']
                    if noisy:p,b=corrupt(p,b,9500+h)
                    for name,artifact,fn in [('v1',v1,v1_features),('v2',v2,features)]:
                        x=fn(p,half['teams'],b,half['pitch'])
                        row=dict(match=match,half=h+1,noisy=noisy,model=name,
                                 **score(half['labels'],predict(artifact['model'],x,artifact['penalty'])))
                        rows.append(row)
                        print(json.dumps({k:v for k,v in row.items() if k!='recall_by_duration'}),flush=True)
        target.write_text(json.dumps(dict(status='Regression on previously inspected external test matches; not a fresh blind test',rows=rows),indent=2))
