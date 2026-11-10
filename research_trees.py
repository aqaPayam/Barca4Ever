"""Research candidates: hierarchical classification and learned boundaries.

Inspired by the problem decomposition of Borghesi et al. 2023 and boundary-based
temporal segmentation (Ishikawa et al. 2021). Original code, not reproductions of
their architectures or reported results. External labels only.
"""
import json
from pathlib import Path
import numpy as np
from learned_possession import viterbi
from research_data import training_arrays,validation,get
from external_idsse import SPLITS
from develop_v2 import score
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'external_reports/research_v3'
MODELS=ROOT/'models/research_v3'


class Hierarchical:
    classes_=np.array([0,1,2])
    def __init__(self,dead,team):self.dead,self.team=dead,team
    def predict_proba(self,x):
        dead=self.dead.predict_proba(x)[:,1]
        team=self.team.predict_proba(x)
        return np.column_stack([(1-dead)*team[:,0],(1-dead)*team[:,1],dead])


def boundary_features(x):
    i=np.arange(len(x))
    # Geometry around a potential change, without class labels or fitted logits.
    return np.column_stack([x,abs(x-x[np.maximum(0,i-3)]),abs(x-x[np.minimum(len(x)-1,i+3)])])


def boundary_targets(y):
    raw=np.r_[False,y[1:]!=y[:-1]]
    target=raw.copy()
    for shift in [1,2,3]:
        target[shift:] |= raw[:-shift]
        target[:-shift] |= raw[shift:]
    return target.astype(np.int8)


def adaptive_decode(probability,boundary,base,scale):
    if not len(probability):return np.empty(0,dtype=np.int8)
    logp=np.log(np.clip(probability,1e-7,1))
    costs=base/(1+scale*np.clip(boundary,0,1))
    back=np.zeros((len(logp),3),dtype=np.int8)
    state=logp[0].copy()
    eye=np.eye(3)
    for i in range(1,len(logp)):
        matrix=state[:,None]-costs[i]*(1-eye)
        back[i]=matrix.argmax(axis=0)
        state=matrix.max(axis=0)+logp[i];state-=state.max()
    out=np.empty(len(logp),dtype=np.int8);out[-1]=state.argmax()
    for i in range(len(logp)-1,0,-1):out[i-1]=back[i,out[i]]
    return out


def evaluate_trials(name,probs,validation_data,bounds=None):
    trials=[]
    combinations=([(p,0) for p in [2,4,8,16,32]] if bounds is None else
                  [(p,s) for p in [8,16] for s in [1,4,16]])
    for penalty,scale in combinations:
        rows=[]
        for j,(h,noisy,d) in enumerate(validation_data):
            pred=(viterbi(probs[j],penalty) if bounds is None else
                  adaptive_decode(probs[j],bounds[j],penalty,scale))
            rows.append(dict(half=h+1,noisy=noisy,**score(d['y'],pred)))
        r=dict(penalty=penalty,boundary_scale=scale,
               spell_f1=float(np.mean([v['spell_f1'] for v in rows])),
               macro_f1=float(np.mean([v['macro_f1'] for v in rows])),rows=rows)
        trials.append(r)
        print(name+' '+json.dumps({k:v for k,v in r.items() if k!='rows'}),flush=True)
    best=max(trials,key=lambda r:(r['spell_f1'],r['macro_f1']))
    (REPORT/(name+'.json')).write_text(json.dumps(dict(status='Validation model selection, not blind evaluation',trials=trials,selected=best),indent=2))
    return best


def fit_boundary(matches):
    bx,by=[],[]
    for match in matches:
        print('Preparing boundary supervision '+match,flush=True)
        for h in [0,1]:
            for noisy in [False,True]:
                for swap in [False,True]:
                    d=get(match,h,noisy,swap)
                    take=np.arange(0,len(d['y']),5)
                    bx.append(boundary_features(d['x'])[take])
                    by.append(boundary_targets(d['y'])[take])
    bx,by=np.concatenate(bx),np.concatenate(by)
    model=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,learning_rate=.08,
        l2_regularization=3,min_samples_leaf=60,early_stopping=False,random_state=20261106)
    model.fit(bx,by,sample_weight=np.where(by,10.,1.))
    return model


if __name__=='__main__':
    REPORT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    x,y,w=training_arrays()
    print('Fitting separate dead/alive and team-control classifiers',flush=True)
    dead=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=.07,
        l2_regularization=3,min_samples_leaf=60,early_stopping=False,random_state=20261106)
    team=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=23,learning_rate=.07,
        l2_regularization=3,min_samples_leaf=60,early_stopping=False,random_state=20261106)
    dead.fit(x,y==2)
    alive=y<2;team.fit(x[alive],y[alive],sample_weight=w[alive])
    model=Hierarchical(dead,team)
    val=validation()
    best=evaluate_trials('hierarchical',[model.predict_proba(d['x']) for h,n,d in val],val)
    from research_trees import Hierarchical as StoredHierarchical
    joblib.dump(dict(model=StoredHierarchical(dead,team),feature_version=2,penalty=best['penalty']),MODELS/'hierarchical.joblib')
    del x,y,w
    bmodel=fit_boundary(SPLITS['train'])
    base=joblib.load(ROOT/'models/external_v2.joblib')
    probabilities=[base['model'].predict_proba(d['x']) for h,n,d in val]
    boundaries=[bmodel.predict_proba(boundary_features(d['x']))[:,1] for h,n,d in val]
    best=evaluate_trials('learned_boundary',probabilities,val,boundaries)
    joblib.dump(dict(model=base['model'],boundary_model=bmodel,feature_version=2,
                    penalty=best['penalty'],boundary_scale=best['boundary_scale'],kind='adaptive_boundary'),MODELS/'learned_boundary.joblib')
