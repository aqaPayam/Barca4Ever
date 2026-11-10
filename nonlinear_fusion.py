"""Conservative nonlinear prediction of relative expert reliability."""
import numpy as np
import learned_possession
from sklearn.ensemble import HistGradientBoostingRegressor
from possession_v2 import spell_weights


def fit_regret(calibration):
    xx,yy,ww=[],[],[]
    for d in calibration:
        take=(np.arange(len(d['y']))%5==0)&(d['y']<3)
        rows=np.flatnonzero(take);labels=d['y'][take]
        t=d['tree'][rows,labels];n=d['neural'][rows,labels]
        xx.append(d['f'][take]);yy.append((n-t)/(n+t+.1));ww.append(spell_weights(d['y'])[take])
    x,y,w=map(np.concatenate,[xx,yy,ww]);w=w/w.mean()
    model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=400,
        learning_rate=.05,l2_regularization=10,early_stopping=False,random_state=20261115)
    model.fit(x,y,sample_weight=w)
    return model


def combine_regret(model,amplitude,tree,neural,f):
    w=.5+amplitude*np.clip(model.predict(f),-1,1)
    return (1-w[:,None])*tree+w[:,None]*neural
