"""Fixed external-only feature cache for controlled architecture comparisons."""
from pathlib import Path
import numpy as np
from external_idsse import load, SPLITS
from possession_v2 import features, spell_weights
from learned_possession import corrupt

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'external_cache'/'research_v3'


def get(match,h,noisy,swap=False,split='train'):
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/'{}_{}_{}_{}_{}.npz'.format(match,h,int(noisy),int(swap),split)
    if not path.exists():
        half=load(match)[h]
        p,b=half['players'],half['ball']
        if split=='train': seed=7300+10*SPLITS['train'].index(match)+h
        elif split=='validation':seed=8400+h
        else:seed=9500+h
        if noisy:p,b=corrupt(p,b,seed)
        y=half['labels']
        teams=1-half['teams'] if swap else half['teams']
        x=features(p,teams,b,half['pitch'])
        w=spell_weights(y)
        if swap:y=np.where(y<2,1-y,y)
        np.savez_compressed(path,x=x,y=y,w=w)
    with np.load(path) as z:return {k:z[k] for k in z.files}


def training_arrays():
    xs,ys,ws=[],[],[]
    for match in SPLITS['train']:
        print('External feature cache '+match,flush=True)
        for h in [0,1]:
            for noisy in [False,True]:
                for swap in [False,True]:
                    d=get(match,h,noisy,swap)
                    ix=(np.arange(len(d['y']))%5==0)&(d['y']<3)
                    xs.append(d['x'][ix]);ys.append(d['y'][ix]);ws.append(d['w'][ix])
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)


def validation():
    return [(h,noisy,get('J03WR9',h,noisy,split='validation'))
            for h in [0,1] for noisy in [False,True]]
