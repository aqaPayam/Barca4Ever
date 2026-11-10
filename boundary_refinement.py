"""Directed transition evidence and signed multiscale 25-Hz geometry."""
import numpy as np

PAIRS=((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))
SELECT=np.array([0,3,6,7,8,9,10,11,12,13,15,17,19,21,23,40,41,42,45,48,49,50,51,52,57,58])


def temporal_features(x):
    if x.ndim!=2 or x.shape[1]!=59:raise ValueError('Expected59 tracking features')
    n=len(x);index=np.arange(n);parts=[x]
    for shift in [2,6,12]:
        left=x[np.maximum(0,index-shift)][:,SELECT]
        right=x[np.minimum(n-1,index+shift)][:,SELECT]
        parts.extend([left,right,right-left])
    return np.column_stack(parts).astype(np.float32)


def targets(y,radius=2):
    y=np.asarray(y,dtype=np.int64)
    target=np.zeros(len(y),np.int8)
    changes=np.flatnonzero((y[1:]!=y[:-1])&(y[:-1]<3)&(y[1:]<3))+1
    if not len(changes):return target
    index=np.arange(len(y));at=np.searchsorted(changes,index)
    before=changes[np.maximum(0,at-1)];after=changes[np.minimum(len(changes)-1,at)]
    nearest=np.where(abs(index-before)<=abs(index-after),before,after)
    near=(abs(index-nearest)<=radius)&(y<3)
    pair_codes=np.zeros((3,3),np.int8)
    for label,(source,destination) in enumerate(PAIRS,1):pair_codes[source,destination]=label
    target[near]=pair_codes[y[nearest[near]-1],y[nearest[near]]]
    return target


def matrix_scores(model,x):
    raw=model.predict_proba(temporal_features(x))
    scores=np.zeros((len(x),3,3),np.float32)
    for column,label in enumerate(model.classes_):
        if label:
            source,destination=PAIRS[int(label)-1]
            scores[:,source,destination]=raw[:,column]
    return scores


def blend_scores(destination,pair,alpha):
    if not 0<=alpha<=1:raise ValueError('Blend alpha outside0..1')
    return (1-alpha)*destination[:,None,:]+alpha*pair


def decode(probability,scores,base=16,scale=8):
    if not len(probability):return np.empty(0,np.int8)
    if scores.shape!=(len(probability),3,3):raise ValueError('Expected Nx3x3 transition evidence')
    logp=np.log(np.clip(probability,1e-7,1))
    costs=base/(1+scale*scores)
    state=logp[0].copy();back=np.zeros((len(probability),3),np.int8);eye=np.eye(3)
    for i in range(1,len(probability)):
        matrix=state[:,None]-costs[i]*(1-eye)
        back[i]=matrix.argmax(0);state=matrix.max(0)+logp[i];state-=state.max()
    result=np.empty(len(probability),np.int8);result[-1]=state.argmax()
    for i in range(len(probability)-1,0,-1):result[i-1]=back[i,result[i]]
    return result
