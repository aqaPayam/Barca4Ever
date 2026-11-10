"""Production boundary-aware sequence decoding; no training or data access."""
import numpy as np


def boundary_features(x):
    i=np.arange(len(x))
    return np.column_stack([x,abs(x-x[np.maximum(0,i-3)]),
                            abs(x-x[np.minimum(len(x)-1,i+3)])])


def adaptive_decode(probability,boundary,base,scale):
    if not len(probability):return np.empty(0,dtype=np.int8)
    logp=np.log(np.clip(probability,1e-7,1))
    costs=base/(1+scale*np.clip(boundary,0,1))
    back=np.zeros((len(logp),3),dtype=np.int8)
    state=logp[0].copy();eye=np.eye(3)
    for i in range(1,len(logp)):
        matrix=state[:,None]-costs[i]*(1-eye)
        back[i]=matrix.argmax(axis=0)
        state=matrix.max(axis=0)+logp[i];state-=state.max()
    out=np.empty(len(logp),dtype=np.int8);out[-1]=state.argmax()
    for i in range(len(logp)-1,0,-1):out[i-1]=back[i,out[i]]
    return out
