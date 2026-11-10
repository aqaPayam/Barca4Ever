"""Destination-specific sequence decoding; pure NumPy, no fitting or data access."""
import numpy as np


def decode(probability,onset,base,scale):
    if not len(probability):return np.empty(0,np.int8)
    logp=np.log(np.clip(probability,1e-7,1))
    cost=base/(1+scale*onset)
    back=np.zeros((len(logp),3),np.int8);state=logp[0].copy();eye=np.eye(3)
    for i in range(1,len(logp)):
        matrix=state[:,None]-cost[i][None,:]*(1-eye)
        back[i]=matrix.argmax(0);state=matrix.max(0)+logp[i];state-=state.max()
    out=np.empty(len(logp),np.int8);out[-1]=state.argmax()
    for i in range(len(logp)-1,0,-1):out[i-1]=back[i,out[i]]
    return out
