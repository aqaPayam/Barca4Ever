"""A small time-varying mixture fitted to external possession-spell scores."""
import numpy as np


def combine_score(gate,tree,neural,f):
    entropy_difference=(f[:,0]-f[:,5])/np.log(3)
    missing_active=(1-f[:,15])*(1-np.maximum(tree[:,2],neural[:,2]))
    weight=np.clip(.5+gate['a']*entropy_difference+gate['b']*missing_active,.1,.9)
    return (1-weight[:,None])*tree+weight[:,None]*neural
