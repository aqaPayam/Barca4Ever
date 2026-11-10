"""Limit a learned reliability combination's departure from frozen V4.1."""
import numpy as np
from scipy.ndimage import maximum_filter1d
from smart_fusion import combine


def blend(gate,variant,tree,neural,f,onset):
    base=.5*(tree+neural)
    learned=combine(gate,tree,neural,f)
    if variant=='shrink25':strength=.25
    elif variant=='shrink50':strength=.5
    else:
        tops_t=tree>=tree.max(1,keepdims=True)-1e-12
        tops_n=neural>=neural.max(1,keepdims=True)-1e-12
        strength=(~np.any(tops_t&tops_n,axis=1)).astype(float)
        if variant=='boundary':
            strength*=maximum_filter1d(onset.max(1),size=25,mode='nearest')
        elif variant!='disagree':raise ValueError('Unknown conservative gate variant')
        strength=strength[:,None]
    return (1-strength)*base+strength*learned
