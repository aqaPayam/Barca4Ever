"""External-label training reference sampling; never used at inference."""
import numpy as np


def spell_refs(sequences,count,rng):
    spells=[]
    for sequence,d in enumerate(sequences):
        y=d['y']
        edges=np.flatnonzero(np.r_[True,y[1:]!=y[:-1],True])
        for start,end in zip(edges[:-1],edges[1:]):
            left,right=max(start,47),min(end,len(y)-16)
            if y[start]<2 and left<right:
                spells.append((sequence,left,right))
    if not spells:raise ValueError('No eligible externally labelled team spells')
    chosen=np.asarray(spells,dtype=np.int64)[rng.integers(len(spells),size=count)]
    target=rng.integers(chosen[:,1],chosen[:,2])
    return np.column_stack([chosen[:,0],target])
