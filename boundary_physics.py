"""Padding-invariant soft touch physics and dense external boundary sampling."""
import numpy as np
import learned_possession  # project-local optional numerical dependencies
from scipy.ndimage import maximum_filter1d
from touch_physics import features as raw_physics,FEATURE_NAMES
from boundary_refinement import temporal_features,targets,PAIRS

DROP_NAMES={'home_player_observed_fraction','away_player_observed_fraction'}
PHYSICS_COLUMNS=np.array([i for i,name in enumerate(FEATURE_NAMES) if name not in DROP_NAMES])
PHYSICS_NAMES=tuple(FEATURE_NAMES[i] for i in PHYSICS_COLUMNS)
assert len(PHYSICS_COLUMNS)==28


def physics_features(players,teams,ball,pitch,fps=25):
    return raw_physics(players,teams,ball,pitch,fps)[:,PHYSICS_COLUMNS]


def combined_features(x,physics):
    if physics.shape!=(len(x),28):raise ValueError('Expected28 selected physics channels')
    return np.column_stack([temporal_features(x),physics]).astype(np.float32)


def dense_sampling(y):
    y=np.asarray(y,dtype=np.int64)
    if not len(y):return np.zeros(0,bool),np.zeros(0,float)
    change=np.r_[False,(y[1:]!=y[:-1])&(y[:-1]<3)&(y[1:]<3)]
    near=maximum_filter1d(change.astype(np.uint8),size=25,mode='constant')>0
    take=(y<3)&(near|(np.arange(len(y))%5==0))
    target=targets(y)
    weight=np.where(target>0,4.,np.where(near,.2,1.))
    return take,weight


def matrix_scores(model,x):
    raw=model.predict_proba(x)
    result=np.zeros((len(x),3,3),np.float32)
    for column,label in enumerate(model.classes_):
        if label:
            source,destination=PAIRS[int(label)-1]
            result[:,source,destination]=raw[:,column]
    return result
