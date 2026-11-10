"""Audited pure geometry: raw observation confidence and actual-time velocity."""
import hashlib
from pathlib import Path
import numpy as np
from external_idsse import load,SPLITS
from learned_possession import corrupt
from possession_v2 import interpolate_short_gaps
from paper2023_data import relative_features

ROOT=Path(__file__).resolve().parent


def object_features(players,teams,ball,pitch,target_fps=25,source_fps=25):
    if target_fps not in [16,25]:raise ValueError('Declared rates only')
    n=len(ball)
    sample=np.rint(np.arange(0,n*target_fps/source_fps)*source_fps/target_fps).astype(int)
    sample=sample[sample<n]
    raw_seen=np.isfinite(ball).all(1)
    filled=interpolate_short_gaps(ball,source_fps)
    xy=np.full((len(sample),36,2),np.nan,np.float32)
    xy[:,0]=filled[sample]
    roles=np.zeros((36,5),np.float32);roles[0,0]=1;roles[1,1]=1
    for team in [0,1]:
        ids=np.flatnonzero(teams==team)
        if len(ids)>17:raise ValueError('Use grouped adapter for extra objects')
        slots=2+17*team+np.arange(len(ids))
        xy[:,slots]=players[sample][:,ids]
        roles[2+17*team:2+17*(team+1),3+team]=1
    seen=np.isfinite(xy).all(2)
    velocity=np.zeros_like(xy)
    elapsed=np.diff(sample)/source_fps
    velocity[1:]=(xy[1:]-xy[:-1])/elapsed[:,None,None]
    velocity_seen=seen&np.r_[seen[:1],seen[:-1]]
    velocity=np.clip(np.nan_to_num(velocity,nan=0),-40,40)
    velocity[~velocity_seen]=-2
    scale=np.asarray(pitch,np.float32)/2
    outside=(np.abs(xy)>scale).any(2)&seen
    position=np.clip(xy/scale,-1,1);position[~seen]=-2
    flags=np.broadcast_to(roles,(len(sample),36,5))
    x=np.concatenate([position,velocity,flags,outside[...,None],(~seen)[...,None]],axis=2).astype(np.float32)
    x=relative_features(x)
    indices=np.arange(n)
    last=np.maximum.accumulate(np.where(raw_seen,indices,-10*source_fps))
    following=np.minimum.accumulate(np.where(raw_seen,indices,n+10*source_fps)[::-1])[::-1]
    confidence=np.column_stack([raw_seen[sample],np.minimum((sample-last[sample])/source_fps,10)/10,
        np.minimum((following[sample]-sample)/source_fps,10)/10]).astype(np.float32)
    confidence=np.broadcast_to(confidence[:,None,:],(len(sample),36,3))
    return np.concatenate([x,confidence],axis=2),sample


def get(match,h,noisy,rate,seed=None):
    assert match in sum(SPLITS.values(),[]),'External matches only'
    if seed is None:seed=8400+h if match=='J03WR9' else 7300+10*SPLITS['train'].index(match)+h
    folder=ROOT/'external_cache/hz25';folder.mkdir(parents=True,exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    target=folder/f'{match}_{h}_{rate}_{int(noisy)}_{seed}_{version}.npz'
    if not target.exists():
        half=load(match)[h];p,b=half['players'],half['ball']
        if noisy:p,b=corrupt(p,b,seed)
        x,samples=object_features(p,half['teams'],b,half['pitch'],rate)
        np.savez_compressed(target,x=x,y=half['labels'][samples],full_y=half['labels'],samples=samples)
    with np.load(target) as z:return {k:z[k] for k in z.files}
