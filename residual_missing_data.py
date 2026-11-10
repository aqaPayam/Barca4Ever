"""Legacy17 geometry unchanged, plus zero-on-observed missing-ball inputs."""
import hashlib
from pathlib import Path
import numpy as np
from external_idsse import load,SPLITS
from learned_possession import corrupt
from paper2023_data import get as legacy_get,relative_features,CACHE as LEGACY_CACHE

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_missing(base,raw_ball,samples,source_fps=25):
    if base.ndim!=3 or base.shape[2]!=17 or len(samples)!=len(base):
        raise ValueError('Expected time-aligned17-channel legacy input')
    seen=np.isfinite(raw_ball).all(1);indices=np.arange(len(raw_ball))
    last=np.maximum.accumulate(np.where(seen,indices,-10*source_fps))
    following=np.minimum.accumulate(np.where(seen,indices,len(raw_ball)+10*source_fps)[::-1])[::-1]
    extra=np.column_stack([~seen[samples],np.minimum((samples-last[samples])/source_fps,10)/10,
        np.minimum((following[samples]-samples)/source_fps,10)/10]).astype(np.float32)
    extra=np.broadcast_to(extra[:,None,:],(len(samples),base.shape[1],3))
    return np.concatenate([base,extra],axis=2)


def get(match,h,noisy,seed=None):
    if match not in sum(SPLITS.values(),[]):raise ValueError('External matches only')
    if seed is None:seed=8400+h if match=='J03WR9' else 7300+10*SPLITS['train'].index(match)+h
    legacy_path=LEGACY_CACHE/f'{match}_{h}_{int(noisy)}_{seed}_short_v1.npz'
    source_path=ROOT/f'external_cache/{match}_{h+1}.npz'
    if not legacy_path.exists() or not all((ROOT/f'external_cache/{match}_{i}.npz').exists() for i in [1,2]):
        raise FileNotFoundError('Prepare and verify original external caches before this experiment')
    source_version=hashlib.sha256(''.join(digest(ROOT/name) for name in ['residual_missing_data.py','paper2023_data.py','learned_possession.py']).encode()).hexdigest()[:12]
    fingerprint=hashlib.sha256((digest(legacy_path)+digest(source_path)+source_version).encode()).hexdigest()
    folder=ROOT/'external_cache/residual_missing';folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'{match}_{h}_{int(noisy)}_{seed}_{fingerprint[:16]}.npz'
    if not target.exists():
        legacy=legacy_get(match,h,noisy,seed=seed,interpolation='short')
        half=load(match)[h]
        assert np.array_equal(legacy['full_y'],half['labels'])
        raw_ball=half['ball']
        if noisy:
            discarded,raw_ball=corrupt(half['players'],raw_ball,seed)
            del discarded
        raw=legacy['x'].copy();raw[:,:,6]=0
        base=relative_features(raw)
        x=append_missing(base,raw_ball,legacy['samples'])
        assert x[:,:,:17].tobytes()==base.tobytes()
        np.savez_compressed(target,x=x,y=legacy['y'],full_y=legacy['full_y'],samples=legacy['samples'],fingerprint=fingerprint)
    with np.load(target) as z:
        assert str(z['fingerprint'])==fingerprint
        return {key:z[key] for key in ['x','y','full_y','samples']}
