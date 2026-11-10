"""External-only object trajectories for the Borghesi 2023 architecture.

Never imports games.py or opens competition data. Labels are returned separately.
"""
from pathlib import Path
import functools
import re
import xml.etree.ElementTree as ET
import numpy as np
from external_idsse import load, paths, SPLITS
from learned_possession import corrupt
from scipy.interpolate import Akima1DInterpolator

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'external_cache/paper2023'
FPS = 16
TARGET = 47
WINDOW = 64


@functools.lru_cache(maxsize=16)
def descriptors(match):
    """Recover slot order and goalkeeper metadata, without parsing frame labels."""
    meta_path, raw_path = paths(match)
    meta = ET.parse(meta_path)
    general = meta.find('.//General')
    team_ids = [general.get('HomeTeamId'), general.get('GuestTeamId')]
    keepers = {p.get('PersonId') for p in meta.iter('Player')
               if p.get('PlayingPosition') == 'TW'}
    sections = {'firstHalf': [], 'secondHalf': []}
    # Headers are much smaller than frame observations. Stream with a tail for
    # a header split across chunks; never expose IDs to the model features.
    with raw_path.open('rb') as stream:
        tail = b''
        while True:
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            block = tail + chunk
            limit = len(block) - 1024
            for m in re.finditer(rb'<FrameSet\s+[^>]*>', block):
                if m.start() >= limit:
                    continue
                e = ET.fromstring(m.group().decode('utf-8')[:-1] + '/>')
                attrs = e.attrib
                if attrs.get('TeamId') in team_ids:
                    sections[attrs['GameSection']].append((team_ids.index(attrs['TeamId']),
                                                         attrs.get('PersonId') in keepers))
            # Headers beginning in the tail are deferred, including complete
            # headers. A header starting before the tail has already been read.
            tail = block[max(0, limit):]
        for m in re.finditer(rb'<FrameSet\s+[^>]*>', tail):
            e = ET.fromstring(m.group().decode('utf-8')[:-1] + '/>')
            a = e.attrib
            if a.get('TeamId') in team_ids:
                sections[a['GameSection']].append((team_ids.index(a['TeamId']),
                                                  a.get('PersonId') in keepers))
    return [sections[k] for k in ['firstHalf', 'secondHalf']]


def object_features(players, teams, ball, pitch, keepers, source_fps=25,
                    interpolation='akima'):
    """Return [time, 36 objects, 11 channels], sampled at 16 Hz.

    Pure geometry transformation. No possession/status/event/clock labels used.
    """
    n = len(ball)
    sample = np.rint(np.arange(0, n * FPS / source_fps) * source_fps / FPS).astype(int)
    sample = sample[sample < n]
    p, b = players[sample], ball[sample].copy()
    raw_ball_seen = np.isfinite(b).all(1)
    good = np.flatnonzero(raw_ball_seen)
    if len(good) >= 3:
        between = np.arange(good[0], good[-1] + 1)
        if interpolation == 'akima':
            b[between] = Akima1DInterpolator(good, b[good], axis=0)(between)
        elif interpolation == 'short':
            for left, right in zip(good[:-1], good[1:]):
                if right-left <= FPS*2:
                    ix = np.arange(left, right+1)
                    b[ix] = b[left] + (b[right]-b[left])*(ix-left)[:, None]/(right-left)
        else:
            raise ValueError(interpolation)
    xy = np.full((len(sample), 36, 2), np.nan, dtype=np.float32)
    xy[:, 0] = b
    roles = np.zeros((36, 5), dtype=np.float32)
    roles[0, 0] = 1  # ball
    roles[1, 1] = 1  # referee: absent in IDSSE, explicitly missing
    for team in [0, 1]:
        ids = np.flatnonzero(teams == team)
        if len(ids) > 17:
            raise ValueError('More than 17 tracked external participants in a team')
        slots = 2 + 17 * team + np.arange(len(ids))
        xy[:, slots] = p[:, ids]
        roles[slots, 2] = np.asarray(keepers)[ids]
        roles[2+17*team:2+17*(team+1), 3+team] = 1
    seen = np.isfinite(xy).all(2)
    velocity = np.zeros_like(xy)
    velocity[1:] = (xy[1:] - xy[:-1]) * FPS
    velocity_seen = seen & np.r_[seen[:1], seen[:-1]]
    velocity = np.clip(np.nan_to_num(velocity, nan=0), -40, 40)
    velocity[~velocity_seen] = -2
    scale = np.asarray(pitch, dtype=np.float32) / 2
    outside = (np.abs(xy) > scale).any(2) & seen
    position = np.clip(xy / scale, -1, 1)
    position[~seen] = -2
    flags = np.broadcast_to(roles, (len(sample), 36, 5))
    x = np.concatenate([position, velocity, flags, outside[..., None],
                        (~seen)[..., None]], axis=2).astype(np.float32)
    return x, sample


def get(match, h, noisy=False, seed=None, interpolation='akima'):
    CACHE.mkdir(parents=True, exist_ok=True)
    if seed is None:
        seed = 8400+h if match == 'J03WR9' else 7300+10*SPLITS['train'].index(match)+h
    target = CACHE / f'{match}_{h}_{int(noisy)}_{seed}_{interpolation}_v1.npz'
    if not target.exists():
        half = load(match)[h]
        desc = descriptors(match)[h]
        if len(desc) != len(half['teams']):
            raise ValueError('External object descriptors do not match cached tracking')
        if [v[0] for v in desc] != half['teams'].tolist():
            raise ValueError('External object teams disagree with cached slot order')
        p, b = half['players'], half['ball']
        if noisy:
            p, b = corrupt(p, b, seed)
        x, samples = object_features(p, half['teams'], b, half['pitch'],
                                     [v[1] for v in desc], interpolation=interpolation)
        np.savez_compressed(target, x=x, y=half['labels'][samples], samples=samples,
                            full_y=half['labels'])
    with np.load(target) as z:
        return {k: z[k] for k in z.files}


def relative_features(x):
    """Our extension: ball-relative geometry/motion, not part of the paper."""
    valid=(x[:,:,10]==0)&(x[:,0:1,10]==0)
    delta=np.clip((x[:,:,:2]-x[:,0:1,:2])*np.array([5.25,3.4]),-10,10)
    motion=np.clip((x[:,:,2:4]-x[:,0:1,2:4])/10,-8,8)
    distance=np.linalg.norm(delta,axis=2,keepdims=True)
    extra=np.concatenate([delta,motion,distance,valid[:,:,None]],axis=2).astype(np.float32)
    extra[~valid]=0
    normalized=x.copy()
    normalized[:,:,2:4]/=np.array([52.5,34],dtype=np.float32)
    seen=x[:,:,10]==0
    velocity_seen=seen&np.r_[seen[:1],seen[:-1]]
    normalized[:,:,2:4][~velocity_seen]=-2
    return np.concatenate([normalized,extra],axis=2)


if __name__ == '__main__':
    for match in SPLITS['train'] + SPLITS['validation']:
        for h in [0, 1]:
            for noisy in [False, True]:
                d = get(match, h, noisy)
                print('External object features prepared', match, h+1, noisy,
                      d['x'].shape, flush=True)
