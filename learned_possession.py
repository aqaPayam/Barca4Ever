"""Geometry-only features and externally trained possession sequence model.

No labels, event outcomes, clock times, player names, or match IDs are features.
Features use offline temporal context and must be computed separately per half.
"""
from pathlib import Path
import sys
import numpy as np

# Project-local optional dependencies installed without altering the bundled runtime.
LOCAL_DEPS = Path(__file__).resolve().parent / ".model_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

LABELS = np.array(["H", "A", "D"])
FEATURE_VERSION = 1


def smooth_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    window = max(1, int(window))
    pad = window // 2
    padded = np.pad(values, ((pad, window - 1 - pad), (0, 0)), mode="edge")
    sums = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(padded, axis=0)])
    return ((sums[window:] - sums[:-window]) / window).astype(np.float32)


def features(players, teams, ball, pitch, fps=25):
    """Coordinates in metres with origin at pitch centre; NaN means absent."""
    n = len(ball)
    if n == 0:
        return np.empty((0, 42), dtype=np.float32)
    if players.shape[0] != n or players.shape[-1] != 2 or ball.shape != (n, 2):
        raise ValueError("Invalid tracking dimensions")
    ball_seen = np.isfinite(ball).all(axis=1)
    player_seen = np.isfinite(players).all(axis=2)
    delta = players - ball[:, None, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    distance[~np.isfinite(distance)] = 100
    nearest = []
    for team in [0, 1]:
        d = np.sort(distance[:, teams == team], axis=1)
        if d.shape[1] < 3:
            d = np.pad(d, ((0, 0), (0, 3 - d.shape[1])), constant_values=100)
        nearest.append(d[:, :3])
    near = np.concatenate(nearest, axis=1)
    team_counts = np.column_stack([player_seen[:, teams == t].sum(axis=1) for t in [0, 1]])
    # Observed speeds over 0.2s, from geometry only (provider S fields excluded).
    lag = max(1, round(0.2 * fps))
    past = np.maximum(0, np.arange(n) - lag)
    p_speed = np.linalg.norm(players - players[past], axis=2) / (lag / fps)
    p_speed[~np.isfinite(p_speed)] = 0
    speed_sum = p_speed.sum(axis=1) / np.maximum(1, player_seen.sum(axis=1))
    pos = np.nan_to_num(ball, nan=0.0)
    edge = np.column_stack([pitch[0] / 2 - abs(pos[:, 0]), pitch[1] / 2 - abs(pos[:, 1])])
    edge[~ball_seen] = 0
    base = np.column_stack([near, near[:, 0] - near[:, 3], ball_seen, team_counts,
                            edge, speed_sum]).astype(np.float32)  # 13
    blocks = [base]
    # Offline context: ball motion over both directions; no filling across halves.
    for seconds in [0.2, 1.0, 2.0]:
        lag = max(1, round(seconds * fps))
        for sign in [-1, 1]:
            other = np.clip(np.arange(n) + sign * lag, 0, n - 1)
            valid = ball_seen & ball_seen[other]
            speed = np.linalg.norm(pos - pos[other], axis=1) / seconds
            speed[~valid] = 0
            blocks.append(np.column_stack([speed, valid]))  # 12
    for seconds in [0.4, 1.0, 2.0]:
        blocks.append(smooth_mean(base[:, [0, 3, 6, 7, 12]], round(seconds * fps)))  # 15
    # Time since/until last ball observation, capped; informative during gaps.
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(ball_seen, idx, -n))
    following = np.minimum.accumulate(np.where(ball_seen, idx, 2 * n)[::-1])[::-1]
    blocks.append(np.column_stack([np.minimum((idx - last) / fps, 10),
                                  np.minimum((following - idx) / fps, 10)]))  # 2
    # Total 42 features. Sanitize and cap to protect against tracking outliers.
    return np.clip(np.nan_to_num(np.column_stack(blocks), nan=0, posinf=100, neginf=-100),
                   -100, 100).astype(np.float32)


def corrupt(players, ball, seed):
    """Label-independent simulated tracking noise; no match-specific fitting.

    Synthetic missingness is not a claim about the challenge's error process.
    """
    rng = np.random.default_rng(seed)
    players = players.copy() + rng.normal(0, 0.25, players.shape).astype(np.float32)
    ball = ball.copy() + rng.normal(0, 0.35, ball.shape).astype(np.float32)
    players[rng.random(players.shape[:2]) < 0.05] = np.nan
    mask = np.zeros(len(ball), dtype=bool)
    # Random contiguous gaps, roughly 35-45% coverage depending on overlaps.
    for start in rng.integers(0, max(1, len(ball)), size=max(1, len(ball) // 90)):
        mask[start:start + int(rng.integers(15, 85))] = True
    ball[mask] = np.nan
    return players, ball


def viterbi(probabilities, penalty):
    """Symmetric switching cost, learned emissions. Penalty tuned externally."""
    if len(probabilities) == 0:
        return np.array([], dtype=np.int8)
    logp = np.log(np.clip(probabilities, 1e-7, 1))
    back = np.zeros((len(logp), 3), dtype=np.int8)
    score = logp[0].copy()
    transition = np.full((3, 3), -float(penalty))
    np.fill_diagonal(transition, 0)
    for i in range(1, len(logp)):
        candidates = score[:, None] + transition
        back[i] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + logp[i]
        score -= score.max()
    result = np.empty(len(logp), dtype=np.int8)
    result[-1] = score.argmax()
    for i in range(len(logp) - 1, 0, -1):
        result[i - 1] = back[i, result[i]]
    return result


def predict(model, x, penalty):
    probability = model.predict_proba(x)
    ordered = np.zeros((len(x), 3))
    ordered[:, model.classes_.astype(int)] = probability
    return viterbi(ordered, penalty)
