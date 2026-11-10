"""Short-spell-aware external model features. Never read match labels here."""
import numpy as np
from learned_possession import features as original_features

FEATURE_VERSION = 2


def interpolate_short_gaps(ball, fps=25, max_gap_s=2):
    result = ball.copy()
    seen = np.isfinite(ball).all(axis=1)
    observed = np.flatnonzero(seen)
    for left, right in zip(observed[:-1], observed[1:]):
        if 1 < right - left <= round(max_gap_s * fps) + 1:
            alpha = np.arange(1, right - left)[:, None] / (right - left)
            result[left + 1:right] = ball[left] + alpha * (ball[right] - ball[left])
    return result


def features(players, teams, ball, pitch, fps=25):
    base = original_features(players, teams, ball, pitch, fps)
    n = len(ball)
    if not n:
        return np.empty((0, 59), dtype=np.float32)
    filled = interpolate_short_gaps(ball, fps)
    valid = np.isfinite(filled).all(axis=1)
    distances = np.linalg.norm(players - filled[:, None, :], axis=2)
    distances[~np.isfinite(distances)] = 100
    nearest = []
    relative_speed, alignment = [], []
    lag = max(1, round(0.2 * fps))
    past = np.maximum(0, np.arange(n) - lag)
    ball_velocity = (filled - filled[past]) / (lag / fps)
    player_velocity = (players - players[past]) / (lag / fps)
    for team in [0, 1]:
        slots = np.flatnonzero(teams == team)
        d = distances[:, slots]
        ordered = np.sort(d, axis=1)
        if len(slots) < 3:
            ordered = np.pad(ordered, ((0, 0), (0, 3 - len(slots))), constant_values=100)
        nearest.append(ordered[:, :3])
        if len(slots):
            closest = slots[d.argmin(axis=1)]
            velocity = player_velocity[np.arange(n), closest]
            relative_speed.append(np.linalg.norm(velocity - ball_velocity, axis=1))
            dot = np.sum(velocity * ball_velocity, axis=1)
            denom = np.linalg.norm(velocity, axis=1) * np.linalg.norm(ball_velocity, axis=1)
            alignment.append(np.divide(dot, denom, out=np.zeros(n), where=denom > 0.01))
        else:
            relative_speed.append(np.full(n, 100.))
            alignment.append(np.zeros(n))
    near = np.column_stack(nearest)
    extra = [near, (near[:, 0] - near[:, 3])[:, None],
             np.column_stack(relative_speed), np.column_stack(alignment)]
    # Unsmeared near-contact context on either side of a possible brief possession.
    for sign in [-1, 1]:
        other = np.clip(np.arange(n) + sign * max(1, round(0.12 * fps)), 0, n - 1)
        extra.append(near[other][:, [0, 3]])
    extra += [valid[:, None], np.linalg.norm(ball_velocity, axis=1)[:, None]]
    return np.column_stack([base, np.clip(np.nan_to_num(np.column_stack(extra),
                                  nan=100, posinf=100, neginf=-100), -100, 100)]).astype(np.float32)


def spell_weights(labels):
    """External training loss weights only; never call with competition labels."""
    edges = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    weights = np.ones(len(labels), dtype=np.float32)
    for start, end in zip(edges[:-1], edges[1:]):
        if labels[start] < 2:
            weights[start:end] = np.clip(np.sqrt(250 / (end - start)), 0.5, 4)
    return weights
