"""Soft XY contact physics; no labels, interpolation, fitting or game access.

Original implementation informed by ELASTIC's contact-candidate construction
and Peral et al.'s touch/possessor decomposition. Formulas and feature order:
TOUCH_PHYSICS_PROTOCOL.md. Inputs are metres about the pitch centre.
"""
import numpy as np


WINDOW_SECONDS = (2 / 25, 6 / 25, 12 / 25)
FEATURE_NAMES = tuple(
    name
    for ms in (80, 240, 480)
    for name in (
        f"home_valley_{ms}ms", f"home_approach_{ms}ms", f"home_recession_{ms}ms",
        f"away_valley_{ms}ms", f"away_approach_{ms}ms", f"away_recession_{ms}ms",
        f"joint_geometry_valid_{ms}ms",
    )
) + (
    "ball_acceleration_240ms", "ball_turn_240ms", "ball_motion_valid_240ms",
    "ball_turn_valid_240ms", "raw_ball_observed", "home_player_observed_fraction",
    "away_player_observed_fraction", "seconds_since_raw_ball", "seconds_until_raw_ball",
)
TEAM_SWAP_INDEX = np.array([
    3, 4, 5, 0, 1, 2, 6,
    10, 11, 12, 7, 8, 9, 13,
    17, 18, 19, 14, 15, 16, 20,
    21, 22, 23, 24, 25, 27, 26, 28, 29,
], dtype=np.int64)
TEAM_SWAP_INDEX.setflags(write=False)


def _indices(n, seconds, fps):
    index = np.arange(n)
    lag = max(1, int(round(seconds * fps)))
    left, right = np.maximum(0, index-lag), np.minimum(n-1, index+lag)
    return left, right, (index-left)/fps, (right-index)/fps


def _whole_window_valid(observed, left, right):
    missing = np.r_[0, np.cumsum(~observed, dtype=np.int64)]
    return missing[right+1] == missing[left]


def features(players, teams, ball, pitch, fps=25):
    """Return finite float32 [frames,30] soft contact/visibility features.

    players: [frames,slots,2], teams: [slots] encoded home0/away1,
    ball: [frames,2], pitch: positive [length,width] in metres, fps: positive.
    Any nonfinite coordinate is missing. Raw gaps are never interpolated.
    Geometry windows require uninterrupted ball observations AND at least one
    observed participant from each team at every frame in the window.
    """
    players, teams, ball = np.asarray(players), np.asarray(teams), np.asarray(ball)
    pitch = np.asarray(pitch, dtype=np.float64)
    if players.ndim != 3 or players.shape[2] != 2:
        raise ValueError("players must have shape [frames,slots,2]")
    n, count, _ = players.shape
    if teams.shape != (count,) or not np.isin(teams, [0, 1]).all():
        raise ValueError("teams must encode every slot as home0 or away1")
    if ball.shape != (n, 2):
        raise ValueError("ball must have shape [frames,2]")
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("pitch must contain positive finite metre dimensions")
    if not np.isscalar(fps) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    b = ball.astype(np.float64, copy=False)
    ball_seen = np.isfinite(b).all(1)
    nearest = np.full((n, 2), np.nan, dtype=np.float64)
    visible_fraction = np.zeros((n, 2), dtype=np.float64)
    # Bounded slot batches avoid a full frames*slots*2 float64 temporary.
    max_distance = 2 * np.hypot(*pitch)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for team in [0, 1]:
            slots = np.flatnonzero(teams == team)
            best = np.full(n, np.inf)
            observed_count = np.zeros(n, dtype=np.int64)
            for start in range(0, len(slots), 16):
                pp = players[:, slots[start:start+16]].astype(np.float64)
                seen = np.isfinite(pp).all(2)
                observed_count += seen.sum(1)
                delta = pp-b[:, None, :]
                distance = np.minimum(np.hypot(delta[..., 0], delta[..., 1]), max_distance)
                distance = np.where(seen & ball_seen[:, None], distance, np.inf)
                best = np.minimum(best, distance.min(1))
            valid = np.isfinite(best)
            nearest[valid, team] = best[valid]
            if len(slots):
                visible_fraction[:, team] = observed_count/len(slots)

        geometry_seen = ball_seen & np.isfinite(nearest).all(1)
        columns = []
        for seconds in WINDOW_SECONDS:
            left, right, dt_left, dt_right = _indices(n, seconds, fps)
            valid = _whole_window_valid(geometry_seen, left, right) & (dt_left > 0) & (dt_right > 0)
            for team in [0, 1]:
                distance = nearest[:, team]
                depth = np.minimum(distance[left], distance[right])-distance
                approach = (distance[left]-distance)/np.maximum(dt_left, 1e-12)
                recession = (distance[right]-distance)/np.maximum(dt_right, 1e-12)
                columns.extend([
                    np.where(valid, np.clip(depth, -20, 20)/10, 0),
                    np.where(valid, np.clip(approach, -50, 50)/10, 0),
                    np.where(valid, np.clip(recession, -50, 50)/10, 0),
                ])
            columns.append(valid.astype(float))

        left, right, dt_left, dt_right = _indices(n, WINDOW_SECONDS[1], fps)
        motion_valid = _whole_window_valid(ball_seen, left, right) & (dt_left > 0) & (dt_right > 0)
        v_left = np.clip((b-b[left])/np.maximum(dt_left[:, None], 1e-12), -100, 100)
        v_right = np.clip((b[right]-b)/np.maximum(dt_right[:, None], 1e-12), -100, 100)
        acceleration = (v_right-v_left)/np.maximum(((dt_left+dt_right)/2)[:, None], 1e-12)
        acceleration_norm = np.hypot(acceleration[:, 0], acceleration[:, 1])
        speed_left = np.hypot(v_left[:, 0], v_left[:, 1])
        speed_right = np.hypot(v_right[:, 0], v_right[:, 1])
        turn_valid = motion_valid & (speed_left > .1) & (speed_right > .1)
        cosine = np.sum(v_left*v_right, axis=1)/np.maximum(speed_left*speed_right, 1e-12)
        turn = (1-np.clip(cosine, -1, 1))/2
        columns.extend([
            np.where(motion_valid, np.minimum(acceleration_norm, 300)/30, 0),
            np.where(turn_valid, turn, 0),
            motion_valid.astype(float), turn_valid.astype(float),
        ])

    index = np.arange(n)
    previous = np.maximum.accumulate(np.where(ball_seen, index, -1))
    following = np.minimum.accumulate(np.where(ball_seen, index, n)[::-1])[::-1]
    since = np.where(previous >= 0, np.minimum((index-previous)/fps, 10), 10)
    until = np.where(following < n, np.minimum((following-index)/fps, 10), 10)
    columns.extend([ball_seen.astype(float), visible_fraction[:, 0], visible_fraction[:, 1], since, until])
    result = np.column_stack(columns).astype(np.float32)
    if result.shape != (n, len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("Nonfinite touch physics result")
    return result
