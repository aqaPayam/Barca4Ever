# Soft touch-physics feature contract

`touch_physics.features(players, teams, ball, pitch, fps=25)` is a pure,
unfitted XY transformation. It returns finite float32 `[frames,30]`.
Coordinates are metres about pitch centre; teams are0(home)/1(away).
There is no interpolation, possession rule, hard event decision or label input.
Only new research files are introduced; no production file is changed.

## Windows and visibility

Half-windows are0.08/0.24/0.48seconds: exactly+/-2/6/12frames at25Hz.
Other frame rates round these physical windows to at least one frame.
Indices clamp at sequence edges and signed rates use their actual elapsed
time, not the nominal lag; both elapsed sides must be positive.

Let d_H(t),d_A(t) be nearest raw player-ball distances for each team. Gross
distance outliers are capped at twice the pitch diagonal. A geometry window
is valid only if the ball and at least one player from EACH team are observed
at EVERY frame inside it. All six geometry values are zero when invalid,
and a separate validity flag disambiguates that zero from physical evidence.
Consequently no missing-ball interval is silently bridged, even when the
window endpoints happen to be observed. Partial player visibility is reported
separately. Ball-motion validity requires only uninterrupted ball observations.

## Exact feature order and formulas

For each half-window in order80/240/480ms, output these seven values:

| Offset within block | Feature | Formula |
|---:|---|---|
|0|home valley depth|clip(min(d_H(left),d_H(right))-d_H(t),-20,20)/10|
|1|home signed approach|clip((d_H(left)-d_H(t))/dt_left,-50,50)/10|
|2|home signed recession|clip((d_H(right)-d_H(t))/dt_right,-50,50)/10|
|3..5|away equivalents|same formulas using d_A|
|6|joint geometry validity|1 when the complete geometry window is available|

Depth is positive for a local distance valley. Approach is positive while
closing; recession is positive while separating. They are continuous features,
not assertions that a touch happened. Blocks occupy columns0..20.

The remaining columns are:

| Index | Feature |
|---:|---|
|21|Ball acceleration magnitude, capped300m/s^2 and divided by30|
|22|Ball direction turn: (1-cos(v_left,v_right))/2, in[0,1]|
|23|Ball-motion window validity|
|24|Turn validity: motion valid and both speed magnitudes above0.1m/s|
|25|Raw ball observed flag|
|26|Fraction of home tracking slots observed at this frame|
|27|Fraction of away tracking slots observed at this frame|
|28|Seconds since the previous raw ball observation, capped10|
|29|Seconds until the next raw ball observation, capped10|

Motion uses the0.24-second half-window. v_left=(b(t)-b(left))/dt_left and
v_right=(b(right)-b(t))/dt_right, with each velocity component guarded to
[-100,100]m/s. Acceleration=(v_right-v_left)/((dt_left+dt_right)/2).
Constant acceleration is recovered exactly before clipping, including when
edge-clamped time steps differ. Undefined directions or invalid windows give
zero evidence and zero corresponding validity. Unavailable previous/next
observations give the capped10-second gap value; observed frames have zero gap.

All guards are fixed engineering scales, not fitted on any match. `pitch`
sets the distance guard; no fixed105x68-m geometry assumption is made.

## Symmetry, missingness and tests

`FEATURE_NAMES` exports all30 names. `TEAM_SWAP_INDEX` gives the exact column
permutation under H/A exchange: each seven-column block swaps its home/away
triplets and retains validity; columns26/27 exchange; all other tail columns
stay unchanged. Player-slot order has no semantic effect. Empty inputs,
all-missing inputs, missing teams and extreme-coordinate guards are tested.
Synthetic tests also cover known contact valleys, acceleration, direction
reversal, physical-frame-rate equivalence and internal-gap validity.

## Paper basis and scope

ELASTIC2026 uses player-ball distance valleys, ball-acceleration peaks and
pitch-boundary proximity to generate possible event times before event/tracking
alignment. Its full algorithm and3m/4m physical filters are not reproduced here.
This module has XY only, no height filter, and produces soft evidence rather
than events. Our implementation is original; no MPL source was copied.

Primary: https://arxiv.org/html/2608.30227v1
Official code/licensing: https://github.com/hyunsungkim-ds/elastic

Peral et al.2025 (Ferraz coauthor) separate possessor inference from touch-event
extraction. That distinction motivates treating these features as boundary
evidence rather than forcing a nearest-player possession assignment.

Primary: https://www.iri.upc.edu/files/scidoc/3004-Temporally-Accurate-Events-Detection-Through-Ball-Possessor-Recognition-in-Soccer.pdf

No competition data, video, labels or prediction contents are used. A later
candidate must be separately declared, fitted and compared using external
match-separated data before it can affect the active model.
