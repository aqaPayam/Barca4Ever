# Physics and dense boundary refinement: declared second candidate

The first directed classifier improved clean CV spellF1 and both short-hit
counts but narrowly reduced noisy spellF1 (.532851 to.532500), failing its
strict acceptance rule. Preserve its source, models and reports unchanged.

Before fitting this distinct candidate, declare: append the soft XY contact
features from touch_physics.py to the previous293 signed temporal inputs.
Drop home_player_observed_fraction and away_player_observed_fraction by their
FEATURE_NAMES. These depend on roster/track-slot count and change under extra
missing padding. The old59 inputs already contain absolute visible counts.
The remaining28 physics channels give321 total inputs and must be invariant
to extra missing player slots. No competition values inform this choice.

Physics adds raw player-ball distance valleys, signed approach/recession,
ball vector acceleration and direction turns, whole-window visibility flags,
and raw-ball gap ages. It does not bridge missing-ball gaps or use height,
event lists, labels or persistent identity features. Its source is unchanged.
See TOUCH_PHYSICS_PROTOCOL.md for paper basis and exact formulas.

Keep the same six ordered-transition targets, radius2, nearest-change ties,
100 iterations,11 leaves,rate.08,L2=5,minleaf80,seed20261116 and16/8 decoding.
Training changes to dense boundary-neighborhood importance sampling: include
all valid frames within +/-12 of an external change; elsewhere every fifth
valid frame. Positive transition frames weight4, near-background weight.2,
remote sampled background weight1. This approximately preserves the previous
objective mass (positive20/background1 with one-in-five sampling) while
retaining very short changes and their nearby negatives. Unknown labels are
excluded; clean/noisy and H/A swap augmentation remain unchanged.

Select alpha0/.25/.5/1 only on WR9, mean organizer spellF1 then macroF1. Alpha0
must exactly reproduce V4.1. A validation replacement must improve spellF1 and
avoid short-hit loss. Freeze alpha before original whole-match folds; refit
boundary models only on their own two fitting matches. CV and then single-
candidate WMX/WN1 reused regression both require higher clean/noisy spellF1 and
no short-hit loss. No evaluated match enters fitting. All7 games are reused
development data, not fresh blind tests or a professional-system ranking.
No provided competition data, video, labels or predictions are accessed.
