# Learned reliability fusion against frozen V4.1

Declare this experiment before observing its results. No competition data,
video, labels or outputs are used. Only the same seven licensed IDSSE matches
are used. V4.1 artifacts, source and predictions remain frozen.

## Fitting and honest comparison

WR9 changes role from validation/selection to combination-fitting data. Its
predictions come from base weights fitted only on WOH/WOY/WPY/WQQ. WR9 was
previously used for architecture/checkpoint selection, so this is reused
development data, not strict blind cross-fitting or a new validation set.
No score on WR9 is used to claim a generalization gain.

For each original whole-match fold, fit its combination on WR9 predictions
from THAT fold's existing base models. The neural initializer, adaptation,
trees and gate exclude the evaluated fold matches. The gate scaler also uses
only WR9. Thus a gate fitted using predictions from a model trained on an
evaluated match cannot leak across the fold. No new base-model fitting occurs.
The baseline uses the exact same probabilities, noise seeds and onset scores.

## Two declared candidates

1. joint: one per-frame learned neural weight for the three-state mixture.
2. factorized: separate learned weights for dead probability and conditional
   home/away probability, with the team weight fitted only on alive labels.

Weights are bounded to [0.1,0.9] by 0.1+0.8*sigmoid(z). Fit a linear logit on
standardized, clipped reliability features: model entropy/margins/dead scores,
agreement and team confidence, plus label-free geometry/visibility/ball-gap
features. Features are invariant to exchanging H/A. No identity, clock, label
or match-ID feature is allowed. Mean weighted log loss plus L2=0.02/2*||beta||^2,
including the intercept, is optimized by deterministic L-BFGS-B, max_iter=300.
Scale from fitting rows only, clip standardized inputs to [-6,6]. Fit every
fifth valid frame; use existing clipped inverse-square-root spell weights,
normalized to mean one. Initial weights are 0.5; no fitted model is pretrained.

Keep V4.1 decoding fixed at base16/scale8, all base weights fixed. Select among
candidates on whole-match development CV, requiring higher spell F1 and no
loss of short-spell hits in BOTH clean/noisy cases versus V4.1. Among eligible
candidates choose mean clean/noisy spell F1, macro F1 breaking ties. Evaluate
only that one selected candidate on WMX/WN1 reused regression. Promotion also
requires higher spell F1 and no short-hit loss in both regression scenarios.
If none qualifies or confirmation fails, retain V4.1. Do not select another
candidate based on regression. Small development gains are not statistical
significance, an independent benchmark or a competition ranking.

Record all trials, hashes, gate fitting matches, source-model fitting matches,
scaler parameters and optimizer status. Assert exact V4.1 reproduction in every
evaluated case. Production changes require synthetic invariance/gradient checks,
saved-model external CPU checks and full-length format checks before local
competition inference. No supplied data is sent to a hosted AI service.
