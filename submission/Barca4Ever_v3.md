# Barca4Ever - Initial Challenge technical report (v3)

## Approach

We infer frame-level home possession (H), away possession (A), and dead ball (D)
from tracking geometry with two histogram gradient boosting classifiers and offline
boundary-aware temporal decoding. Unknown frames may be labelled X. Model training and parameter
selection used external matches only. The competition match is inference-only.

The supplied tactical ATD XML maps track slots to teams. The tactical ATD TXT
provides normalized player and ball XY observations. We convert them to metres
relative to the pitch centre. ATD record 1 maps to submission frame 0; the loader
rejects missing, duplicate, reordered or malformed records and inconsistent slot
counts. Team assignments come from metadata, not player appearance or names.

We use 59 features. The original 42 include three nearest player-ball distances per team, their nearest
distance difference, ball visibility, observed track counts per team, distances
to the pitch boundaries, mean observed player speed, ball motion at temporal
scales of 0.2, 1 and 2 seconds in both directions, contextual feature averages
over 0.4, 1 and 2 seconds, and time to/from observed ball positions. The use of
future observations is intentional: this is offline full-match inference.

Seventeen additional features describe nearest-team distances after short-gap
ball interpolation, relative player-ball speed, velocity alignment, and contact
context at 0.12 seconds on both sides. Interpolation fills only internal ball gaps
at most two seconds and never extrapolates beyond observed endpoints. Original
visibility and missingness features are retained. No labels enter interpolation.

The classifier has 200 boosting iterations, 23 leaves maximum per tree, learning
rate 0.07, L2 regularization 3, and minimum leaf size 60. It produces H/A/D
probabilities. A separate boundary classifier predicts transition evidence from
177 features: the 59 inputs and absolute differences from three frames before
and after. It has 140 boosting iterations, 15 leaves, learning rate 0.08, L2 3,
and minimum leaf size 60. External label changes expanded by three frames are
its targets, with positive weight 10. Its output b is an uncalibrated score.

Viterbi decoding maximizes summed log class probabilities minus a cost for
each switch, C_t = 16 / (1 + 4 b_t). This discourages fragmentation while
permitting quicker changes near predicted transitions. Base 16 and scale 4 were
selected on a separate external validation match using possession-spell F1.
No competition labels, video, clock times, event outcomes,
match IDs or player names enter fitting or parameter selection.

## Assets and missing/noisy observations

Competition inputs used: tactical automatic tracking and its metadata only.
Broadcast video/tracking, tactical video, Smart Tagging and homography are not
used. We do not compare predictions against the supplied video.

Missing positions remain missing; an absent ball is not automatically classified
as dead. Observation indicators, temporal context and sequence decoding support
inference through missing frames. Extremely large feature values are clipped to
bound the effect of outliers. Complete player-tracking outages of at least ten
seconds separate processing blocks; those gaps are X rather than assumed D.
This reset threshold is a fixed engineering assumption, not a parameter fitted
on the competition match. No manually specified match-specific frame intervals
are used. Track slots are not assumed to identify a player throughout the game.

## External training data and label provenance

The only labelled dataset used is IDSSE, originally supplied by DFL / Sportec
Solutions and published by Manuel Bassek, Robert Rein, Hendrik Weber and Daniel
Memmert, Scientific Data 12, 195 (2025):
https://doi.org/10.1038/s41597-025-04505-y.
Dataset: https://doi.org/10.6084/m9.figshare.28196177.
Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.

We use the source's published per-frame BallPossession and BallStatus fields.
Active/home becomes H, active/away becomes A, inactive becomes D, and missing
labels become X. These are DFL/Sportec provider labels, with tracking quality
control described in the paper; our team did not hand-label these matches.
No human-produced data about the competition match is used.

Training matches: J03WOH, J03WOY, J03WPY, J03WQQ. Validation: J03WR9.
Original v1 held-out tests: J03WMX and J03WN1. V2 and V3 reuse these for regression,
so their later results are not fresh blind tests. Both halves of a match always
belong to the same split. Every fifth labelled training frame is retained.
Clean/noisy and team-swapped copies produce 454,656 training examples; those
augmented examples are not statistically independent frames or matches.
For team-possession frames, training loss weights are
clip(sqrt(250 / spell_duration_frames), 0.5, 4), derived only from external labels.
Dead-ball frames retain weight 1. This increases the influence of brief control.

Noise augmentation adds Gaussian XY errors (0.25 m for players, 0.35 m for the
ball), 5% independent player dropouts, and randomly placed contiguous ball gaps
of 15-84 frames. These simulations are independent of labels. We document them
as stress tests, not as a reconstruction of the competition error process.

V2's temporal-penalty grid was 2, 4, 8, 16 and 32, selected on J03WR9 only.
V3 retains its emission classifier and searches boundary bases 8/16 and scales
1/4/16, selecting 16/4 on J03WR9. The architecture search evaluated 26 alternative
validation configurations: five hierarchical, six boosted-boundary, nine TCN
checkpoint/penalty combinations, four TCN/tree blends, and two exploratory TCN
boundary decoder settings. Our original compact TCN and hierarchical models
did not beat V2 on validation. Fusion improved validation but reduced short-spell
recall in cross-validation. Those negative results are retained. V3's settings
were frozen before cross-validation and its regression comparison on WMX/WN1.
Neither fitting nor parameter selection used the supplied public ground truth.

## Possession interpretation

Control and dead-ball evidence are learned from the external provider labels.
The sequence decoder favours continuity, reducing isolated team changes caused
by momentary proximity or ambiguous tracking, and lowers the cost of changes near
predicted boundaries to retain more brief possessions. Loose-ball and deflection cases
do not have an independent output class. Restarts are inferred as D-to-team
transitions; there is no manually assigned restart instant. We do not claim
these learned rules exactly reproduce Metrica's definition of controlled
possession, nor do we hand-correct predictions to enforce agreement.

## External evaluation

Using the organizers' metric implementation on external labels only, mean spell
F1 weights halves equally. Short recall means true possessions under two seconds
matched at IoU 0.5. Validation averaged clean/noisy halves: V2 0.4865, V3 0.5132.

| Evaluation | Tracking | V2 spell F1 | V3 spell F1 | V3 macro F1 | V2 short recall | V3 short recall |
|---|---|---:|---:|---:|---:|---:|
| Cross-validation | Clean | 0.5887 | 0.6245 | 0.9235 | 73/463 | 111/463 |
| Cross-validation | Simulated noise | 0.4170 | 0.4178 | 0.8392 | 51/463 | 61/463 |
| Reused-test regression | Clean | 0.5564 | 0.5896 | 0.9139 | 60/362 | 81/362 |
| Reused-test regression | Simulated noise | 0.3906 | 0.4079 | 0.8279 | 37/362 | 46/362 |

Cross-validation retrains both models on WOH/WOY and evaluates WPY/WQQ, then
reverses the roles. Each evaluated match is absent from fitting in its fold.
The noisy cross-validation overall change is essentially flat, and some halves
declined. Clean/noisy cases share labels and are not independent samples. All
seven external games have appeared in development; architecture acceptance
uses these comparisons. None is a fresh blind test. Acceptance checks were
development checks rather than a preregistered blind-test protocol. No parameters
changed after the comparisons.

Full per-half results, negative trials and artifact/source hashes
are included in external_reports/ and MODEL_RESEARCH_RESULTS.md. No competition
quality score is available or inferred from these external results.

## Limitations

Seven previously used external matches provide limited independent evidence. The external tracking
is cleaner and more complete than automatic challenge tracking, and provider
possession definitions differ. Noise simulation does not reproduce all occlusion,
identity-switch or calibration errors. The model can miss referee whistles,
stationary restarts, and stoppages not evident in geometry. Temporal penalties
may suppress short genuine possessions. Long abstentions are not necessarily
harmless under the scoring rule. Further development would require additional
licensed external matches and a fresh independent test protocol.

## Reproduction

Use Python 3.12 and the pinned requirements-model.txt dependencies. The public
code repository must include models/external_v3.joblib and its attribution file.
Place the organizer-supplied tactical folder under a local bundle root, then run:

```text
python infer_boundary.py --model models/external_v3.joblib --data-dir /local/bundle --out Barca4Ever.csv
python evaluation/validate_submission.py Barca4Ever.csv
```

On the prepared Windows workstation, Run-Barca4Ever.ps1 also checks the model
checksum and saves the format-validation log locally. No fitting is performed
during inference. The output manifest records model and CSV SHA-256 hashes.
The CSV contains exactly 199,319 indexed rows. Keep the original model artifact
to reproduce the exact output; re-training is documented separately in
V3_MODEL_GUIDE.md. Twenty-seven synthetic/unit tests and full-length synthetic CLI checks
were completed before competition inference.

All competition data, predictions and local validation logs remain outside the
public code archive. Source files are not redistributed, and no competition
data or extracts are sent to hosted AI models by these programs.

## Research references

Borghesi et al. (2023), Using Temporal Convolutional Networks to estimate ball
possession in soccer games, https://doi.org/10.1016/j.eswa.2023.119780. This
addresses H/A/dead possession with trajectory TCNs. Our feature-based TCN is an
original smaller design, not a reproduction or reuse of their confidential data.

Abu Farha and Gall (CVPR 2019), MS-TCN, https://arxiv.org/abs/1903.01945.
Motivates frame-resolution temporal convolutions in our neural experiment.

Ishikawa et al. (WACV 2021), Alleviating Over-Segmentation Errors by Detecting
Action Boundaries, https://arxiv.org/abs/2007.06866. Motivates separate class
and boundary evidence; our adaptive-cost decoder is our own design, not ASRF.
Published metrics/datasets differ and cannot be compared directly with ours.
