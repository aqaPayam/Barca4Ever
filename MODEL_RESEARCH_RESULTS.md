# Model research results: boundary-aware v3

The selected model is the strongest tested candidate in this search. This
does not establish a globally optimal model or a competition score.

## Research and implemented candidates

We reviewed exact-task football possession TCN research and temporal action
segmentation research. Primary sources and design decisions are recorded in
[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). Implementations are original;
we did not copy paper code or obtain their confidential training data/weights.

Twenty-six validation configurations were evaluated in addition to the V2
control: five hierarchical dead/alive + team penalties; six boosted boundary
decoder settings; nine compact TCN checkpoint/penalty pairs; four TCN/tree
blends and two exploratory TCN-boundary decoder settings.

The neural model has 59 inputs, 32 channels, five dilated residual blocks
(dilations 1/2/4/8/16), no temporal pooling, and possession/boundary heads.
It was trained from scratch on external data, with short-spell weighting.
Neither the TCN alone nor its blend is used in production. The blend gained
on validation but reduced short-spell recall in cross-validation.

## Selected design

Input: player/ball XY tracking, team metadata and missingness. The unchanged
V2 classifier estimates home/away/dead from 59 geometry/motion features.
A second boosted classifier estimates transition evidence from 177 features:
the 59 inputs and absolute differences from three frames before/after.
External label changes widened by three frames provide its supervision.
It has 140 iterations, 15 leaves, learning rate 0.08, L2 3 and minimum leaf
size 60; positive boundary frames carry weight 10. Its output b is a score,
not a calibrated probability because training uses unequal class weights.

The decoder maximizes summed log class probabilities minus switch costs
C_t = 16 / (1 + 4 b_t). This permits faster changes where boundary evidence
is strong and stronger continuity where it is weak. Output remains one
H/A/D/X label per frame. Unknown long tracking outages use the prior X rule.

## External results

Spell F1 is the organizer metric averaged equally over evaluated halves.
Short recall counts possessions under two seconds matched at IoU 0.5.
Clean/noisy variants share labels and are not independent samples.

| Evaluation | Tracking | Model | Spell F1 | Macro F1 | Short recall |
|---|---|---|---:|---:|---:|
| validation_selection | mixed | v2 | 0.4865 | 0.8814 | 31/188 (16.5%) |
| validation_selection | mixed | hierarchical | 0.4829 | 0.8788 | 31/188 (16.5%) |
| validation_selection | mixed | learned_boundary | 0.5132 | 0.8857 | 40/188 (21.3%) |
| validation_selection | mixed | tcn | 0.4809 | 0.8708 | 24/188 (12.8%) |
| validation_selection | mixed | fusion | 0.5124 | 0.8862 | 30/188 (16.0%) |
| cross_validation | clean | learned_boundary | 0.6245 | 0.9235 | 111/463 (24.0%) |
| cross_validation | clean | v2 | 0.5887 | 0.9182 | 73/463 (15.8%) |
| cross_validation | noise | learned_boundary | 0.4178 | 0.8392 | 61/463 (13.2%) |
| cross_validation | noise | v2 | 0.4170 | 0.8368 | 51/463 (11.0%) |
| cross_validation | clean | fusion | 0.5994 | 0.9151 | 69/463 (14.9%) |
| cross_validation | clean | tcn | 0.5547 | 0.8979 | 64/463 (13.8%) |
| cross_validation | noise | fusion | 0.4151 | 0.8362 | 32/463 (6.9%) |
| cross_validation | noise | tcn | 0.3917 | 0.8128 | 28/463 (6.0%) |
| reused_test_regression | clean | learned_boundary | 0.5896 | 0.9139 | 81/362 (22.4%) |
| reused_test_regression | clean | v2 | 0.5564 | 0.9073 | 60/362 (16.6%) |
| reused_test_regression | noise | learned_boundary | 0.4079 | 0.8279 | 46/362 (12.7%) |
| reused_test_regression | noise | v2 | 0.3906 | 0.8259 | 37/362 (10.2%) |

## Evidence and limits

Training/selection: four IDSSE matches, validation J03WR9, CC BY 4.0.
Cross-validation: fit WOH/WOY and evaluate WPY/WQQ, then reverse. Models
were fitted again within each fold; both halves stay together. Boundary
parameters and neural training length were fixed before these comparisons.
The fitted model never saw its fold evaluation match. Architecture selection
still uses development evidence; these are not a new independent blind test.

The noisy cross-validation spell F1 change (0.4170 to 0.4178) is essentially
flat and should not be treated as a reliable overall gain. Some individual
halves declined. The clean gain and short-spell recall gains are clearer.
The WMX/WN1 results are regression on previously inspected external matches;
no parameter changes followed that check. No significance or ranking claim
is made from seven previously used games or synthetic noise.

All competition assets remain inference-only. We did not score predictions
against the competition ground truth, compare them against its video, or
fit/tune on the supplied match. Its CSV is checked only for file validity.
New licensed matches are needed for a genuinely fresh blind assessment.

## Reproduction

V3 inference uses requirements-model.txt only, without PyTorch. For research
also install requirements-research.txt. See V3_MODEL_GUIDE.md for commands.
All trials and per-half results, including negative results, are retained
in external_reports/research_v3/. Data attribution is in EXTERNAL_DATA.md.
