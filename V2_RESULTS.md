# Short-possession model v2: external results

The competition match was not used for fitting, tuning, selection or quality scoring.
V1 and its submission remain available as a fallback.

## Changes

- Add 17 features (59 total): short-gap ball interpolation, nearest-team geometry
  after interpolation, relative player/ball motion, directional alignment and
  near-contact context at 0.12 seconds. Interpolation never extrapolates beyond
  observations and leaves gaps longer than two seconds missing.
- Weight externally labelled team spells by clipped square-root inverse duration:
  clip(sqrt(250 / duration_frames), 0.5, 4). Dead-ball frames keep weight 1.
- Fit 200 boosting iterations with 23 leaves and select a temporal switch
  penalty from 2/4/8/16/32 using only J03WR9. Selected penalty: 8.

## Evidence

Whole-match cross-validation retrains BOTH models on identical two-match folds
from the original four training games. Each evaluated match is absent from
both fitted models in its fold. These are development data, not a new blind test.
The WMX/WN1 comparison reuses previously inspected test games and is explicitly
a regression check. No parameter changes followed those regression results.

Short-spell recall counts true possessions under two seconds recovered at IoU 0.5.
Clean and noisy scenarios reuse the same underlying labels; their counts are
not independent observations. Score means weight halves equally.

| Evaluation | Noise | Model | Spell F1 | Macro F1 | Short-spell recall |
|---|---|---|---:|---:|---:|
| whole_match_cross_validation | False | v1 | 0.5571 | 0.9094 | 41/463 (8.9%) |
| whole_match_cross_validation | False | v2 | 0.5887 | 0.9182 | 73/463 (15.8%) |
| whole_match_cross_validation | True | v1 | 0.4005 | 0.8290 | 28/463 (6.0%) |
| whole_match_cross_validation | True | v2 | 0.4170 | 0.8368 | 51/463 (11.0%) |
| reused_test_regression | False | v1 | 0.5144 | 0.8959 | 38/362 (10.5%) |
| reused_test_regression | False | v2 | 0.5564 | 0.9073 | 60/362 (16.6%) |
| reused_test_regression | True | v1 | 0.3647 | 0.8153 | 22/362 (6.1%) |
| reused_test_regression | True | v2 | 0.3906 | 0.8259 | 37/362 (10.2%) |
| validation_selection | mixed | v1 | 0.4615 | 0.8761 | 16/188 (8.5%) |
| validation_selection | mixed | v2 | 0.4865 | 0.8814 | 31/188 (16.5%) |

## Limits

Data remains limited and provider-specific. Synthetic noise does not reproduce
all challenge tracking errors. External improvements do not establish a
competition score or rank. A genuinely new blind test requires additional
licensed matches not already used in this project.
Full data attribution is in EXTERNAL_DATA.md. Exact results are retained in
external_reports/v2_validation.json, v2_crosscheck.json and v2_regression.json.
