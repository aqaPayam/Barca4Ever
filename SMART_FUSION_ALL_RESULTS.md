# Learned combinations compared with V4.1

This cycle implemented and tested four families of time-varying reliability
combination. Every original model weight and the 16/8 decoder stayed fixed.
The component predictions, original match splits and corruption seeds match
the V4.1 benchmark. Baseline spell scores reproduced exactly throughout.

## Fitting and evaluation roles

WR9 changes from validation to gate-fitting/calibration data. Each gate sees
WR9 predictions from base weights that were not fitted on WR9. However, WR9
was previously used to select architecture/checkpoints; it is reused
development data, not strict new blind cross-fitting. No WR9 fitting score
is used to claim generalization. The same seven licensed IDSSE games are used.

Each whole-match fold has its own gate. The scaler, base trees, neural
initializer, neural adaptation and gate fitting exclude the evaluated fold
matches. CV screens candidates; only a qualifying single choice goes to
WMX/WN1 reused regression. These comparisons are development evidence, not
nested blind evaluation, statistical significance or competition rankings.

## Whole-match comparison

| Family | Candidate | Clean spell F1 | Noisy spell F1 | Clean short /463 | Noisy short /463 |
|---|---|---:|---:|---:|---:|
| Baseline | V4.1 | 0.7105 | 0.5329 | 191 | 79 |
| Linear | joint | 0.7061 | 0.5247 | 197 | 87 |
| Linear | factorized | 0.7052 | 0.5277 | 198 | 89 |
| Conservative | joint:shrink25 | 0.7083 | 0.5270 | 192 | 79 |
| Conservative | joint:shrink50 | 0.7084 | 0.5285 | 194 | 82 |
| Conservative | joint:disagree | 0.7095 | 0.5273 | 195 | 86 |
| Conservative | joint:boundary | 0.7070 | 0.5248 | 193 | 81 |
| Conservative | factorized:shrink25 | 0.7094 | 0.5275 | 192 | 79 |
| Conservative | factorized:shrink50 | 0.7095 | 0.5294 | 195 | 83 |
| Conservative | factorized:disagree | 0.7059 | 0.5301 | 196 | 88 |
| Conservative | factorized:boundary | 0.7068 | 0.5281 | 196 | 83 |
| Nonlinear | regret25 | 0.7087 | 0.5286 | 192 | 78 |
| Nonlinear | regret40 | 0.7075 | 0.5298 | 192 | 79 |
| Spell-score fitted | score_gate | 0.7088 | 0.5348 | 184 | 79 |

Selection rule: improve spell F1 and avoid a short-hit decline in BOTH
clean/noisy cases. Extra short recovery alone does not qualify. Short
recovery counts correctly matched team possessions under two seconds at
IoU0.5; spell F1 is the exact organizer mean over IoU0.3 through0.7.

The spell-score fitted algorithm internally selects two bounded parameters
from 25 settings on each fold WR9 calibration match. Those are fitting
trials, not extra independent test results. All calibration choices are
retained in models/smart_fusion/score_fold*.json.

No candidate passed the complete acceptance rule. V4.1 remains the active
model; its artifact, runner and local prediction file are preserved.
No failed candidate was evaluated to search the regression matches for
a favorable replacement. Research code, fitted gates and negative results
are retained for reproducibility, separate from the active model.

## Implementations and checks

The first linear gate used bounded likelihood fitting; factorized used
separate dead/conditional-team gates. Conservative variants restricted
their strength or activation. The nonlinear gate predicted relative
true-label probability advantage. The final small reliability rule fitted
spell scores on calibration only. Protocols were recorded before each family
was evaluated; see the four *_PROTOCOL.md files and per-half JSON reports.

Six synthetic checks passed: finite-difference gradient validation, learning
a reliability signal, probability sums and bounds, H/A exchange invariance
including ties, exact zero-parameter baseline and model serialization.
No competition tracking, video, labels or output-quality checks were used.
External-data attribution and CC BY4.0 licence: EXTERNAL_DATA.md.
