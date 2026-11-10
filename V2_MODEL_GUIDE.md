# Previous candidate: short-possession model v2

V5 is the current candidate; see V5_MODEL_GUIDE.md and MULTIAGENT_RESULTS.md. This file
preserves V2's method and reproduction commands.

V2 supersedes v1 as the submission candidate based on external development
evidence. Original v1 models, selection records and the previous CSV are retained.
See V2_RESULTS.md for the full comparison and its limitations.

## Inference

Python 3.12; install requirements-model.txt. Use the included trusted model:

```text
python infer_learned.py --model models/external_v2.joblib --data-dir /local/bundle --out Barca4Ever.csv
python evaluation/validate_submission.py Barca4Ever.csv
```

To run V2 on this workstation, set Run-Barca4Ever.ps1 -Model models/external_v2.joblib; it checks its SHA-256
against models/external_v2.joblib.metadata.json. The bundle needs only
tactical/BAR-ATH_tactical_atd.xml and tactical/BAR-ATH_tactical_atd.txt.
The inference program contains no network calls. It does not fit, tune, read
ground truth or compare predictions to video. Keep all restricted data and outputs
local. No competition quality score is known.

## Method

The original 42 geometry/context features are supplemented with 17 features for
near-contact geometry and motion. Linear ball interpolation is restricted to
internal gaps at most two seconds; no extrapolation is performed. Raw visibility
and gap features remain available, so interpolated coordinates are not presented
as actual detections. Relative speed/alignment use the closest player on each
team; these can be affected by automatic tracking identity switches.

The classifier has 200 boosting iterations, 23 leaves, learning rate 0.07,
L2 regularization 3 and minimum leaf size 60. External training labels define
loss weights clip(sqrt(250 / spell_duration_frames), 0.5, 4) for team possessions;
dead-ball frames have weight 1. This raises the influence of short possessions.
No competition labels are used. The offline Viterbi switch penalty is 8, selected
on external validation match J03WR9 from 2/4/8/16/32.

## Reproduce training and checks in a fresh experiment directory

1. download_external.py retrieves and checksum-verifies licensed IDSSE data.
2. develop_v2.py fits on the four original training matches and selects on WR9,
   writing external_v2_experimental.joblib and v2_validation.json.
3. crosscheck_v2.py retrains both model versions on identical two-match folds.
4. develop_v2.py --stage regression compares the frozen versions on the two
   previously inspected test games; this is not a fresh blind test.
5. report_v2.py aggregates results; promote_v2.py checks external gains in overall
   spellF1 and short-spell recall before creating the canonical v2 artifact.

The acceptance checks were development checks, not a preregistered blind protocol.
Do not delete published result files to tune against their matches while calling
them new independent tests. Reproduction of the existing experiment is different
from selecting new parameters. A new blind test requires additional matches.

All data attribution and licensing remain as documented in EXTERNAL_DATA.md.
MODEL_GUIDE.md describes v1 historically. V2 model metadata records hashes of
the feature, training and inference code. No competition inference result was
used to choose or revise the algorithm.
