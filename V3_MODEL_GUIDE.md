# Boundary-aware model v3

This file preserves V3. The current candidate is V5; see V5_MODEL_GUIDE.md
and MULTIAGENT_RESULTS.md for its extension and comparisons.

V3 is the strongest tested candidate in the architecture search recorded in
MODEL_RESEARCH_RESULTS.md. It retains V2's possession classifier and adds an
externally supervised change detector plus adaptive sequence decoding. V1 and
V2 remain unchanged as fallbacks. This is our design, informed by published
research; the organizers did not prescribe it.

## Input, model and output

The input is player/ball XY tracking in metres, team assignments, and observation
indicators. There are 59 possession features from possession_v2.py. These summarize
team-ball distances, motion, relative speed, alignment, visibility and nearby
temporal context. No player IDs, event labels, clocks, names, match IDs or
possession labels are inference inputs.

The first classifier is the unchanged weighted V2 histogram gradient boosting
model: 200 iterations, 23 leaves, learning rate 0.07, L2 3, minimum leaf size 60.
It estimates H/A/D class probabilities. Training upweights short external spells
by clip(sqrt(250 / duration_frames), 0.5, 4).

The boundary classifier uses 177 features: x[t], abs(x[t]-x[t-3]), and
abs(x[t]-x[t+3]), clamped at block endpoints. It has 140 boosting iterations,
15 leaves, learning rate 0.08, L2 3, minimum leaf size 60. Boundary targets are
external label changes expanded by three frames on either side; positives have
weight 10. Because of that weighting, its output b is an uncalibrated score.

Viterbi decoding maximizes sum(log p[t,label[t]]) minus C[t] for each change,
where C[t] = 16 / (1 + 4*b[t]). It retains continuity while allowing a change
more readily when the detector sees transition evidence. This differs from V2's
fixed switch cost 8. The inference is offline and may use future observations.

Output: 199,319 frame,label rows with H/A/D/X. Player tracking outages of at least
10 seconds split processing blocks and receive X. No fit, tuning or manual
correction happens during competition inference.

## Training and selection

IDSSE, DFL/Sportec, Bassek et al. (2025), CC BY 4.0:
https://doi.org/10.6084/m9.figshare.28196177.
Training: J03WOH, J03WOY, J03WPY, J03WQQ. Validation: J03WR9.
Keep both halves of every match in the same split. The boundary model uses the
same clean/noisy and swapped-team external sequences, subsampled every fifth
frame, as the possession model. No competition data enters training or selection.

Boundary base penalties 8/16 and scales 1/4/16 were selected by mean validation
spell F1. Other families: hierarchical dead/alive + team classification, an
original dilated TCN, TCN/tree probability fusion, and exploratory decoding using
the TCN boundary head. All trial results, including failures, are retained.

Cross-validation refits on WOH/WOY and evaluates WPY/WQQ, then reverses the roles.
The settings were frozen before those comparisons. The WMX/WN1 comparison is
regression on previously inspected matches. Architecture acceptance uses these
development results, so none is a fresh independent blind test. In particular,
the noisy cross-validation spell F1 gain is negligible; do not claim robustness
to arbitrary tracking failures or a known competition score.

## Local inference

Python 3.12 with requirements-model.txt. PyTorch is not required for V3:

```text
python infer_boundary.py --model models/external_v3.joblib --data-dir /local/bundle --out Barca4Ever.csv
python evaluation/validate_submission.py Barca4Ever.csv
```

On Windows, Run-Barca4Ever.ps1 defaults to V3, verifies the artifact checksum,
and saves the validator log locally. To reproduce V2 use its explicit --model
argument with infer_learned.py, or the wrapper's -Model option. Keep restricted
inputs, CSVs, manifests and logs local. Only load trusted serialized models.

## Research reproduction

Use a fresh copy/experiment directory to preserve original reports. Install
requirements-model.txt, download the external dataset with download_external.py
and read EXTERNAL_DATA.md. The files below access external data only:

```text
python develop_v2.py --stage fit
python promote_v2.py
python research_trees.py
python research_crosscheck.py
python research_regression.py
python report_research.py
python promote_v3.py
```

promote_v2.py also requires the historical V1/V2 validation/cross/regression
reports. The archived V2 emission artifact is provided for exact reproduction;
research_trees.py uses it directly. Re-training can vary with software versions,
so keep the frozen artifacts for exact inference reproduction.

Optional neural experiments additionally use requirements-research.txt:

```text
python research_tcn.py
python research_fusion.py
python research_fusion_crosscheck.py
```

The neural experiment trains from scratch on external data, never the supplied
video. Its weights are retained for research but are not part of V3 inference.

## Verification

Twenty-seven synthetic/unit tests passed, including decoder equivalence and
neural chunk-context checks. A full-length generated tracking bundle tests V3's
inference and file validation independently of competition inputs:

```text
python -m unittest discover -s tests -v
python smoke_model.py --engine boundary --model models/external_v3.joblib
```

The two neural tests require the optional runtime and otherwise skip. The prepared
workstation ran them successfully. Model/source hashes and external acceptance
checks are in models/external_v3.joblib.metadata.json. Those checks are development
acceptance checks, not a preregistered blind-test protocol.
