# Barca4Ever external-model results

This report evaluates external data only. No competition score is available.

## Data and protocol

IDSSE (DFL/Sportec), CC BY 4.0. Four training matches, one validation
match and two independent held-out Bundesliga matches. Published external
BallPossession/BallStatus fields provide labels; features use XY geometry
only. Source and changes are described in EXTERNAL_DATA.md.

The model used 454,656 frame examples including clean/noisy and team-swapped
copies. This is not 454,656 independent observations. The classifier was
fixed before validation selected the sequence switch penalty. Test labels
were not used to fit or choose parameters.

Selected penalty: 16. Validation mean spellF1: 0.4615.

## Held-out summary

Arithmetic means over four halves from two matches; not four independent matches.

| Input scenario | Model | Possession-spell F1 | Macro F1 |
|---|---|---:|---:|
| clean | prototype_v0.1 | 0.3898 | 0.7220 |
| clean | published_nearest_smooth50 | 0.2964 | 0.3788 |
| clean | learned_external_v1 | 0.5144 | 0.8959 |
| corrupted | prototype_v0.1 | 0.2758 | 0.6838 |
| corrupted | published_nearest_smooth50 | 0.2017 | 0.3837 |
| corrupted | learned_external_v1 | 0.3647 | 0.8153 |

Nearest-player smooth50 is a reimplementation of the published baseline
on the external coordinate arrays, using the same observations as our model.

## Per-half results

| Match | Half | Scenario | Model | Spell F1 | Macro F1 |
|---|---:|---|---|---:|---:|
| J03WMX | 1 | clean | prototype_v0.1 | 0.4126 | 0.7910 |
| J03WMX | 1 | clean | published_nearest_smooth50 | 0.3159 | 0.4576 |
| J03WMX | 1 | clean | learned_external_v1 | 0.5071 | 0.9304 |
| J03WMX | 1 | corrupted | prototype_v0.1 | 0.3534 | 0.7546 |
| J03WMX | 1 | corrupted | published_nearest_smooth50 | 0.2068 | 0.4582 |
| J03WMX | 1 | corrupted | learned_external_v1 | 0.3704 | 0.8545 |
| J03WMX | 2 | clean | prototype_v0.1 | 0.4148 | 0.7406 |
| J03WMX | 2 | clean | published_nearest_smooth50 | 0.2996 | 0.3682 |
| J03WMX | 2 | clean | learned_external_v1 | 0.5198 | 0.8912 |
| J03WMX | 2 | corrupted | prototype_v0.1 | 0.2755 | 0.6944 |
| J03WMX | 2 | corrupted | published_nearest_smooth50 | 0.1898 | 0.3681 |
| J03WMX | 2 | corrupted | learned_external_v1 | 0.3701 | 0.8181 |
| J03WN1 | 1 | clean | prototype_v0.1 | 0.3904 | 0.7517 |
| J03WN1 | 1 | clean | published_nearest_smooth50 | 0.3002 | 0.3967 |
| J03WN1 | 1 | clean | learned_external_v1 | 0.5260 | 0.9086 |
| J03WN1 | 1 | corrupted | prototype_v0.1 | 0.2744 | 0.7074 |
| J03WN1 | 1 | corrupted | published_nearest_smooth50 | 0.2288 | 0.4137 |
| J03WN1 | 1 | corrupted | learned_external_v1 | 0.3477 | 0.8202 |
| J03WN1 | 2 | clean | prototype_v0.1 | 0.3412 | 0.6046 |
| J03WN1 | 2 | clean | published_nearest_smooth50 | 0.2698 | 0.2927 |
| J03WN1 | 2 | clean | learned_external_v1 | 0.5046 | 0.8534 |
| J03WN1 | 2 | corrupted | prototype_v0.1 | 0.2000 | 0.5786 |
| J03WN1 | 2 | corrupted | published_nearest_smooth50 | 0.1817 | 0.2949 |
| J03WN1 | 2 | corrupted | learned_external_v1 | 0.3707 | 0.7682 |

## Limitations

Two matches provide limited evidence. Provider definitions and tracking
quality differ from the competition. Synthetic noise is independent of
labels but cannot establish robustness to actual occlusion, tracking
identity errors or homography errors. No competition result is inferred
from these numbers. The learned model has not read the competition match.

Test results were inspected after final selection; model parameters were
not changed afterwards. Future work must preserve this distinction or
declare this test set development data and acquire a new independent test.

## Reproduction

See MODEL_GUIDE.md. Exact trial history and code hashes are in
external_reports/frozen_selection.json; per-half metrics are in
external_reports/heldout_results.json.

Frozen artifact SHA-256: `85dc389a5019b8a04a5664034e25b32e60d20a2fe0b2efa7dbbbd5354868c2f8`.
