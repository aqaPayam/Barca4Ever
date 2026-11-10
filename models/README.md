# Barca4Ever trained artifacts

## Active candidate V5: contact timing and directed transitions

external_v5.joblib combines a 20-channel contact network, the preserved V2
geometry classifier and a directed H/A/D transition classifier. It retains raw
ball observability, corrects sampled velocity timing and uses fresh references
each fine-tuning epoch. The selected 16 Hz, equal neural/tree mixture and pure
directed boundary evidence passed clean/noisy whole-match and reused-regression
checks. Source/evidence hashes and all eight quality gates are in the metadata.
See ../V5_MODEL_GUIDE.md and ../MULTIAGENT_RESULTS.md for exact reproduction,
alternatives and limitations. Canonical CPU inference uses refined_infer.py.
V4.1 is preserved. All seven games are reused development evidence.

## Previous candidate v4.1: decoder refinement

external_v4_1.joblib preserves all V4 weights and changes the boundary scale
from 4 to 8, giving switch cost 16/(1+8b). The single setting selected from
six predefined external validation settings passed clean/noisy spell-F1 and
short-recovery checks in both whole-match and reused-test comparisons. V4
is preserved. Small development gains do not establish statistical significance.
See ../V4_1_MODEL_GUIDE.md and ../V4_DECODER_RESULTS.md. The metadata records
parent/source/model hashes, exact changes and external acceptance evidence.

## Previous candidate v4: contact fusion

external_v4.joblib combines the V2 geometric classifier with an externally
trained trajectory/contact network and a destination-specific change detector.
It uses equal neural/tree probability weights and decoding cost 16/(1+4b).
The neural architecture extends Borghesi et al. (2023), with ball-relative
features, an immediate contact path and learned distance-biased attention.
Keeper-role inputs remain zero after external fine-tuning; only XY and team
metadata are required. It is our adaptation, not the authors' original weights.

See ../V4_MODEL_GUIDE.md, ../PAPER_2023_PROTOCOL.md and ../PAPER_2023_RESULTS.md.
The final artifact passed its own clean/noisy validation,
whole-match cross-validation and reused-test regression checks. Both
neural fitting stages and all trees exclude each fold's evaluated games.
Checksums, pinned source hashes, architecture, attribution and acceptance
checks are in external_v4.joblib.metadata.json. CPU inference uses
infer_contact.py and requirements-contact.txt. Prior artifacts are preserved.

All versions use the IDSSE training matches and CC BY 4.0 attribution below.
Seven games provide development evidence; later results are not fresh blind
tests or proof of professional-system superiority. paper2023/ retains the
paper-based reimplementation, optimization changes, contact extension,
role-free adaptation and fold checkpoints, including negative experiments.
No competition observations or author pretrained weights are included.

## Previous model v3

external_v3.joblib contains the unchanged V2 possession classifier plus an
externally trained boundary classifier. Adaptive transition cost is 16/(1+4b).
The boundary score b is not calibrated because positive training frames are
weighted. Feature version remains 2; kind is adaptive_boundary. Inference uses
infer_boundary.py and needs no PyTorch. See ../V3_MODEL_GUIDE.md and
../MODEL_RESEARCH_RESULTS.md. Checksums, source hashes, attribution and development
acceptance checks are in external_v3.joblib.metadata.json. V1 and V2 are preserved.

V3's noisy cross-validation overall improvement over V2 was negligible.
research_v3/ retains experimental models, including neural and hierarchical
candidates not selected for production. These were trained from scratch on
external data; no competition data or paper pretrained weights are included.

## Previous model v2

external_v2.joblib is the previous candidate. It adds near-contact features and
short-spell-aware training weights; its feature version is 2 and switch penalty
is 8. See ../V2_MODEL_GUIDE.md and ../V2_RESULTS.md. Its checksum and code hashes
are in external_v2.joblib.metadata.json. It uses the same CC BY 4.0 IDSSE source
and training/validation matches attributed below. Preserve that attribution for
both artifacts. The earlier v1 artifact is retained unchanged as a fallback.

## Original model v1

`external_v1.joblib` is the team's own histogram gradient boosting model with its
externally selected temporal penalty. It is an executable Python serialization;
load only this trusted local artifact, not arbitrary joblib files from others.

Training: IDSSE matches J03WOH, J03WOY, J03WPY, J03WQQ.
Validation: J03WR9. Final penalty: 16.
Held-out test: J03WMX, J03WN1. No fitting or parameter changes followed testing.

Data source: DFL / Sportec Solutions; published by Bassek, Rein, Weber and Memmert,
Scientific Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y.
Dataset: https://doi.org/10.6084/m9.figshare.28196177, CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/.
Changes: positional feature extraction, simulated tracking corruption, frame
subsampling, team-swap augmentation, trained model and sequence decoding.
Preserve this attribution when distributing the artifact and report.

Feature schema and caveats: ../MODEL_GUIDE.md.
Complete scores: ../EXTERNAL_RESULTS.md.
Artifact SHA-256: ../external_reports/frozen_selection.json.
No competition data was read, used to train, or used to select this artifact.
