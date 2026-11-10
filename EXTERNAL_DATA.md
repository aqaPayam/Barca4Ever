# External-data provenance and attribution

Selected: IDSSE, published by Manuel Bassek, Robert Rein, Hendrik Weber and Daniel
Memmert, "An integrated dataset of spatiotemporal and event data in elite soccer",
Scientific Data 12, 195 (2025), DOI https://doi.org/10.1038/s41597-025-04505-y.
Original data: DFL / Sportec Solutions; deposit https://doi.org/10.6084/m9.figshare.28196177.
Licence confirmed both in the paper and Figshare article API: CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/.

Local changes: XML parsing, coordinate features, separating published possession
and ball-status labels from inputs, simulated tracking noise/dropouts, trained
classifiers, trajectory/contact neural networks, external-only role-free
fine-tuning, raw-observation/gap-time inputs, source-time velocity correction,
directed transition learning and temporal decoding. Preserve this acknowledgement in public code
and the technical report. Files are downloaded from the original deposit, with
size and published MD5 verification; original metadata is stored locally in
external_data/idsse/source_manifest.json.

The labels are the source's BallPossession (1 home, 2 away) and BallStatus
(0 inactive, 1 active); inactive is mapped to D, unknown to X. These are external
provider labels, not newly hand-labelled by our team. Their definitions may differ
from Metrica's. No competition match labels, video or human-produced data are
used. No label-bearing fields enter the classifier features.

Whole-match split fixed before training:

- Train: J03WOH, J03WOY, J03WPY, J03WQQ.
- Validation: J03WR9. Select symmetric temporal switch penalty here only.
- Test: J03WMX, J03WN1. Evaluate the frozen artifact once, then report honestly.

These are the original V1 roles. Development through V5 has now inspected all seven
games; WMX/WN1 are explicitly reused-test regression data for those versions,
not fresh blind tests. New candidate acceptance uses external development
cross-validation. See PAPER_2023_RESULTS.md and MODEL_RESEARCH_RESULTS.md for
the evaluation limits and full comparison history. Each V4 fold's initializer,
fine-tuned network, possession trees and boundary trees exclude evaluated games.
V5 uses the same licensed data and exclusions, with fresh frame/spell references
for each external fine-tuning epoch. Its whole-match comparisons and negative
experiments are recorded in MULTIAGENT_RESULTS.md and V5_MODEL_GUIDE.md.

Two Bundesliga matches are held out against four second-division training
matches. This is deliberately a cross-match and cross-division test, not a random
frame split. Clean tracking and a deterministic synthetic corruption scenario are
reported separately. Corruption masks use no labels. Corruption is a stress test,
not a reconstruction of the competition's actual tracking errors.

Alternatives investigated: Metrica sample-data has community-use language and
attribution instructions but less explicit licensing. SkillCorner is another
candidate, but not used in this experiment. SoccerNet video requires additional
access/licensing arrangements and is unnecessary for this geometry-only model.
