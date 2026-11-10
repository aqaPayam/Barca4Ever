# Paper-driven experiments after V4.1

Research reviewed on 6 November 2026. This note proposes external-data
experiments; no competition inputs, video, ground truth or prediction contents
were opened. Production files were not changed during this review.

## Recommended order

1. Native-rate contact adaptation, with explicit preprocessing controls.
2. Physics-based touch/reception features and a boundary-focused objective.
3. A player-only latent possession-path branch for long ball gaps.

These target information and timing weaknesses rather than another probability
mixture. None is assumed to improve the organizer score until tested.

## Concrete audit of our implementation

`paper2023_data.object_features` rounds 25-Hz sample indices to obtain 16 Hz,
but then multiplies each position difference by a fixed 16. The source time
steps alternate between 1/25 and 2/25 seconds. Thus a constant-speed trajectory
gets approximately 0.64x/1.28x alternating velocity estimates. Native 25-Hz
input removes this approximation. A useful control is a 16-Hz representation
whose differences are divided by the actual source-index time delta.

The function also calculates `raw_ball_seen` but does not pass it to the
network. After interpolation, the missingness flag and relative pair-valid
indicator treat filled samples as observed. The tree retains raw observation
and gap-age information, whereas the neural branch cannot distinguish real
and interpolated ball geometry explicitly. Test an extra raw-ball-observed
channel. Zero-pad the pretrained stem and instant-path input weights so this
new input initially preserves the original function, then fine-tune externally.
Changing the existing missingness flag directly would also suppress filled
relative geometry, so it is a different intervention.

Relative geometry currently assumes a 105x68-m pitch when converting normalized
coordinates into 10-m units. This is documented, but an actual-pitch-scale
control is another possible ablation. Keep it separate from the native-rate test.

## Experiment 1: native-rate temporal contact path

Keep the shared per-object TCN, two attention branches and role-free inputs.
Fine-tune a copied checkpoint at 25 Hz, with all frame/time conversions explicit.
A 64-step input now spans 2.56 seconds, not the paper's four seconds. With
target index47, lookahead becomes 16/25 seconds. Declare the shorter context
as an adaptation; do not silently claim identical physical receptive fields.

Evaluate the native-rate variant against V4.1 and, if needed, a 16-Hz
actual-delta velocity control. Keep original V4.1 probability outputs as an
unchanged long-context reference. Optional subsequent augmentation can switch
within-team track identities partway through a window, preserving instantaneous
player sets and team labels while breaking artificial reliance on stable IDs.

Borghesi et al. specify coordinate differences divided by the frame period,
but do not unambiguously settle before/after-coordinate-scaling velocity units.
Consequently, our normalized-velocity interpretation is not a proven paper
replication error. Kernel/block/width choices are still documented assumptions.
The paper's two-branch shared-TCN and separate attention layout is present in
our implementation.

Source: https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf

## Experiment 2: touch candidates and explicit transition supervision

ELASTIC extracts possible contact times from player-ball distance valleys,
ball-acceleration peaks and distance-to-pitch-boundary valleys. Its published
filters reject player-ball distance above3m or ball height above4m. It then
aligns a known event sequence with those candidates using dynamic programming.
This is event synchronization, not an end-to-end possession classifier.

Transfer the candidate features to our boundary learner at native 25 Hz:
local nearest-player distance minima, approaching/receding slopes, acceleration
and direction changes, boundary proximity, and raw/interpolated visibility.
Use multiple physical windows and missingness flags. Do not make a 3m or
height filter a hard rule without external validation, particularly because
our current production representation is XY only. Instead use features or
soft evidence. Keep V4.1 emissions frozen for an isolated boundary experiment.

For a subsequent auxiliary contact head, external ELASTIC corrected events
can supply reception/control times, while external provider H/A/D remains
the team-possession target. This tests timing supervision directly. Do not use
evaluation-match event annotations to fit its corresponding fold.

ELASTIC code is MPL-2.0; its benchmark event data is explicitly redistributed
under CC BY4.0 with Sportec/DFL attribution. It re-annotates our already-used
WMX/WN1/WPY games, so it is not new independent-match evidence.

Paper: https://arxiv.org/html/2608.30227v1
Code/benchmark: https://github.com/hyunsungkim-ds/elastic

## Experiment 3: player-only latent path for missing-ball intervals

PathCRF (KDD2026) is a particularly relevant new source. It infers a path
through player-control self-loops, player-to-player in-flight edges and outside
states, using player trajectories alone. A Set Attention backbone produces
emissions and dynamic CRF transitions; impossible edge transitions are masked.
Auxiliary sender/receiver and emission losses matter in its ablations.
Its headline event F1 is75.69% at a one-second tolerance; this is not our spell F1.

A compact adaptation could reuse our per-player trajectory embeddings but add
control/in-flight/outside latent states and an auxiliary player-only branch.
Train with masked-ball examples and legal transition constraints, allowing
one-touch actions. Decode team ownership from the control/flight path, retaining
the passing team's possession during flight unless opposing control is inferred.
This would add information during ball gaps rather than merely reweighting
two estimates built around the same missing observation.

Official code and weights are public under MPL-2.0. The supplied checkpoint140
has 3.43M parameters, 5-Hz sampling and a 10-second window. Its args specify
LOMOCV test index6: sorted Sportec files imply WR9 test, WQQ validation, and
WMX/WN1/WOH/WOY/WPY training. These weights therefore overlap our original
regression and CV games. They cannot be presented as a matched held-out
comparison there; retrain within our folds or evaluate only a genuinely excluded
match with explicit unequal-data-budget caveats.

Additional protocol hazards: official preprocessing uses true live-play episodes,
skips short or NaN-possessor episodes, and evaluation corrects episode endings
and can skip red-card phases. An honest full H/A/D comparison must not inherit
those ground-truth-derived shortcuts. Track roles/identities and preprocessing
also differ substantially from automatic competition tracking.

Paper: https://arxiv.org/html/2602.12080v2
Code: https://github.com/hyunsungkim-ds/pathcrf
Checkpoint config: https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/main/saved/140/args.json
Split code: https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/main/train.py

## Video paper and validation cautions

Peral et al. (VISAPP2025; Ferraz is a coauthor) use player-centric video tubes,
ResNet50 with temporal shift, and a second possessor-selection stage using
ConvTasNet/TDNN ideas. A background state covers the interval between players'
touch sequences. This supports treating control, transit and dead ball as
distinct latent concepts. The original video setup is not a drop-in tracking
model; I found no official released weights/code link in the paper/author page.
The useful immediate transfer is its touch/possession decomposition, not an
assertion that we have reproduced its video system.

Paper: https://www.iri.upc.edu/files/scidoc/3004-Temporally-Accurate-Events-Detection-Through-Ball-Possessor-Recognition-in-Soccer.pdf

Bischofberger et al. (PLOS2024) show how dependent tracking/event labels can
inflate event results: pass F-scores around0.87-0.92 on some feeds fell to0.71
on an independent event/position dataset. This is not proof of leakage in
IDSSE, but motivates caution about our single-provider, reused-match evidence.
PathCRF also reports a strong cross-provider drop. Broader independent raw
tracking remains necessary before making professional-system comparisons.

Validation paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298107
