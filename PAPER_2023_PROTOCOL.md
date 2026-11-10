# 2023 possession architecture: replication and improvement protocol

Source: Borghesi et al., ESWA 223 (2023), 119780,
https://doi.org/10.1016/j.eswa.2023.119780.
Accepted manuscript: https://iris.polito.it/handle/11583/2976470,
Sections 3-5, especially Table 1 and the strongest two-branch/self-attention
configuration in Table 2. Original implementation and weights were not supplied.
This is an independently implemented, externally retrained adaptation, not the
authors' trained system and not a reproduction of its confidential-dataset score.

## Architecture to retain

- Input 64 timesteps; target at zero-based offset 47 (48th element).
- Independent, shared trajectory embedding for all objects. Initial 1x1
  convolution and batch normalization; dilated temporal convolution, dropout,
  1x1 convolution repeated in blocks; final dropout and batch normalization.
- Last embedding timestep, separate ball-query self-attention aggregation for
  dead/alive and conditional home/away; ELU dense layers of 64 and 32 units.
- Joint loss alpha*BCE(dead) + (1-alpha)*BCE(home), with home loss zero on dead
  frames. Two branches share the trajectory embedding, not attention or FFNs.
- Eleven channels: XY, XY velocity, ball/referee/goalkeeper flags, home/away
  flags, outside-pitch flag and missingness. Missing numeric channels use -2.
- Akima ball interpolation, pitch-scaled XY in [-1,1], fixed clipping of extreme
  coordinates, Xavier-normal convolution/attention and Xavier-uniform FFN.

## Necessary adaptations and unspecified settings

External IDSSE is at 25 Hz; sample trajectories at 16 Hz to retain the paper's
4-second window, then interpolate probabilities back to all 25-Hz scoring frames.
No labels are interpolated for evaluation. No referee positions exist in these
external streams: retain a missing referee slot, not a fabricated trajectory.
Goalkeeper flags use external match metadata only. Reserve 17 slots per team,
with absent players padded as missing. Same object order has no semantic role.

The paper does not fix embedding/attention widths, convolution kernel, number of
blocks, dropout, alpha, batch size or epoch count. Initial explicit choices:
width=64, attention=64, kernel=2, dilations=1/2/4/8/16/32 (receptive field 64),
dropout=0.1, alpha=0.5, batch=128, eight epochs of 100,000 sampled windows.
Use causal padding inside the 64-step window; the target offset makes inference
acausal. Batch normalization uses the object, time and batch axes. XY clipping
uses the pitch extent; velocities are finite differences in metres/second with
a fixed [-40,40] guard. These missing details limit fidelity and must be disclosed.

The first run retains the reported Adam learning rate 1e-5 and epoch decay 0.7,
clean training samples, unweighted loss and best validation frame accuracy.
Also test a clearly labelled adaptation: learning rate 0.001, cosine decay,
clean/noisy inputs, team swaps and externally computed short-spell loss weights.
These are our changes, not paper hyperparameters. No pretrained weights are used.

## Fixed external evaluation

Train WOH/WOY/WPY/WQQ; select on WR9, keeping both halves together. All seven
matches have already appeared in prior development; none is a fresh blind test.
Report raw hierarchical-threshold predictions separately from optional Viterbi
decoding selected only on WR9. Decoder grid 0/2/4/8/16 plus the original hard
threshold rule. Compare V3 on identical clean/noisy external inputs.

For promising improvements, fit within the same two folds as prior comparisons:
WOH/WOY -> WPY/WQQ and reverse; evaluation corruption seeds remain
11200 + 100*fold_index + 10*match_index + half_index. Retrain every learned
component within each fold. Freeze improved settings before inspecting folds.
Use existing WMX/WN1 only for explicitly labelled regression after freezing.

Primary measure: organizer spellF1. Also report macroF1, global frame accuracy,
and short-spell recall (under two seconds at IoU 0.5). Retain failures, training
curves, per-half scores, seeds, runtime, source and artifact hashes. Check
permutation invariance, dead-frame loss masking, target alignment and chunked
inference against direct sliding-window inference using synthetic inputs.

Any later extension must be logged before inspecting its new evaluation results.
A higher score here beats a particular retrained implementation on these external
data, not every professional model. Competition inputs/video/labels are excluded
from every research program. V3 and its CSV stay frozen until a replacement has
supporting external evidence and passes the production checks.

## Additional independent extension, declared before results

While preparing the neural runtime, test a destination-specific boundary model.
Keep V2 emissions and 177 boundary features, but replace binary change/no-change
supervision with background/start-H/start-A/start-D. Keep the V3 tree sizes,
positive weight 10, and external-only fitting. Decode with a cost per destination,
C[t,j] = base/(1+scale*b[t,j]), selected over base 8/16 and scale 4/16 on WR9.
This is our extension, not part of the 2023 paper. It must pass cross-match checks
before replacing V3. Preserve its results even if it fails.

## Neural contact extension, declared before paper training results

Also test a relative recipe: retain the two-branch architecture but add six
object channels (ball-relative XY scaled by 10 m, relative XY velocity scaled
by 10 m/s, distance in 10 m units, and valid object-ball pair indicator).
Relative XY uses the standard 105x68 m coordinate scale as an explicit engineering
approximation. Zero these extra channels when either object or ball is missing.
Replace unrestricted Akima filling with internal linear interpolation limited to
two seconds. Keep adapted optimization, noise, team swaps and short-spell weights.
This is a bundled original extension, not an isolated attribution of gains to
one component, and not a faithful paper configuration. Select epochs 2/4/8 and
the same decoder grid using external validation spell F1 only.

After inspecting external duration counts but before training the relative
recipe, add a declared spell-focused sampler: 75% of training references are
uniform eligible frames; 25% first choose a team-possession spell uniformly and
then a valid frame within it. Keep short-spell weights as well. The median of
the 463 short training-game spells is 1.2 seconds (29 under 0.2 s, 43 from
0.2-0.5 s, 109 from 0.5-1 s, 282 from 1-2 s). Thus most are not single-frame
artifacts. This sampler is another part of our bundled extension, not the paper.

The initial training-driver source is preserved byte-for-byte in
research_snapshots/train_paper2023_before_sampler.py. The final driver adds the
relative-only sampler; its paper branch is unchanged. Numeric implementation
choices include GPU mixed precision, gradient norm clipping at 5, and batch
normalization epsilon 0.001/momentum 0.01.

Use batch 256 for the two modified recipes to improve GPU throughput; the paper
recipe retains the declared batch 128. Evaluate all frozen full models on the
same original WMX/WN1 regression games as V3, with unchanged seeds 9500+h.
Cross-validation retraining is required for promising neural replacements, not
for every noncompetitive intermediate profile. Report the distinction explicitly;
do not imply a baseline was cross-validated if only regression was performed.

Before relative training, also scale its original velocity channels into
normalized-coordinate units per second (divide by 52.5/34); preserve -2 for
missing velocities. Its additional relative velocities retain 10 m/s units.
The paper does not unambiguously fix whether velocity is computed before or
after coordinate scaling. The paper and adapted profiles retain the original
metres/second interpretation; the relative bundle tests the scaled alternative.
The earlier preprocessing source is preserved in research_snapshots/.

Before its first fit, the relative recipe also gains a direct target-frame
geometry path: shared per-object MLP 17->64->64 with ReLU and LayerNorm, added
to each TCN trajectory embedding before attention. It uses the target frame
at offset 47, not the final input frame. This gives brief contacts a short
path to the classifier while retaining trajectory context. It is our extension.
The old architecture source is preserved in research_snapshots/; paper/adapted
profiles retain instant_skip=False, so their behavior is unchanged.

Attention diagnostics on 64 clean external WR9 windows found the paper home
branch spread attention across about 22.6 effective objects. Before the first
relative fit, add a declared contact prior: subtract softplus(k)*distance/10m
from attention logits for valid object-ball pairs, initializing k=2 and learning
it on external training data. Mask currently missing objects; the possession
branch pools player values only, retaining a ball fallback when none are seen.
The dead branch may still pool the ball. This geometry attention is enabled only
for the relative extension; baseline behavior and weight compatibility remain.

After fitting relative, test four externally selected probability fusions:
neural weight 0.25/0.5, and binary-V3/destination-specific boundary decoding.
Keep decoder base16/scale4 fixed. This is a small declared fusion test, not
assumed to help. Select on WR9 only, then evaluate the frozen setting on the
same reused WMX/WN1 regression games. Promising replacements require refitting
every learned component inside the existing whole-match folds before promotion.

Before inspecting blend results, test three generic/destination boundary mixtures
(destination weight 0.25/0.5/0.75), keeping V2 emissions and decoder16/4 fixed.
This follows the observed short-recall trade-off of the destination detector.
Select on WR9; refit both boundary classifiers inside the prior folds before
acceptance. A replacement intended to improve short recall should improve
overall spell F1 and avoid short-recall regressions against V3 in both scenarios.
This remains development work on reused data, not a new blind test.

After relative training, before the first fusion results, extend the probability
weight grid to include 1.0 (six settings total). Relative achieved validation
spellF1 0.5720 at fixed penalty16 but short recovery38/188 versus V3's40/188.
Weight1 tests the neural classifier with learned boundaries directly, permitting
brief switches without weakening continuity everywhere. Weights0.25/0.5 still
test probability mixtures. The selection remains external WR9 only.

For deployment without goalkeeper role metadata, predeclare a fixed geometry
fallback: at least 25 observed frames; median absolute X within the end penalty
area (pitch half-length minus16.5m), median absolute Y <=20.16m, and median
absolute X within5m of the deepest teammate. This permits fragmented tracks.
No parameter is fitted to the supplied match. Check against external role
metadata and external possession scoring before enabling it. Unknown roles can
also remain zero; that separate absence stress test is retained. Missing extra
objects are masked; all observed players are retained at inference.

## Role-free contact adaptation

A local pass/fail input-schema check, displaying no supplied metadata contents,
found that the competition input does not supply recognized goalkeeper-role
fields. No supplied video, ground truth or predictions were inspected or scored.
The fixed geometry fallback recovered only12/20 external keeper roles (1 false
positive) and is retained as a diagnostic, not the preferred production input.
To remove this dependency, predeclare a rolefree recipe: initialize from the
external relative checkpoint, set the keeper input channel to zero in training
and inference, and fine-tune two epochs of100k windows, Adam1e-4/cosine to3e-5,
batch256, seed20261112. Keep the contact architecture and other augmentations.
Evaluate with the same six probability/boundary combinations on WR9. Repeat
the two-epoch adaptation inside each fold, initializing only from that fold's
eight-epoch external checkpoint. No learned component may see its evaluated
match during fitting. Compare on identical external inputs and preserve prior
metadata-dependent comparisons. Production may use this role-free model only
after its own complete validation, cross-validation and regression checks.
