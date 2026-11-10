# Contact-network fusion: model v4

V5 is the current candidate; see V5_MODEL_GUIDE.md and MULTIAGENT_RESULTS.md.
This file preserves V4. V4.1's decoder-only refinement is documented separately
in V4_1_MODEL_GUIDE.md and V4_DECODER_RESULTS.md.

The frozen candidate combines a tracking-based contact network with the existing
possession classifier and an externally learned change detector. The full
architecture search and measured comparisons are in PAPER_2023_RESULTS.md.
Its own validation, whole-match cross-validation and reused-test regression
acceptance checks passed before freezing. These are development checks,
not a preregistered blind-test protocol or proof of professional-system superiority.

## Inputs and architecture

Only tactical player/ball XY tracking and team metadata are required. The model
does not require player names, persistent player IDs, goalkeeper roles, game
clocks, Smart Tagging, homography or video. The keeper-role channel remains zero
in both fine-tuning and production inference, eliminating an unavailable feature.

The tree classifier retains V2's 59 geometry/motion/context inputs, 200 boosting
iterations, 23 leaves, learning rate 0.07, L2=3 and minimum leaf size 60.

The neural branch is an original extension of the two-branch/self-attention
architecture described by Borghesi et al. (2023), ESWA223,119780,
https://doi.org/10.1016/j.eswa.2023.119780. It uses independent shared per-object
trajectory encoders, then separate ball-query attention and dead/alive and
conditional home/away FFNs. This is not the authors' code or original trained
model; unavailable settings and adaptations are documented in PAPER_2023_PROTOCOL.md.

The encoder uses width 64, a 1x1 stem, batch normalization, six causal temporal
convolution/dropout/1x1 blocks with kernel 2 and dilations 1/2/4/8/16/32, and final
dropout/normalization. A 64-step window at 16 Hz predicts its 48th frame (index 47),
so the offline model sees about one second beyond the target. Two ELU FFNs have
64/32 hidden units. Their outputs are P(dead) and P(home | alive). Home loss is
zero on dead frames.

Its 17 channels comprise the paper's 11 geometric/role/visibility channels
(with keeper role zero) plus relative ball XY and velocity, distance, and pair
validity. Raw XY is pitch-normalized; original velocities use normalized-coordinate
units/second. Added relative positions use 10m units and velocities10m/s units,
with the standard 105x68 m pitch scale as an explicit approximation. Missing values
retain their indicators. Ball interpolation fills internal gaps up to 2 seconds
only. A target-frame per-object MLP 17->64->64 with LayerNorm adds an immediate
contact path to the trajectory embeddings. Learned positive distance penalties
focus attention on nearby observed objects. The possession branch pools player
values, with a ball fallback if no players are observed.

The training representation pads to 17 player slots per team, plus ball and a
missing referee slot. Production retains every observed input slot in padded
groups; additional missing objects do not affect the contact network's output.
No player is discarded to enforce 36 objects. Within-window identity switches
remain a limitation even though ordering and additional missing padding are
handled. The target-frame geometry path reduces reliance on stable trajectories.

Neural joint probabilities are [(1-d)*h,(1-d)*(1-h),d]. They are interpolated
back to every 25 Hz frame and averaged with the tree probabilities using a weight
selected on external validation. The boundary type and weight are recorded in
the frozen artifact's metadata. Decoding maximizes summed log probability minus
switch costs 16/(1+4*b[t,destination]). A binary change detector shares the same
score across destinations; a state detector supplies separate H/A/D onset
scores. Its positive training weight means these scores are not calibrated
probabilities. The primary comparison uses the exact organizer spellF1 scorer.

## External-only fitting

Source: IDSSE, DFL/Sportec, Bassek et al. (2025), CC BY 4.0,
https://doi.org/10.6084/m9.figshare.28196177. See EXTERNAL_DATA.md.

Train WOH/WOY/WPY/WQQ; validation WR9. Both halves stay within their match.
The contact network first trains eight epochs of 100k windows, batch 256,
Adam 0.001/cosine and seed 20261107. Fine-tuning zeros keeper roles and uses two
epochs, Adam 0.0001/cosine and seed 20261112. Initializers must have the exact same
fitting-match list as the fine-tuning run. No external evaluated fold match is
included in either fitting stage. Source hashes and initializer hashes are saved.

Training mixes clean/noisy observations and team swaps. Seventy-five percent of
references sample eligible frames; 25% sample team spells uniformly then a frame
within the spell. The loss also uses V2's clipped inverse-square-root duration
weights. Every frame, label and role used for fitting comes from licensed external
matches. No competition tracking, video or ground truth is used to fit or tune.

The paper-based structure, original reported learning-rate recipe, improved
optimization, contact extension, role-free adaptation and probability fusions
are retained separately. Negative results are not discarded. All seven external
games have appeared in development. Original WMX/WN1 tests are reused regression
data; subsequent results are not fresh blind tests. Synthetic corruption does
not reproduce all real automatic-tracking errors or establish a competition rank.

## Reproduce inference

Python 3.12; CPU inference is canonical and does not require a GPU:

```text
python -m pip install -r requirements-contact.txt
python infer_contact.py --model models/external_v4.joblib --data-dir /local/bundle --out Barca4Ever.csv
python evaluation/validate_submission.py Barca4Ever.csv
```

The bundle root contains tactical/BAR-ATH_tactical_atd.xml and its TXT. The
artifact includes the trained neural state as NumPy arrays, tree classifier,
change detector, architecture and decoding settings. Preserve its CC BY 4.0 data
attribution. Load only the team's trusted artifact. Conversion verifies every
neural weight and counter against the original checkpoint without precision loss.

Run-Barca4Ever.ps1 verifies the model checksum and routes to the matching engine.
It keeps validation details local. The CSV manifest records model/output hashes
and inference device. Long player-tracking outages of at least 10 seconds split
blocks and receive X, using the same fixed rule as V3. No fit or adjustment
happens during competition inference. Keep restricted inputs and outputs local.

V1/V2/V3 artifacts and previous CSVs are preserved as fallbacks. Use the wrapper's
explicit -Model option for an earlier model. The official validator checks file
format only; the hidden competition quality score remains unknown.

## Research reproduction

Use a fresh experiment directory to preserve the original evidence. GPU fitting
uses requirements-paper2023.txt; CPU fitting uses requirements-contact.txt but
is slower. The original trained artifacts are included for exact inference.

```text
python train_paper2023.py --recipe paper
python train_paper2023.py --recipe adapted --batch 256
python train_paper2023.py --recipe relative --batch 256
python train_paper2023.py --recipe rolefree --initialize models/paper2023/relative.pt --epochs 2 --batch 256 --seed 20261112
python evaluate_paper2023.py --recipe rolefree
python fuse_paper2023.py --recipe rolefree --stage validation
python fuse_paper2023.py --recipe rolefree --stage regression
python crosscheck_fused_paper2023.py --rolefree
python promote_contact.py
```

The fold adaptation requires the corresponding relative_fold1/2 checkpoints,
trained by the metadata-dependent crosscheck_fused_paper2023.py first. The tree
boundary fitting and cache creation steps are in PAPER_2023_PROTOCOL.md and the
research scripts; no evaluated match may appear in its fold's cache fitting data.

Synthetic tests cover loss masking, frame alignment, direct-window/dense
equivalence, permutation invariance, additional missing slots, contact priors,
role handling and exact destination-cost decoding. A full-length synthetic
pipeline check uses generated observations, not competition data:

```text
python -m unittest discover -s tests
python smoke_model.py --engine contact --model models/external_v4.joblib
```
