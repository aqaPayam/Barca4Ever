# Barca4Ever - Initial Challenge technical report (v4.1)

## Approach

We estimate home possession (H), away possession (A) and dead ball (D)
from tactical tracking using a contact network, a geometric boosted
classifier, and an externally learned state-change detector. Unknown long
tracking outages receive X. All fitting and parameter selection use
licensed external matches. The supplied game is inference-only.

The ATD XML maps tracking slots to teams; TXT observations are converted
from normalized XY into metres about the pitch centre. ATD record 1 maps
to output frame 0. The loader rejects missing, duplicate, reordered and
malformed rows and inconsistent slot counts. Team mapping uses metadata,
not names or appearance. No clock, match ID or player ID is a feature.

The geometric classifier has 59 distance/motion/visibility/context inputs
and 200 histogram boosting iterations, 23 leaves, learning rate 0.07, L2=3
and minimum leaf size 60. Its externally trained probabilities are mixed
with the neural probabilities at neural weight 0.50.

The neural architecture extends Borghesi et al. (2023), ESWA 223, 119780,
https://doi.org/10.1016/j.eswa.2023.119780. Independent shared trajectory
TCNs feed separate ball-query attention and dead/alive and conditional
home/away FFNs. Six kernel 2 blocks with dilations 1/2/4/8/16/32 use width 64,
dropout 0.1 and batch normalization. Two ELU FFNs have 64/32 hidden units.
A 64-step window at 16 Hz predicts element 48, using future context offline.

Our extensions add ball-relative contact/motion inputs, a target-frame
geometry MLP skip, and learned distance-biased attention. The 17-channel
layout includes XY/velocity, object/team flags, outside/missing indicators
and relative XY/velocity/distance/validity. The goalkeeper channel is
constant zero after a separate external fine-tuning phase, so production
requires only tracking and team metadata. It does not guess player roles.
Additional missing slots are masked and all observed actors are retained.
Neural posteriors are interpolated to every 25 Hz frame before mixing.

The selected boundary type is state. A 140-iteration, 15-leaf
boosted classifier uses 177 geometry/context difference inputs, rate 0.08,
L2=3 and minimum leaf size 60. External changes expanded by three frames
provide targets; positive frames have weight 10. Destination-specific
H/A/D onset scores or a shared binary score are used as selected externally.
The scores are uncalibrated. Viterbi maximizes summed log probability
minus a cost 16/(1+8*b[t,destination]) on each switch.

## Assets, missingness and possession interpretation

Only supplied tactical ATD tracking and metadata are used. Tactical and
broadcast video, broadcast tracking, Smart Tagging and homography are not
used. Predictions are not checked against video. No supplied data or
extracts are sent to hosted AI models by the inference programs.

Internal ball gaps up to 2 seconds may be interpolated, without extrapolation.
Longer gaps remain missing and observations carry missingness indicators.
An absent ball is not automatically dead. Complete player-tracking outages
of at least 10 seconds reset blocks and receive X. This is a fixed engineering
rule, with no manually specified intervals from the supplied game.

Control, loose-ball and deflection behavior are learned from external
provider labels. The decoder favours continuity but permits quicker changes
near learned boundaries. Restarts are D-to-team transitions; there is no
independent loose-ball class or manually chosen restart time. These rules
may differ from Metrica controlled-possession definitions.

## External data, training and selection

Only IDSSE was used: DFL/Sportec Solutions data published by Bassek, Rein,
Weber and Memmert, Scientific Data 12, 195 (2025),
https://doi.org/10.1038/s41597-025-04505-y. Dataset:
https://doi.org/10.6084/m9.figshare.28196177, CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/. Preserve this attribution
with the trained artifact. We use the source BallPossession/BallStatus:
active/home -> H, active/away -> A, inactive -> D, unknown -> X. We did not
hand-label these games or use human-produced labels about the competition.

Train: J03WOH, J03WOY, J03WPY, J03WQQ. Validation: J03WR9. Original
V1 held-out tests: J03WMX/J03WN1. All seven have since appeared in
development; later tests are explicitly reused regression. Both halves
stay within the same split. Whole-match CV fits WOH/WOY and evaluates
WPY/WQQ, then reverses. Every learned component is fitted only on its
fold fitting matches, including both neural training stages.

The contact network trains eight epochs of 100k windows, then fine-tunes
two epochs with keeper flags zero. Adam learning rates are 0.001 and 0.0001,
with cosine decay, batch 256 and seeds 20261107/20261112. Initializers must
have the same fitting-match list as fine-tuning. Training includes clean
and label-independent noisy observations and team swaps. Noise adds XY
jitter, 5% player drops and 15-84-frame ball gaps. These simulate stress,
not the exact challenge tracking error process.

Training uses 75% uniformly sampled eligible frames and 25% uniform team
spells followed by a random frame inside each. Possession loss weights are
clip(sqrt(250/duration_frames),0.5,4); dead frames retain weight 1. Home/away
loss is zero on dead frames. These changes increase brief-control exposure.

The paper-based reimplementation, optimization changes, contact extension
and role-free adaptation remain separate experiments. Original author
weights/code were unavailable and some settings were unspecified; this is
not a reproduction of the authors private-dataset score. Six fusion
settings (neural weights 0.25/0.5/1, binary/destination boundaries) are
selected on WR9 by spellF1. Negative trials and source/initializer hashes
are retained. No competition labels, video or output quality are used.

## Matched external evaluation

The exact organizer scorer is applied to external labels. Mean spellF1
weights halves equally. Short recall counts true possessions under 2 seconds
matched to the correct team at IoU 0.5. Clean/noisy cases share labels and
are not independent observations. The V3 control reproduced exactly in CV.

| Evaluation | Tracking | V3 spellF1 | V4.1 spellF1 | V4.1 macroF1 | V3 short | V4.1 short |
|---|---|---:|---:|---:|---:|---:|
| Cross-validation | Clean | 0.6245 | 0.7105 | 0.9373 | 111/463 | 191/463 |
| Cross-validation | Simulated noise | 0.4178 | 0.5329 | 0.8760 | 61/463 | 79/463 |
| Reused-test regression | Clean | 0.5896 | 0.6808 | 0.9299 | 81/362 | 128/362 |
| Reused-test regression | Simulated noise | 0.4079 | 0.4968 | 0.8607 | 46/362 | 64/362 |

These are development comparisons, not nested validation or fresh blind
tests. Architecture acceptance uses this evidence; no statistical-significance
or professional-system superiority claim is made. Complete trial history
and per-half results are in PAPER_2023_RESULTS.md and external_reports/.

## Limitations and reproduction

Seven games and one tracking provider provide limited independent evidence.
Automatic challenge tracking has identity switches, missingness and geometric
errors not reproduced fully by synthetic noise. Within-window identity
changes can hurt the trajectory branch. XY alone can miss whistles, body
orientation, aerial contacts and stationary restarts. Short possessions
remain difficult. No actual competition score or rank is known.

Python 3.12; install pinned requirements-contact.txt. CPU inference is
canonical; GPU is optional for research. The trusted artifact includes
neural weights as NumPy arrays plus the tree models, with exact conversion
checks and model/source hashes. Keep the original artifact for reproduction.

```text
python infer_contact.py --model models/external_v4_1.joblib --data-dir /local/bundle --out Barca4Ever.csv
python evaluation/validate_submission.py Barca4Ever.csv
```

The bundle root contains tactical/ with the supplied XML/TXT. The Windows
wrapper verifies the model checksum and stores validator details locally.
The output has 199,319 indexed rows; its manifest records model/output hashes.
45 synthetic/unit checks and full-length synthetic CLI validation were
completed before local competition inference. Validation checks file format,
not football quality. No fitting or tuning occurs during inference.

All competition inputs, intermediate data, predictions and validation logs
remain outside the public code archive. Reproduction details and full
architecture/settings are in V4_MODEL_GUIDE.md and PAPER_2023_PROTOCOL.md.

## V4.1 decoder refinement

All V4 neural and tree weights are unchanged. After an automatic external
validation diagnostic, six decoder settings were declared before testing:
base costs 8/12/16 and boundary scales 4/8. The single WR9-selected setting
was 16/8. It passed clean/noisy spell-F1 and short-recovery acceptance
checks in both whole-match comparison and reused-test regression. Every
V4 control reproduced exactly. Small gains are development evidence,
with no claim of statistical significance. V4 artifacts and outputs are
preserved. See V4_DECODER_PROTOCOL.md, V4_DECODER_RESULTS.md and
V4_1_MODEL_GUIDE.md for the refinement and its reproduction commands.
