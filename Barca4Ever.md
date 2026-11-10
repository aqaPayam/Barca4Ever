# Barca4Ever - Initial Challenge technical report (V5)

## Approach and inputs

We estimate home possession H, away possession A and dead ball D from
tactical player/ball XY tracking and team assignments. Long complete player
tracking outages of at least 10 seconds receive X and split inference blocks.
All fitting, hyperparameter selection and quality evaluation use licensed
external data only. The supplied match is used solely for local inference.

The unchanged strict ATD loader maps slots to teams, converts normalized
XY to pitch-centred metres, and maps ATD record 1 to output frame 0. It
rejects missing, duplicate/reordered/malformed rows and inconsistent slot
counts. No player name, match ID, event outcome, clock or possession label
enters a feature. Video, Smart Tagging, broadcast tracking and homography
are not used. XY alone cannot identify every whistle or aerial contact.

## Possession probabilities

The geometric branch is the preserved V2 histogram gradient boosting
classifier: 59 geometry/motion/visibility inputs, 200 iterations, 23 leaves,
learning rate 0.07, L2=3, minleaf 60. Its probabilities are averaged equally
with a contact-network branch extending Borghesi et al. (2023),
https://doi.org/10.1016/j.eswa.2023.119780. Original author weights and
confidential data were unavailable; this is an independently trained
adaptation with declared assumptions, not their reported model score.

The shared per-object trajectory TCN has width 64, a 1x1 stem and six causal
kernel 2 blocks with dilations 1/2/4/8/16/32, dropout 0.1 and batch normalization.
Separate ball-query attention and ELU 64/32 FFNs predict dead probability
and conditional home probability. Home loss is zero on dead labels.
Relative ball geometry/motion, a target-frame MLP and learned distance-biased
attention add direct contact evidence. Missing objects are masked; every
observed tracking actor is retained in padded groups, without a 36-object cap.

V5 has 20 input channels. The original 17 are recomputed with source-rate
short-gap interpolation and velocities divided by actual elapsed source
time, correcting rounded 16 Hz interval variation. Three extra channels
retain raw ball visibility and time since/until raw observation, capped
at 10 seconds and scaled by 10. Filled geometry remains separate from raw
visibility. Keeper roles are always zero; player role metadata is unnecessary.

Neural inputs use 16 Hz, 64 samples and target 47 (48th element), about 4 seconds
of trajectory with 1 second of future trajectory. Gap-age features may use
up to 10 seconds of future observation evidence, so this is offline inference.
The native 25 Hz candidate was evaluated and rejected on external validation.
Marginals are interpolated to every 25 Hz frame; joint probabilities are
[(1-dead)*home,(1-dead)*(1-home),dead].

## Directed transition decoding

A new classifier predicts background plus six ordered transitions:
H->A,H->D,A->H,A->D,D->H,D->A. It uses 293 label-free inputs: 59 base
features plus before/after/signed-difference context for 26 selected features
at +/-2, 6, 12 frames. Histogram boosting uses 100 iterations, 11 leaves,
learning rate 0.08, L2=5, minleaf 80, seed 20261116. Targets are nearest external
changes expanded by two frames, earlier change breaking ties. Fitting
samples every fifth valid frame; positive targets have weight 20.

Viterbi maximizes summed log possession probability minus switch cost
16/(1+8*b[t,previous,next]). Unlike V4.1, each directed score lowers only
its corresponding switch cost. Scores are uncalibrated because positives
are weighted. WR9 selected pure directed evidence (alpha 1) for the final
composite. The old destination detector is retained in the artifact for
reproduction but contributes zero at this alpha. This is an original
paper-informed boundary design, not a reproduction of ASRF or ELASTIC.

Internal ball gaps up to 2 seconds may be interpolated without extrapolation;
longer gaps stay missing. Missing ball does not automatically mean dead.
Loose balls, deflections and restarts are learned through external H/A/D
labels and directed changes; there is no separate loose-ball class or
manually selected restart interval from the supplied game.

## External fitting and selection

IDSSE only: DFL/Sportec data, Bassek et al., Scientific Data 12, 195 (2025),
https://doi.org/10.1038/s41597-025-04505-y. Deposit:
https://doi.org/10.6084/m9.figshare.28196177, CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/. Preserve this attribution
with the trained artifact. BallStatus 0 maps to D; live BallPossession 1/2
maps to H/A; unknown maps to X. Labels are isolated from feature inputs.

Base fit: J03WOH/J03WOY/J03WPY/J03WQQ; validation: J03WR9; original
held-out games J03WMX/J03WN1 are now explicitly reused regression. All seven
have appeared in development. Whole-match CV fits WOH/WOY and evaluates
WPY/WQQ, then reverses; both halves stay together. Every neural initializer,
fine-tune, tree and directed detector excludes its evaluated fold matches.

V5 initializes from the corresponding frozen role-free checkpoint. Its
new input columns are zero-padded initially. It fine-tunes two epochs of
100k windows, batch 256, Adam 1e-4/cosine to 3e-5, seed 20261116. Each epoch
draws new references: 75% eligible frames and 25% uniform team spells. Loss
weights are clip(sqrt(250/duration_frames),0.5,4). Clean/noisy observations
and team swaps are included. Label-independent corruption reuses the
original jitter, 5% player drops and 15-84-frame ball gaps. Synthetic noise
does not reproduce all automatic-tracking errors or identity switches.

Rates 16/25 and neural mixture weights 0.25/0.5/0.75 were selected on WR9.
The composite tested directed weights 0/0.5/1 on WR9 before CV. Final rate 16,
neural weight 0.5 and directed weight 1 then passed the declared clean/noisy
CV and regression checks. Among fully qualifying methods, the composite
had the highest WR9 spell F1. Failed individual methods, alternate physics
and symmetry results are retained in MULTIAGENT_RESULTS.md. No released
PathCRF weights overlapping evaluated matches were used.

## Matched external results

Exact organizer spell F1 is primary, with mean halves weighted equally.
Short recovery counts same-team possessions under 2 seconds at IoU 0.5.
Clean/noisy scenarios share labels and are not independent samples.

| Comparison | Tracking | V4.1 spell F1 | V5 spell F1 | V4.1 short | V5 short |
|---|---|---:|---:|---:|---:|
| Whole-match CV | clean | 0.7105 | 0.7133 | 191/463 | 206/463 |
| Whole-match CV | noise | 0.5329 | 0.5344 | 79/463 | 80/463 |
| Reused regression | clean | 0.6808 | 0.6847 | 128/362 | 132/362 |
| Reused regression | noise | 0.4968 | 0.5140 | 64/362 | 71/362 |

These are small, repeated development gains, not statistical significance,
a fresh blind test, a professional-system comparison or a competition score.
Seven matches from one provider give limited independent evidence. Actual
tracking identity switches, body orientation and aerial contacts remain
limitations. Provider possession definitions may differ from Metrica.

## Reproduce and verify

Python 3.12, requirements-contact.txt. Canonical CPU inference uses 4 threads.
The trusted artifact contains NumPy neural state plus trained tree models,
with exact serialization checks and pinned model/source/evidence hashes.
No external training dataset is needed to regenerate the submission.

```text
python -m pip install -r requirements-contact.txt
python refined_infer.py --model models/external_v5.joblib --data-dir /local/bundle --out Barca4Ever.csv
python evaluation/validate_submission.py Barca4Ever.csv
```

The bundle contains tactical/BAR-ATH_tactical_atd.xml and its TXT. The
Windows wrapper verifies the artifact checksum and stores validator details
locally. The file has 199,319 indexed rows. Its manifest records model/output
hashes. 82 synthetic/unit tests, four saved-model external CPU comparisons
and full-length synthetic CLI/format checks precede local inference.
Format validation does not establish football quality. No fitting or
adjustment happens during inference, and predictions are not checked against
the supplied video or ground truth. No supplied data or extracts are sent
to hosted AI services. Private inputs, intermediates, outputs and validator
logs remain outside the public code package. V4.1 and older fallbacks are
preserved. Details: V5_MODEL_GUIDE.md and MULTIAGENT_RESULTS.md.
