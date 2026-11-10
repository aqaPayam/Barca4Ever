# Transition-pair boundary refinement

Research-only candidate; V4.1 remains frozen until all acceptance and
production checks are complete. No competition inputs or predictions
are read by the research program.

## Method

V4.1 predicts only the incoming state at a potential change. This candidate
predicts six ordered source/destination transitions plus background. A
separate score controls H-to-A versus D-to-A, for example. Signed before/
after geometry at +/-2,6,12 frames supplements the original 59 inputs,
for293 total. The original control used absolute differences at +/-3.

The classifier is a100-iteration,11-leaf histogram gradient booster with
rate0.08,L2=5,minleaf80,seed20261116. Changes within two frames provide
targets; nearest change wins, with earlier changes winning ties. Positive
frames have weight20. Training samples every fifth valid frame, using
clean/noisy observations and H/A swap augmentation. No classifier or
network probability output is a feature, so training the boundary model
does not rely on in-sample expert predictions.

Positive-label weighting means the directed outputs are cost-control scores,
not calibrated probabilities of a real-world transition.

A directed-score Viterbi decoder retains base16/scale8. New scores blend
with frozen V4.1 destination scores at alpha0,.25,.5,1, selected on WR9.
Alpha0 is mathematically and empirically the exact V4.1 control.

WR9 selected alpha: 1.0.

## External comparison

| Stage | Tracking | Alpha | Spell F1 | Macro F1 | Short recovery |
|---|---|---:|---:|---:|---:|
| validation | clean | 0 | 0.7415 | 0.9480 | 39/94 |
| validation | noise | 0 | 0.5471 | 0.8850 | 23/94 |
| validation | clean | 0.25 | 0.7387 | 0.9472 | 38/94 |
| validation | noise | 0.25 | 0.5442 | 0.8844 | 23/94 |
| validation | clean | 0.5 | 0.7418 | 0.9479 | 39/94 |
| validation | noise | 0.5 | 0.5500 | 0.8852 | 24/94 |
| validation | clean | 1.0 | 0.7423 | 0.9482 | 40/94 |
| validation | noise | 1.0 | 0.5561 | 0.8879 | 25/94 |
| crosscheck | clean | 0 | 0.7105 | 0.9373 | 191/463 |
| crosscheck | noise | 0 | 0.5329 | 0.8760 | 79/463 |
| crosscheck | clean | 1.0 | 0.7126 | 0.9377 | 192/463 |
| crosscheck | noise | 1.0 | 0.5325 | 0.8762 | 84/463 |

## Acceptance and limits

- validation: passed {'validation_spell': True, 'validation_short': True}
- crosscheck: failed {'False_spell': True, 'False_short': True, 'True_spell': False, 'True_short': True}

Training uses only the four licensed external IDSSE training games; WR9
selects alpha. Promising designs are fitted again on each original two-match
fold and compared on the other two matches. WMX/WN1 reused regression is
only reached after both clean/noisy CV gates pass. Every comparison demands
higher spell F1 and no short-hit loss, and V4.1 controls reproduce exactly.

All seven games have appeared in development. This is not a new blind
benchmark, statistical-significance claim or professional-system ranking.
The method is an original refinement inspired by boundary supervision
(Ishikawa et al., WACV2021) and possession/event timing research; it is not
their ASRF architecture. Sources and fixed choices are recorded in
boundary_refinement_protocol.md. IDSSE attribution: EXTERNAL_DATA.md.

Six synthetic checks passed: target direction and unknown handling, H/A
mapping and nearest-boundary ties, signed temporal direction, exact V4.1
control recovery, ordered-edge probability routing, and brute-force
verification of the decoder objective.
