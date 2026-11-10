# Smarter model combination versus V4.1

Two bounded, learned reliability combinations were implemented and declared
before evaluation. Joint learns one neural weight; factorized learns separate
weights for dead probability and conditional team possession. Both use
model confidence/agreement and tracking visibility, ball-gap and contact
geometry. No player identity, clock, match ID or label is an input.

WR9 now fits the combination and its scaler. The base weights never fit
WR9, but WR9 previously selected architecture/checkpoints, so these are
reused development predictions rather than strict new blind cross-fitting.
No WR9 fitting score is claimed as a generalization improvement.

Each original whole-match fold fits its own gate on WR9 predictions from
only that fold base model. The evaluated matches are excluded from every
component fitting step. Thus no gate uses predictions from a model fitted
on its evaluation match. V4.1 controls reproduce exactly, with the same
observations, corruption seeds and fixed 16/8 state-boundary decoding.

CV selects the candidate subject to clean/noisy spell-F1 improvement and
no short-hit loss. Only that candidate is checked on the same WMX/WN1
reused regression. Protocol: SMART_FUSION_PROTOCOL.md.

| Evaluation | Candidate | Tracking | V4.1 spell F1 | Candidate spell F1 | V4.1 short | Candidate short |
|---|---|---|---:|---:|---:|---:|
| Whole-match development CV | joint | clean | 0.7105 | 0.7061 | 191/463 | 197/463 |
| Whole-match development CV | joint | noise | 0.5329 | 0.5247 | 79/463 | 87/463 |
| Whole-match development CV | factorized | clean | 0.7105 | 0.7052 | 191/463 | 198/463 |
| Whole-match development CV | factorized | noise | 0.5329 | 0.5277 | 79/463 | 89/463 |

Selected on CV: none.
External acceptance: failed; V4.1 remains the active model.

All seven licensed IDSSE games have appeared in development. These are
not nested independent tests, statistical-significance claims, professional
benchmarks or competition scores. WR9 has a different fitting role here
than in V4.1. The same external-data budget is used, with no extra matches.
No supplied competition match, video, labels or prediction quality is used.

Tests verify analytic gradients numerically, correct probability sums,
H/A exchange invariance and learning from a synthetic reliability signal.
All trials are retained as research evidence. Only candidates that pass
the predefined comparisons may change the active prediction program.

Reproduce in a fresh experiment directory after V4.1 fold preparation:

```text
python -m unittest tests.test_smart_fusion -v
python research_smart_fusion.py --stage crosscheck
python research_smart_fusion.py --stage regression
python report_smart_fusion.py
```

Skip regression if no candidate passed CV. Data licence and attribution:
EXTERNAL_DATA.md (IDSSE, DFL/Sportec, Bassek et al. 2025, CC BY 4.0).
