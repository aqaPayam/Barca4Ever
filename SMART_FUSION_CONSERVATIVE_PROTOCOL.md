# Conservative learned fusion: declared follow-up

The first two learned gates improved short recovery but reduced overall CV
spell F1, so neither was selected or evaluated for regression acceptance.
Retain their weights and initial results. This follow-up changes only how
the learned probability estimate is blended back toward V4.1. It reuses the
same fit-only-WR9 gates, base predictions, fold exclusions and fixed decoder.

Before inspecting these results, declare four variants for each of the joint
and factorized gates (eight candidates total):

- shrink25: 25% gate prediction plus 75% V4.1 prediction.
- shrink50: 50% gate prediction plus 50% V4.1 prediction.
- disagree: use the learned estimate only where the component top-label sets
  do not overlap; otherwise keep V4.1. Sets handle ties symmetrically.
- boundary: use disagreement times the maximum state-onset score within
  +/-12 frames as the gate blend strength. This is a fixed half-second window,
  not a parameter selected from match labels. Onset scores are uncalibrated.

The probability estimate is always a convex blend. No weights or scalers are
refitted. CV selects among candidates that improve spell F1 and do not lose
short hits in both clean/noisy scenarios. Only one selected candidate then
undergoes the same clean/noisy regression gates on WMX/WN1. Do not select another
candidate after regression. WR9 fitting and development-reuse limitations from
SMART_FUSION_PROTOCOL.md apply; CV is used for architecture/variant selection,
not presented as independent blind testing. Keep all negative evidence and
preserve V4.1 unless all required checks pass. No competition inputs are read.
