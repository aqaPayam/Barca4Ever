# Predeclared composite: observability neural input plus directed boundaries

This optional composite uses no new learned weights beyond the independently
declared HZ25_PROTOCOL.md and boundary_refinement_protocol.md experiments.
Declare before inspecting composite results. Base trees and all input/scoring
seeds stay unchanged. No competition data or prediction contents are used.

Use only the neural rate/mixing weight selected on WR9 in hz25_selection.json.
With this neural emission mixture, evaluate directed-boundary blending alpha
0,0.5,1 on WR9. Alpha0 is the selected new neural candidate with old boundaries.
Consider the composite only if it strictly improves WR9 spell F1 without losing
short hits versus both V4.1 AND that neural-only candidate. Rank eligible
choices by mean spell F1, macro F1 breaking ties. No CV/regression is used to
choose alpha or rate. If no choice qualifies, retain the independent candidates.

For any selected composite, each original fold must use ONLY its own fine-tuned
neural checkpoint, tree classifier, old destination detector and new directed
detector. No evaluated match may be fitted by any component or initializer.
Compare exact V4.1 controls; higher spell F1 and no short-hit loss are required
in BOTH clean/noisy CV and reused WMX/WN1 regression. Confirm only the WR9 choice,
no fallback search after regression. Retain all results, including failures.

This remains seven-match external development, not a fresh blind benchmark or
proof of professional-system superiority. Neural input fixes are an original
paper-informed extension; the directed detector is not the original ASRF model.
