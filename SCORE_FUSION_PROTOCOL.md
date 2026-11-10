# Spell-score fitted reliability rule: final family in this cycle

The linear, conservative and nonlinear frame-loss gates failed overall CV
acceptance. Keep all evidence. This final predefined family fits the actual
spell-score objective on WR9, not per-frame likelihood or expert regret.

Use neural weight clip(0.5 + a*c + b*m,0.1,0.9), where
c=(entropy(tree)-entropy(neural))/log(3), and
m=(1-ball_observed)*(1-max(tree_dead,neural_dead)). Features are label-free,
H/A invariant, and vary per frame. The second term permits a different mixture
when the ball is missing and the models still consider play alive.

For EACH fold independently select a,b from {-0.2,-0.1,0,0.1,0.2} squared,
using only that fold's WR9 calibration predictions and labels. Select mean
organizer spell F1 over four calibration cases, then macro F1, then smallest
squared parameter magnitude. The equal mixture is in the grid. Calibration
scores are fitting scores, not evidence of generalization. Keep base weights
and 16/8 decoding fixed. Fit full-model a,b separately on full-model WR9 only.

Compare this SINGLE fitting algorithm with V4.1 on the original whole-match
folds. Require higher spell F1 and no short-hit loss in both clean/noisy cases.
Only if it passes, evaluate its full-model fit on WMX/WN1 with the same gates.
No alternative may be selected from regression. Fold/source fitting exclusions
and WR9 reuse limitations of SMART_FUSION_PROTOCOL.md apply. No competition data.
If it fails, finish this experiment cycle with V4.1 active and report failure.
