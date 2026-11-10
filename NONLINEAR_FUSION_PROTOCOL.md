# Nonlinear reliability follow-up

The two linear gates and eight conservative variants all failed clean/noisy
overall CV spell-F1 acceptance, despite better short recovery. Preserve those
results and V4.1. Declare this different fitting objective before evaluating it.

On the SAME WR9 calibration rows and the SAME fold-specific base predictions,
fit a histogram gradient boosting regressor to the normalized true-label
probability advantage (neural_y-tree_y)/(neural_y+tree_y+0.1). Fit every fifth
valid frame using existing spell weights normalized to mean one. Reliability
inputs are unchanged and H/A invariant. Parameters: 100 iterations, 7 leaves,
minimum leaf400, learning rate0.05, L2=10, early_stopping=False, seed20261115.

Two declared strengths: neural weight = 0.5 + amplitude*clip(prediction,-1,1),
amplitude0.25 or0.4. Thus changes remain close to the equal mixture when no
clear average model advantage is learned. Use the same 16/8 decoder unchanged.

Apply the SAME whole-match CV selection and single-candidate reused-regression
confirmation gates. In both clean/noisy cases require higher spell F1 and no
loss of short-spell hits. If neither passes, preserve V4.1. No selection from
regression alternatives. Original WR9 role-change, exclusions and development
reuse limitations in SMART_FUSION_PROTOCOL.md apply. No competition access.
