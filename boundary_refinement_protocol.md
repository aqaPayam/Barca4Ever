# Transition-pair boundary refinement

Declare before fitting: preserve V4.1 emissions, base cost16, scale8, every
existing artifact and prediction. This research reads licensed external IDSSE
only, never competition tracking, video, ground truth or prediction contents.

Ishikawa et al. (WACV2021), Alleviating Over-Segmentation Errors by Detecting
Action Boundaries, motivates separate boundary evidence. This is an original
transition-specific classifier and decoder, not their ASRF implementation.
Football event/touch timing research motivates retaining signed local motion
and 25-Hz resolution. Exact-task possession context: Borghesi et al. (2023).

The old boundary classifier detects which state begins but ignores which state
it leaves. Its features use absolute differences at +/-3 frames. This candidate
predicts background plus six ordered transitions H->A,H->D,A->H,A->D,D->H,D->A.
It adds signed before/after geometry at +/-2,6,12 frames to the current59 inputs.
The selected26 contact/motion/visibility channels give293 total inputs. These
are label-free and use no expert probability outputs, avoiding in-sample
expert-prediction supervision during boundary fitting.

Targets use the nearest external change within two frames; ties prefer the
earlier change. Unknown labels are excluded. Training every fifth valid frame,
clean/noisy, H/A-swap augmentation. Positive transition frames receive weight20.
Histogram gradient boosting:100 iterations,11 leaves,rate0.08,L2=5,minleaf80,
early_stopping=False,seed20261116. These settings are fixed before results.

At inference a score matrix b[t,previous,next] lowers only that directed switch
cost:16/(1+8*b). Blend new pair evidence and V4.1 destination evidence with
alpha0,0.25,0.5,1. Select on external WR9 only, mean organizer spellF1 then
macroF1; alpha0 is the exact V4.1 control. No new network weights are fitted.

If validation improves spellF1 and avoids short-hit loss, freeze alpha before
the original whole-match folds. Refit boundary classifier only on that fold's
two fitting matches. All other base-model and decoder components are the same
fold-specific frozen components and seeds as V4.1. Confirm the selected design
on WMX/WN1 reused regression only if it passes both clean/noisy CV gates. Each
comparison requires higher spellF1 and no loss of short hits. Controls must
reproduce V4.1 exactly. All7 matches are reused development data, not fresh
blind tests. Failed trials are retained and must not alter the active model.
