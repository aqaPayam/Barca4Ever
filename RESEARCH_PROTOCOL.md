# Architecture search: purpose, sources and evaluation rules

Goal: strongest tested possession-segment predictor under the available external
data, measured primarily by organizer spellF1. No claim of global optimality or a
competition leaderboard score. V2 stays active until a replacement passes checks.

## Primary research sources

1. Borghesi, Costa, Morra and Lamberti (2023), *Using Temporal Convolutional Networks
   to estimate ball possession in soccer games*, Expert Systems with Applications
   223, 119780. https://doi.org/10.1016/j.eswa.2023.119780
   Institutional record: https://iris.polito.it/handle/11583/2976470.
   Studies the exact dead/home/away problem using trajectory TCNs and also studies
   decomposing dead/alive and team control. Their confidential data and reported
   accuracy cannot be compared with our external spellF1. We are not reproducing
   their original per-object architecture or claiming access to their weights.
2. Abu Farha and Gall (CVPR 2019), *MS-TCN: Multi-Stage Temporal Convolutional
   Network for Action Segmentation*. https://arxiv.org/abs/1903.01945.
   Motivates dilated residual temporal convolutions without temporal pooling to
   retain frame resolution and context. Our implementation is compact, single-stage,
   built from scratch and applied to geometry rather than image descriptors.
3. Ishikawa et al. (WACV 2021), *Alleviating Over-Segmentation Errors by Detecting
   Action Boundaries*. https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html.
   Motivates separate boundary evidence. Our adaptive Viterbi costs and joint TCN
   head are our own designs, not an implementation of the published ASRF model.
4. *Automatic event detection in football using tracking data* (Sports Engineering
   2022). https://doi.org/10.1007/s12283-022-00381-6.
   Motivates distinguishing close contact from genuine control using ball motion
   and context. A simple possession-zone rule cannot directly solve the challenge's
   controlled team possession and noisy tracking; V2 already includes contact/motion.

No paper-specific implementation code or pretrained weights are copied. Third-party
libraries retain their own licences. External data remains IDSSE, CC BY 4.0.

## Fixed candidate families

- V2: weighted boosted classifier plus fixed-cost Viterbi, retained as control.
- Hierarchical boosted classifiers: dead/alive first, then home/away conditional
  on play being alive. Fixed tree sizes; penalty grid 2/4/8/16/32.
- Learned boundary classifier plus V2 emissions: boundary supervision uses only
  external changes, widened by three frames. Transition costs decrease where a
  change is predicted. Base penalties 8/16; boundary scales 1/4/16.
- Compact original TCN: 59 invariant inputs, 32 channels, five residual layers,
  dilations 1/2/4/8/16, no temporal pooling, possession and boundary heads. External
  crops train at 25 fps with central-frame loss and a 64-frame context margin.
  AdamW, fixed seed, short-spell weights, 1200 steps; compare checkpoints at
  400/800/1200 using penalty 2/8/16. CPU runtime limits the initial experiment;
  GPU training is not necessary to establish whether this small architecture helps.
- One simple fusion test after individual models: 25% and 50% TCN probability
  mixed with V2 probabilities, penalties 4/8. This is not assumed to help.

## Evaluation and limits

Train on the original four training matches; validation selection only on J03WR9,
using identical clean and deterministic-noisy inputs for every candidate.
Select by mean spellF1; report macroF1 and short-spell recall too. None of this is
a new blind test: all seven IDSSE matches have appeared in prior development.
For any promising replacement, retrain both control and candidate within the same
two-match cross-validation folds before promotion. No evaluated outer-fold match
may appear in that fold's fitting data. Freeze design/parameters before inspecting
those fold results. A genuinely fresh blind test needs new licensed matches.

Do not open the competition video or prediction file to select models. Do not
read its ground truth. Do not adjust parameters based on competition inference.
V2, its recorded results and its existing submission remain intact during research.
Failures and negative comparisons must be retained, not omitted from the report.
