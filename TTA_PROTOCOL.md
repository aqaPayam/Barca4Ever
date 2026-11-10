# Frozen-network symmetry averaging

Declare three external WR9 trials before scoring: NN joint-posterior average
of original and restored H/A-swapped inputs; original and180-degree pitch
reflection; and original plus both transformed predictions. Keep the same
V4.1 tree mixture0.5, state-onset detector and16/8 decoding. No weights fit.

Transform original11-channel inputs first and then derive relative17 channels.
Zero keeper roles. H/A swap exchanges the17-slot team groups and role flags,
then restores output H/A columns. Reflection negates valid coordinates and
valid velocities, preserving missing -2 sentinels, roles and visibility. Teams
remain unchanged. Restore joint posterior labels before averaging, and average
at the full25Hz output rate after each path's normal marginal upsampling.

The base role-free network and all original files/artifacts remain unchanged.
Only licensed external feature caches and original component caches are read.
No competition data, video, labels, predictions or output quality are accessed.
CPU-only inference, OMP/BLAS/Torch2 threads; GPU is reserved for other training.

On WR9 choose the highest mean spell F1 (macro tie-break) among candidates with
strictly higher spell F1 and no short-hit decline versus V4.1 (0.6442778883,
62/188). Only that candidate proceeds to the original whole-match folds and
reused WMX/WN1 regression. Every learned component stays fold-specific, with
the corresponding role-free checkpoint and tree/onset caches. No new fitting.

Controls must reproduce V4.1 exactly, including macro F1 and duration recalls.
CV and regression each require spell-F1 improvement and no short-hit loss in
BOTH clean/noisy scenarios. No fallback is chosen after inspection. If a stage
fails, preserve V4.1. All seven matches are reused development data; no fresh
blind-test, statistical-significance or professional-superiority claim is made.

Retain all original baselines and every trial. Prediction caches are stored
only in external_cache/tta, with source-model/input/transform fingerprints.
