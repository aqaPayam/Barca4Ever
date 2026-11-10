# Residual missing-ball adaptation with protected legacy weights

Declare before fitting. This is a NEW Pareto-anchor experiment, not a relaxation
of acceptance rules for any previously failed model. Keep V4.1 and all original
files intact. Only the same licensed external IDSSE matches are used.

Preserve the exact legacy11-channel short-interpolated input and relative17
features, including their16Hz sampling/velocities and original preprocessing.
Zero keeper roles as in the existing role-free model. Append3 channels to all
objects: raw ball-MISSING flag, time since and until a raw ball observation,
capped at10seconds and divided by10. These are all zero for fully observed
ball sequences. Use the same source frame indices, raw data and original
label-independent corruption seeds. No label enters the input.

Initialize from the corresponding role-free17-channel checkpoint through the
existing function-preserving20-channel expansion. Freeze every old parameter
and buffer. Only the newly zero-initialized3 input columns in encoder.stem.weight
and instant.0.weight may learn. Mask all gradients in their first17 columns;
restore those protected columns after every optimizer step as an additional
byte-exact safeguard. No weight decay. Keep model.eval() throughout fitting so
batch normalization buffers and dropout never change. Record protected tensor
byte checksums and compare EVERY old weight/buffer before/after training.

Fixed2epochs of100k windows per epoch, fresh75% uniform frames/25% uniform team
spells EACH epoch, existing spell weights and H/A swap augmentation. Batch256,
Adam5e-4, cosine decay to1e-4, seed20261117. If GPU memory fails, stop and restart
from the original initializer/RNG with batch128; record that memory contingency
before restarting. No checkpoint or mathematical hyperparameter selection.

Keep tree mixture0.5 and state-boundary16/8 decoding fixed. On WR9 require
clean spell F1 >= V4.1, strictly higher noisy spell F1 and no short-hit loss in
either scenario. The frozen legacy function is mathematically retained when
all extra channels are zero;17/20-channel finite-precision discrepancies are
tested and reported rather than silently changing acceptance.

Only if WR9 qualifies, adapt each CV fold from its OWN corresponding role-free
checkpoint using only the same fitting-match list. Every learned component
excludes evaluated fold matches. Require the SAME Pareto rule in original
whole-match CV and then WMX/WN1 reused regression: clean spell >=, noisy spell >,
short hits >= in both scenarios. No fallback selected after inspection.

All seven matches are reused development data, not a fresh blind benchmark.
Source/model/input hashes, memory settings, histories and proof of unchanged
old tensors are retained. No competition input, video, label, prediction or
quality check is used. No model/default/production changes without complete
qualification and separate production checks. Producer missing20 mode is
deferred until qualification.
