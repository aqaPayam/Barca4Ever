# Dense physics boundary research

This is a separate research candidate, preserving all first transition-pair
and V4.1 artifacts. The first candidate narrowly failed its noisy CV gate
despite improving clean F1 and both short-hit counts. No production source
or competition prediction is modified by these research programs.

## Declared method

Append28 soft XY touch-physics inputs to the293 signed temporal geometry
inputs, giving321. The source helper produces30 channels; the two roster-
fraction channels are excluded by FEATURE_NAMES because they change under
extra missing track-slot padding. The selected28 inputs passed an explicit
padding-invariance check. TOUCH_PHYSICS_PROTOCOL.md gives the exact formulas
and primary-paper basis (ELASTIC and possessor/touch recognition).

Raw ball acceleration, direction changes, contact valleys, approach/recession
and gap-validity evidence do not bridge missing raw-ball windows. These are
continuous unfitted features, not hand-produced labels or event times.

Dense sampling includes all valid frames within12 frames of an external
change, and every fifth frame elsewhere. Positive weight4, nearby background
weight0.2 and remote sampled background weight1 approximately preserve the
old one-in-five sampling mass while retaining very short changes and their
nearby negatives. Unknown labels are excluded; clean/noisy and H/A-swap
augmentation remain. The seven-class directed detector and 16/8 decoder
are otherwise the same100-iteration,11-leaf model as the first candidate.

Alpha0/.25/.5/1 is chosen only on external WR9, before any fold evaluation.
The alpha0 control must reproduce V4.1 exactly in all metrics.

| Stage | Tracking | Alpha | Spell F1 | Macro F1 | Short recovery |
|---|---|---:|---:|---:|---:|
| validation | clean | 0 | 0.7415 | 0.9480 | 39/94 |
| validation | noise | 0 | 0.5471 | 0.8850 | 23/94 |
| validation | clean | 0.25 | 0.7363 | 0.9473 | 37/94 |
| validation | noise | 0.25 | 0.5484 | 0.8845 | 23/94 |
| validation | clean | 0.5 | 0.7388 | 0.9475 | 37/94 |
| validation | noise | 0.5 | 0.5515 | 0.8848 | 23/94 |
| validation | clean | 1.0 | 0.7410 | 0.9474 | 42/94 |
| validation | noise | 1.0 | 0.5542 | 0.8873 | 25/94 |
| crosscheck | clean | 0 | 0.7105 | 0.9373 | 191/463 |
| crosscheck | noise | 0 | 0.5329 | 0.8760 | 79/463 |
| crosscheck | clean | 1.0 | 0.7149 | 0.9375 | 198/463 |
| crosscheck | noise | 1.0 | 0.5331 | 0.8760 | 86/463 |
| regression | clean | 0 | 0.6808 | 0.9299 | 128/362 |
| regression | noise | 0 | 0.4968 | 0.8607 | 64/362 |
| regression | clean | 1.0 | 0.6844 | 0.9314 | 132/362 |
| regression | noise | 1.0 | 0.5020 | 0.8614 | 67/362 |

## Acceptance and limitations

- validation: passed {'validation_spell': True, 'validation_short': True}
- crosscheck: passed {'False_spell': True, 'False_short': True, 'True_spell': True, 'True_short': True}
- regression: passed {'False_spell': True, 'False_short': True, 'True_spell': True, 'True_short': True}

Every acceptance comparison requires higher spell F1 and no loss of short
hits. Full fitting uses only WOH/WOY/WPY/WQQ; promising candidates refit only
on each original two-match fold. WMX/WN1 reused regression is reached only
after both clean/noisy CV gates pass. All seven matches are reused
development data, not a new blind benchmark or significance claim.

The geometry and sampling changes form one declared bundle; gains cannot
be attributed to either alone without a new ablation. No supplied match,
video, possession labels or prediction-quality checks enter this research.
17 related synthetic checks passed:6 directed-boundary,2 dense-physics
integration and9 soft-physics checks. Data attribution: EXTERNAL_DATA.md
(licensed IDSSE, DFL/Sportec, Bassek et al.2025, CC BY4.0).
