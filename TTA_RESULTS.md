# Frozen-network symmetry averaging

Three candidates were declared in TTA_PROTOCOL.md before scoring: average
the original NN joint probabilities with restored H/A-swapped predictions,
with a180-degree pitch reflection, or with both. Tree mixing stays0.5 and
the state-boundary decoder stays16/8. Every model weight is unchanged.

Transform original11-channel inputs, preserve missing sentinels and roles,
then derive relative17-channel features. Restore H/A output columns before
averaging full25Hz joint posteriors. All new forward passes are CPU-only
with2threads. Existing original-path posteriors may be reused bit-exactly
from the verified external component caches; provenance is recorded in
external_reports/tta_original_reuse_audit.json.

## External WR9 selection

| Candidate | Mixed spell F1 | Short hits /188 | Clean spell F1 | Noisy spell F1 |
|---|---:|---:|---:|---:|
| v4.1 | 0.6443 | 62 | 0.7415 | 0.5471 |
| swap | 0.6468 | 62 | 0.7370 | 0.5566 |
| reflection | 0.6440 | 64 | 0.7397 | 0.5483 |
| both | 0.6481 | 64 | 0.7392 | 0.5569 |

Selected: both.
Selection requires strictly higher mixed spell F1 and no loss of short hits.
Only that selected candidate proceeds; there is no fallback after CV.

## Whole-match development comparison

| Model | Clean spell F1 | Noisy spell F1 | Clean short | Noisy short |
|---|---:|---:|---:|---:|
| v4.1 | 0.7105 | 0.5329 | 191/463 | 79/463 |
| both | 0.7075 | 0.5376 | 186/463 | 81/463 |

Stage acceptance: failed; preserve V4.1.

## Reproducibility and limits

The original whole-match folds retain their own role-free NN and tree/onset
models, with evaluated games excluded from all fitting. Controls must match
V4.1 exactly in spell F1, macro F1 and duration recalls. These are the same
seven licensed external IDSSE games already used in development, not fresh
blind tests, evidence of statistical significance or a competition ranking.
No supplied competition input, video, label or prediction quality is used.

Three synthetic tests pass: H/A transform and output-label inverses,
reflection inverse with actual sign changes and missing padding preserved,
and normalized joint-probability averaging. Original sources and artifacts
remain untouched. Passing external comparisons is distinct from production
validation and promotion, which are not performed by these scripts.

```text
python -m unittest tests.test_tta_features -v
python tta_research.py --stage validation
python tta_research.py --stage crosscheck
python tta_research.py --stage regression
python tta_report.py
```

Each later stage requires the prior acceptance gate. Existing evidence is
preserved; reproduce in a fresh experiment directory. Dependencies, base
fitting and external data attribution: V4_MODEL_GUIDE.md, V4_1_MODEL_GUIDE.md
and EXTERNAL_DATA.md. The data licence is CC BY4.0.
