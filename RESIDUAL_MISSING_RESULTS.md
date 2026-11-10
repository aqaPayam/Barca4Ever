# Protected missing-ball residual

A new Pareto-anchor experiment preserves the legacy input preprocessing
and every original weight/buffer. Only384 new input coefficients learn
from raw ball-missing and observation-gap channels. Those channels are
zero on fully observed sampled sequences. This is a distinct candidate,
not a relaxation of criteria for a previously failed fine-tuned model.

All parameters except two input-weight tensors are frozen. Gradients on
their first17 columns are masked, and protected columns are copied back
after each optimizer step. Batch normalization/dropout remain in eval
mode throughout. Every original tensor is compared byte-for-byte before
and after fitting. No original artifact or production source is changed.

Fixed two epochs of100k fresh75/25 frame/spell windows, batch256,
Adam5e-4/cosine to1e-4, seed20261117. No checkpoint/parameter selection.
Batch128 is a predeclared memory-only restart contingency. Inputs retain
legacy16Hz geometry/velocity/interpolation exactly; the tree mixture stays
0.5 and the state-boundary decoder stays16/8.

## Matched external results

| Stage | Tracking | V4.1 spell F1 | Residual spell F1 | V4.1 short | Residual short |
|---|---|---:|---:|---:|---:|
| validation | clean | 0.7415 | 0.7415 | 39/94 | 39/94 |
| validation | noisy | 0.5471 | 0.5463 | 23/94 | 22/94 |

Each stage requires clean spell F1 >= V4.1, noisy spell F1 > V4.1 and
no short-hit loss in either scenario. Short recovery counts team spells
under two seconds at IoU0.5. Fully observed function preservation is a
mathematical property; finite-precision comparisons are tested separately.

validation acceptance: failed; preserve the qualified baseline.

## Preservation evidence

- full: 68 old tensors checked; byte-exact = True; unchanged checksum `1970b00b0401560b69879c7680e77309c4ed8e7bdf8d7367b9b89b04aaec997f`.

Three synthetic tests passed: first17 input bytes unchanged and zero-on-
observed extras; zero old-column gradients and byte-exact old tensors after
an Adam update; and matching old-network function on zero-extra windows
even after learning nonzero new columns. Source/model/reference-content
hashes, batch settings and epoch proofs are retained in model fit reports.

All data comes from the same seven licensed IDSSE matches. Initializers,
residual fitting and tree/onset models exclude each evaluated CV match.
WR9 is external validation; WMX/WN1 are reused regression. All seven games
have appeared in development, so these are not fresh blind tests, proof of
statistical significance or competition rankings. No supplied competition
input, video, label, prediction or quality score is used.

Producer edits and promotion are deferred until qualification and the root
global winner decision. No artifact/default/production change is performed
by these experiment scripts. Data attribution: EXTERNAL_DATA.md, CC BY4.0.

```text
python -m unittest tests.test_residual_missing -v
python residual_missing_train.py --initialize models/paper2023/rolefree.pt --out models/residual_missing/full.pt --batch 256
python residual_missing_research.py --stage validation
python residual_missing_research.py --stage crosscheck
python residual_missing_research.py --stage regression
python residual_missing_report.py
```

Use `--batch 256` (or the declared128 memory contingency); reproduce in a
fresh experiment directory and run later stages only after prior gates.
