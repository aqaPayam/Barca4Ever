# V4 decoder experiment

Frozen V4 weights and the same external observations were used throughout.
Six settings were declared in V4_DECODER_PROTOCOL.md before results were
inspected: base costs 8/12/16 and boundary scales 4/8. The single setting
selected on WR9 was then compared with V4 on the original whole-match
folds and reused WMX/WN1 regression. V4 controls reproduced exactly.

Selected cost: 16/(1+8*onset); V4 cost: 16/(1+4*onset).

| Comparison | Tracking | V4 spell F1 | Trial spell F1 | V4 short | Trial short |
|---|---|---:|---:|---:|---:|
| validation | mixed | 0.6371 | 0.6443 | 55/188 | 62/188 |
| crosscheck | clean | 0.7050 | 0.7105 | 171/463 | 191/463 |
| crosscheck | noise | 0.5315 | 0.5329 | 68/463 | 79/463 |
| regression | clean | 0.6705 | 0.6808 | 109/362 | 128/362 |
| regression | noise | 0.4863 | 0.4968 | 53/362 | 64/362 |

Acceptance gates: all passed.

These are reused external development comparisons, not fresh blind tests.
Small gains do not establish statistical significance or professional-system
superiority. Clean/noisy rows share labels. No supplied competition match,
video, possession labels or output-quality checks were used. Frozen V4
artifacts and predictions are preserved. Per-half results and gate details
are in external_reports/v4_decoder_*.json.

## Diagnostic motivation and remaining work

On frozen V4 external validation, 11 noisy short spells were matched by raw
labels but lost after decoding. Among misses, 20 spells had a correct
majority from one component which was lost by probability averaging.
These are overlapping descriptive symptoms, not guaranteed recoverable
gains or an oracle performance target. The diagnostic is reproducible with
diagnose_v4_short.py and its source/model hashes are retained.

Further hypotheses are reliability-dependent fusion learned from strictly
out-of-fold predictions, corruption of player identities during external
training, and an additional 25-Hz contact/boundary path. Each needs its own
validation and whole-match comparison. Broader licensed external matches
and a genuinely unseen benchmark are needed to assess transfer reliably.
