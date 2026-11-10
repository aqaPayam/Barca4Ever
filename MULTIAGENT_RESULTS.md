# Paper-driven multi-agent model improvement

V5 selects the composite of observability-aware 16 Hz contact emissions and
directed transition boundaries. It is the highest-WR9 candidate that passed
all declared clean/noisy whole-match CV and reused-test regression checks.
The selection rule was recorded in MULTIAGENT_SELECTION_PROTOCOL.md. V4.1
is preserved as a fallback. Saved-model software checks are recorded separately.

## What the agents contributed

- Research: audited Borghesi (2023), Peral/Ferraz (2025), ELASTIC (2026) and PathCRF (2026);
  translated contact physics into tested soft features. PathCRF released
  weights overlap our evaluation matches and were not used.
- Audit: verified labels, timing and preprocessing; found lost raw ball
  observability, rounded 16 Hz velocity approximations and repeated training
  references. Verified new inputs and exact zero-padded initialization.
- Implementation: fitted directed and physics boundary models, evaluated
  symmetry averaging and a protected missing-input adapter; built and
  tested the new production producer.
- Integration: fitted the new neural inputs, selected and confirmed the
  composite and prepared the reproducible release.

Primary links and detailed caveats: RESEARCH_AGENT_METHODS.md and
PAPER_RESEARCH_NOTES_2026_11_06.md. These are original adaptations, not
reproductions of the papers' private-data scores or pretrained products.

## Reused external validation selection

| Candidate | Spell F1 | Short /188 |
|---|---:|---:|
| V4.1 | 0.6443 | 62 |
| Observability 16Hz | 0.6523 | 64 |
| Observability 25Hz | 0.6270 | 65 |
| Directed boundary | 0.6492 | 65 |
| Physics+dense boundary | 0.6476 | 67 |
| Composite | 0.6542 | 67 |
| Symmetry averaging | 0.6481 | 64 |

Native 25 Hz was rejected. Observability 16 Hz was selected before CV. The
composite alpha 1 was selected on WR9 before its CV scores were inspected.
The residual adapter preserved all 68 old tensors but failed noisy validation;
no CV/regression followed. Symmetry and the first directed model failed
their standalone CV gates; no standalone regression was searched.

## Whole-match comparisons

| Candidate | Tracking | Spell F1 | Short /463 |
|---|---|---:|---:|
| V4.1 | clean | 0.7105 | 191 |
| V4.1 | noise | 0.5329 | 79 |
| Observability16 | clean | 0.7064 | 193 |
| Observability16 | noise | 0.5364 | 79 |
| Directed boundary | clean | 0.7126 | 192 |
| Directed boundary | noise | 0.5325 | 84 |
| Physics+dense boundary | clean | 0.7149 | 198 |
| Physics+dense boundary | noise | 0.5331 | 86 |
| Composite | clean | 0.7133 | 206 |
| Composite | noise | 0.5344 | 80 |
| Symmetry averaging | clean | 0.7075 | 186 |
| Symmetry averaging | noise | 0.5376 | 81 |

## Single-choice reused regression

| Candidate | Tracking | Spell F1 | Short /362 |
|---|---|---:|---:|
| V4.1 | clean | 0.6808 | 128 |
| V4.1 | noise | 0.4968 | 64 |
| Physics+dense boundary | clean | 0.6844 | 132 |
| Physics+dense boundary | noise | 0.5020 | 67 |
| Composite | clean | 0.6847 | 132 |
| Composite | noise | 0.5140 | 71 |

## Limits and integrity

All 7 licensed IDSSE matches have appeared in development. Both halves stay
together and each learned fold component excludes the evaluated matches.
Scores use the exact organizer scorer and original corruption seeds. All
V4.1 controls reproduce exactly. Clean/noisy scenarios share labels. Small
gains do not establish statistical significance or professional superiority.
No competition inputs, video, labels or prediction quality were used.

One damaged external parsed cache was rebuilt from checksum-verified
original XML. Rebuilt clean/noisy geometry and object features match the
historical caches exactly; the damaged copy was preserved. Evidence:
external_reports/agent_cache_integrity_repair.json. All 14 external half
frame-ID sequences are contiguous. See V5_MODEL_GUIDE.md for final inputs,
training and inference; the physics alternative remains a research result.
