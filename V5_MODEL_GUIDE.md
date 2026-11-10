# V5: contact timing and directed possession changes

V5 is the active model. It combines a new 20-channel contact network with the
preserved V2 geometry classifier and a new directed transition detector.
V4.1 and all earlier artifacts remain available. This is our externally trained
adaptation of published ideas, not an author's pretrained model or published score.

## What changed

The contact network still uses the Borghesi et al. (2023) shared trajectory TCN,
two attention branches, ball-relative features and immediate contact path from
V4. Three new channels distinguish actual ball observations from interpolated
positions: raw visibility, time since an observation and time until the next
observation. Gap times are capped at 10 seconds and scaled by 10.

Short gaps are interpolated on the original 25 Hz source before 16 Hz sampling.
Velocities use the actual elapsed source time between sampled frames. New input
weights start at zero, retaining the corresponding role-free initializer; two
epochs then fine-tune the network on external data. Each epoch draws fresh
100,000 frame/spell references. Keeper-role inputs stay zero. Every observed
actor is retained in padded object groups, including extra tracking actors.

The 16 Hz network uses 64 samples, target index 47 and about one second of future
trajectory. Gap-time inputs may use ten seconds of future observation evidence.
It is an offline method. The native 25 Hz alternative had worse validation spell
F1 and was rejected. Equal tree/neural probability weighting was selected on WR9.

The new boundary classifier has seven classes: background and the six ordered
H/A/D changes. Its 293 inputs combine 59 base features with selected features
before, after and across each potential change. A score for H-to-A lowers only
that switch cost. Viterbi uses 16/(1+8*b[t,previous,next]); boundary scores are
weighted-training evidence, not calibrated probabilities. WR9 selected alpha 1,
so the old destination detector is retained in the artifact but contributes zero.

See HZ25_PROTOCOL.md, boundary_refinement_protocol.md and
COMBINED_REFINEMENT_PROTOCOL.md for fitting and selection recipes. The research
agent also implemented soft contact-physics features informed by ELASTIC and
touch/reception work. That alternative passed its checks but ranked below the
composite on WR9. It is not part of the active model. PathCRF's released weights
overlap our evaluation games and were not used. Detailed primary references and
limitations are in PAPER_RESEARCH_NOTES_2026_11_06.md.

## Matched external evidence

| Comparison | Tracking | V4.1 spell F1 | V5 spell F1 | V4.1 short | V5 short |
|---|---|---:|---:|---:|---:|
| Whole-match CV | clean | 0.7105 | 0.7133 | 191/463 | 206/463 |
| Whole-match CV | noise | 0.5329 | 0.5344 | 79/463 | 80/463 |
| Reused regression | clean | 0.6808 | 0.6847 | 128/362 | 132/362 |
| Reused regression | noise | 0.4968 | 0.5140 | 64/362 | 71/362 |

The exact organizer spell scorer is used. Short recovery means a same-team
possession under two seconds matched at IoU 0.5. Scores average halves equally;
short counts are summed. These are not frame accuracy percentages.

All seven licensed IDSSE games have appeared in development. Every fold excludes
its evaluated whole matches from all fitting, including neural initialization.
The two original held-out games are now reused regression data. Clean/noisy runs
share labels, and simulated corruption does not reproduce every automatic
tracking error. The small gains are development evidence, not statistical
significance, professional superiority or a competition score.

MULTIAGENT_SELECTION_PROTOCOL.md records the integration rule: among candidates
passing their declared CV and regression checks, select the highest WR9 spell
F1. The composite won that rule. MULTIAGENT_RESULTS.md retains alternatives and
failed experiments; external_reports/multiagent_summary.json has exact values.

## Run locally

Use Python 3.12 with requirements-contact.txt. The trusted artifact contains
neural NumPy state, the classifiers and decoding settings. Canonical inference
uses CPU with four threads; no training dataset or GPU is needed.

```powershell
python -m pip install -r requirements-contact.txt
.\Run-Barca4Ever.ps1 -DataRoot "C:\path\to\bundle"
```

The bundle root contains tactical/BAR-ATH_tactical_atd.xml and its TXT file.
The wrapper checks the artifact checksum, selects refined_infer.py and stores
validator details locally. Direct equivalent:

```text
python refined_infer.py --model models/external_v5.joblib --data-dir /local/bundle --out Barca4Ever.csv
python evaluation/validate_submission.py Barca4Ever.csv
```

The producer writes 199,319 indexed rows and a manifest with model/output/source
hashes. Long complete player-tracking outages of at least ten seconds receive X.
No fitting, tuning or quality comparison happens during inference. Do not upload
the supplied match or extracts to hosted AI services. Do not compare predictions
with the supplied video or ground truth. Keep predictions, intermediates and
validator logs local; the validator checks format only.

V5 model SHA-256:
9b1169c5bf920a1bff811ba23a15b907c983c7364626823ff10ccdf5da595d72.
Metadata pins production source and external evidence hashes. The release copy
is byte-identical to the candidate used for software checks. V4.1 can be run with
the wrapper's explicit -Model models/external_v4_1.joblib option.

## Software checks and research reproduction

82 synthetic/unit tests passed. The saved production engine exactly reproduced
spell F1, macro F1 and short counts in four external CPU comparisons. A generated
199,319-row ATD fixture passed actual CLI inference and the official validator.
These software checks do not establish possession quality on the supplied match.

```text
python -m unittest discover -s tests -v
python check_refined_engine.py --model models/external_v5.joblib
python check_refined_pipeline.py --model models/external_v5.joblib
```

Reports: external_reports/refined_production_check.json and
external_reports/refined_pipeline_check.json. Both record unchanged model/source
hashes. The pipeline checker uses only generated observations under local_outputs.

For training reproduction, use a separate fresh experiment directory and retain
the original evidence. First follow V4_MODEL_GUIDE.md for licensed data, base
trees, boundary caches and corresponding role-free fold checkpoints. The current
source and included checkpoints support V5 reproduction; GPU fitting uses
requirements-paper2023.txt. Run research sequentially to avoid simultaneous
writes to a cold parsed-data cache.

```text
python hz25_research.py --stage validation --rate 16
python hz25_research.py --stage validation --rate 25
python hz25_select.py
python hz25_research.py --stage crosscheck
python boundary_refinement_research.py --stage validation
python boundary_refinement_research.py --stage crosscheck
python combined_refinement_research.py --stage validation
python combined_refinement_research.py --stage crosscheck
python combined_refinement_research.py --stage regression
python promote_refined.py --method combined --out models/reproduced_v5.joblib
```

Scripts preserve existing frozen artifacts and evidence. Only qualifying choices
proceed to regression; standalone neural/directed candidates failed their gates.
No standalone regression is claimed for them. The first 16 Hz validation source
is preserved in research_snapshots/hz25_research_before_components_cache.py;
later memoization changes no feature or scoring mathematics. A repaired external
parsed cache was checked against checksum-verified original XML and historical
clean/noisy features; external_reports/agent_cache_integrity_repair.json records
the repair. The damaged copy remains local.

## Attribution

IDSSE: DFL / Sportec Solutions; Bassek, Rein, Weber and Memmert (2025),
Scientific Data 12, 195, https://doi.org/10.1038/s41597-025-04505-y.
Deposit https://doi.org/10.6084/m9.figshare.28196177, CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/.
Changes include parsing, features, simulated corruption, external training,
fine-tuning and decoding. Preserve this attribution with the trained artifacts.
