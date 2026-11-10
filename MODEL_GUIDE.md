# External model: development and local inference

This document records v1. The current submission candidate is V5; see
V5_MODEL_GUIDE.md and MULTIAGENT_RESULTS.md for its method,
evidence and commands.

## Reproduce

Install Python 3.12 and requirements-model.txt. Download the external dataset with
`python download_external.py` (about 2.6 GB; original Figshare files; checksum
verified). See EXTERNAL_DATA.md for attribution and licence.

Run with 4 CPU threads, for example in PowerShell:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
python train_external.py fit
python select_temporal.py
python train_external.py test
```

The test command intentionally refuses to overwrite an existing held-out report.
Test results must not become development labels. Do not delete the guard to select
model changes against those matches. Any future experiment must declare fresh
independent test data or honestly describe reused tests as development data.

## Method

Input is player and ball XY geometry only. Coordinates are metres relative to
pitch centre. No player identity, game clock, frame ID, event outcome, possession
label or ball-status label is a feature. Features are calculated separately for
each external half, preventing temporal context from crossing halftime.

The 42 features comprise: three closest player-ball distances for each team;
nearest-distance difference; observed-ball indicator; two team track counts;
two boundary distances; mean player speed over 0.2 s; ball displacement speeds
and joint-observation indicators looking both directions at 0.2, 1 and 2 s;
centred averages of nearest distances, distance difference, ball visibility and
player speed at 0.4, 1 and 2 s; elapsed/remaining time to a ball observation.

A histogram gradient boosting classifier learns H/A/D probabilities from four
external matches. Every fifth frame is retained for fitting. Team-swapped copies
balance home and away. Clean and simulated-noisy tracking are used. A symmetric
Viterbi decoder minimizes label flicker; its switch penalty is selected from a
predeclared list on one separate external validation match by mean spellF1, then
macroF1 as a tie-break. It is an offline method, using future observations.
The initial grid was 0/1/2/4/8/16. Because the best validation score was at 16,
select_temporal.py extends it to 32/64/128 before any test scores are viewed;
the complete selection history is retained. Validation scores are selection
results, not independent generalization estimates.

Simulation adds Gaussian XY noise (0.25 m players, 0.35 m ball), 5% independent
player dropouts and contiguous random ball gaps. It is independent of labels.
It does not pretend to model real occlusions or provider error rates exactly.

## Run on competition data yourself, locally

On this workstation, with the download extracted under data/, a single command
checks the frozen model hash, performs local inference, and validates the CSV:

```powershell
.\Run-Barca4Ever.ps1
```

Or pass `-DataRoot 'C:\path\to\bundle'`. It only needs the tactical ATD XML and
TXT. Video and broadcast files are not required. Format-check details are written
to a local log instead of printed; no performance check or parameter adjustment
on the supplied match is performed.

```powershell
python infer_learned.py --data-dir 'C:\path\to\bundle'
python evaluation/validate_submission.py evaluation/results/Barca4Ever.csv
```

Only load the team's own trained models/external_v1.joblib: joblib files can run
code and must not be loaded from untrusted sources. The model uses tactical ATD
only. It never reads the competition video, ground truth or event annotations.
Long complete tracking outages separate context blocks automatically. Such gaps
are labelled X, not assumed dead-ball events. Output CSV and its manifest stay
local, excluded from Git. Do not upload them or source data into hosted AI tools.

## Scope and limitations

The external held-out report measures only two independent Bundesliga matches.
DFL tracking has human correction and clean coverage; challenge tracking is
automatic with different missingness, identity switches and geometry noise.
Possession definitions can differ. Match-independent performance and synthetic
corruption results do not establish a competition score or leaderboard position.
Data-driven dead-ball detection can miss whistles and motionless restarts.
No video or Smart Tagging information is incorporated yet.

Per-half scores are averaged equally in report summaries; halves are not
independent matches. Four halves in the test report still mean two matches.
All training data and tuning decisions are external. The supplied match is
inference-only under the organizer's November 3 clarification.
