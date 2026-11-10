# Barca4Ever - frame-level football possession

Externally trained tracking contact network, geometric classifier and learned
state-change decoding, developed for More Than A Hack 2027. See Barca4Ever.md
for the technical report and MULTIAGENT_RESULTS.md for the latest comparisons.
The neural architecture extends Borghesi et al. (2023); original author weights
were unavailable and this is our retrained adaptation. Competition quality
has not been evaluated. No competition data is distributed in this repository.

## Reproduce the submission

Python 3.12:

```text
python -m pip install -r requirements-contact.txt
python refined_infer.py --model models/external_v5.joblib --data-dir /path/to/organizer/bundle --out Barca4Ever.csv
python evaluation/validate_submission.py Barca4Ever.csv
```

The bundle root contains tactical/BAR-ATH_tactical_atd.xml and
tactical/BAR-ATH_tactical_atd.txt. Use the included, trusted
models/external_v5.joblib. CPU inference is canonical; no GPU is required.
Do not load joblib files from untrusted sources.
Run-Barca4Ever.ps1 offers a Windows wrapper with checksum verification.

The output manifest records the output and model hashes. Keep restricted inputs,
predictions, and intermediate files local. No network calls, training, fitting,
video checks or parameter tuning are performed during inference. Do not upload
the provided data to hosted AI services.

## External training and evaluation

See V5_MODEL_GUIDE.md, MULTIAGENT_RESULTS.md, EXTERNAL_DATA.md, models/README.md
and external_reports/.
Dataset: IDSSE, DFL/Sportec, Bassek et al. (2025), CC BY 4.0.
DOI: https://doi.org/10.6084/m9.figshare.28196177.
Four training matches, one validation match and two original held-out matches.
All seven are now previously inspected development/regression data; later results
are not new blind tests. Each evaluated whole-match fold is excluded from every
learned component's fitting, including the neural initializer and adaptation.

The externally selected model artifact is included so organizers can reproduce
inference without downloading or re-training the external dataset. For a fresh
training reproduction, follow V5_MODEL_GUIDE.md in a fresh experiment directory. The regression command
refuses to overwrite its existing report; retain original evidence and reproduce
in a separate fresh experiment directory rather than using tests for tuning.

## Tests

```text
python -m unittest discover -s tests -v
python check_refined_engine.py --model models/external_v5.joblib
python check_refined_pipeline.py --model models/external_v5.joblib
```

Unit tests use synthetic data only. The engine checker uses licensed external
data; the pipeline checker writes a full-length synthetic integration check
under local_outputs. That generated file is not a competition submission.
V5 inference and neural tests use PyTorch from requirements-contact.txt.
Earlier V1/V2/V3/V4/V4.1 artifacts and source are preserved. V5 adds raw ball
observability, accurate timing inputs and directed boundaries. The active model, research
candidates and negative comparison results are included for transparency.

## Attribution

The public competition starter code (including games.py and evaluation utilities)
comes from https://github.com/metrica-sports/more-than-a-hack-2027 under its MIT
licence, included as LICENSE. Team modifications and external dataset provenance
are documented in the technical report. Preserve the external-data attribution
when distributing the trained artifact.
