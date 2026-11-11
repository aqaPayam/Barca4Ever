# Reviewing Barca4Ever V5

This guide points to the selected submission model, its evidence and the commands that regenerate the possession file.

## Start with the report

The four-page paper defines the task and metric, explains the method, then presents the external experiments, limitations and reproduction steps. It is compiled from the included LaTeX source. [PDF](submission/Barca4Ever.pdf) · [LaTeX source](submission/Barca4Ever.tex) · [Browser-readable version](Barca4Ever.md)

## Follow the active inference path

| Review question | Entry point |
| --- | --- |
| Which artifact produces the file? | [models/external_v5.joblib](models/external_v5.joblib) and its [metadata](models/external_v5.joblib.metadata.json) |
| How are inputs read and indexed? | [infer_learned.py](infer_learned.py) and [games.py](games.py) |
| How are observation confidence and actor groups handled? | [hz25_data.py](hz25_data.py) and [refined_infer.py](refined_infer.py) |
| How does the contact model work? | [paper2023_model.py](paper2023_model.py) |
| How are possession changes decoded? | [boundary_refinement.py](boundary_refinement.py) |
| Where are the training recipe and comparisons? | [V5 model guide](V5_MODEL_GUIDE.md) and [development results](MULTIAGENT_RESULTS.md) |
| Where is the latest software verification? | [November 11 verification](external_reports/v5_software_verification_2026_11_11.json) |

## Regenerate and validate

Run these commands from the repository root with Python 3.12 and the organizer's local data bundle.

```text
python -m pip install -r requirements-contact.txt
python refined_infer.py --model models/external_v5.joblib --data-dir "/path/to/organizer/bundle" --out local_outputs/Barca4Ever.csv
python evaluation/validate_submission.py local_outputs/Barca4Ever.csv
```

The bundle contains `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt`. The generated CSV has 199,319 indexed rows. Its local manifest records model and output hashes.

## Interpret the evidence

Software checks establish indexing, execution and file validity. The football-quality tables use licensed IDSSE labels across previously inspected development matches. Noisy short possessions remain difficult. The report records those limitations alongside the measured gains.

The public repository retains earlier versions and rejected research trials for traceability. The regeneration command explicitly selects V5. Keep organizer data, generated predictions and local logs outside the public package.

## Build the paper

The current PDF was compiled with Tectonic 0.17.0. From the repository root, run

```text
tectonic --untrusted --outdir submission submission/Barca4Ever.tex
```

The first run may fetch standard TeX resources. The LaTeX source contains the figure, equations, tables and references.
