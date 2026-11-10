# V4.1: decoder refinement

This file preserves V4.1. The active candidate is V5; see V5_MODEL_GUIDE.md
and MULTIAGENT_RESULTS.md for its method and comparisons.

V4.1 preserves every V4 neural and tree weight and changes boundary_scale from
4 to 8. The switch cost is now 16/(1+8*onset), with the same destination-state
detector and equal neural/tree probability mixture. V4_MODEL_GUIDE.md documents
the full architecture, inputs, training, external licence and limitations.

Six settings were declared in V4_DECODER_PROTOCOL.md before testing. Selection
used only external WR9; the single chosen setting was then checked on the same
whole-match folds and reused WMX/WN1 regression. Every V4 control reproduced
exactly. Spell F1 improved and short recovery did not decline in either scenario
of either comparison. Full results and small-gain caveats: V4_DECODER_RESULTS.md.
No competition match or video was used for selection or checking quality.

The artifact is models/external_v4_1.joblib. Its metadata includes the parent
model hash, model/source hashes, exact changes, acceptance gates and hashes of
the protocol and external evidence. Promotion verifies that all unchanged
artifact components survive serialization exactly. V4 remains available.

## Inference

Python 3.12 and the same pinned CPU dependencies as V4:

```text
python -m pip install -r requirements-contact.txt
python infer_contact.py --model models/external_v4_1.joblib --data-dir /local/bundle --out Barca4Ever.csv
python evaluation/validate_submission.py Barca4Ever.csv
```

Run-Barca4Ever.ps1 selects the active artifact, checks its checksum and retains
format-validation details locally. Supply -Model models/external_v4.joblib to
run the preserved V4. Outputs and all restricted inputs must remain local.

## Reproduce the experiment

Use a fresh experiment directory. Follow V4_MODEL_GUIDE.md first, including
its fold-specific neural and tree fitting/cache steps. Then run:

```text
python diagnose_v4_short.py
python research_v4_decoder.py --stage validation
python research_v4_decoder.py --stage crosscheck
python research_v4_decoder.py --stage regression
python report_v4_decoder.py
python promote_v4_decoder.py
python check_contact_engine.py --model models/external_v4_1.joblib --reference external_reports/v4_decoder_validation.json --out external_reports/contact_decoder_production_check.json
python smoke_model.py --engine contact --model models/external_v4_1.joblib
```

The experiment and promotion scripts preserve existing evidence. No additional
neural fitting is needed. All source data is licensed IDSSE (CC BY 4.0), with
the attribution in EXTERNAL_DATA.md preserved. The seven matches have appeared
in development; this is not a fresh blind benchmark or a competition score.
