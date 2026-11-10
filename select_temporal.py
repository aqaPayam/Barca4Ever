"""Validation-only search extension, before any held-out evaluation.

The initial best value was on the edge of the grid. Extend that grid, without
refitting the classifier or viewing test matches. This is validation selection,
not another independent validation estimate.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from train_external import ARTIFACT, REPORTS, build_x, metric
from external_idsse import load, SPLITS
from learned_possession import predict
import joblib


if __name__ == "__main__":
    if (REPORTS / "heldout_results.json").exists():
        raise RuntimeError("Held-out evaluation already done; selection is closed")
    artifact = joblib.load(ARTIFACT)
    selection_path = REPORTS / "frozen_selection.json"
    frozen = json.loads(selection_path.read_text())
    validation = [(half["labels"], build_x(half, noise, 8400 + h))
                  for match in SPLITS["validation"] for h, half in enumerate(load(match))
                  for noise in [False, True]]
    for penalty in [32, 64, 128]:
        scores = [metric(y, predict(artifact["model"], x, penalty)) for y, x in validation]
        row = dict(penalty=penalty,
                   spell_f1_mean=float(np.mean([s["spell_f1_mean"] for s in scores])),
                   macro_f1=float(np.mean([s["macro_f1"] for s in scores])))
        frozen["validation_trials"].append(row)
        print(json.dumps(row), flush=True)
    best = max(frozen["validation_trials"], key=lambda r: (r["spell_f1_mean"], r["macro_f1"]))
    artifact["penalty"] = best["penalty"]
    joblib.dump(artifact, ARTIFACT)
    frozen["selected"] = best
    frozen["model_sha256"] = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    frozen["selection_note"] = "Extended validation grid to 32/64/128 after initial optimum hit grid edge; no test results viewed"
    frozen["code_sha256"]["select_temporal.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    selection_path.write_text(json.dumps(frozen, indent=2))
    print("Final selection " + json.dumps(best), flush=True)
