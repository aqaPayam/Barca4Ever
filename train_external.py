"""Reproducible external-only fit, validation selection and held-out evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from learned_possession import features, corrupt, predict, LABELS, FEATURE_VERSION
from external_idsse import load, SPLITS
from barca4ever import Config, decode
from evaluation import possession_metrics as pm
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "models" / "external_v1.joblib"
REPORTS = ROOT / "external_reports"


def metric(y, p):
    result = pm.evaluate(y, p, bootstrap_n=0)
    return {k: result[k] for k in ["spell_f1_mean", "macro_f1", "accuracy", "n_gt_spells", "n_pred_spells"]}


def build_x(half, noise=False, seed=0):
    players, ball = half["players"], half["ball"]
    if noise:
        players, ball = corrupt(players, ball, seed)
    return features(players, half["teams"], ball, half["pitch"])


def reference(x):
    """v0.1 on exactly the same feature inputs as the learned system."""
    c = Config()
    evidence = np.full(len(x), "X", dtype="<U1")
    winner = x[:, 3] < x[:, 0]
    distance = np.minimum(x[:, 0], x[:, 3])
    control = (distance <= c.control_radius_m) & (abs(x[:, 6]) >= c.team_margin_m) & (x[:, 7] > 0)
    evidence[control] = np.where(winner[control], "A", "H")
    evidence[(x[:, 7] > 0) & (np.min(x[:, 10:12], axis=1) < -c.boundary_margin_m)] = "D"
    return pm.label_codes(decode(evidence, 25, c))


def published_baseline(x):
    home = (x[:, 0] <= x[:, 3]) & (x[:, 7] > 0) & (x[:, 0] < 100)
    away = (x[:, 3] < x[:, 0]) & (x[:, 7] > 0) & (x[:, 3] < 100)
    ix = np.arange(len(x))
    counts = []
    for mask in [home, away]:
        cs = np.r_[0, np.cumsum(mask)]
        counts.append(cs[np.minimum(len(x), ix + 51)] - cs[np.maximum(0, ix - 50)])
    h, a = counts
    return np.where(h + a == 0, 2, np.where(h >= a, 0, 1)).astype(np.int8)


def fit():
    xs, ys = [], []
    for m, match in enumerate(SPLITS["train"]):
        print("Preparing TRAIN " + match, flush=True)
        for h, half in enumerate(load(match)):
            # Clean and corrupted observations; no labels enter feature generation.
            for variant in range(2):
                x = build_x(half, variant == 1, 7300 + 10 * m + h)
                y = half["labels"]
                take = (np.arange(len(y)) % 5 == 0) & (y < 3)
                xs.append(x[take])
                ys.append(y[take])
                # Team swap enforces home/away balance without new match labels.
                teams = 1 - half["teams"]
                p, b = half["players"], half["ball"]
                if variant:
                    p, b = corrupt(p, b, 7300 + 10 * m + h)
                swapped = features(p, teams, b, half["pitch"])
                ys.append(np.where(y[take] < 2, 1 - y[take], y[take]))
                xs.append(swapped[take])
    model = HistGradientBoostingClassifier(max_iter=130, max_leaf_nodes=15,
                                           learning_rate=0.08, l2_regularization=2,
                                           min_samples_leaf=60, early_stopping=False,
                                           random_state=20261104)
    x, y = np.concatenate(xs), np.concatenate(ys)
    print("Fitting {} EXTERNAL frames, {} features".format(*x.shape), flush=True)
    model.fit(x, y)
    validation = []
    for match in SPLITS["validation"]:
        for h, half in enumerate(load(match)):
            for noise in [False, True]:
                validation.append((half["labels"], build_x(half, noise, 8400 + h)))
    trials = []
    for penalty in [0, 1, 2, 4, 8, 16]:
        scores = [metric(y, predict(model, x, penalty)) for y, x in validation]
        row = dict(penalty=penalty,
                   spell_f1_mean=float(np.mean([s["spell_f1_mean"] for s in scores])),
                   macro_f1=float(np.mean([s["macro_f1"] for s in scores])))
        trials.append(row)
        print("Validation " + json.dumps(row), flush=True)
    best = max(trials, key=lambda t: (t["spell_f1_mean"], t["macro_f1"]))
    artifact = dict(model=model, penalty=best["penalty"], feature_version=FEATURE_VERSION,
                    splits=SPLITS, licence="CC BY 4.0", dataset_doi="10.6084/m9.figshare.28196177",
                    training_samples=len(np.concatenate(ys)),
                    feature_count=42, seed=20261104)
    ARTIFACT.parent.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(artifact, ARTIFACT)
    frozen = {"model_sha256": hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
              "splits": SPLITS, "validation_trials": trials, "selected": best,
              "test_policy": "No fit or parameter changes after reading held-out results",
              "label_provenance": "Sportec/DFL published BallStatus and BallPossession fields",
              "feature_exclusions": ["BallStatus", "BallPossession", "events", "frame ids", "clock", "identity"]}
    frozen["source_manifest_sha256"] = hashlib.sha256(
        (ROOT / "external_data" / "idsse" / "source_manifest.json").read_bytes()).hexdigest()
    frozen["code_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in ["learned_possession.py", "external_idsse.py", "train_external.py"]}
    (REPORTS / "frozen_selection.json").write_text(json.dumps(frozen, indent=2))
    print("Frozen model: " + frozen["model_sha256"], flush=True)


def evaluate():
    if (REPORTS / "heldout_results.json").exists():
        raise RuntimeError("Held-out report already exists. Do not repeatedly tune against test matches.")
    artifact = joblib.load(ARTIFACT)
    frozen = json.loads((REPORTS / "frozen_selection.json").read_text())
    if hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() != frozen["model_sha256"]:
        raise ValueError("Model changed since validation selection")
    rows = []
    for match in SPLITS["test"]:
        for h, half in enumerate(load(match)):
            for noise in [False, True]:
                x = build_x(half, noise, 9500 + h)
                for name, p in [("prototype_v0.1", reference(x)),
                                ("published_nearest_smooth50", published_baseline(x)),
                                ("learned_external_v1", predict(artifact["model"], x, artifact["penalty"]))]:
                    row = dict(match=match, half=h + 1, scenario="corrupted" if noise else "clean",
                               model=name, **metric(half["labels"], p))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    summary = []
    for scenario in ["clean", "corrupted"]:
        for name in ["prototype_v0.1", "published_nearest_smooth50", "learned_external_v1"]:
            group = [r for r in rows if r["scenario"] == scenario and r["model"] == name]
            summary.append(dict(scenario=scenario, model=name,
                                spell_f1_mean=float(np.mean([r["spell_f1_mean"] for r in group])),
                                macro_f1=float(np.mean([r["macro_f1"] for r in group]))))
    report = dict(frozen_model_sha256=frozen["model_sha256"], rows=rows, summary=summary,
                  caveats=["Only two independent held-out matches", "DFL labels differ from Metrica definitions",
                           "Synthetic noise is not actual challenge tracking noise", "No competition score predicted"])
    (REPORTS / "heldout_results.json").write_text(json.dumps(report, indent=2))
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "test"])
    args = parser.parse_args()
    fit() if args.stage == "fit" else evaluate()
