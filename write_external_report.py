"""Render the recorded held-out results; does not fit or select anything."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    report = json.loads((ROOT / "external_reports" / "heldout_results.json").read_text())
    selection = json.loads((ROOT / "external_reports" / "frozen_selection.json").read_text())
    lines = ["# Barca4Ever external-model results", "",
             "This report evaluates external data only. No competition score is available.", "",
             "## Data and protocol", "",
             "IDSSE (DFL/Sportec), CC BY 4.0. Four training matches, one validation",
             "match and two independent held-out Bundesliga matches. Published external",
             "BallPossession/BallStatus fields provide labels; features use XY geometry",
             "only. Source and changes are described in EXTERNAL_DATA.md.", "",
             "The model used 454,656 frame examples including clean/noisy and team-swapped",
             "copies. This is not 454,656 independent observations. The classifier was",
             "fixed before validation selected the sequence switch penalty. Test labels",
             "were not used to fit or choose parameters.", "",
             "Selected penalty: {}. Validation mean spellF1: {:.4f}.".format(
                 selection["selected"]["penalty"], selection["selected"]["spell_f1_mean"]), "",
             "## Held-out summary", "",
             "Arithmetic means over four halves from two matches; not four independent matches.", "",
             "| Input scenario | Model | Possession-spell F1 | Macro F1 |",
             "|---|---|---:|---:|"]
    for row in report["summary"]:
        lines.append("| {} | {} | {:.4f} | {:.4f} |".format(
            row["scenario"], row["model"], row["spell_f1_mean"], row["macro_f1"]))
    lines += ["", "Nearest-player smooth50 is a reimplementation of the published baseline",
              "on the external coordinate arrays, using the same observations as our model.", "",
              "## Per-half results", "",
              "| Match | Half | Scenario | Model | Spell F1 | Macro F1 |",
              "|---|---:|---|---|---:|---:|"]
    for row in report["rows"]:
        lines.append("| {} | {} | {} | {} | {:.4f} | {:.4f} |".format(
            row["match"], row["half"], row["scenario"], row["model"],
            row["spell_f1_mean"], row["macro_f1"]))
    lines += ["", "## Limitations", "",
              "Two matches provide limited evidence. Provider definitions and tracking",
              "quality differ from the competition. Synthetic noise is independent of",
              "labels but cannot establish robustness to actual occlusion, tracking",
              "identity errors or homography errors. No competition result is inferred",
              "from these numbers. The learned model has not read the competition match.", "",
              "Test results were inspected after final selection; model parameters were",
              "not changed afterwards. Future work must preserve this distinction or",
              "declare this test set development data and acquire a new independent test.", "",
              "## Reproduction", "",
              "See MODEL_GUIDE.md. Exact trial history and code hashes are in",
              "external_reports/frozen_selection.json; per-half metrics are in",
              "external_reports/heldout_results.json.", "",
              "Frozen artifact SHA-256: `{}`.".format(report["frozen_model_sha256"]), ""]
    (ROOT / "EXTERNAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
