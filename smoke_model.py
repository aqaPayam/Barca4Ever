"""Full-length CLI integration check using generated tracking, never gated data."""
from pathlib import Path
import argparse
import subprocess
import sys
from games import source_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/external_v1.joblib')
    parser.add_argument('--engine', choices=['learned', 'boundary', 'changes', 'contact'], default='learned')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    bundle = root / "local_outputs" / "synthetic_bundle"
    folder = bundle / "tactical"
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / "BAR-ATH_tactical_atd.xml"
    meta.write_text('<root><Team id="h"/><Team id="a"/>' +
                    '<Player teamId="h"/>' * 3 + '<Player teamId="a"/>' * 3 + '</root>')
    raw = folder / "BAR-ATH_tactical_atd.txt"
    with raw.open("w") as stream:
        for i in range(source_frames()):
            if 90000 <= i < 91000:
                people, ball = ";".join(["NaN,NaN,NaN"] * 6), "NaN,NaN,NaN"
            else:
                people = "0.25,0.5,0;0.3,0.5,0;0.3,0.4,0;0.75,0.5,0;0.7,0.5,0;0.7,0.4,0"
                ball = "0.25,0.5,0" if (i // 500) % 2 == 0 else "0.75,0.5,0"
            stream.write("{}:{}:{}\n".format(i + 1, people, ball))
    out = root / "local_outputs" / "SYNTHETIC_MODEL_CHECK.csv"
    script = {'boundary':'infer_boundary.py','changes':'infer_changes.py','contact':'infer_contact.py','learned':'infer_learned.py'}[args.engine]
    subprocess.run([sys.executable, str(root / script), "--data-dir", str(bundle),
                    "--model", str(root / args.model), "--out", str(out)], check=True)
    subprocess.run([sys.executable, str(root / "evaluation" / "validate_submission.py"), str(out)], check=True)
    print("Synthetic integration check completed; this is NOT a competition submission.")
