import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


@unittest.skipUnless(POWERSHELL, "PowerShell is required to test the launcher")
class RunWrapperPathTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / "local_outputs" / ("wrapper_test_" + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, folder)
        self.project = folder / "published code"
        self.caller = folder / "caller"
        self.bundle = self.caller / "bundle [data]"
        self.receipt = folder / "receipt.json"
        self.output = folder / "synthetic.csv"
        self.project.mkdir()
        self.caller.mkdir()
        tactical = self.bundle / "tactical"
        tactical.mkdir(parents=True)
        for suffix in ("xml", "txt"):
            (tactical / f"BAR-ATH_tactical_atd.{suffix}").write_text("synthetic")
        shutil.copyfile(ROOT / "Run-Barca4Ever.ps1", self.project / "Run-Barca4Ever.ps1")
        model = self.project / "models" / "external_v5.joblib"
        model.parent.mkdir()
        model.write_bytes(b"synthetic launcher model")
        metadata = {
            "kind": "refined_contact",
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        }
        Path(str(model) + ".metadata.json").write_text(json.dumps(metadata))
        self.interpreter = self.caller / "synthetic interpreter.ps1"
        self.interpreter.write_text(
            """$ErrorActionPreference = 'Stop'
if ($args[2] -eq 'refined_infer.py') {
    $index = [Array]::IndexOf($args, '--data-dir')
    $root = $args[$index + 1]
    if (-not (Test-Path -LiteralPath (Join-Path $root 'tactical/BAR-ATH_tactical_atd.txt'))) {
        throw 'The launcher passed a data path that cannot be found.'
    }
    @{data_dir = (Convert-Path -LiteralPath $root)} | ConvertTo-Json | Set-Content -LiteralPath $env:BARCA_WRAPPER_TEST_RECEIPT -Encoding UTF8
}
$global:LASTEXITCODE = 0
""",
            encoding="utf-8",
        )

    def run_wrapper(self, data_root, python_path=None):
        environment = os.environ.copy()
        environment["BARCA_WRAPPER_TEST_RECEIPT"] = str(self.receipt)
        result = subprocess.run(
            [
                POWERSHELL, "-NoProfile", "-File", str(self.project / "Run-Barca4Ever.ps1"),
                "-DataRoot", str(data_root), "-Output", str(self.output),
                "-Python", str(python_path or self.interpreter),
            ],
            cwd=self.caller,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = json.loads(self.receipt.read_text(encoding="utf-8-sig"))
        self.assertEqual(Path(receipt["data_dir"]), self.bundle)

    def test_absolute_data_path_from_another_folder(self):
        self.run_wrapper(self.bundle)

    def test_relative_data_path_from_another_folder(self):
        self.run_wrapper(self.bundle.relative_to(self.caller))

    def test_relative_python_path_from_another_folder(self):
        self.run_wrapper(self.bundle, ".\\" + self.interpreter.name)


if __name__ == "__main__":
    unittest.main()
