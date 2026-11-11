param(
    [string]$DataRoot = (Join-Path $PSScriptRoot 'data'),
    [string]$Output = 'evaluation/results/Barca4Ever.csv',
    [string]$Model = 'models/external_v5.joblib',
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $Python = if (Test-Path -LiteralPath $bundledPython) { $bundledPython } else { 'python' }
}
$DataRoot = (Resolve-Path -LiteralPath $DataRoot).ProviderPath
if (Test-Path -LiteralPath $Python -PathType Leaf) {
    $Python = (Resolve-Path -LiteralPath $Python).ProviderPath
}
foreach ($name in @('BAR-ATH_tactical_atd.xml', 'BAR-ATH_tactical_atd.txt')) {
    if (-not (Test-Path -LiteralPath (Join-Path $DataRoot "tactical\$name"))) {
        throw "Missing tactical tracking file: $name. Extract the supplied folders under the data root first."
    }
}
$model = if ([IO.Path]::IsPathRooted($Model)) { $Model } else { Join-Path $PSScriptRoot $Model }
$selectionFile = if (Test-Path -LiteralPath "$model.metadata.json") { "$model.metadata.json" } else { Join-Path $PSScriptRoot 'external_reports\frozen_selection.json' }
$selection = Get-Content -LiteralPath $selectionFile -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash.ToLowerInvariant() -ne $selection.model_sha256) {
    throw 'Model checksum differs from the externally selected artifact.'
}
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
Push-Location $PSScriptRoot
try {
    $inferenceScript = if ($selection.kind -eq 'refined_contact') { 'refined_infer.py' } elseif ($selection.kind -eq 'contact_fusion') { 'infer_contact.py' } elseif ($selection.kind -in @('state_boundary', 'blended_boundary')) { 'infer_changes.py' } elseif ($selection.kind -eq 'adaptive_boundary') { 'infer_boundary.py' } else { 'infer_learned.py' }
    & $Python -X utf8 $inferenceScript --data-dir $DataRoot --model $model --out $Output
    if ($LASTEXITCODE -ne 0) { throw 'Local inference failed.' }
    $validationLog = "$Output.validation.txt"
    & $Python -X utf8 evaluation/validate_submission.py $Output *> $validationLog
    if ($LASTEXITCODE -ne 0) { throw "Submission format check failed; inspect $validationLog locally." }
    Write-Output 'Inference and format validation complete. No training or tuning was performed.'
    Write-Output "Output: $Output. Keep the file and logs local."
}
finally { Pop-Location }
