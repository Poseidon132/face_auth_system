# FaceAuth v1.5: run the same predeclared protocol on val and test.
# Usage (after conda activate faceauth):
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_lfw_1n_windows.ps1
# No real identities/photos/embeddings are written to the report.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $commonArgs = @(
        '--images', 'data/lfw_home/lfw_funneled',
        '--sizes', '10', '50', '100',
        '--enroll-per-person', '2',
        '--queries-per-person', '2',
        '--unknown-queries', '300',
        '--seed', '20261021',
        '--target-fpir', '0.01',
        '--face-selection', 'lfw-largest'
    )
    Write-Host 'FaceAuth v1.5: validation and threshold calibration' -ForegroundColor Cyan
    & python scripts/eval_identification.py --stage val @commonArgs --calibration-out reports/lfw_1n_thresholds_val.json --out reports/lfw_1n_val.json
    if ($LASTEXITCODE -ne 0) { throw 'LFW 1:N validation failed; test not started.' }
    Write-Host 'FaceAuth v1.5: fixed-threshold identity-disjoint test' -ForegroundColor Cyan
    & python scripts/eval_identification.py --stage test @commonArgs --threshold-file reports/lfw_1n_thresholds_val.json --out reports/lfw_1n_test.json
    if ($LASTEXITCODE -ne 0) { throw 'LFW 1:N test failed.' }
    Write-Host 'DONE: reports/lfw_1n_test.csv and reports/lfw_1n_test.json' -ForegroundColor Green
}
finally {
    Pop-Location
}
