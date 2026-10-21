# Run in an activated (faceauth) conda environment: powershell -File scripts/launch_windows.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "Checking model files ..."
python scripts/download_models.py --verify-only
if ($LASTEXITCODE -ne 0) {
  Write-Host "Models missing: running official downloader ..."
  python scripts/download_models.py
  if ($LASTEXITCODE -ne 0) { throw "Model download failed. See README manual download instructions." }
}
python scripts/verify_env.py
if ($LASTEXITCODE -ne 0) { throw "Environment verification failed." }
Write-Host "Launching local Streamlit UI ..."
python -m streamlit run app.py --server.fileWatcherType none
