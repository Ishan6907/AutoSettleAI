# Auto-Settle AI: local run script (PowerShell)
#   .\run.ps1 -Install          # pip install -r requirements.txt
#   .\run.ps1 -Test             # unit tests + exploit suite
#   .\run.ps1 -App -NoQwen      # web app in deterministic template mode
param(
    [switch]$Install,
    [switch]$Test,
    [switch]$App,
    [switch]$NoQwen
)

$ErrorActionPreference = "Stop"
$PY = "python"

if ($Install) {
    Write-Host "Installing requirements..." -ForegroundColor Cyan
    & $PY -m pip install -r requirements.txt
}

if ($Test) {
    Write-Host "Running engine unit tests..." -ForegroundColor Cyan
    & $PY -m unittest discover -s autosettle_engine/tests -t .
    Write-Host "Running vision unit tests..." -ForegroundColor Cyan
    & $PY -m unittest discover -s autosettle_vision/tests -t .
    Write-Host "Running exploit / attack suite..." -ForegroundColor Cyan
    & $PY -m autosettle_engine.exploit_suite
}

if ($App) {
    if ($NoQwen) {
        Write-Host "Starting web app (no Qwen, deterministic template mode)..." -ForegroundColor Cyan
        & $PY -m autosettle_vision.app --no-qwen
    } else {
        Write-Host "Starting web app (loads Qwen; needs GPU)..." -ForegroundColor Cyan
        & $PY -m autosettle_vision.app
    }
}

if (-not ($Install -or $Test -or $App)) {
    Write-Host "No action specified. Use -Install, -Test, or -App [-NoQwen]."
}
