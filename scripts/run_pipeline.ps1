<#
.SYNOPSIS
  Reproduce the whole ML pipeline locally from a clean checkout (Windows PowerShell).
  ./scripts/run_pipeline.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt

function Step($name, [scriptblock]$block) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$name failed" }
}

Step "Lint"              { & $py -m flake8 src tests scripts dags }
Step "Unit tests"        { & $py -m pytest --cov=src --cov-report=term }
Step "Download + clean"  { & $py scripts/download_data.py }
Step "EDA"               { & $py -m src.eda.run_eda }
Step "Train + track"     { & $py -m src.models.train }
Step "Batch inference"   { & $py -m src.inference.predict }

Write-Host "`nDone. MLflow UI:  $py -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
Write-Host "API:             $py -m uvicorn src.api.main:app --port 8000"
