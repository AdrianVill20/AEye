$ErrorActionPreference = "Stop"

# This script lives in integration/. It builds the project's virtual
# environment at integration/.venv (Python 3.12) with all dependencies, so the
# venv lives next to the project. run.bat and the training command both use it.
$integrationDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $integrationDir

Write-Host "Checking Python 3.12..."
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    throw "Python launcher 'py' was not found. Install Python 3.12 and try again."
}

$python312 = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
if (-not $python312) {
    throw "Python 3.12 is not installed. Install Python 3.12 and try again."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in $integrationDir\.venv..."
    & py -3.12 -m venv .venv
}

Write-Host "Upgrading pip/setuptools/wheel..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

Write-Host "Installing project requirements..."
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Environment setup complete."
Write-Host "Run the app with:"
Write-Host "  .\.venv\Scripts\python.exe app\main.py"
Write-Host "Or double-click integration\run.bat"
