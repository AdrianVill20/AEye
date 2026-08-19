$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Checking Python 3.11..."
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    throw "Python launcher 'py' was not found. Install Python 3.11+ and try again."
}

$python311 = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
if (-not $python311) {
    throw "Python 3.11 is not installed. Install Python 3.11 and try again."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    & py -3.11 -m venv .venv
}

Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing project requirements..."
python -m pip install -r requirements.txt

Write-Host "" 
Write-Host "Environment setup complete."
Write-Host "Run the app with:"
Write-Host "  python app\main.py"
Write-Host "Or from the project root:"
Write-Host "  .\.venv\Scripts\python.exe app\main.py"
