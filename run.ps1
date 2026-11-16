<#
.SYNOPSIS
    Runs DigitalMind: prepares the environment, loads the demo data, starts the app.

.EXAMPLE
    .\run.ps1
    .\run.ps1 -Reset -NoReindex
#>
param(
    [switch]$Reset,
    [switch]$NoDemo,
    [switch]$NoReindex,
    [switch]$SkipDownload,
    # NB: not $Host - that is a read-only PowerShell automatic variable.
    [string]$BindAddress = "",
    [int]$Port = 0,
    [switch]$Setup,
    [switch]$Evaluate,
    [switch]$Test
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Find-Python {
    # DigitalMind is pinned to Python 3.11. Prefer it, fall back to whatever is on PATH.
    foreach ($version in @("3.11", "3.12")) {
        $candidate = Get-Command "py" -ErrorAction SilentlyContinue
        if ($candidate) {
            try {
                $check = & py "-$version" -c "import sys; print(sys.version_info[:2] == (3, 11))" 2>$null
                if ($LASTEXITCODE -eq 0) { return @("py", "-$version") }
            } catch { }
        }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) { Write-Error "Python not found. Install Python 3.11 from python.org." }
    return @("python")
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if ($Setup -or -not (Test-Path $VenvPython)) {
    $cmd = Find-Python
    Write-Host "Creating virtual environment with $($cmd -join ' ') ..." -ForegroundColor Cyan
    & $cmd[0] $cmd[1..($cmd.Length - 1)] -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed." }
}

Write-Host "Installing Python dependencies ..." -ForegroundColor Cyan
& $VenvPython -m pip install --quiet --upgrade pip
& $VenvPython -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Error "Dependency install failed." }

if (-not (Test-Path (Join-Path $Root "frontend\node_modules"))) {
    Write-Host "Installing frontend dependencies ..." -ForegroundColor Cyan
    Push-Location frontend
    npm install --silent
    Pop-Location
}

if ($Test) {
    Write-Host "`nRunning tests ..." -ForegroundColor Cyan
    & $VenvPython -m pytest -q
    exit $LASTEXITCODE
}

if ($Evaluate) {
    Write-Host "`nRunning the full evaluation suite ..." -ForegroundColor Cyan
    & $VenvPython -m backend.scripts.generate_dataset
    & $VenvPython -m backend.scripts.fetch_public_data
    & $VenvPython -m backend.scripts.train_risk --model tfidf
    & $VenvPython -m backend.scripts.evaluate_demo_scenarios
    & $VenvPython -m backend.scripts.evaluate_extraction
    & $VenvPython -m backend.scripts.evaluate_entity_resolution
    & $VenvPython -m backend.scripts.evaluate_risk
    & $VenvPython -m backend.scripts.evaluate_retrieval
    & $VenvPython -m backend.scripts.error_analysis
    & $VenvPython -m backend.scripts.make_report
    Write-Host "`nEvaluation complete. See reports/evaluation_report.md" -ForegroundColor Green
    exit 0
}

$runArgs = @("-m", "backend.run")
if ($Reset) { $runArgs += "--reset" }
if ($NoDemo) { $runArgs += "--no-demo" }
if ($NoReindex) { $runArgs += "--no-reindex" }
if ($SkipDownload) { $runArgs += "--skip-download" }
if ($BindAddress) { $runArgs += @("--host", $BindAddress) }
if ($Port -gt 0) { $runArgs += @("--port", $Port) }

& $VenvPython @runArgs
