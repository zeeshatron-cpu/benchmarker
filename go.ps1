# Plug-and-go launcher for the real Fenzo comparison (Windows PowerShell).
#
#   1. Copy .env.example to .env and paste your API keys into it (once).
#   2. Capture a Fenzo login once:
#        playwright codegen --save-storage=fenzo_auth.json https://fenzo.ai/home
#   3. Run:  ./go.ps1  "Explain a balance sheet in one paragraph."
#
# Passing a query is optional; omit it to just run whatever is already recorded.

param([string]$Query)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env — open it, paste your API keys, then run ./go.ps1 again." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path "fenzo_auth.json")) {
    Write-Host "fenzo_auth.json not found. Capture a Fenzo login first:" -ForegroundColor Yellow
    Write-Host "  playwright codegen --save-storage=fenzo_auth.json https://fenzo.ai/home"
    exit 1
}

if ($Query) {
    python -m benchmarker -c config.compare.yaml record "$Query"
}

python -m benchmarker -c config.compare.yaml run

# Open the newest report.
$report = Get-ChildItem reports\*.md | Sort-Object LastWriteTime | Select-Object -Last 1
if ($report) { notepad $report.FullName }
