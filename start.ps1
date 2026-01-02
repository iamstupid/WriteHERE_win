param(
  [int]$BackendPort = 5001,
  [int]$FrontendPort = 3000
)

$ErrorActionPreference = "Stop"

function Write-Log([string]$Message) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Write-Host "$ts - $Message"
}

Write-Log "Starting WriteHERE on Windows..."
Write-Log "BackendPort=$BackendPort FrontendPort=$FrontendPort"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-not (Test-Path -Path "venv")) {
  Write-Log "Creating Python venv..."
  python -m venv venv
}

Write-Log "Activating venv..."
& (Join-Path $repoRoot "venv\\Scripts\\Activate.ps1")

Write-Log "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -v -e .
python -m pip install -r backend\\requirements.txt

Write-Log "Starting backend..."
$backend = Start-Process -FilePath python -ArgumentList @("backend\\server.py", "--port", "$BackendPort") -WorkingDirectory $repoRoot -PassThru
Write-Log "Backend PID=$($backend.Id)"

Write-Log "Starting frontend..."
if (-not (Test-Path -Path "frontend\\node_modules")) {
  Write-Log "Installing frontend dependencies..."
  Push-Location "frontend"
  npm install
  Pop-Location
}

$env:PORT = "$FrontendPort"
$env:REACT_APP_BACKEND_PORT = "$BackendPort"
$env:BROWSER = "none"
$npmCmd = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Path
if (-not $npmCmd) { $npmCmd = "npm.cmd" }
$frontend = Start-Process -FilePath $npmCmd -ArgumentList @("start") -WorkingDirectory (Join-Path $repoRoot "frontend") -PassThru
Write-Log "Frontend PID=$($frontend.Id)"

Write-Log "Open http://localhost:$FrontendPort"
Start-Sleep -Seconds 2
Start-Process "http://localhost:$FrontendPort" | Out-Null

try {
  Write-Log "Press Ctrl+C to stop."
  Wait-Process -Id @($backend.Id, $frontend.Id)
} finally {
  Write-Log "Stopping..."
  if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
  if (-not $frontend.HasExited) { Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue }
  Write-Log "Stopped."
}
