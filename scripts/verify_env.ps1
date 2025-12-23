Write-Host "Verifying environment (.venv_311)..."

if (-not (Test-Path -Path ".venv_311")) {
    Write-Host "ERROR: .venv_311 not found. Run scripts/setup_env.ps1 first." -ForegroundColor Red
    exit 1
}

# Check python version
$py = ".\.venv_311\Scripts\python.exe"
$ver = & $py --version 2>&1
Write-Host "Python in venv:" $ver
if ($ver -notmatch "3\.11\.") {
    Write-Host "WARNING: Expected Python 3.11.x in .venv_311." -ForegroundColor Yellow
}

# Compare pip freeze to locked requirements
if (-not (Test-Path -Path "requirements.lock")) {
    Write-Host "No requirements.lock found. Run scripts/setup_env.ps1 to generate it." -ForegroundColor Yellow
    exit 0
}

& $py -m pip freeze | Out-File -Encoding utf8 current.freeze

$diff = Compare-Object -ReferenceObject (Get-Content requirements.lock) -DifferenceObject (Get-Content current.freeze)
if ($diff) {
    Write-Host "Dependency differences detected:" -ForegroundColor Yellow
    $diff | ForEach-Object { Write-Host $_ }
    exit 2
} else {
    Write-Host "Dependencies match requirements.lock" -ForegroundColor Green
}

# Check ffmpeg availability
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    & ffmpeg -version | Select-Object -First 1 | Write-Host
} else {
    Write-Host "ffmpeg not found on PATH. Install ffmpeg and ensure it's available." -ForegroundColor Yellow
}
