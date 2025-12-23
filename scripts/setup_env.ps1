param(
    [string]$PythonPath = ""
)

# Create a reproducible Python 3.11 venv and install pinned requirements
Write-Host "Setting up environment (.venv_311)..."

if (-not (Test-Path -Path ".venv_311")) {
    if ($PythonPath -ne "") {
        $py = $PythonPath
    } else {
        # Try to find or install `uv` if possible
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uv) {
            Write-Host "uv not found. Attempting to install 'uv' using recommended methods..."

            # 1) Try WinGet
            if (Get-Command winget -ErrorAction SilentlyContinue) {
                Write-Host "Trying winget install..."
                try { & winget install --id=astral-sh.uv -e --silent | Out-Null } catch { }
            }
            $uv = Get-Command uv -ErrorAction SilentlyContinue
            if (-not $uv) {
                # 2) Try Scoop
                if (Get-Command scoop -ErrorAction SilentlyContinue) {
                    Write-Host "Trying scoop install..."
                    try { & scoop install main/uv | Out-Null } catch { }
                }
            }
            $uv = Get-Command uv -ErrorAction SilentlyContinue
            if (-not $uv) {
                # 3) Try pipx install if available
                if (Get-Command pipx -ErrorAction SilentlyContinue) {
                    Write-Host "Trying pipx install uv..."
                    try { & pipx install uv | Out-Null } catch { }
                }
            }
            $uv = Get-Command uv -ErrorAction SilentlyContinue
            if (-not $uv) {
                # 4) Fallback to pip --user
                Write-Host "Trying pip --user install uv..."
                try {
                    & python -m pip install --user uv | Out-Null
                    # Add common user script locations to PATH for this session so uv can be found
                    $userScripts1 = Join-Path $env:APPDATA "Python\Scripts"
                    $userScripts2 = Join-Path $env:LOCALAPPDATA "Programs\Python\Scripts"
                    if (Test-Path $userScripts1) { $env:Path = $userScripts1 + ";" + $env:Path }
                    if (Test-Path $userScripts2) { $env:Path = $userScripts2 + ";" + $env:Path }
                } catch {
                    Write-Host "Automatic uv install failed or was skipped: $_" -ForegroundColor Yellow
                }
            }
            $uv = Get-Command uv -ErrorAction SilentlyContinue
        }

        if ($uv) {
            Write-Host "uv found: ensuring Python 3.11 is installed via uv..."
            & uv python install 3.11 | Out-Null
            # Search uv-managed installations and pick a python.exe that reports 3.11
            $uvPythonRoot = Join-Path $env:APPDATA "uv\python"
            $candidates = @(Get-ChildItem -Path $uvPythonRoot -Recurse -Filter python.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
            $py = $null
            foreach ($cand in $candidates) {
                try {
                    $ver = & $cand --version 2>&1
                    if ($ver -match 'Python\s+3\.11') {
                        $py = $cand
                        break
                    }
                } catch {
                    # ignore and continue
                }
            }
            if (-not $py) {
                if ($candidates.Count -gt 0) {
                    Write-Host "No Python 3.11 found under uv; first candidate will be used but may be incompatible:" -ForegroundColor Yellow
                    Write-Host $candidates[0]
                    $py = $candidates[0]
                } else {
                    $py = "python"
                }
            }
        } else {
            $py = "python"
        }
    }

    Write-Host "Using Python executable: $py"

    # Validate selected python is 3.11.x; fail fast if not
    try {
        $selVer = & $py --version 2>&1
    } catch {
        $selVer = ""
    }
    if ($selVer -notmatch 'Python\s+3\.11') {
        # Try Python launcher fallback: py -3.11
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) {
            try {
                $py311 = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
                if ($py311) {
                    Write-Host "Found Python 3.11 via 'py -3.11': $py311"
                    $py = $py311.Trim()
                    $selVer = & $py --version 2>&1
                }
            } catch {
                # ignore
            }
        }
    }
    if ($selVer -notmatch 'Python\s+3\.11') {
        Write-Host "ERROR: Selected Python ($py) is: $selVer" -ForegroundColor Red
        Write-Host "This setup requires Python 3.11. Install via 'uv python install 3.11' or provide -PythonPath pointing to a Python 3.11 executable, or install Python 3.11 and ensure it's on PATH." -ForegroundColor Red
        exit 1
    }

    & $py -m venv .venv_311
} else {
    Write-Host ".venv_311 already exists. Skipping venv creation."
}

# Activate and install
Write-Host "Activating venv and installing requirements..."
. .\.venv_311\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Prefer reproducible lockfile when present
if (Test-Path "requirements.lock") {
    Write-Host "Found requirements.lock — installing exact pinned dependencies..."
    pip install -r requirements.lock
} else {
    Write-Host "No requirements.lock found — installing top-level requirements (requirements.txt) and generating lockfile..."
    pip install -r requirements.txt
    pip freeze > requirements.lock
}

Write-Host "Done. Activate with: .\.venv_311\Scripts\Activate.ps1"
