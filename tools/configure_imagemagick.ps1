<#
PowerShell helper: find ImageMagick and add its folder to the current user's PATH
and set IMAGEMAGICK_BINARY to the magick.exe path. Safe to run multiple times.

Usage (run from repo root):
  powershell -ExecutionPolicy Bypass -File .\tools\configure_imagemagick.ps1
#>

Write-Output "Searching for ImageMagick installations..."
$candidates = @(
    'C:\Program Files\ImageMagick*',
    'C:\Program Files (x86)\ImageMagick*'
)

$found = $null
foreach ($pat in $candidates) {
    Get-ChildItem -Path (Split-Path $pat) -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $dir = $_.FullName
        $exe = Join-Path $dir 'magick.exe'
        if (Test-Path $exe) { $found = $exe; return }
    }
    if ($found) { break }
}

if (-not $found) {
    Write-Warning "No magick.exe found in common locations. If you installed ImageMagick, run this script again with the install path as first argument."
    if ($args.Count -ge 1) {
        $candidate = $args[0]
        $exe = Join-Path $candidate 'magick.exe'
        if (Test-Path $exe) { $found = $exe }
    }
}

if (-not $found) {
    Write-Output "No ImageMagick binary located. Exiting without changes."
    exit 0
}

$folder = Split-Path $found -Parent
Write-Output "Found magick at: $found"
Write-Output "Adding $folder to user PATH and setting IMAGEMAGICK_BINARY (User scope)."

$oldPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($oldPath -notlike "*$folder*") {
    $newPath = if ($oldPath) { "$oldPath;$folder" } else { $folder }
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    Write-Output "User PATH updated. Please restart your shell/IDE."
} else {
    Write-Output "User PATH already contains $folder"
}

[Environment]::SetEnvironmentVariable('IMAGEMAGICK_BINARY', $found, 'User')
Write-Output "Set IMAGEMAGICK_BINARY to $found"

Write-Output "Done. Open a new PowerShell and run: magick -version"
