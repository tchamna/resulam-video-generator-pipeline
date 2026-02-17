#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Test script for audio production pipeline
.DESCRIPTION
    Tests the pipeline with limited data to ensure no breaking changes.
    Creates detailed log and verifies folder structure.
.EXAMPLE
    .\test_pipeline.ps1
#>

param(
    [Parameter(Mandatory=$false)]
    [int]$StartSentence = 1,
    
    [Parameter(Mandatory=$false)]
    [int]$EndSentence = 5,
    
    [Parameter(Mandatory=$false)]
    [string]$Language = "Twi",
    
    [Parameter(Mandatory=$false)]
    [switch]$FullPipeline,
    
    [Parameter(Mandatory=$false)]
    [switch]$CleanBefore,
    
    [Parameter(Mandatory=$false)]
    [switch]$NoCleanup
)

$ErrorActionPreference = "Stop"
$WarningPreference = "Continue"

# Colors for output
$colors = @{
    success = "Green"
    error = "Red"
    warning = "Yellow"
    info = "Cyan"
    step = "Magenta"
}

function Write-Colored {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Section {
    param([string]$Title)
    Write-Host "`n" + ("=" * 60) -ForegroundColor $colors.step
    Write-Colored "▶ $Title" $colors.step
    Write-Host ("=" * 60) -ForegroundColor $colors.step
}

# Set up paths
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LangPhrasebook = "$(${Language}Phrasebook"
$PrivateAssetsBase = Join-Path $ScriptRoot "private_assets" "Languages" $LangPhrasebook
$ResultsAudios = Join-Path $PrivateAssetsBase "Results_Audios"
$ResultsAudiosNormalPace = Join-Path $PrivateAssetsBase "Results_Audios_normal_pace"
$LogFile = Join-Path $ScriptRoot "pipeline_test_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Section "Pipeline Test Configuration"
Write-Host "Start Sentence: $StartSentence"
Write-Host "End Sentence: $EndSentence"
Write-Host "Language: $Language"
Write-Host "Full Pipeline: $FullPipeline"
Write-Host "Clean Before: $CleanBefore"
Write-Host "Log File: $LogFile"
Write-Host "Results Dir: $ResultsAudios"

# Clean before if requested
if ($CleanBefore) {
    Write-Section "Cleaning Previous Results"
    $dirs_to_clean = @(
        $ResultsAudios,
        $ResultsAudiosNormalPace
    )
    
    foreach ($dir in $dirs_to_clean) {
        if (Test-Path $dir) {
            Write-Colored "🧹 Removing $dir" $colors.warning
            Remove-Item -Path $dir -Recurse -Force -ErrorAction SilentlyContinue
            Write-Colored "✓ Removed" $colors.success
        }
    }
}

# Run test
Write-Section "Running Pipeline Test"

try {
    # Set environment variables
    $env:START_SENTENCE = $StartSentence
    $env:END_SENTENCE = $EndSentence
    $env:USE_PRIVATE_ASSETS = "1"
    
    if ($FullPipeline) {
        Write-Colored "Running FULL pipeline..." $colors.info
        $cmd = "python .\step0_main_pipeline.py"
    } else {
        Write-Colored "Running step1 audio processing (test mode)..." $colors.info
        $cmd = "python .\step1_audio_processing_v2.py"
    }
    
    # Run with tee to both console and log
    Write-Host "`nCommand: $cmd`n"
    Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath $LogFile
    
    Write-Colored "`n✅ Pipeline execution completed" $colors.success
}
catch {
    Write-Colored "❌ Pipeline execution failed: $_" $colors.error
    exit 1
}

# Verify results
Write-Section "Verifying Output Structure"

$expected_folders = @(
    "lecture_gen1_normalized",
    "lecture_gen2_normalized_padded",
    "lecture_gen3_bilingual_sentences",
    "bilingual_sentences_chapters",
    "bilingual_sentences_chapters_background"
)

$all_exist = $true
foreach ($folder in $expected_folders) {
    $path = Join-Path $ResultsAudios $folder
    if (Test-Path $path) {
        $fileCount = @(Get-ChildItem -Path $path -File).Count
        Write-Colored "✓ $folder ($fileCount files)" $colors.success
    } else {
        Write-Colored "✗ $folder - NOT FOUND" $colors.error
        $all_exist = $false
    }
}

# Check normal pace folders if full pipeline
if ($FullPipeline) {
    Write-Host "`nNormal Pace Folders:"
    $np_folders = @(
        "normal_rythm",
        "lecture_gen1_normalized",
        "lecture_gen2_normalized_padded",
        "lecture_gen3_bilingual_sentences",
        "bilingual_sentences_chapters"
    )
    
    foreach ($folder in $np_folders) {
        $path = Join-Path $ResultsAudiosNormalPace $folder
        if (Test-Path $path) {
            $fileCount = @(Get-ChildItem -Path $path -File).Count
            Write-Colored "✓ $folder ($fileCount files)" $colors.success
        } else {
            Write-Colored "✗ $folder - NOT FOUND" $colors.error
            $all_exist = $false
        }
    }
}

# Summary
Write-Section "Test Summary"
if ($all_exist) {
    Write-Colored "✅ All expected folders created successfully!" $colors.success
    Write-Colored "📁 Results saved to: $ResultsAudios" $colors.info
    if ($FullPipeline) {
        Write-Colored "📁 Normal Pace results saved to: $ResultsAudiosNormalPace" $colors.info
    }
} else {
    Write-Colored "⚠️  Some folders missing - check log for errors" $colors.warning
}

Write-Colored "📋 Log file: $LogFile" $colors.info

# Check log for errors
Write-Host "`nChecking log for errors..."
$errors = @(Get-Content $LogFile | Select-String -Pattern "error|failed|exception" -ErrorAction SilentlyContinue)
if ($errors.Count -gt 0) {
    Write-Colored "⚠️  Found potential errors in log:" $colors.warning
    $errors | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Colored "✓ No errors detected in log" $colors.success
}

# Optional cleanup suggestion
if (-not $NoCleanup) {
    Write-Host "`n" + ("=" * 60)
    Write-Colored "Tip: To clean results, run:" $colors.info
    Write-Host "  Remove-Item -Path '$ResultsAudios' -Recurse -Force"
    Write-Host "  Remove-Item -Path '$ResultsAudiosNormalPace' -Recurse -Force"
}

Write-Host "`n" + ("=" * 60)
Write-Colored "✅ Test completed successfully!" $colors.success
Write-Host ("=" * 60)
