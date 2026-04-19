param(
    [ValidateSet("smoke-5", "smoke-10", "pilot-100", "standard-3000", "standard-5000")]
    [string]$Profile = "smoke-10",
    [string]$VenvPath = ".venv-seed",
    [string]$EnvFile = ".env.seed",
    [string]$Source = "",
    [ValidateSet("auto", "direct", "heuristic", "llm")]
    [string]$EdgeMapper = "auto",
    [string]$LlmModelName = "",
    [string]$LlmDevice = "cpu",
    [string]$RelationFilter = "",
    [switch]$AllowEmptyGold,
    [switch]$BuildTrmArrays,
    [switch]$RunSystemSmoke,
    [switch]$DryRun
)

. "$PSScriptRoot\\MiniMedLocalCommon.ps1"

Import-MiniMedEnvFile -EnvFile $EnvFile
Initialize-MiniMedCacheDirs
if (-not $DryRun) {
    Assert-MiniMedVenvExists -VenvPath $VenvPath
}

$repoRoot = Get-MiniMedRepoRoot
$venvPython = Get-MiniMedVenvPython -VenvPath $VenvPath
$seedDefaults = Get-MiniMedSeedDefaults
if ([string]::IsNullOrWhiteSpace($Source)) {
    $Source = $seedDefaults.Source
}
if ([string]::IsNullOrWhiteSpace($LlmModelName)) {
    $LlmModelName = $seedDefaults.LlmModelName
}

$profileMap = @{
    "smoke-5" = @{ TrainLimit = 5; ValidationLimit = 1; TrainMaxSaved = 5; ValidationMaxSaved = 1 }
    "smoke-10" = @{ TrainLimit = 10; ValidationLimit = 2; TrainMaxSaved = 10; ValidationMaxSaved = 2 }
    "pilot-100" = @{ TrainLimit = 100; ValidationLimit = 10; TrainMaxSaved = 100; ValidationMaxSaved = 10 }
    "standard-3000" = @{ TrainLimit = 3000; ValidationLimit = 300; TrainMaxSaved = 3000; ValidationMaxSaved = 300 }
    "standard-5000" = @{ TrainLimit = 5000; ValidationLimit = 500; TrainMaxSaved = 5000; ValidationMaxSaved = 500 }
}
$selected = $profileMap[$Profile]
$outputRoot = Resolve-MiniMedRepoPath (Join-Path "data\\seed_runs" $Profile)
$rawDir = Join-Path $outputRoot "raw"
$qualityDir = Join-Path $outputRoot "quality"
$trainQualityDir = Join-Path $qualityDir "train"
$validationQualityDir = Join-Path $qualityDir "validation"
$trmOutputDir = Join-Path $outputRoot "trm_medical"

if (-not $DryRun) {
    New-Item -ItemType Directory -Path $rawDir -Force | Out-Null
    New-Item -ItemType Directory -Path $trainQualityDir -Force | Out-Null
    New-Item -ItemType Directory -Path $validationQualityDir -Force | Out-Null
    if ($BuildTrmArrays) {
        New-Item -ItemType Directory -Path $trmOutputDir -Force | Out-Null
    }
}

$trainSeedPath = Join-Path $rawDir "trm_seed_train.jsonl"
$validationSeedPath = Join-Path $rawDir "trm_seed_validation.jsonl"
$trainSummaryPath = Join-Path $trainQualityDir "seed_quality_summary.json"
$validationSummaryPath = Join-Path $validationQualityDir "seed_quality_summary.json"

function Invoke-SeedPreparation {
    param(
        [string]$Split,
        [string]$OutputPath,
        [int]$Limit,
        [int]$MaxSaved
    )

    $command = @(
        $venvPython,
        "scripts/prepare_medreason_seed.py",
        "--source", $Source,
        "--split", $Split,
        "--output-jsonl", $OutputPath,
        "--edge-mapper", $EdgeMapper,
        "--llm-model-name", $LlmModelName,
        "--llm-device", $LlmDevice,
        "--limit", "$Limit",
        "--max-saved", "$MaxSaved"
    )
    if (-not [string]::IsNullOrWhiteSpace($RelationFilter)) {
        $command += @("--relation-filter", $RelationFilter)
    }
    if ($AllowEmptyGold) {
        $command += "--allow-empty-gold"
    }
    Invoke-MiniMedCommand -Command $command -DryRun:$DryRun
}

function Invoke-QualitySplit {
    param(
        [string]$InputJsonl,
        [string]$OutputDir
    )

    $command = @(
        $venvPython,
        "scripts/filter_seed_quality.py",
        "--input-jsonl", $InputJsonl,
        "--output-dir", $OutputDir
    )
    Invoke-MiniMedCommand -Command $command -DryRun:$DryRun
}

function Get-GoldishCount {
    param(
        [string]$SummaryPath
    )

    if (-not (Test-Path $SummaryPath)) {
        return 0
    }
    $payload = Get-Content $SummaryPath -Raw | ConvertFrom-Json
    return [int]$payload.goldish_records
}

function Invoke-TrmBuild {
    param(
        [string]$InputJsonl,
        [string]$SplitName
    )

    $command = @(
        $venvPython,
        "scripts/build_trm_dataset.py",
        "--input-jsonl", $InputJsonl,
        "--output-dir", $trmOutputDir,
        "--split", $SplitName,
        "--device", "cpu"
    )
    Invoke-MiniMedCommand -Command $command -DryRun:$DryRun
}

Write-MiniMedSection -Title "Seed Build"
Write-Host "Repo root: $repoRoot"
Write-Host "Profile: $Profile"
Write-Host "Source: $Source"
Write-Host "Edge mapper: $EdgeMapper"
Write-Host "LLM model: $LlmModelName"
Write-Host "Output root: $outputRoot"

if ($RunSystemSmoke) {
    Invoke-MiniMedCommand -Command @($venvPython, "scripts/smoke_test.py") -DryRun:$DryRun
}

if (
    ($EdgeMapper -in @("auto", "llm")) -and
    [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY) -and
    [string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY) -and
    ($LlmModelName -eq "heuristic")
) {
    Write-Warning "No OPENAI_API_KEY/GEMINI_API_KEY found and LLM model is heuristic. Quality split will likely produce only silver/reject rows."
}

Invoke-SeedPreparation -Split "train" -OutputPath $trainSeedPath -Limit $selected.TrainLimit -MaxSaved $selected.TrainMaxSaved
Invoke-SeedPreparation -Split "validation" -OutputPath $validationSeedPath -Limit $selected.ValidationLimit -MaxSaved $selected.ValidationMaxSaved

Invoke-QualitySplit -InputJsonl $trainSeedPath -OutputDir $trainQualityDir
Invoke-QualitySplit -InputJsonl $validationSeedPath -OutputDir $validationQualityDir

if ($BuildTrmArrays -and -not $DryRun) {
    $trainGoldishPath = Join-Path $trainQualityDir "trm_seed_goldish.jsonl"
    $validationGoldishPath = Join-Path $validationQualityDir "trm_seed_goldish.jsonl"
    if ((Get-GoldishCount -SummaryPath $trainSummaryPath) -gt 0) {
        Invoke-TrmBuild -InputJsonl $trainGoldishPath -SplitName "train"
    }
    else {
        Write-Warning "Skipping TRM train split because train goldish count is zero."
    }
    if ((Get-GoldishCount -SummaryPath $validationSummaryPath) -gt 0) {
        Invoke-TrmBuild -InputJsonl $validationGoldishPath -SplitName "val"
    }
    else {
        Write-Warning "Skipping TRM val split because validation goldish count is zero."
    }
}

Write-MiniMedSection -Title "Artifacts"
Write-Host "Train seed: $trainSeedPath"
Write-Host "Validation seed: $validationSeedPath"
Write-Host "Train summary: $trainSummaryPath"
Write-Host "Validation summary: $validationSummaryPath"
if ($BuildTrmArrays) {
    Write-Host "TRM dataset dir: $trmOutputDir"
}
