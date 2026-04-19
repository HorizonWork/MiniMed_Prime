param(
    [string]$VenvPath = ".venv-seed",
    [string]$EnvFile = ".env.seed",
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

Write-MiniMedSection -Title "System Smoke"
Write-Host "Repo root: $repoRoot"
Write-Host "Judge model: $env:MINIMED_JUDGE_MODEL"
Write-Host "Synthesis model: $env:MINIMED_SYNTHESIS_MODEL"

Invoke-MiniMedCommand -Command @($venvPython, "scripts/smoke_test.py") -DryRun:$DryRun
