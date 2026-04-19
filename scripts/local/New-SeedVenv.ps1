param(
    [string]$VenvPath = ".venv-seed",
    [string]$EnvFile = ".env.seed",
    [string]$PythonVersion = "3.10",
    [switch]$ForceRecreate,
    [switch]$SkipTorch,
    [switch]$DryRun
)

. "$PSScriptRoot\\MiniMedLocalCommon.ps1"

Import-MiniMedEnvFile -EnvFile $EnvFile
Initialize-MiniMedCacheDirs

$repoRoot = Get-MiniMedRepoRoot
$venvRoot = Resolve-MiniMedRepoPath $VenvPath
$venvPython = Get-MiniMedVenvPython -VenvPath $VenvPath
$bootstrapPython = Get-MiniMedBootstrapPython -PythonVersion $PythonVersion
$requirementsPath = Resolve-MiniMedRepoPath "requirements-local-seed.txt"

Write-MiniMedSection -Title "Seed Venv Bootstrap"
Write-Host "Repo root: $repoRoot"
Write-Host "Venv path: $venvRoot"
Write-Host "Requirements: $requirementsPath"

if ($ForceRecreate -and (Test-Path $venvRoot)) {
    Write-Host "Removing existing venv: $venvRoot"
    if (-not $DryRun) {
        Remove-Item -LiteralPath $venvRoot -Recurse -Force
    }
}

if (-not (Test-Path $venvPython)) {
    Invoke-MiniMedCommand -Command ($bootstrapPython + @("-m", "venv", $venvRoot)) -DryRun:$DryRun
}
else {
    Write-Host "Existing venv detected: $venvPython"
}

Invoke-MiniMedCommand -Command @($venvPython, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<82") -DryRun:$DryRun

if (-not $SkipTorch) {
    Invoke-MiniMedCommand -Command @(
        $venvPython, "-m", "pip", "install",
        "torch",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ) -DryRun:$DryRun
}

Invoke-MiniMedCommand -Command @($venvPython, "-m", "pip", "install", "-r", $requirementsPath) -DryRun:$DryRun
Invoke-MiniMedCommand -Command @(
    $venvPython, "-c",
    "import sys, torch, spacy; print({'python': sys.version.split()[0], 'torch': getattr(torch, '__version__', 'missing'), 'cuda': bool(getattr(torch, 'cuda', None) and torch.cuda.is_available()), 'spacy': spacy.__version__})"
) -DryRun:$DryRun

Write-MiniMedSection -Title "Next"
Write-Host "Activate manually if needed:"
Write-Host "  $venvRoot\\Scripts\\Activate.ps1"
Write-Host "Then run:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/local/Invoke-SystemSmoke.ps1"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/local/Invoke-SeedBuild.ps1 -Profile smoke-10"
