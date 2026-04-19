Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-MiniMedRepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
}

function Resolve-MiniMedRepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    return Join-Path (Get-MiniMedRepoRoot) $RelativePath
}

function Import-MiniMedEnvFile {
    param(
        [string]$EnvFile = ".env.seed"
    )

    $candidate = Resolve-MiniMedRepoPath $EnvFile
    if (-not (Test-Path $candidate)) {
        Write-Host "Env file not found, skipping: $candidate"
        return
    }

    foreach ($line in Get-Content $candidate) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $parts = $trimmed -split "=", 2
        if ($parts.Count -ne 2) {
            continue
        }
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        Set-Item -Path "Env:$name" -Value $value
    }
}

function Initialize-MiniMedCacheDirs {
    foreach ($envName in @("HF_HOME", "TRANSFORMERS_CACHE", "PIP_CACHE_DIR", "TMP", "TEMP")) {
        $value = [Environment]::GetEnvironmentVariable($envName, "Process")
        if ([string]::IsNullOrWhiteSpace($value)) {
            continue
        }
        if (-not (Test-Path $value)) {
            New-Item -ItemType Directory -Path $value -Force | Out-Null
        }
    }
}

function Get-MiniMedBootstrapPython {
    param(
        [string]$PythonVersion = "3.10"
    )

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $py) {
        return @("py", "-$PythonVersion")
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        return @($python.Source)
    }

    throw "Could not find a bootstrap Python executable. Install Python $PythonVersion first."
}

function Get-MiniMedVenvPython {
    param(
        [string]$VenvPath = ".venv-seed"
    )

    return Resolve-MiniMedRepoPath (Join-Path $VenvPath "Scripts\\python.exe")
}

function Assert-MiniMedVenvExists {
    param(
        [string]$VenvPath = ".venv-seed"
    )

    $venvPython = Get-MiniMedVenvPython -VenvPath $VenvPath
    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment is missing: $venvPython. Run scripts/local/New-SeedVenv.ps1 first."
    }
}

function Format-MiniMedCommand {
    param(
        [string[]]$Command
    )

    return ($Command | ForEach-Object {
        if ($_ -match "\s") {
            '"' + $_ + '"'
        }
        else {
            $_
        }
    }) -join " "
}

function Invoke-MiniMedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Command,
        [string]$WorkingDirectory = (Get-MiniMedRepoRoot),
        [switch]$DryRun
    )

    $display = Format-MiniMedCommand -Command $Command
    Write-Host ">> $display"
    if ($DryRun) {
        return
    }

    Push-Location $WorkingDirectory
    try {
        if ($Command.Count -gt 1) {
            & $Command[0] @($Command[1..($Command.Count - 1)])
        }
        else {
            & $Command[0]
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $display"
        }
    }
    finally {
        Pop-Location
    }
}

function Write-MiniMedSection {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title
    )

    Write-Host ""
    Write-Host "=== $Title ==="
}

function Get-MiniMedSeedDefaults {
    return @{
        Source = if ([string]::IsNullOrWhiteSpace($env:MEDREASON_SOURCE)) { "data/medreason" } else { $env:MEDREASON_SOURCE }
        EdgeMapper = if ([string]::IsNullOrWhiteSpace($env:MEDREASON_EDGE_LLM)) { "auto" } else { "auto" }
        LlmModelName = if ([string]::IsNullOrWhiteSpace($env:MEDREASON_EDGE_LLM)) { "heuristic" } else { $env:MEDREASON_EDGE_LLM }
    }
}
