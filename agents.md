# MiniMed Prime Agent Playbook

## Skill: Push Kaggle Source

Use this workflow when the user asks to push updated MiniMed Prime code/assets to Kaggle.

### Targets

- Project repo: `D:\CapstoneProjectSP26\MiniMed_Prime`
- Kaggle upload folder: `D:\CapstoneProjectSP26\kaggle_assets\minimed-prime-source-upload`
- Kaggle dataset: `huynhnhuthuyk18hcm/minimed-prime-source`
- Required Kaggle config file: `D:\CapstoneProjectSP26\kaggle_assets\.kaggle\kaggle.json`

Do not use or commit any other `kaggle.json` unless explicitly requested. Never print the Kaggle key.

### Preflight

Run from the project repo:

```powershell
Set-Location D:\CapstoneProjectSP26\MiniMed_Prime
git status --short
Test-Path D:\CapstoneProjectSP26\kaggle_assets\.kaggle\kaggle.json
$env:KAGGLE_CONFIG_DIR = 'D:\CapstoneProjectSP26\kaggle_assets\.kaggle'
kaggle --version
```

If `kaggle.json` is missing, stop and ask the user to restore it at:

```text
D:\CapstoneProjectSP26\kaggle_assets\.kaggle\kaggle.json
```

### Validate Before Packaging

Run the relevant tests for the files changed. Typical commands:

```powershell
python -m py_compile src\models\trm_wrapper.py src\layers\layer4_judges.py scripts\audit_checkpoint.py scripts\test_trm_isolated.py
python -m pytest tests\test_layer4_judges.py tests\test_schemas.py -q
python -m pytest tests\test_layer3_trm.py tests\test_trm_trainer.py -q
```

Use narrower tests if the change is small, but record exactly what passed or failed.

### Update Source Version Marker

Update both project and upload marker files. Use a clear version string tied to the ticket or change.

```powershell
$repo = 'D:\CapstoneProjectSP26\MiniMed_Prime'
$upload = 'D:\CapstoneProjectSP26\kaggle_assets\minimed-prime-source-upload'
$versionText = @'
MiniMed Prime Kaggle source package
version: YYYY-MM-DD-ticket-XXX-short-description
notes:
- Short bullet describing the code change
- Short bullet describing tests or operational impact
'@
Set-Content -Path (Join-Path $repo 'KAGGLE_SOURCE_VERSION.txt') -Value $versionText -Encoding UTF8
Set-Content -Path (Join-Path $upload 'KAGGLE_SOURCE_VERSION.txt') -Value $versionText -Encoding UTF8
```

### Rebuild Kaggle Source Package

Copy root docs and rebuild archive files:

```powershell
$repo = 'D:\CapstoneProjectSP26\MiniMed_Prime'
$upload = 'D:\CapstoneProjectSP26\kaggle_assets\minimed-prime-source-upload'

Copy-Item -LiteralPath (Join-Path $repo 'README_KAGGLE_VI.md') -Destination $upload -Force
Copy-Item -LiteralPath (Join-Path $repo 'TRAINING_KAGGLE_LOCAL_VI.md') -Destination $upload -Force
Copy-Item -LiteralPath (Join-Path $repo 'IMPLEMENTATION_AUDIT.md') -Destination $upload -Force
Copy-Item -LiteralPath (Join-Path $repo 'requirements-integration.txt') -Destination $upload -Force

Compress-Archive -LiteralPath (Join-Path $repo 'src') -DestinationPath (Join-Path $upload 'src.zip') -Force
Compress-Archive -LiteralPath (Join-Path $repo 'scripts') -DestinationPath (Join-Path $upload 'scripts.zip') -Force
Compress-Archive -LiteralPath (Join-Path $repo 'tools') -DestinationPath (Join-Path $upload 'tools.zip') -Force
Compress-Archive -LiteralPath (Join-Path $repo 'judges') -DestinationPath (Join-Path $upload 'judges.zip') -Force
Compress-Archive -LiteralPath (Join-Path $repo 'prompts') -DestinationPath (Join-Path $upload 'prompts.zip') -Force
Compress-Archive -LiteralPath (Join-Path $repo 'schemas') -DestinationPath (Join-Path $upload 'schemas.zip') -Force
Compress-Archive -LiteralPath (Join-Path $repo 'tests') -DestinationPath (Join-Path $upload 'tests.zip') -Force
Compress-Archive -LiteralPath (Join-Path $repo 'notebooks') -DestinationPath (Join-Path $upload 'notebooks.zip') -Force
```

Verify important files are inside the zip archives:

```powershell
@'
import zipfile
from pathlib import Path
base = Path(r'D:\CapstoneProjectSP26\kaggle_assets\minimed-prime-source-upload')
checks = {
    'src.zip': ['src/models/trm_wrapper.py', 'src/layers/layer4_judges.py'],
    'scripts.zip': ['scripts/audit_checkpoint.py', 'scripts/test_trm_isolated.py'],
    'tools.zip': ['tools/trace_one_sample.py'],
    'judges.zip': ['judges/posthoc_auditor.py'],
    'prompts.zip': ['prompts/posthoc_auditor_system.txt'],
    'schemas.zip': ['schemas/posthoc_auditor_schema.py'],
    'tests.zip': ['tests/test_layer4_judges.py'],
}
for archive_name, probes in checks.items():
    with zipfile.ZipFile(base / archive_name) as zf:
        names = set(zf.namelist())
    print(archive_name)
    for probe in probes:
        print(' ', probe, probe in names)
'@ | python -
```

### Push Kaggle Dataset Version

Use only the configured Kaggle config directory:

```powershell
$env:KAGGLE_CONFIG_DIR = 'D:\CapstoneProjectSP26\kaggle_assets\.kaggle'
kaggle datasets version `
  -p D:\CapstoneProjectSP26\kaggle_assets\minimed-prime-source-upload `
  -m "TICKET-XXX short Kaggle source update"
```

Verify Kaggle accepted the files:

```powershell
$env:KAGGLE_CONFIG_DIR = 'D:\CapstoneProjectSP26\kaggle_assets\.kaggle'
kaggle datasets files huynhnhuthuyk18hcm/minimed-prime-source --page-size 200 |
  Select-String -Pattern 'KAGGLE_SOURCE_VERSION|src/src/models/trm_wrapper.py|src/src/layers/layer4_judges.py|scripts/scripts/audit_checkpoint.py|scripts/scripts/test_trm_isolated.py'
```

Kaggle may index zip contents as nested paths such as `src/src/...` and `scripts/scripts/...`. That is expected for this dataset layout.

### Push Code Branch

Do not commit credentials or local artifacts. Keep `external/`, `data/`, `.env*`, and `kaggle.json` out of git.

```powershell
Set-Location D:\CapstoneProjectSP26\MiniMed_Prime
git status --short
git switch -c ticket-XXX-short-name
git add src scripts tests KAGGLE_SOURCE_VERSION.txt agents.md
git status --short
git commit -m "TICKET-XXX: short description"
git push -u origin ticket-XXX-short-name
```

If already on the intended branch:

```powershell
git add src scripts tests KAGGLE_SOURCE_VERSION.txt agents.md
git commit -m "TICKET-XXX: short description"
git push
```

### Final Response Checklist

Report:

- Kaggle dataset id and version message.
- Files verified in Kaggle listing.
- Tests run and results.
- Git branch name, commit hash, and push status if a branch push was requested.
- Any skipped step and the exact reason.
