$ErrorActionPreference = "Stop"

param(
  [Parameter(Mandatory=$true)]
  [string]$SpaceRepo,  # e.g. "mano678/DART_1"

  [Parameter(Mandatory=$false)]
  [string]$WorkDir = "$env:USERPROFILE\Desktop\hf_space_sync"
)

Write-Host "== Digital Twin -> Hugging Face Space sync ==" -ForegroundColor Cyan
Write-Host "SpaceRepo: $SpaceRepo"
Write-Host "WorkDir:   $WorkDir"

if (-not (Test-Path $WorkDir)) {
  New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
}

$spaceUrl = "https://huggingface.co/spaces/$SpaceRepo"
$clonePath = Join-Path $WorkDir ($SpaceRepo.Replace("/", "__"))

Write-Host "`n[1/5] Ensuring HF CLI installed..." -ForegroundColor Yellow
python -m pip install -U huggingface_hub | Out-Null

Write-Host "`n[2/5] Cloning (or updating) Space repo..." -ForegroundColor Yellow
if (Test-Path $clonePath) {
  Set-Location $clonePath
  git fetch --all
  git reset --hard origin/main
} else {
  Set-Location $WorkDir
  git clone $spaceUrl $clonePath
  Set-Location $clonePath
}

Write-Host "`n[3/5] Copying project files into Space repo..." -ForegroundColor Yellow
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Clean Space working tree except .git
Get-ChildItem -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force

# Copy everything from project root
Copy-Item -Recurse -Force (Join-Path $projectRoot "*") $clonePath

Write-Host "`n[4/5] Committing changes..." -ForegroundColor Yellow
Set-Location $clonePath
git add .
try {
  git commit -m "Sync Space with GitHub project"
} catch {
  Write-Host "No new changes to commit." -ForegroundColor DarkGray
}

Write-Host "`n[5/5] Pushing to Hugging Face..." -ForegroundColor Yellow
Write-Host "If this fails with auth error: run 'huggingface-cli login' and retry." -ForegroundColor Magenta
git push

Write-Host "`nDone. Now go to Space -> Settings -> Factory rebuild." -ForegroundColor Green

