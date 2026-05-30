# ============================================================================
# purge_secrets_from_git.ps1
# Removes sensitive files from entire git history using git-filter-repo.
#
# PREREQUISITES:
#   pip install git-filter-repo
#
# USAGE:
#   1. BACK UP your repo first!
#   2. Run this script from the repo root.
#   3. Force-push to overwrite remote history: git push --force --all
#
# WARNING: This rewrites git history. All collaborators must re-clone.
# ============================================================================

Write-Host "============================================" -ForegroundColor Yellow
Write-Host "  Git History Secret Purge Tool" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow
Write-Host ""

# Check if git-filter-repo is installed
$filterRepo = git filter-repo --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] git-filter-repo not found. Install it with:" -ForegroundColor Red
    Write-Host "  pip install git-filter-repo" -ForegroundColor Cyan
    exit 1
}

Write-Host "[INFO] git-filter-repo found: $filterRepo" -ForegroundColor Green
Write-Host ""

# Files to purge from history
$filesToPurge = @(
    "config/settings.yaml",
    "config/google_vision_key.json"
)

Write-Host "[WARNING] This will permanently rewrite git history!" -ForegroundColor Red
Write-Host "Files to purge from ALL commits:" -ForegroundColor Yellow
foreach ($f in $filesToPurge) {
    Write-Host "  - $f" -ForegroundColor Cyan
}
Write-Host ""

$confirm = Read-Host "Type 'PURGE' to continue (any other input cancels)"
if ($confirm -ne "PURGE") {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "[1/3] Purging files from history..." -ForegroundColor Green

foreach ($f in $filesToPurge) {
    Write-Host "  Removing: $f" -ForegroundColor Cyan
    git filter-repo --invert-paths --path $f --force
}

Write-Host ""
Write-Host "[2/3] Cleaning up reflog and garbage..." -ForegroundColor Green
git reflog expire --expire=now --all
git gc --prune=now --aggressive

Write-Host ""
Write-Host "[3/3] Done!" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Verify the repo is clean: git log --all --diff-filter=A -- config/settings.yaml" -ForegroundColor Cyan
Write-Host "  2. Force-push to remote:    git push --force --all" -ForegroundColor Cyan
Write-Host "  3. Rotate ALL API keys on their provider dashboards:" -ForegroundColor Cyan
Write-Host "     - DeepL:   https://www.deepl.com/your-account/keys" -ForegroundColor White
Write-Host "     - Google:  https://aistudio.google.com/app/apikey" -ForegroundColor White
Write-Host "     - Groq:    https://console.groq.com/keys" -ForegroundColor White
Write-Host "     - Sarvam:  https://dashboard.sarvam.ai" -ForegroundColor White
Write-Host "  4. Tell all collaborators to re-clone the repo." -ForegroundColor Cyan
