# Run ChessPlayer in development mode.
# Must be invoked from the repo root: .\scripts\run_dev.ps1

$RepoRoot = Split-Path -Parent $PSScriptRoot
$AppDir   = Join-Path $RepoRoot "src\chessplayer"
$Venv     = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"

if (Test-Path $Venv) {
    . $Venv
}

Set-Location $AppDir
python main.py @args
