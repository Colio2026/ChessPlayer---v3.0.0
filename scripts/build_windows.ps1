# Build a standalone Windows executable with PyInstaller.
# Run from the repo root: .\scripts\build_windows.ps1
#
# Output: dist\ChessPlayer\ChessPlayer.exe
#
# NOTE: Stockfish is NOT bundled. Users must supply their own binary and
# point engine.path in config to it. See README.md > Stockfish Setup.

$RepoRoot = Split-Path -Parent $PSScriptRoot
$AppDir   = Join-Path $RepoRoot "src\chessplayer"
$Venv     = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"

if (Test-Path $Venv) {
    . $Venv
}

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "PyInstaller not found. Installing..."
    pip install pyinstaller
}

Set-Location $RepoRoot

pyinstaller `
    --name ChessPlayer `
    --windowed `
    --distpath dist `
    --workpath build `
    --specpath build `
    --add-data "assets;assets" `
    --add-data "config\default.yaml;config" `
    --add-data "src\chessplayer\qml;qml" `
    --paths "src\chessplayer" `
    src\chessplayer\main.py

Write-Host ""
Write-Host "Build complete: dist\ChessPlayer\ChessPlayer.exe"
Write-Host "Remember to place your Stockfish binary at the path set in config/default.yaml."
