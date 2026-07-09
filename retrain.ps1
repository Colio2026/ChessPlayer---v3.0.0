# retrain.ps1 -- Full pipeline: parse -> train -> evaluate -> inspect
# Run from the project root: .\retrain.ps1
#
# Data sources (in order):
#   1. Annotated PGNs  -- dynamic/meta concepts (tempo, combination, fortification)
#   2. Lichess CSV     -- tactical concepts (pin, fork, sacrifice) + all algo labels
#   3. Game databases  -- structural/strategic concepts (battery, blockade, space)
#      Sources: data/Caissabase.pgn, data/Carlsen.pgn, data/lichess_elite_2020-10.pgn

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# -- 1. Parse all annotated PGNs ----------------------------------------------
Step "Parsing annotated PGNs"
python -m tools.parse_annotated_pgn `
    --input data/annotated_pgns `
    --output data/training_raw.jsonl
if (-not $?) { Write-Error "Parse step failed"; exit 1 }

# -- 2. Ingest Lichess puzzle CSV (tag mapping + algorithmic labels) -----------
Step "Ingesting Lichess puzzle CSV"
python -m tools.ingest_lichess_csv `
    --input data/lichess_db_puzzle.csv `
    --output data/training_raw.jsonl `
    --append
if (-not $?) { Write-Error "Lichess CSV ingest failed"; exit 1 }

# -- 3. Ingest game databases (algorithmic labels on master positions) ----------
# Caissabase + Carlsen + Lichess Elite give rich positions for structural
# concepts that puzzles don't tag: battery, blockade, pawn_storm, space_advantage,
# minority_attack, color_complex, development_lead, piece_activity, etc.
Step "Ingesting game databases"
$db_files = @()
foreach ($f in @("data/Caissabase.pgn", "data/Carlsen.pgn", "data/lichess_elite_2020-10.pgn")) {
    if (Test-Path $f) { $db_files += $f }
}
if ($db_files.Count -eq 0) {
    Write-Host "  WARNING: no game database PGNs found in data/ - skipping this step" -ForegroundColor Yellow
} else {
    Write-Host "  Found: $($db_files -join ', ')"
    $db_args = @("tools/ingest_game_database.py", "--input") + $db_files + @("--output", "data/training_raw.jsonl", "--append")
    & python $db_args
    if (-not $?) { Write-Error "Game database ingest failed"; exit 1 }
}

# -- 4. Train -----------------------------------------------------------------
Step "Training"
python -m src.chess_coach.ml.train
if (-not $?) { Write-Error "Train step failed"; exit 1 }

# -- 5. Calibrate per-class thresholds ----------------------------------------
Step "Evaluating + calibrating thresholds"
python -m src.chess_coach.ml.evaluate
if (-not $?) { Write-Error "Evaluate step failed"; exit 1 }

# -- 6. Inspect neuron weights ------------------------------------------------
Step "Inspecting weights"
python tools/inspect_weights.py
if (-not $?) { Write-Error "Inspect step failed"; exit 1 }

Write-Host "`nDone. Best checkpoint at data/classifier_best.pt" -ForegroundColor Green
