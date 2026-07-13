# retrain_and_reparse.ps1 -- Full pipeline: parse -> ingest -> train -> evaluate
# Run from the project root: .\retrain_and_reparse.ps1
#
# Data sources (all append into training_raw.jsonl):
#   1. data/annotated_pgns/   (lichess_studies/ + Raw_pgn/)
#   2. data/lichess_db_puzzle.csv
#   3. data/Caissabase.pgn, data/Carlsen.pgn, data/lichess_elite_2020-10.pgn (if present)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# -- 1. Parse annotated PGNs (lichess_studies/ + Raw_pgn/) --------------------
#Step "Parsing annotated PGNs"
#python tools/parse_annotated_pgn.py --input data/annotated_pgns --output data/training_raw.jsonl
#if (-not $?) { Write-Error "Parse step failed"; exit 1 }

# -- 2. Ingest Lichess puzzle CSV ----------------------------------------------
Step "Ingesting Lichess puzzle CSV"
python tools/ingest_lichess_csv.py --input data/lichess_db_puzzle.csv --output data/training_raw.jsonl --append
if (-not $?) { Write-Error "Lichess CSV ingest failed"; exit 1 }

# -- 3. Ingest game databases (optional) --------------------------------------
Step "Ingesting game databases"
$db_files = @()
foreach ($f in @("data/Caissabase.pgn", "data/Carlsen.pgn", "data/lichess_elite_2020-10.pgn")) {
    if (Test-Path $f) { $db_files += $f }
}
if ($db_files.Count -eq 0) {
    Write-Host "  No game database PGNs found in data/ -- skipping" -ForegroundColor Yellow
} else {
    Write-Host "  Found: $($db_files -join ', ')"
    $db_args = @("tools/ingest_game_database.py", "--input") + $db_files + @("--output", "data/training_raw.jsonl", "--append")
    & python $db_args
    if (-not $?) { Write-Error "Game database ingest failed"; exit 1 }
}

# -- 4. Train ------------------------------------------------------------------
Step "Training"
python -m src.chess_coach.ml.train
if (-not $?) { Write-Error "Train step failed"; exit 1 }

# -- 5. Calibrate + evaluate ---------------------------------------------------
Step "Calibrating thresholds + evaluating"
python -m src.chess_coach.ml.evaluate --calibrate
if (-not $?) { Write-Error "Evaluate step failed"; exit 1 }

# -- 6. Inspect weights --------------------------------------------------------
Step "Inspecting weights"
python tools/inspect_weights.py
if (-not $?) { Write-Error "Inspect step failed"; exit 1 }

Write-Host "`nDone. Best checkpoint: data/classifier_best.pt" -ForegroundColor Green
