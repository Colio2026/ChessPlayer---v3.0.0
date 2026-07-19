# retrain_and_reparse.ps1 -- Full pipeline: parse -> ingest -> build caches -> train -> evaluate
# Run from the project root: .\retrain_and_reparse.ps1
#
# Phase 4-B changes active in this script:
#   - parse_annotated_pgn.py  outputs history_rich (144-dim GRU) + 1811-dim algo features
#   - ingest_lichess_csv.py   outputs history_rich=[] + 1811-dim algo features
#   - Silver labels added for deterministic concepts on unlabeled positions
#   - Training uses --phase4 flag
#
# Data sources (all append into training_raw.jsonl):
#   1. data/annotated_pgns/   (lichess_studies/ + Raw_pgn/)
#   2. data/lichess_db_puzzle.csv
#   3. data/Caissabase.pgn, data/Carlsen.pgn, data/lichess_elite_2020-10.pgn (if present)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# -- 1. Parse annotated PGNs (lichess_studies/ + Raw_pgn/) ---------------------
Step "Parsing annotated PGNs"
python tools/parse_annotated_pgn.py --input data/annotated_pgns --output data/training_raw.jsonl
if (-not $?) { Write-Error "Parse step failed"; exit 1 }

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

# -- 3b. Build algo feature cache (strips 1811-float arrays from JSONL → binary) --
# Reduces dataset RAM from ~13 GB to ~1.5 GB. Must run once after any parse/ingest.
Step "Building algo feature cache"
python tools/build_algo_cache.py
if (-not $?) { Write-Error "Algo cache build failed"; exit 1 }

# -- 3c. Build Stockfish classical eval cache ---------------------------------
# Runs SF on every position (depth-0 classical eval) and stores 14 features per
# position. If Stockfish is not found a zero cache is written and training still
# works — set STOCKFISH_PATH env var or pass --sf-path to populate the cache.
Step "Building Stockfish classical eval cache"
python tools/build_sf_cache.py
if (-not $?) { Write-Error "SF cache build failed"; exit 1 }

# -- 4. Train ------------------------------------------------------------------
Step "Training (Phase 4)"
python -m src.chess_coach.ml.train --phase4
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
