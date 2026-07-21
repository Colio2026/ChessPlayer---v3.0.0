# retrain_and_reparse.ps1 -- Phase 4B pipeline: parse -> ingest -> cache -> train -> eval
# Run from project root: .\retrain_and_reparse.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# -- 1. Parse annotated PGNs --------------------------------------------------
# Step "Parsing annotated PGNs"
# python tools/parse_annotated_pgn.py --input data/annotated_pgns --output data/training_raw.jsonl
# if (-not $?) { Write-Error "Parse step failed"; exit 1 }

# -- 2. Ingest Lichess puzzle CSV ----------------------------------------------
# Step "Ingesting Lichess puzzle CSV"
# python tools/ingest_lichess_csv.py --input data/lichess_db_puzzle.csv --output data/training_raw.jsonl --append
# if (-not $?) { Write-Error "Lichess CSV ingest failed"; exit 1 }

# -- 3. Ingest game databases (skipped if files absent) -----------------------
# --limit 50000 caps games per file so Caissabase (4.25 GB) doesn't run forever.
# Step "Ingesting game databases"
# $db_files = @()
# foreach ($f in @("data/Caissabase.pgn", "data/Carlsen.pgn", "data/lichess_elite_2020-10.pgn")) {
#     if (Test-Path $f) { $db_files += $f }
# }
# if ($db_files.Count -eq 0) {
#     Write-Host "  No game database PGNs found -- skipping" -ForegroundColor Yellow
# } else {
#     $db_args = @("tools/ingest_game_database.py", "--input") + $db_files + @("--output", "data/training_raw.jsonl", "--append", "--limit", "50000")
#     & python $db_args
#     if (-not $?) { Write-Error "Game database ingest failed"; exit 1 }
# }

# -- 4. Build algo + v3 cache --------------------------------------------------
# Strips algo_features from JSONL, writes algo_cache.npy (13.35 GB, 1811-dim)
# and v3_cache.npy (435 MB, 59-dim). Stamps _ac indices. Use --force to rebuild.
# algo_cache.npy (13.35 GB) and v3_cache.npy (435 MB) are current -- skipping.
# Step "Building algo + v3 cache"
# python tools/build_algo_cache.py --force
# if (-not $?) { Write-Error "Algo cache build failed"; exit 1 }

# -- 5. Build Stockfish classical eval cache -----------------------------------
# sf_cache.npy exists (83 MB) -- skipping.
# Step "Building Stockfish classical eval cache"
# python tools/build_sf_cache.py
# if (-not $?) { Write-Error "SF cache build failed"; exit 1 }

# -- 6. Build NNUE Feature Transformer cache -----------------------------------
# NOTE: Neural net cache is built and valid but not used in Phase 4B training.
# It will be used downstream in the coach layer for position correctness gating.
# nnue_cache.npy (15.09 GB) exists -- skipping.
# Step "Building NNUE feature transformer cache"
# python tools/build_nnue_cache.py --force
# if (-not $?) { Write-Error "NNUE cache build failed"; exit 1 }

# -- 7. Build board tensor cache -----------------------------------------------
# board_cache.npy (7.38 GB) is current -- skipping.
# Step "Building board tensor cache"
# python tools/build_board_cache.py
# if (-not $?) { Write-Error "Board cache build failed"; exit 1 }

# -- 8. Build RAG coaching index -----------------------------------------------
# eco_db.json + rag_index.jsonl are current -- skipping.
# Step "Building RAG coaching index"
# python tools/build_rag_index.py
# if (-not $?) { Write-Error "RAG index build failed"; exit 1 }

# -- 9. Train (Phase 4B) -------------------------------------------------------
Step "Training (Phase 4B)"
python -m src.chess_coach.ml.train --phase4
if (-not $?) { Write-Error "Train step failed"; exit 1 }

# -- 10. Calibrate + evaluate --------------------------------------------------
Step "Calibrating thresholds + evaluating"
python -m src.chess_coach.ml.evaluate --calibrate
if (-not $?) { Write-Error "Evaluate step failed"; exit 1 }

# -- 11. Inspect weights -------------------------------------------------------
Step "Inspecting weights"
python tools/inspect_weights.py
if (-not $?) { Write-Error "Inspect weights failed"; exit 1 }

Write-Host "`nDone. Best checkpoint: data/classifier_best.pt" -ForegroundColor Green
