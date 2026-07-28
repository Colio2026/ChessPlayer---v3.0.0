# Phase 6B (MoE): builds on Phase 4C caches.  New cache: tb_cache.npy (tablebase WDL/DTZ).
# Steps 1-8 (ingest / caches) unchanged from Phase 4C -- commented out.
# Step 8.5 (TB cache) already built (162,924 positions / 7.6% coverage) -- commented out.
# Run steps 9B → 10 → 11 → 12 → 13 to train, calibrate, and audit Phase 6B.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# # -- 1. Parse annotated PGNs --------------------------------------------------
# # Uncomment when re-ingesting new annotated PGNs.
# Step "Parsing annotated PGNs"
# python tools/parse_annotated_pgn.py --input data/annotated_pgns --output data/training_raw.jsonl
# if (-not $?) { Write-Error "Parse step failed"; exit 1 }

# -- 2. Ingest Lichess puzzle CSV ----------------------------------------------
# Uncomment when re-scraping Lichess puzzles (e.g. after adding x_ray / shouldering synonyms).
# Step "Ingesting Lichess puzzle CSV"
# python tools/ingest_lichess_csv.py --input data/lichess_db_puzzle.csv --output data/training_raw.jsonl --append
# if (-not $?) { Write-Error "Lichess CSV ingest failed"; exit 1 }

# -- 3. Ingest game databases --------------------------------------------------
# --limit 50000 caps games per file so Caissabase (4.25 GB) doesn't run forever.
# Uncomment when adding new game database PGNs.
# Step "Ingesting game databases"
# $db_files = @()
# foreach ($f in @("data/Caissabase.pgn", "data/Carlsen.pgn", "data/lichess_elite_2020-10.pgn")) {
#     if (Test-Path $f) { $db_files += $f }
# }
# if ($db_files.Count -eq 0) {
#     Write-Host "  No game database PGNs found -- skipping" -ForegroundColor Yellow
# } else {
#     $db_args = @("tools/ingest_game_database.py", "--input") + $db_files + @("--output", "data/training_raw.jsonl", "--append", "--limit", "5000")
#     & python $db_args
#     if (-not $?) { Write-Error "Game database ingest failed"; exit 1 }
# }

# -- 4. Build algo + v3 cache (float16) ----------------------------------------
# Converts algo_cache.npy from float32 (~32GB) to float16 (~16GB).
# Reads existing training_raw.jsonl (_ac indices already stamped) -- no JSONL rewrite.
# board_cache, sf_cache, tb_cache untouched.
# Step "Building algo + v3 cache (float16)"
# python tools/build_algo_cache.py --force
# if (-not $?) { Write-Error "Algo cache build failed"; exit 1 }

# -- 5. Build Stockfish classical eval cache -----------------------------------
# sf_cache.npy unchanged (SF features not modified) -- skipping.
# Step "Building Stockfish classical eval cache"
# python tools/build_sf_cache.py --force
# if (-not $?) { Write-Error "SF cache build failed"; exit 1 }

# -- 6. Build NNUE Feature Transformer cache -----------------------------------
# Phase 5 abandoned -- not used in Phase 4C training. Skipping.
# python tools/build_nnue_cache.py --force

# -- 7. Build board tensor cache -----------------------------------------------
# board_cache.npy unchanged (board encoding not modified) -- skipping.
# Step "Building board tensor cache"
# python tools/build_board_cache.py --force
# if (-not $?) { Write-Error "Board cache build failed"; exit 1 }

# -- 8. Build RAG coaching index -----------------------------------------------
# eco_db.json + rag_index.jsonl unchanged -- skipping.
# Step "Building RAG coaching index"
# python tools/build_rag_index.py --force
# if (-not $?) { Write-Error "RAG index build failed"; exit 1 }

# -- 8.5. Build TB (tablebase WDL/DTZ) cache ----------------------------------
# Already built 2026-07-21: 162,924 covered / 7.6%, 25.7 MB.
# Re-run only if training_raw.jsonl is re-ingested with new positions.
# Step "Building TB cache"
# python tools/build_tablebase_cache.py --force
# if (-not $?) { Write-Error "TB cache build failed"; exit 1 }

# -- 9. Train ------------------------------------------------------------------
# Phase 4C (archived -- superseded by Phase 6B):
# Step "Training Nimzo-Net"
# python -m src.chess_coach.ml.train --phase4

# Phase 6B: 5-expert MoE + ECO embedding + TB gate conditioning.
# Writes data/classifier_best.pt  (phase6=True flag baked into checkpoint).
# Step "Training (Phase 6B MoE)"
# python -m src.chess_coach.ml.train --phase6 --batch-size 1024
# if (-not $?) { Write-Error "Train step failed"; exit 1 }

# -- 10. Calibrate + evaluate --------------------------------------------------
# Step "Calibrating thresholds + evaluating"
# python -m src.chess_coach.ml.evaluate --calibrate
# if (-not $?) { Write-Error "Evaluate step failed"; exit 1 }

# Timestamp prefix shared by all eval result files from this run
$ts = Get-Date -Format "yyyy-MM-dd_HHmm"

# -- 11. Inspect weights -------------------------------------------------------
Step "Inspecting weights"
$inspectLog = "results\${ts}_inspect.txt"
python -u tools/inspect_weights.py | Tee-Object -Variable _inspect
if ($LASTEXITCODE -ne 0) { Write-Error "Inspect weights failed"; exit 1 }
$_inspect | Out-File -FilePath $inspectLog -Encoding UTF8
Write-Host "  Saved -> $inspectLog" -ForegroundColor DarkGray

# -- 12. Hysteresis threshold survey -------------------------------------------
# Verifies per-concept ACTIVATE thresholds (data/activate_thresholds.json) and
# global HOLD=0.40 are well-placed against the real probability distribution.
Step "Surveying hysteresis thresholds"
$hysteresisLog = "results\${ts}_hysteresis.txt"
python -u tools/survey_hysteresis.py --n 2000 | Tee-Object -Variable _hysteresis
if ($LASTEXITCODE -ne 0) { Write-Error "Hysteresis survey failed"; exit 1 }
$_hysteresis | Out-File -FilePath $hysteresisLog -Encoding UTF8
Write-Host "  Saved -> $hysteresisLog" -ForegroundColor DarkGray

# -- 13. False-positive / false-negative audit ---------------------------------
# Ranks concepts by FP rate and samples real positions from the test split.
# Prints Lichess analysis links -- open 10-20 manually to distinguish model errors
# from label noise. Audit the 8 worst concepts by default.
Step "Auditing concept false positives"
$auditLog = "results\${ts}_audit.txt"
python -u tools/audit_concepts.py --n 8 --samples 10 | Tee-Object -Variable _audit
if ($LASTEXITCODE -ne 0) { Write-Error "Concept audit failed"; exit 1 }
$_audit | Out-File -FilePath $auditLog -Encoding UTF8
Write-Host "  Saved -> $auditLog" -ForegroundColor DarkGray

# -- 14. Consistency check (optional, runs on a sample game) -------------------
# Replay a game through coach.analyze() and verify concept transitions happen
# at meaningful move boundaries, not on quiet moves.
# Uncomment and supply UCI moves to run:
  
Write-Host "`nDone. Best checkpoint: data/classifier_best.pt" -ForegroundColor Green
Write-Host "Eval logs: results/" -ForegroundColor Green
Write-Host "Next: open Lichess URLs from ${auditLog} for manual spot-checks." -ForegroundColor Yellow
