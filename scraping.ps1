# scraping.ps1  —  Collect annotated PGN training data from web sources
# Run from project root:  .\scraping.ps1 -Token lip_xxxx
#
# WHAT EACH SCRAPER FILLS
# ───────────────────────
# 1. scrape_lichess_studies.py  (API, requests)
#    Concept-targeted: searches Lichess study API for each of the 49 concepts
#    by keyword and downloads annotated PGNs into:
#        data/annotated_pgns/lichess_studies/<concept>/batch_NNNN.pgn
#    The parse step's folder_concept injection guarantees the concept label on
#    every example from that folder — no keyword match required in the prose.
#
#    PRIMARY TARGET: data-starved concepts from the Phase 4B eval —
#      initiative     (~17K examples)  — search terms expanded 2026-07-22
#      interference   (~18K examples)  — search terms expanded 2026-07-22
#      x_ray          (~19K examples)
#      shouldering    (~1.3K examples) — smallest dataset of all 49 concepts
#      zwischenzug    (~51K)            — new terms: "before recapturing" etc.
#      prophylaxis    (~52K)            — new terms: "preventive move" etc.
#
# 2. scrape_chessgames.py  (Playwright, chessgames.com)
#    Crawls chessgames.com for human-expert annotated master games.
#    Not concept-targeted; relies on keyword matching in { } comment blocks.
#    Best concept coverage from this source:
#      outpost, blockade, bad_bishop, good_bishop, piece_activity
#      pawn_chain, pawn_storm, space_advantage, initiative, prophylaxis
#      rook_endgame, opposition, battery, weak_square
#    Output: data/annotated_pgns/batch_NNNN.pgn  (parsed as Raw_pgn)
#
# 3. scrape_gameknot.py  (Playwright, gameknot.com)
#    Same as chessgames — keyword-matched prose from a different annotator pool.
#    Complementary coverage for concepts that depend on descriptive language:
#      king_activity, king_safety, pawn_majority, development_lead
#      mating_attack, sacrifice, clearance, deflection
#    Output: data/annotated_pgns/<slug>.pgn
#
# AFTER SCRAPING
#   Run .\retrain_and_reparse.ps1 steps 1-3 to ingest the new PGNs into
#   training_raw.jsonl, then step 4 to rebuild algo_cache (3779-dim).
#
# ─────────────────────────────────────────────────────────────────────────────

param(
    [string]$Token         = "",
    [int]   $MaxPerConcept = 200,
    [switch]$StudiesOnly,
    [switch]$AnnotatedOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "  [!] $msg" -ForegroundColor Yellow }
function Ok($msg)   { Write-Host "  [+] $msg" -ForegroundColor Green }

$ts     = Get-Date -Format "yyyy-MM-dd_HHmm"
$logDir = "results"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

# -- 1. Lichess Studies (API) -------------------------------------------------
if (-not $AnnotatedOnly) {
    Step "Lichess Studies  (concept-targeted, all 49 concepts)"

    if (-not $Token) {
        Warn "No -Token provided. Lichess rate-limits unauthenticated requests more aggressively."
        Warn "Get a free OAuth token at:  lichess.org/account/oauth/token  (tick 'study:read')"
    }

    $studiesLog  = "$logDir\${ts}_scrape_lichess_studies.txt"
    $studiesArgs = @(
        "assets/scraping/scrape_lichess_studies.py",
        "--max-per-concept", $MaxPerConcept
    )
    if ($Token) { $studiesArgs += @("--token", $Token) }

    python @studiesArgs | Tee-Object -FilePath $studiesLog
    if ($LASTEXITCODE -ne 0) { Write-Error "Lichess studies scrape failed — see $studiesLog"; exit 1 }
    Ok "Log -> $studiesLog"
}

# -- 2. Chessgames.com (Playwright) -------------------------------------------
if (-not $StudiesOnly) {
    Step "Chessgames.com  (expert-annotated master games, keyword-matched)"

    $chessgamesLog = "$logDir\${ts}_scrape_chessgames.txt"
    try {
        python assets/scraping/scrape_chessgames.py | Tee-Object -FilePath $chessgamesLog
        if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
        Ok "Log -> $chessgamesLog"
    } catch {
        Warn "Chessgames scrape failed: $_"
        Warn "Make sure playwright is installed:  pip install playwright beautifulsoup4 && python -m playwright install chromium"
    }
}

# -- 3. GameKnot (Playwright) --------------------------------------------------
if (-not $StudiesOnly) {
    Step "GameKnot  (annotated master games, keyword-matched)"

    $gameknotLog = "$logDir\${ts}_scrape_gameknot.txt"
    try {
        python assets/scraping/scrape_gameknot.py | Tee-Object -FilePath $gameknotLog
        if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
        Ok "Log -> $gameknotLog"
    } catch {
        Warn "GameKnot scrape failed: $_"
        Warn "Make sure playwright is installed:  pip install playwright && python -m playwright install chromium"
    }
}

Write-Host "`nDone. Scraping logs saved to $logDir\" -ForegroundColor Green
Write-Host @"

Next steps:
  1. Run .\retrain_and_reparse.ps1 steps 1-3   (parse new PGNs + ingest Lichess CSV + game DBs)
  2. Run step 4                                  (rebuild algo_cache.npy  3779-dim)
  3. Train:  python -m src.chess_coach.ml.train --phase4
"@ -ForegroundColor Yellow
