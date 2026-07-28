# download_syzygy.ps1
# Downloads all 3-4-5 piece Syzygy tablebase files (.rtbw and .rtbz) from
# tablebase.sesse.net into data\syzygy\.
#
# Usage:
#   .\tools\download_syzygy.ps1
#
# Total size: ~938 MB (WDL + DTZ for all 3-4-5 piece material combinations)
# These files are required for Phase 6 tablebase integration.
# python-chess reads them via chess.syzygy.open_tablebase("data/syzygy")

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$BaseUrl  = "https://tablebase.sesse.net/syzygy/3-4-5/"
$OutDir   = Join-Path $PSScriptRoot "..\data\syzygy"

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
    Write-Host "Created $OutDir"
}

Write-Host "Fetching file listing from $BaseUrl ..."
$html = (Invoke-WebRequest -Uri $BaseUrl -UseBasicParsing).Content

# Parse all .rtbw and .rtbz filenames from the directory listing
$files = [regex]::Matches($html, 'href="([^"]+\.(rtbw|rtbz))"') |
         ForEach-Object { $_.Groups[1].Value } |
         Where-Object { $_ -notmatch "^/" } |
         Sort-Object -Unique

if ($files.Count -eq 0) {
    Write-Error "No .rtbw/.rtbz files found in directory listing. The page structure may have changed."
    exit 1
}

Write-Host "$($files.Count) files to download."
$downloaded = 0
$skipped    = 0
$failed     = 0
$i          = 0

foreach ($f in $files) {
    $i++
    $dest = Join-Path $OutDir $f
    $url  = $BaseUrl + $f

    if (Test-Path $dest) {
        $skipped++
        Write-Host "[$i/$($files.Count)] SKIP  $f  (already exists)"
        continue
    }

    Write-Host "[$i/$($files.Count)] $f ..." -NoNewline
    try {
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
        $sizeMb = [math]::Round((Get-Item $dest).Length / 1MB, 1)
        Write-Host "  OK  (${sizeMb} MB)"
        $downloaded++
    } catch {
        Write-Host "  FAILED: $_"
        $failed++
        if (Test-Path $dest) { Remove-Item $dest }
    }
}

$totalMb = [math]::Round(
    (Get-ChildItem $OutDir -File | Measure-Object -Property Length -Sum).Sum / 1MB, 0
)

Write-Host ""
Write-Host "Done."
Write-Host "  Downloaded : $downloaded"
Write-Host "  Skipped    : $skipped  (already had)"
Write-Host "  Failed     : $failed"
Write-Host "  Total size : ${totalMb} MB in $OutDir"
Write-Host ""
Write-Host "Verify with python-chess:"
Write-Host '  python -c "import chess, chess.syzygy; tb = chess.syzygy.open_tablebase(\"data/syzygy\"); b = chess.Board(\"8/8/8/3k4/8/8/3PK3/8 w - - 0 1\"); print(tb.probe_wdl(b))"'
