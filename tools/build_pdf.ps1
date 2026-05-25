# tools/build_pdf.ps1 -- Assemble the defense PDF from report/full_report.md
#
# Usage (from repo root):
#   .\tools\build_pdf.ps1
#   .\tools\build_pdf.ps1 -Engine xelatex -Toc
#   .\tools\build_pdf.ps1 -Output report\DLP_Rapport_Final_Ziouine_Nacir_Taroujena.pdf
#
# Requires: pandoc + a LaTeX engine (xelatex preferred for French + UTF-8).
#   - Pandoc: https://pandoc.org/installing.html
#   - MiKTeX or TeX Live for xelatex
#
# Maouia template path (if available): set $env:MAOUIA_TEMPLATE to the .tex file.

param(
    [string]$Source   = "report\full_report.md",
    [string]$Output   = "report\DLP_Rapport_Final_Ziouine_Nacir_Taroujena.pdf",
    [string]$Engine   = "xelatex",
    [switch]$Toc      = $false,
    [switch]$NoFigs   = $false
)

$ErrorActionPreference = "Stop"

# --- preflight ----------------------------------------------------------------
function Test-Cmd($name) {
    return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

if (-not (Test-Cmd "pandoc")) {
    Write-Error "pandoc not found on PATH. Install from https://pandoc.org/installing.html"
}
if (-not (Test-Cmd $Engine)) {
    Write-Error "$Engine not found on PATH. Install MiKTeX or TeX Live."
}
if (-not (Test-Path $Source)) {
    Write-Error "Source file not found: $Source"
}

# --- diagrams check (best-effort) --------------------------------------------
$diagDir = "report\diagrams"
$pumls   = Get-ChildItem $diagDir -Filter "*.puml" -ErrorAction SilentlyContinue
$pngs    = Get-ChildItem $diagDir -Filter "*.png"  -ErrorAction SilentlyContinue
if ($pumls.Count -gt 0 -and $pngs.Count -lt $pumls.Count -and -not $NoFigs) {
    Write-Warning "PlantUML PNGs missing or stale ($($pngs.Count)/$($pumls.Count))."
    Write-Warning "Render via: plantuml -tpng $diagDir\*.puml"
    Write-Warning "Continuing -- diagram references in the PDF may be broken."
}

# --- pandoc invocation --------------------------------------------------------
# NOTE: $args is a PowerShell automatic variable - renamed to $pandocArgs to
# avoid collision (read-only in StrictMode; splatting via @args is ambiguous).
$pandocArgs = @(
    $Source,
    "-o", $Output,
    "--pdf-engine=$Engine",
    "--from=markdown+yaml_metadata_block+pipe_tables+task_lists+fenced_code_blocks",
    "-V", "geometry:margin=2.5cm",
    "-V", "fontsize=11pt",
    "-V", "linestretch=1.15",
    "-V", "lang=fr",
    "-V", "documentclass=article",
    "-V", "colorlinks=true",
    "-V", "linkcolor=NavyBlue",
    "-V", "urlcolor=NavyBlue",
    "--highlight-style=tango"
)
if ($Toc) { $pandocArgs += "--toc"; $pandocArgs += "--toc-depth=2" }

if ($env:MAOUIA_TEMPLATE -and (Test-Path $env:MAOUIA_TEMPLATE)) {
    $pandocArgs += "--template=$($env:MAOUIA_TEMPLATE)"
    Write-Host "Using Maouia template: $env:MAOUIA_TEMPLATE" -ForegroundColor Green
} else {
    Write-Host "Maouia template not configured. Using pandoc default." -ForegroundColor Yellow
    Write-Host "  To use Maouia: `$env:MAOUIA_TEMPLATE = 'C:\path\to\maouia.tex'" -ForegroundColor Yellow
}

Write-Host "pandoc " ($pandocArgs -join " ") -ForegroundColor Cyan
& pandoc @pandocArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "pandoc failed (exit $LASTEXITCODE). Inspect the log above."
}

$sizeKilobytes = [math]::Round((Get-Item $Output).Length / 1024, 1)
$msg = "OK -> $Output ($sizeKilobytes kilobytes)"
Write-Host $msg -ForegroundColor Green
