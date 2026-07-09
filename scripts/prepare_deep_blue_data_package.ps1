<#
.SYNOPSIS
Prepare or dry-run the Deep Blue Data package for the GNN Benchmark manuscript.

.DESCRIPTION
This script assembles the analysis-facing data package described in
docs/DATA_PACKAGE.md. By default it runs in DryRun mode: it writes manifests and
summaries, but does not copy large data files. Use Copy mode only after checking
the manifest.

The script accepts either the consolidated prediction snapshot itself or an
already staged public data package as the main source. Optional arguments add
the corrected edge-hop h >= 14, delta = 0.05 counterfactual-copying materials.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts/prepare_deep_blue_data_package.ps1 `
  -ConsolidatedRoot "Z:\Tomer\gnn_benchmark_consolidated_20260530" `
  -OutputRoot "Z:\Tomer\gnn_benchmark_public_data_20260708" `
  -CounterfactualRoot "Z:\Tomer\fallback_fingerprint_v1_2_16_W_edgehop14_delta005" `
  -CounterfactualPpgnSympairRoot "Z:\Tomer\fallback_fingerprint_v1_2_16_W_edgehop14_delta005_ppgn_sympair" `
  -CounterfactualSummaryRoot "C:\Users\tomers\Documents\DCG revision\fallback_fingerprint_edgehop14_copy_diagnostic_20260620" `
  -Mode DryRun

.EXAMPLE
# After inspecting the dry-run manifests:
powershell -ExecutionPolicy Bypass -File scripts/prepare_deep_blue_data_package.ps1 `
  -ConsolidatedRoot "Z:\Tomer\gnn_benchmark_consolidated_20260530" `
  -OutputRoot "Z:\Tomer\gnn_benchmark_public_data_20260708" `
  -CounterfactualRoot "Z:\Tomer\fallback_fingerprint_v1_2_16_W_edgehop14_delta005" `
  -CounterfactualPpgnSympairRoot "Z:\Tomer\fallback_fingerprint_v1_2_16_W_edgehop14_delta005_ppgn_sympair" `
  -CounterfactualSummaryRoot "C:\Users\tomers\Documents\DCG revision\fallback_fingerprint_edgehop14_copy_diagnostic_20260620" `
  -Mode Copy -ComputeSha256
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ConsolidatedRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [ValidateSet('DryRun', 'Copy', 'Verify')]
    [string]$Mode = 'DryRun',

    [string]$CounterfactualRoot = '',

    [string]$CounterfactualPpgnSympairRoot = '',

    [string]$CounterfactualSummaryRoot = '',

    [switch]$ComputeSha256,

    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-ExistingDirectory {
    param(
        [AllowEmptyString()][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label,
        [switch]$Required
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        if ($Required) {
            throw "$Label path is empty."
        }
        return $null
    }

    if (Test-Path -LiteralPath $Path -PathType Container) {
        return (Resolve-Path -LiteralPath $Path).Path
    }

    if ($Required) {
        throw "$Label path does not exist: $Path"
    }

    Write-Warning "$Label path not found; skipping optional source: $Path"
    return $null
}

function Join-RelPath {
    param([Parameter(Mandatory = $true)][string[]]$Parts)
    return ($Parts -join '/').Replace('\', '/')
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$Base,
        [AllowEmptyString()][string]$Path
    )

    $baseFull = [System.IO.Path]::GetFullPath($Base).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    $pathFull = [System.IO.Path]::GetFullPath($Path)
    $baseUri = [System.Uri]::new($baseFull + [System.IO.Path]::DirectorySeparatorChar)
    $pathUri = [System.Uri]::new($pathFull)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($pathUri).ToString()).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

function Add-ManifestFiles {
    param(
        [AllowEmptyCollection()][System.Collections.Generic.List[object]]$Rows,
        [Parameter(Mandatory = $true)][string]$Category,
        [Parameter(Mandatory = $true)][string]$SourceDir,
        [Parameter(Mandatory = $true)][string]$PackageDir,
        [string]$Filter = '*',
        [bool]$Recurse = $true,
        [bool]$Required = $true
    )

    if (-not (Test-Path -LiteralPath $SourceDir -PathType Container)) {
        if ($Required) {
            throw "Required source folder missing for ${Category}: $SourceDir"
        }
        Write-Warning "Optional source folder missing for ${Category}: $SourceDir"
        return
    }

    $files = @(Get-ChildItem -LiteralPath $SourceDir -File -Filter $Filter -Recurse:$Recurse)
    if ($Required -and $files.Count -eq 0) {
        throw "No files found for required category ${Category}: $SourceDir"
    }

    foreach ($file in $files) {
        $rel = Get-RelativePath -Base $SourceDir -Path $file.FullName
        $destRel = Join-RelPath @($PackageDir, $rel)
        $Rows.Add([pscustomobject]@{
            category = $Category
            source_path = $file.FullName
            package_path = $destRel
            source_size_bytes = [int64]$file.Length
            source_last_write_time_utc = $file.LastWriteTimeUtc.ToString('o')
            status = 'planned'
            sha256 = ''
        })
    }
}

function Resolve-ConsolidatedSource {
    param([Parameter(Mandatory = $true)][string]$Root)

    $rootResolved = Resolve-ExistingDirectory -Path $Root -Label 'ConsolidatedRoot' -Required
    $nested = Join-Path $rootResolved 'predictions/consolidated'
    if (Test-Path -LiteralPath $nested -PathType Container) {
        return [pscustomobject]@{
            package_source_root = $rootResolved
            consolidated_root = (Resolve-Path -LiteralPath $nested).Path
            source_is_package = $true
        }
    }

    return [pscustomobject]@{
        package_source_root = ''
        consolidated_root = $rootResolved
        source_is_package = $false
    }
}

function First-ExistingDirectory {
    param([AllowEmptyCollection()][string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate -PathType Container)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Copy-ManifestRows {
    param(
        [AllowEmptyCollection()][object[]]$Rows,
        [Parameter(Mandatory = $true)][string]$DestinationRoot,
        [switch]$ForceCopy
    )

    foreach ($row in $Rows) {
        $dest = Join-Path $DestinationRoot $row.package_path
        $destDir = Split-Path -Parent $dest
        if (-not (Test-Path -LiteralPath $destDir -PathType Container)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }

        if (Test-Path -LiteralPath $dest -PathType Leaf) {
            $existing = Get-Item -LiteralPath $dest
            if (-not $ForceCopy -and [int64]$existing.Length -eq [int64]$row.source_size_bytes) {
                continue
            }
        }

        Copy-Item -LiteralPath $row.source_path -Destination $dest -Force:$ForceCopy
    }
}

function Verify-ManifestRows {
    param(
        [AllowEmptyCollection()][object[]]$Rows,
        [Parameter(Mandatory = $true)][string]$DestinationRoot
    )

    $problems = New-Object System.Collections.Generic.List[object]
    foreach ($row in $Rows) {
        $dest = Join-Path $DestinationRoot $row.package_path
        if (-not (Test-Path -LiteralPath $dest -PathType Leaf)) {
            $problems.Add([pscustomobject]@{ package_path = $row.package_path; issue = 'missing' })
            continue
        }
        $existing = Get-Item -LiteralPath $dest
        if ([int64]$existing.Length -ne [int64]$row.source_size_bytes) {
            $problems.Add([pscustomobject]@{
                package_path = $row.package_path
                issue = "size_mismatch expected=$($row.source_size_bytes) actual=$($existing.Length)"
            })
        }
    }
    return $problems
}

function Add-Checksums {
    param(
        [AllowEmptyCollection()][object[]]$Rows,
        [Parameter(Mandatory = $true)][string]$DestinationRoot
    )

    foreach ($row in $Rows) {
        $dest = Join-Path $DestinationRoot $row.package_path
        if (Test-Path -LiteralPath $dest -PathType Leaf) {
            $row.sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $dest).Hash.ToLowerInvariant()
        }
    }
}

function Write-DataPackageReadme {
    param([Parameter(Mandatory = $true)][string]$Root)

    $readme = @'
# GNN Benchmark Public Data Package

This package accompanies the manuscript "A Controlled in Silico Benchmark for
GNN Prediction of Tissue Dynamics" and is intended for use with:

https://github.com/Stern-Lab-UM/GNN-Benchmark-Code

## Contents

- `predictions/consolidated/`: model prediction files and split manifests.
- `embeddings/per_graph/`: saved spring-embedding outputs for test graphs.
- `analysis_tables/analyzer_cache/revision_2026/`: analysis summaries used to
  rebuild figures without reparsing every prediction file.
- `figures/`: generated manuscript/revision figures from the final analysis
  pipeline.
- `final_models/consolidated/`: trained model checkpoints for provenance and
  optional reuse.
- `manuscript_analyses/feature_head_ablation_20260619/`: feature/head ablation
  outputs.
- `manuscript_analyses/counterfactual_copying_edgehop14_delta005/`: corrected
  counterfactual-copying analysis materials for edge-hop h >= 14 and
  delta = 0.05, including the symmetric-pair PPGN rerun.
- `manifests/`: file-level manifest, category summary, and optional checksums.

## Reproducing Analyses from the Package

In MATLAB:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code'))
report = GNNBenchmark_run_from_data_package('/path/to/gnn_benchmark_public_data_<date>');
```

The runner reparses predictions by default and writes regenerated summaries and
figures to a separate output folder.

## Notes

The counterfactual-copying analysis treats perturbed pre-T1 input lengths as a
copy target, not as physical ground truth. Raw directed PPGN outputs from the
obsolete run should not be used for manuscript conclusions; the package uses
the symmetric-pair PPGN rerun.

For questions, contact Tomer Stern, University of Michigan, tomers@umich.edu.
'@

    $readmePath = Join-Path $Root 'README_DATA_PACKAGE.md'
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($readmePath, $readme, $utf8NoBom)
}

$resolved = Resolve-ConsolidatedSource -Root $ConsolidatedRoot
$sourceConsolidated = $resolved.consolidated_root
$sourcePackage = $resolved.package_source_root

$outputResolved = [System.IO.Path]::GetFullPath($OutputRoot)
if (-not (Test-Path -LiteralPath $outputResolved -PathType Container)) {
    New-Item -ItemType Directory -Path $outputResolved -Force | Out-Null
}
$manifestDir = Join-Path $outputResolved 'manifests'
New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null

$rows = New-Object System.Collections.Generic.List[object]

Add-ManifestFiles -Rows $rows -Category 'prediction_file' -SourceDir $sourceConsolidated -PackageDir 'predictions/consolidated' -Filter '*.pred.txt' -Recurse $false -Required $true
Add-ManifestFiles -Rows $rows -Category 'split_file' -SourceDir (Join-Path $sourceConsolidated 'splits') -PackageDir 'predictions/consolidated/splits' -Filter '*' -Recurse $true -Required $true

$embeddingCandidates = @((Join-Path $sourceConsolidated 'embeddings/per_graph'))
if ($sourcePackage) { $embeddingCandidates += (Join-Path $sourcePackage 'embeddings/per_graph') }
$embeddingSource = First-ExistingDirectory $embeddingCandidates
if ($embeddingSource) {
    Add-ManifestFiles -Rows $rows -Category 'embedding_output' -SourceDir $embeddingSource -PackageDir 'embeddings/per_graph' -Filter '*' -Recurse $true -Required $false
}

$cacheCandidates = @(
    (Join-Path $sourceConsolidated '_analyzer_cache/revision_2026'),
    (Join-Path $sourceConsolidated '_analyzer_cache/revision_codex_2026')
)
if ($sourcePackage) {
    $cacheCandidates += (Join-Path $sourcePackage 'analysis_tables/analyzer_cache/revision_2026')
    $cacheCandidates += (Join-Path $sourcePackage 'analysis_tables/analyzer_cache/revision_codex_2026')
}
$cacheSource = First-ExistingDirectory $cacheCandidates
if ($cacheSource) {
    Add-ManifestFiles -Rows $rows -Category 'analysis_cache' -SourceDir $cacheSource -PackageDir 'analysis_tables/analyzer_cache/revision_2026' -Filter '*' -Recurse $true -Required $false
}

$figureCandidates = @((Join-Path $sourceConsolidated '_figures'))
if ($sourcePackage) { $figureCandidates += (Join-Path $sourcePackage 'figures') }
$figureSource = First-ExistingDirectory $figureCandidates
if ($figureSource) {
    Add-ManifestFiles -Rows $rows -Category 'figure_output' -SourceDir $figureSource -PackageDir 'figures' -Filter '*' -Recurse $true -Required $false
}

$modelCandidates = @()
if ($sourcePackage) { $modelCandidates += (Join-Path $sourcePackage 'final_models/consolidated') }
$modelSource = First-ExistingDirectory $modelCandidates
if ($modelSource) {
    Add-ManifestFiles -Rows $rows -Category 'model_checkpoint' -SourceDir $modelSource -PackageDir 'final_models/consolidated' -Filter '*' -Recurse $true -Required $false
} else {
    Add-ManifestFiles -Rows $rows -Category 'model_checkpoint' -SourceDir $sourceConsolidated -PackageDir 'final_models/consolidated' -Filter '*.model.pth' -Recurse $false -Required $false
}

$ablationCandidates = @((Join-Path $sourceConsolidated 'feature_head_ablation_20260619'))
if ($sourcePackage) { $ablationCandidates += (Join-Path $sourcePackage 'manuscript_analyses/feature_head_ablation_20260619') }
$ablationSource = First-ExistingDirectory $ablationCandidates
if ($ablationSource) {
    Add-ManifestFiles -Rows $rows -Category 'feature_head_ablation' -SourceDir $ablationSource -PackageDir 'manuscript_analyses/feature_head_ablation_20260619' -Filter '*' -Recurse $true -Required $false
}

$cfRoot = Resolve-ExistingDirectory -Path $CounterfactualRoot -Label 'CounterfactualRoot' -Required:$false
if ($cfRoot) {
    foreach ($subdir in @('metadata', 'data', 'results_predict_existing_lh_all5', 'analysis_predict_existing_lh_all5')) {
        $src = Join-Path $cfRoot $subdir
        if (Test-Path -LiteralPath $src -PathType Container) {
            Add-ManifestFiles -Rows $rows -Category 'counterfactual_copying' -SourceDir $src -PackageDir (Join-RelPath @('manuscript_analyses/counterfactual_copying_edgehop14_delta005/mpnn_pna_edgehop14', $subdir)) -Filter '*' -Recurse $true -Required $false
        }
    }
}

$cfPpgnRoot = Resolve-ExistingDirectory -Path $CounterfactualPpgnSympairRoot -Label 'CounterfactualPpgnSympairRoot' -Required:$false
if ($cfPpgnRoot) {
    foreach ($subdir in @('results_predict_existing_lh_ppgn_sympair', 'analysis_predict_existing_lh_ppgn_sympair')) {
        $src = Join-Path $cfPpgnRoot $subdir
        if (Test-Path -LiteralPath $src -PathType Container) {
            Add-ManifestFiles -Rows $rows -Category 'counterfactual_copying_ppgn_sympair' -SourceDir $src -PackageDir (Join-RelPath @('manuscript_analyses/counterfactual_copying_edgehop14_delta005/ppgn_sympair', $subdir)) -Filter '*' -Recurse $true -Required $false
        }
    }
}

$cfSummaryRoot = Resolve-ExistingDirectory -Path $CounterfactualSummaryRoot -Label 'CounterfactualSummaryRoot' -Required:$false
if ($cfSummaryRoot) {
    Add-ManifestFiles -Rows $rows -Category 'counterfactual_copying_final_summary' -SourceDir $cfSummaryRoot -PackageDir 'manuscript_analyses/counterfactual_copying_edgehop14_delta005/final_summary' -Filter '*' -Recurse $false -Required $false
}

$manifestPath = Join-Path $manifestDir 'public_data_manifest.csv'
$summaryPath = Join-Path $manifestDir 'public_data_summary_by_category.csv'

if ($Mode -eq 'Copy') {
    Write-Host "[prepare_deep_blue_data_package] Copying $($rows.Count) files to $outputResolved"
    Copy-ManifestRows -Rows $rows.ToArray() -DestinationRoot $outputResolved -ForceCopy:$Force
    Write-DataPackageReadme -Root $outputResolved
}

if ($Mode -eq 'Verify') {
    $problems = @(Verify-ManifestRows -Rows $rows.ToArray() -DestinationRoot $outputResolved)
    $problemPath = Join-Path $manifestDir 'verification_problems.csv'
    $problems | Export-Csv -LiteralPath $problemPath -NoTypeInformation
    if ($problems.Count -gt 0) {
        Write-Warning "Verification found $($problems.Count) problems; see $problemPath"
    }
}

if ($ComputeSha256) {
    if ($Mode -eq 'DryRun') {
        Write-Warning 'ComputeSha256 requested in DryRun mode; checksums require copied package files and will be skipped.'
    } else {
        Write-Host '[prepare_deep_blue_data_package] Computing SHA256 checksums; this can take a while.'
        Add-Checksums -Rows $rows.ToArray() -DestinationRoot $outputResolved
    }
}

$rows | Export-Csv -LiteralPath $manifestPath -NoTypeInformation
$summary = $rows |
    Group-Object category |
    ForEach-Object {
        [pscustomobject]@{
            category = $_.Name
            file_count = $_.Count
            total_size_bytes = [int64](($_.Group | Measure-Object -Property source_size_bytes -Sum).Sum)
            total_size_gb = [math]::Round((($_.Group | Measure-Object -Property source_size_bytes -Sum).Sum / 1GB), 3)
        }
    } |
    Sort-Object category
$summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation

if ($Mode -ne 'Copy') {
    Write-Host "[prepare_deep_blue_data_package] $Mode complete. No large data files were copied."
} else {
    Write-Host '[prepare_deep_blue_data_package] Copy complete.'
}
Write-Host "Manifest: $manifestPath"
Write-Host "Summary:  $summaryPath"
$summary | Format-Table -AutoSize
