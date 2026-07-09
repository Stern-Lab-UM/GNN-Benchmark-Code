# Final Data Sources for Public Deposit

This file records the source folders that should be used when assembling the
public Deep Blue Data package. It is intentionally path-oriented so the final
package can be rebuilt without searching through older run folders.

Do not treat similarly named scratch, quarantine, or obsolete fallback folders
as final sources unless this file is explicitly updated.

## Primary Analysis-Ready Source

Use this folder as the main consolidated source:

```text
Z:\Tomer\gnn_benchmark_consolidated_20260530
```

Expected components:

- Prediction files: `*.pred.txt`
- Final checkpoints: `*.model.pth`
- Split files: `splits\`
- Per-graph embeddings: `embeddings\per_graph\`
- Analyzer cache: `_analyzer_cache\revision_2026\`
  - If only `_analyzer_cache\revision_codex_2026\` exists in the current
    source snapshot, package it as
    `analysis_tables\analyzer_cache\revision_2026\`.
- Figures: `_figures\`
- Feature/head ablation: `feature_head_ablation_20260619\`

## Existing Staged Package

An earlier staged package exists here:

```text
Z:\Tomer\gnn_benchmark_public_data_20260627
```

It is a useful starting point, but before final upload it should be refreshed or
checked against this file because it predates the final naming cleanup and the
corrected PPGN symmetric-pair counterfactual-copying note.

## Corrected Counterfactual Copying Analysis

Use the edge-hop h >= 14, delta = 0.05 counterfactual perturbation inputs and
outputs. Do not use the older hop-8 or symmetric-distance versions.

Base MPNN/PNA source:

```text
Z:\Tomer\fallback_fingerprint_v1_2_16_W_edgehop14_delta005
```

Include these subfolders if present:

- `metadata\`
- `data\`
- `results_predict_existing_lh_all5\`
- `analysis_predict_existing_lh_all5\`

Corrected PPGN symmetric-pair rerun:

```text
Z:\Tomer\fallback_fingerprint_v1_2_16_W_edgehop14_delta005_ppgn_sympair
```

Include these subfolders if present:

- `results_predict_existing_lh_ppgn_sympair\`
- `analysis_predict_existing_lh_ppgn_sympair\`

Local final combined summaries:

```text
C:\Users\tomers\Documents\DCG revision\fallback_fingerprint_edgehop14_copy_diagnostic_20260620
```

Important files:

- `edgehop14_copy_diagnostic_full_2x2_FINAL_SYMPAIR_RERUN.csv`
- `PPGN_SYMMETRIC_PAIR_CORRECTION_NOTE.md`

Any raw directed PPGN outputs marked obsolete are not manuscript-final.

## Upstream Raw/Generated Simulation Provenance

The analysis-facing package does not need to duplicate every raw simulator
scratch folder, because the code repository can regenerate these data. If full
raw provenance is desired for a larger archival deposit, the relevant source
families are:

```text
Z:\Tomer\DCG 2024\Data\Raw vertex models
Z:\Tomer\DCG 2024\Data\Raw graphs
Z:\Tomer\DCG 2024\New code version\All vt2d\kA
Z:\Tomer\DCG 2024\New code version\All vt2d\Shear
Z:\Tomer\DCG 2024\New code version\All vt2d\Tissue size
Z:\Tomer\DCG 2024\New code version\All vt2d\Flip two
```

These are provenance folders, not the main analysis-ready package.

## Exclusions

Do not include these as final public sources:

- `Z:\Tomer\gnn_benchmark_consolidated_20260530\prev`
- `Z:\Tomer\_stale_consolidated_orphans`
- `Z:\Tomer\_embed_lh\run\work` as the primary embedding source
- Older fallback/counterfactual variants containing `hop8`
- Raw directed PPGN counterfactual files marked obsolete
- Temporary local screenshots, listener logs, scratch queues, and cluster
  command folders
