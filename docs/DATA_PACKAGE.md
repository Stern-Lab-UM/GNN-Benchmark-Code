# Public Data Package Layout

The manuscript data should be deposited outside the Git repository. The
recommended archive layout is a curated, analysis-facing package derived from
the durable outputs of `GNNBenchmark_run_publication_pipeline`, not a complete
copy of a pipeline run directory. Transient run folders such as `bo_runs/`,
`best_hps/`, `logs/`, `generated_data/`, and `staged_inputs/` are intentionally
omitted unless they are needed for a specific deposited diagnostic.

A prepared data package should look like this:

```text
gnn_benchmark_public_data_<date>/
  README_DATA_PACKAGE.md
  predictions/consolidated/
    *.pred.txt
    splits/<split_key>/{train.inds,val.inds,test.inds,_applies_to.txt}
  embeddings/per_graph/
  analysis_tables/analyzer_cache/revision_2026/
  figures/
    01_standard_v1/
    02_hexagonality/
    03_condition_comparisons/
    04_two_T1_events/
    05_summary_panels/
    06_embedding_error_bounds/
    07_counterfactual_copying/
  final_models/consolidated/
  manuscript_analyses/feature_head_ablation_20260619/
  manifests/
    public_data_manifest.csv
    public_data_summary_by_category.csv
    source_vs_package_counts.csv
    compute_sha256_manifest.ps1
```

For analysis-only reproduction, use the MATLAB data-package runner. It treats
the downloaded package as input and writes rebuilt summaries, regenerated
figures, embedding-bound diagnostics, and a JSON/MAT report under a separate output
folder:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code'))
package_root = '/path/to/gnn_benchmark_public_data_20260627';
report = GNNBenchmark_run_from_data_package(package_root);
```

To choose the destination explicitly:

```matlab
report = GNNBenchmark_run_from_data_package(package_root, ...
    'output_root', '/path/to/reanalysis_outputs');
```

By default the runner reparses the consolidated prediction files instead of
trusting cached summaries. If called with `rebuild_summaries=false`, it will use
existing summaries from `analysis_tables/analyzer_cache/revision_2026/`, with silent compatibility for older cache-folder names when needed. It
also analyzes `embeddings/per_graph/` when that folder is present. Embedding example panels inside the main plotting script are
off by default because they require vt2d geometry and a spring executable; the
saved per-graph embedding outputs are sufficient for the manuscript
embedding-error bound analysis.

The `final_models/consolidated/` folder is included for provenance and optional
reuse. It is a flat checkpoint archive and does not match the training-stage
folder layout (`final_models/<job_id>/seed_<n>/`) produced during a live pipeline
run. Manuscript figure reproduction should not require retraining if
`predictions/consolidated/`, `splits/`, and the saved embedding outputs are
present.
