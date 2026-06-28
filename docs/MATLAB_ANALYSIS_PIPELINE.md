# MATLAB Analysis Pipeline

This document describes the MATLAB scripts used to turn saved prediction files
into numerical summaries and manuscript figures.

The repository is split into two stages:

1. Model fitting and prediction are implemented in Python/PyTorch under
   `models/`.
2. Analysis and plotting of saved prediction files are implemented in MATLAB
   under `analysis/matlab/`.

The MATLAB scripts do not reimplement Bayesian optimization or neural-network
training. They assume that prediction files have already been generated for the
requested model, dataset, training seed, and cohort size.

## Data Root

All MATLAB scripts use a user-supplied data root. This can be either the public`r`ndata package root or the consolidated prediction snapshot folder inside it.
The folder should contain prediction text files and split files in this layout:

```text
<package_root>/
  predictions/consolidated/<task>_<model>_<W|UW>_<size>_s<seed>.pred.txt
  predictions/consolidated/splits/<key>/train.inds
  predictions/consolidated/splits/<key>/val.inds
  predictions/consolidated/splits/<key>/test.inds
```

Set the data root in one of three ways:

```matlab
data_root = '/path/to/gnn_benchmark_public_data_20260627';
GNNBenchmark_plot_everything
```

or:

```matlab
setenv('GNN_BENCHMARK_DATA_ROOT', '/path/to/gnn_benchmark_public_data_20260627')
GNNBenchmark_plot_everything
```

or copy `analysis/matlab/GNNBenchmark_local_config_template.m` to
`analysis/matlab/GNNBenchmark_local_config.m` and edit the local paths there. The local
config file is ignored by git.

## Main Entry Points

From MATLAB:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code/analysis/matlab'))
```

Rebuild all numeric summaries from prediction files:

```matlab
GNNBenchmark_rebuild_all_summaries
```

Generate the manuscript/revision figure set from the summaries:

```matlab
GNNBenchmark_plot_everything
```

Run only selected revision datasets:

```matlab
datasets = {'Flip_two'};
rebuild_summaries = true;
plot_after_summary = true;
GNNBenchmark_run_revision_analyses
```

Run one dataset directly:

```matlab
GNNBenchmark_CONFIG = struct();
GNNBenchmark_CONFIG.dataset = 'v1_W';
GNNBenchmark_CONFIG.data_root = data_root;
GNNBenchmark_plot_results
clear GNNBenchmark_CONFIG
```

## What Each Script Does

- `GNNBenchmark_rebuild_all_summaries.m`
  rebuilds the canonical result summaries from prediction files for v1 weighted,
  v1 unweighted, hexagonality, kA, shear, Flip_two, and tissue-size datasets.

- `GNNBenchmark_analyze_results.m`
  parses prediction files, loads train/validation/test split indices, builds
  per-graph/per-hop structures, adds the identity baseline, and writes
  `results_summary.mat`.

- `GNNBenchmark_plot_everything.m`
  is the top-level plotting wrapper for the current manuscript figure set.

- `GNNBenchmark_plot_results.m`
  plots per-dataset MAE, log2(MAE), nMAE, hexagonality, fallback, scatter, and
  embedding panels.

- `GNNBenchmark_run_revision_analyses.m`
  orchestrates the revision-specific datasets and calls the per-dataset plotter,
  calibration helper, composite builder, and Flip_two-specific analysis.

- `GNNBenchmark_plot_Flip_two_interaction.m`
  performs the two-T1 analysis: nearest-T1 hop profiles, single-T1 versus
  two-T1 comparisons, inter-T1 separation curves, interaction-zone summaries,
  and distance heatmaps.

- `GNNBenchmark_consolidated_paths.m`
  maps analyzer prefixes to consolidated prediction filenames and split folders.

- `GNNBenchmark_publication_config.m`
  resolves user-local paths without hard-coding a drive, account, or cluster.


## Focused Manuscript Diagnostics

The manuscript-specific numerical checks that do not belong to a single figure
panel are also MATLAB entry points under `analysis/matlab/`:

- `GNNBenchmark_analyze_embedding_error_bounds.m` scans saved per-graph embedding outputs,
  writes graph-level prediction/embedding MAE tables, fits the strict log-log
  upper envelope, and plots the embedding-error ratio by model and cohort.
- `GNNBenchmark_analyze_counterfactual_copying.m` compares regular and counterfactually
  perturbed prediction files for the distal fallback/copying diagnostic. Pass
  `inds_dir` to restrict the calculation to `test.inds`; otherwise the function
  warns and uses all graphs in the files.

Example:

```matlab
GNNBenchmark_analyze_embedding_error_bounds( ...
    'embedding_root', fullfile(data_root, 'embeddings', 'per_graph'), ...
    'output_dir', '/path/to/reanalysis_outputs/analysis_tables/embedding_error_bounds')

GNNBenchmark_analyze_counterfactual_copying( ...
    'regular_pred_root', data_root, ...
    'counterfactual_pred_root', '/path/to/counterfactual_predictions', ...
    'inds_dir', '/path/to/standard_2_16_split', ...
    'h_min', 14, ...
    'delta', 0.05)
```
## Output Folders

For the public data package, the recommended entry point is
`GNNBenchmark_run_from_data_package`, which treats the package as input and
writes regenerated outputs under a separate `reanalysis_outputs/` folder or an
explicit `output_root`. Standalone scripts also recognize both the package root
and `predictions/consolidated`.

```text
<output_root>/analysis_tables/analyzer_cache/revision_2026/
<output_root>/figures/
```

Generated summaries, `.mat` caches, `.fig` files, exported images, prediction
files, embeddings, and trained checkpoints are intentionally not tracked by git.

## Embedding Example Figures

Standalone embedding example figures require additional local paths:

- `embed_engine`
- `embed_workdir`
- `embed_vt2d_std`
- `embed_vt2d_rev`

Set these through `GNNBenchmark_CONFIG`, environment variables, or
`GNNBenchmark_local_config.m`. If these paths are not configured, disable embedding
examples:

```matlab
GNNBenchmark_CONFIG.embed_examples = false;
```

The embedding plots color edges by the linear per-edge embedding error using a
turbo colormap with percentile clipping controlled by
`GNNBenchmark_CONFIG.embed_color_percentiles`.

The spring-relaxation engine source is included in
`external/spring_embed/`. Build it locally, then set `embed_engine` to the
compiled executable:

```bash
cd external/spring_embed
make
```

```matlab
setenv('GNN_BENCHMARK_EMBED_ENGINE', '/path/to/GNN-Benchmark-Code/external/spring_embed/build/spring_embed')
```

Compiled binaries and generated engine outputs are intentionally not tracked by
git. The committed engine uses `kA = 0` and 2000 integration updates.

The top-level publication pipeline can also generate embeddings directly from
its own prediction files. If `embedding_root` is omitted,
`GNNBenchmark_run_publication_pipeline` builds the spring engine when needed,
matches each generated prediction block to its `.vt2d` geometry, and writes
per-graph outputs under `<output_root>/embeddings/per_graph/`.

## Statistical Conventions

The plotting code uses the test split for manuscript analyses. For per-hop and
per-size summaries, errors are averaged within a graph, then across graphs, then
across the five training seeds. The shaded bands and bar errors used in the main
plots are standard deviations across seed-level means unless a specific figure
documents another convention.

nMAE is reported as an identity-normalized log2 error relative to the
pre-T1-length identity baseline.

## Scope

The MATLAB pipeline is complete for post-prediction analysis and figure
generation. The Bayesian optimization, five-seed training, and prediction-file
generation are performed by the Python/PyTorch model code and associated launch
commands.
