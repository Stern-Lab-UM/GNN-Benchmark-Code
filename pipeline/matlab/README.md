# MATLAB Publication Pipeline

`GNNBenchmark_run_publication_pipeline.m` is the MATLAB top-level orchestrator for the
publication code. It coordinates the existing repository components rather than
duplicating their logic:

- `data_generation/vertex_model/` regenerates vertex-model graph files.
- `training/bayesopt/` runs Bayesian optimization and stores partial results.
- `models/mpnn/` and `models/ppgn/` train and predict.
- `external/spring_embed/` builds the spring-relaxation executable for embedding predictions.
- `analysis/matlab/` rebuilds summaries and plots manuscript figures.

## Mini End-To-End Run

The mini run uses the same code paths as the full run but limits the data,
trials, epochs, seeds, and vertex-model simulator durations:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code'))
manifest = GNNBenchmark_run_publication_pipeline( ...
    'mode', 'mini', ...
    'output_root', '/path/to/scratch/gnn_benchmark_pipeline_mini', ...
    'mini_graphs_per_dataset', 6, ...
    'mini_split_counts', [4 1 1], ...
    'mini_simulation_times', [1 7 2], ...
    'n_trials', 1, ...
    'bo_max_epochs', 3, ...
    'final_epochs', 3, ...
    'seeds', 0, ...
    'cuda', 0);
```

``mini_simulation_times`` rescales the three C-simulator relaxation windows used only for mini smoke tests. Leave it empty, or run publication mode, to use the publication defaults compiled in the simulator.

Mini mode still exercises Bayesian optimization, but it caps the PPGN learning-rate search range with `mini_ppgn_max_learning_rate` (default `1.1e-5`). The BO result is saved for inspection; by default, final PPGN mini training then uses deterministic smoke-test hyperparameters (`mini_ppgn_use_fixed_hps = true`) and at least `mini_ppgn_final_epochs = 15`. This keeps the tiny installation example from depending on a single random categorical BO draw while preserving the full BO code path. Publication mode uses the full manuscript search space and BO-selected hyperparameters. Mini analysis also fails loudly if any prediction file produces a non-finite or implausibly large MAE above `mini_max_prediction_mae` (default `10`).

For a CPU-only smoke test, set `'cuda', -1` and optionally restrict models:

```matlab
manifest = GNNBenchmark_run_publication_pipeline( ...
    'mode', 'mini', ...
    'models', {'GraphSAGE'}, ...
    'weights', {'W'}, ...
    'cuda', -1);
```

The embedding stage builds `external/spring_embed/build/spring_embed` if needed, embeds the held-out mini test graph for each generated prediction file, and writes outputs under:

```text
<output_root>/embeddings/per_graph/
<output_root>/figures/06_embedding_error_bounds/
```

The mini figure is a simple MAE smoke-test plot under:

```text
<output_root>/figures/00_mini_smoke/mini_prediction_mae.png
```

## Integration-Scale Run

Integration mode sits between the tiny mini example and the full publication run.
It is intended for an overnight GPU/CPU-node test that exercises the manuscript
pipeline shape: all generated condition families, canonical V1 cohort split
names, Bayesian optimization, final training, prediction, spring embedding,
summary rebuilding, counterfactual copy diagnostics, and the manuscript/revision
plotting wrappers.

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code'))
manifest = GNNBenchmark_run_publication_pipeline( ...
    'mode', 'integration', ...
    'output_root', '/path/to/scratch/gnn_benchmark_pipeline_integration', ...
    'workers', 12, ...
    'cuda', 0, ...
    'integration_graphs_per_dataset', 18, ...
    'integration_split_counts', [12 3 3], ...
    'integration_simulation_times', [2 12 4], ...
    'n_trials', 2, ...
    'bo_max_epochs', 6, ...
    'final_epochs', 10, ...
    'seeds', 0:1, ...
    'embedding_max_graphs_per_prediction', 2);
```

The integration data are intentionally small, but the generated split folders use
canonical manuscript names such as `standard_2_16` and
`training_set_1_16_cells`. This lets the standard analyzer and figure scripts run
without a separate testing branch. PPGN is skipped for the 484- and 784-cell
conditions by default, matching the manuscript-scale availability; set
`'include_ppgn_large_tissues', true` only when you explicitly want to try those
large PPGN jobs.

Integration mode sets `counterfactual=true` by default. It trains the normal
models, then re-predicts the perturbed weighted standard 16-cohort inputs with
the same checkpoints and runs `GNNBenchmark_analyze_counterfactual_copying`. It
does not retrain models on perturbed data.

To smoke-test the hexagonality plotter without regenerating the separate full
hexagonality dataset, integration mode copies the standard weighted `2_8`
predictions to a clearly marked `Hexagonality_*_W_2_8` alias. The alias is only a
pipeline/plotting test artifact and is not a publication hexagonality dataset.
Disable it with `'integration_include_hex_alias', false`.

The expensive embedded example panels can be controlled independently from the
embedding-error analysis. Integration mode enables them by default; disable them
with `'plot_embedding_examples', false` if you only want the saved per-graph
embedding diagnostics.

## Publication-Scale Run

Publication mode uses all generated revision datasets by default and asks at
startup whether expensive stages may reuse cached outputs:

```matlab
manifest = GNNBenchmark_run_publication_pipeline( ...
    'mode', 'publication', ...
    'output_root', '/path/to/project/gnn_benchmark_publication_run', ...
    'workers', 20, ...
    'cuda', 0);
```

``mini_simulation_times`` rescales the three C-simulator relaxation windows used only for mini smoke tests. Leave it empty, or run publication mode, to use the publication defaults compiled in the simulator.

Long-stage cache reuse is decided once at the beginning for:

- `data_generation`
- `bayesopt`
- `final_training`
- `prediction`
- `embedding`

For noninteractive runs, pass a cache policy:

```matlab
cache_policy = struct( ...
    'data_generation', true, ...
    'bayesopt', true, ...
    'final_training', true, ...
    'prediction', false, ...
    'embedding', true);

manifest = GNNBenchmark_run_publication_pipeline( ...
    'mode', 'publication', ...
    'prompt_cache', false, ...
    'cache_policy', cache_policy);
```

## Output Layout

Each run is self-contained:

```text
<output_root>/generated_data/vertex_model/
<output_root>/bo_runs/
<output_root>/best_hps/
<output_root>/staged_inputs/
<output_root>/final_models/
<output_root>/predictions/raw/
<output_root>/predictions/consolidated/
<output_root>/embeddings/per_graph/
<output_root>/embeddings/work/
<output_root>/analysis_tables/
<output_root>/figures/00_mini_smoke/
<output_root>/figures/01_standard_v1/
<output_root>/figures/02_hexagonality/
<output_root>/figures/03_condition_comparisons/
<output_root>/figures/04_two_T1_events/
<output_root>/figures/05_summary_panels/
<output_root>/figures/06_embedding_error_bounds/
<output_root>/figures/07_counterfactual_copying/
<output_root>/logs/
<output_root>/manifests/
```

`manifests/pipeline_manifest.json` and `.mat` are updated after every stage.
They record options, paths, git commit, cache policy, stage status, and outputs.

## Notes

- The pipeline does not write model checkpoints, predictions, figures, or caches
  into source-code directories unless the user explicitly chooses such an output
  root.
- PPGN is invoked through `python -m gnn_benchmark_ppgn.main` with the curated source tree on
  `PYTHONPATH`, so a fresh clone does not require a globally installed `gnn_benchmark_ppgn`
  console script.
- If `embedding_root` is omitted, the pipeline generates per-graph spring embeddings
  from its own prediction files and generated `.vt2d` geometries. In mini mode it
  embeds one held-out test graph per prediction file by default; in publication mode
  it can embed all test graphs, so use the embedding cache policy for long runs.
- Existing full manuscript embedding caches can still be analyzed by passing
  `'embedding_root', '/path/to/embeddings/per_graph'`.
