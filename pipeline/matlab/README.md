# MATLAB Publication Pipeline

`DCG_run_publication_pipeline.m` is the MATLAB top-level orchestrator for the
publication code. It coordinates the existing repository components rather than
duplicating their logic:

- `data_generation/vertex_model/` regenerates vertex-model graph files.
- `training/bayesopt/` runs Bayesian optimization and stores partial results.
- `models/mpnn/` and `models/ppgn/` train and predict.
- `analysis/matlab/` rebuilds summaries and plots manuscript figures.

## Mini End-To-End Run

The mini run uses the same code paths as the full run but limits the data,
trials, epochs, seeds, and vertex-model simulator durations:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code/pipeline/matlab'))
manifest = DCG_run_publication_pipeline( ...
    'mode', 'mini', ...
    'output_root', '/path/to/scratch/dcg_pipeline_mini', ...
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

For a CPU-only smoke test, set `'cuda', -1` and optionally restrict models:

```matlab
manifest = DCG_run_publication_pipeline( ...
    'mode', 'mini', ...
    'models', {'GraphSAGE'}, ...
    'weights', {'W'}, ...
    'cuda', -1);
```

The mini figure is a simple MAE smoke-test plot under:

```text
<output_root>/figures/mini_prediction_mae.png
```

## Publication-Scale Run

Publication mode uses all generated revision datasets by default and asks at
startup whether expensive stages may reuse cached outputs:

```matlab
manifest = DCG_run_publication_pipeline( ...
    'mode', 'publication', ...
    'output_root', '/path/to/project/dcg_publication_run', ...
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

manifest = DCG_run_publication_pipeline( ...
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
<output_root>/analysis_tables/
<output_root>/figures/
<output_root>/logs/
<output_root>/manifests/
```

`manifests/pipeline_manifest.json` and `.mat` are updated after every stage.
They record options, paths, git commit, cache policy, stage status, and outputs.

## Notes

- The pipeline does not write model checkpoints, predictions, figures, or caches
  into source-code directories unless the user explicitly chooses such an output
  root.
- PPGN is invoked through `python -m dcg.main` with the curated source tree on
  `PYTHONPATH`, so a fresh clone does not require a globally installed `dcg`
  console script.
- Full manuscript embedding caches can be analyzed by passing
  `'embedding_root', '/path/to/embeddings/per_graph'`. Embedding example panels
  still depend on the existing figure scripts and `DCG_EMBED_ENGINE`/vt2d path
  configuration.
