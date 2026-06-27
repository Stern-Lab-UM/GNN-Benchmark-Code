# MATLAB Analysis Scripts

These scripts parse saved prediction files, compute manuscript summary
statistics, and generate MATLAB figures. They are the post-prediction analysis
pipeline.

Before running:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code/analysis/matlab'))
setenv('GNN_BENCHMARK_DATA_ROOT', '/path/to/gnn_benchmark_consolidated_20260530')
```

or create an untracked `GNNBenchmark_local_config.m` from
`GNNBenchmark_local_config_template.m`.

Typical full workflow:

```matlab
GNNBenchmark_rebuild_all_summaries
GNNBenchmark_plot_everything
```

Focused diagnostics can be run independently:

```matlab
GNNBenchmark_analyze_embedding_error_bounds
GNNBenchmark_analyze_counterfactual_copying( ...
    'regular_pred_root', data_root, ...
    'counterfactual_pred_root', '/path/to/counterfactual_predictions', ...
    'inds_dir', '/path/to/split_dir')
```

The model training and prediction steps are implemented in Python/PyTorch under
`models/`. MATLAB starts from prediction files already written to disk.
