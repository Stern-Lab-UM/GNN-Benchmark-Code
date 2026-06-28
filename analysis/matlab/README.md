# MATLAB Analysis Scripts

These scripts parse saved prediction files, compute manuscript summary
statistics, and generate MATLAB figures. They are the post-prediction analysis
pipeline.

Before running:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code/analysis/matlab'))
setenv('GNN_BENCHMARK_DATA_ROOT', '/path/to/gnn_benchmark_public_data_20260627')
```

or create an untracked `GNNBenchmark_local_config.m` from
`GNNBenchmark_local_config_template.m`.

For a downloaded public data package, the safest one-command entry point is:

```matlab
report = GNNBenchmark_run_from_data_package('/path/to/gnn_benchmark_public_data_20260627');
```

Standalone analysis scripts also accept either the package root or the
`predictions/consolidated` folder as `GNN_BENCHMARK_DATA_ROOT`.

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


## Figure Output Layout

Figure-writing scripts use nalysis/matlab/GNNBenchmark_figure_paths.m so regenerated figures land in manuscript-oriented folders such as igures/01_standard_v1/, igures/03_condition_comparisons/, igures/04_two_T1_events/, and igures/05_summary_panels/. Explicit igures_output_dir overrides are still honored for one-off runs.
