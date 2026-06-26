# MATLAB Analysis Scripts

These scripts parse saved prediction files, compute manuscript summary
statistics, and generate MATLAB figures. They are the post-prediction analysis
pipeline.

Before running:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code/analysis/matlab'))
setenv('DCG_DATA_ROOT', '/path/to/gnn_benchmark_consolidated_20260530')
```

or create an untracked `DCG_local_config.m` from
`DCG_local_config_template.m`.

Typical full workflow:

```matlab
DCG_rebuild_all_summaries
DCG_plot_everything
```

The model training and prediction steps are implemented in Python/PyTorch under
`models/`. MATLAB starts from prediction files already written to disk.
