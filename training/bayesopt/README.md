# Bayesian Optimization

This folder contains the MATLAB Bayesian-optimization drivers used to tune the
GNN hyperparameters before final five-seed training.

The optimization functions are source-only wrappers around MATLAB `bayesopt`.
They do not include generated `.mat` optimization histories, prediction files,
checkpoints, or cluster logs.

## Files

- `optimize_MPNN.m` runs Bayesian optimization for GraphSAGE, GAT, GIN, and PNA
  by launching the MPNN trainer once per trial and reading the validation-loss
  curve written by that trial.
- `optimize_PPGN.m` runs Bayesian optimization for PPGN by launching `dcg train`
  once per trial and reading `metrics.csv`.
- `run_or_resume_bayesopt.m` reloads a partial `.mat` checkpoint and reseeds a
  fresh `bayesopt` call with previous observations.
- `parse_bayesopt_log.m` reconstructs prior observations from a saved bayesopt
  diary log when no partial checkpoint exists.
- `email_bayesopt_outputfcn.m` is an optional per-iteration notification hook.
- `DCG_bayesopt_search_spaces.m` returns the final V1 search spaces used for the
  revision runs.

## Search Spaces

The final V1 Bayesian-optimization setup matches the manuscript Methods. All
models used 20 objective evaluations with 6 initial seed points, training seed
0, a 120-epoch BO cap, early stopping patience of 40 epochs, scheduler patience
of 20 epochs, and early-stopping `min_delta = 1e-4`.

For GraphSAGE, GAT, GIN, and PNA, Bayesian optimization searched:

- learning rate: log-uniform over `[1e-4, 1e-2]`
- hidden-channel width: `{64, 128}`
- dropout: `{0, 0.1, 0.2}`
- batch size: `{1, 2, 4}`

The MPNN depth was fixed at 16 layers, scheduler factor at 0.75, and weight
decay at 0.

For PPGN, Bayesian optimization searched:

- learning rate: log-uniform over `[1e-5, 1e-2]`
- batch size: `{2, 4, 8, 16, 32}`
- scheduler factor: `{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}`
- gradient clipping: `{0.001, 0.01, 0.1, 1}`

The PPGN architecture was fixed at `block_features = [400, 400, 400]` and
`depth_of_mlp = 2`.

## Environment Configuration

`optimize_MPNN.m` defaults to the trainer in this repository:

```text
models/mpnn/trainer_final.py
```

Override it when needed with either the `trainer_py` name-value argument or the
`DCG_MPNN_TRAINER` environment variable. The Python executable defaults to
`python`; override with the `python` argument or `DCG_PYTHON`.

Both optimizers accept a `module_prefix` string. This string is prepended to each
trial's shell command and should activate the relevant Python environment or HPC
module stack. The environment-variable defaults are:

```bash
export DCG_MPNN_MODULE_PREFIX='source /path/to/mpnn_env/bin/activate; '
export DCG_PPGN_MODULE_PREFIX='source /path/to/ppgn_env/bin/activate; '
```

If no prefix is supplied, the functions only set conservative CPU-thread limits.
That is enough when MATLAB is already running inside the intended environment.

## MPNN Example

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code/training/bayesopt'))
spaces = DCG_bayesopt_search_spaces();

hp = spaces.mpnn_v1_l16.hp_ranges;
hp.num_layers = {'16'};   % fixed final depth
hp.factor = {'0.75'};     % fixed final LR-scheduler factor

results = optimize_MPNN( ...
    '/path/to/v1_2_16_W_weighted.txt', ...
    '/path/to/standard_2_16_split', ...
    'GIN', hp, spaces.mpnn_v1_l16.n_trials, ...
    'max_epochs', spaces.mpnn_v1_l16.max_epochs, ...
    'patience', spaces.mpnn_v1_l16.patience, ...
    'early_stop_patience', spaces.mpnn_v1_l16.early_stop_patience, ...
    'early_stop_min_delta', spaces.mpnn_v1_l16.early_stop_min_delta, ...
    'num_seed_points', spaces.mpnn_v1_l16.num_seed_points, ...
    'module_prefix', getenv('DCG_MPNN_MODULE_PREFIX'));
```

## PPGN Example

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code/training/bayesopt'))
spaces = DCG_bayesopt_search_spaces();

results = optimize_PPGN( ...
    '/path/to/Training set lengths_to_lengths/training_set_2_16_cells.txt', ...
    '/path/to/standard_2_16_split', ...
    spaces.ppgn_v1.hp_ranges, spaces.ppgn_v1.n_trials, ...
    'max_epochs', spaces.ppgn_v1.max_epochs, ...
    'early_stop', spaces.ppgn_v1.early_stop, ...
    'patience', spaces.ppgn_v1.patience, ...
    'threshold', spaces.ppgn_v1.threshold, ...
    'num_seed_points', spaces.ppgn_v1.num_seed_points, ...
    'block_features', spaces.ppgn_v1.fixed.block_features, ...
    'depth_of_mlp', spaces.ppgn_v1.fixed.depth_of_mlp, ...
    'module_prefix', getenv('DCG_PPGN_MODULE_PREFIX'));
```

## Notes

The BO objective is the smoothed minimum validation loss by default. Final
reported model performance should still be computed from independent final
training runs over the intended seed set, not from the single-seed BO trials.
