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
decay at 0. By default the revision length-informed architecture passes raw
edge attributes to the final regression head. Set `ablate_head_edge_attr = true`
in `optimize_MPNN` to run the head-input ablation; the backbone still receives
the same edge attributes.

For PPGN, Bayesian optimization searched:

- learning rate: log-uniform over `[1e-5, 1e-2]`
- batch size: `{2, 4, 8, 16, 32}`
- scheduler factor: `{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}`
- gradient clipping: `{0.001, 0.01, 0.1, 1}`

The PPGN architecture was fixed at `block_features = [400, 400, 400]` and
`depth_of_mlp = 2`. By default all `RegularBlock` skip connections are kept.
Set `disable_first_skip = true` in `optimize_PPGN` to run the first-skip
ablation.

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

## Output Files and Restart Safety

Each optimizer writes all run products under `output_dirname`. If
`output_dirname` is not supplied, the default location is:

```text
<dataset parent>/Bayesian_optimization_results/
```

For each optimization run, the main files are:

- `<run_name>.mat` - final saved MATLAB result, written only after `bayesopt`
  finishes all requested objective evaluations.
- `<run_name>.partial.mat` - per-iteration checkpoint written by MATLAB
  `bayesopt` through `OutputFcn = @saveToFile` and `SaveFileName`. This file is
  updated after each completed objective evaluation.
- `<run_name>.log` - text diary containing the run configuration, Bayesian
  optimization iteration table, objective values, and status messages.

The restart behavior is handled by `run_or_resume_bayesopt.m`. If a
`.partial.mat` checkpoint exists when an optimizer starts, the helper loads the
stored `BayesianOptimization` object and inspects the completed objective trace.
If the checkpoint already contains the requested number of evaluations, it is
returned as the final result. Otherwise, completed finite-objective observations
are passed back to a fresh `bayesopt` call as `InitialX`, `InitialObjective`, and
`InitialObjectiveEvaluationTimes`. This avoids MATLAB's native `resume()` path,
because `resume()` reuses the old objective-function closure and can therefore
retain stale options such as an old GPU ID, Python environment, or command
prefix.

Crash recovery is therefore at the trial level. A crash or wall-time kill after
trial `k` finishes preserves trials `1:k` in `<run_name>.partial.mat`; restarting
the same command continues from those observations. A crash in the middle of a
single training trial may require that one trial to be re-run, because
`bayesopt` only checkpoints after an objective evaluation returns.

`optimize_MPNN.m` creates one shared working directory per optimization run and
reuses the MPNN processed-data cache across trials. `optimize_PPGN.m` creates a
`trials_PPGN_<...>/` directory under `output_dirname`; trial-specific `dcg`
outputs are written there while the objective is evaluated. By default, large
per-trial PPGN output directories are cleaned after the relevant metric is read;
set `keep_trial_dirs = true` when debugging failed trials.

If a `.partial.mat` file is missing but a previous text diary exists,
`prior_log` can be used to reconstruct completed BO observations with
`parse_bayesopt_log.m`. This is a fallback path for interrupted historical runs
and is less complete than the `.partial.mat` checkpoint.

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

% Ablation variant:
% results = optimize_MPNN(..., 'ablate_head_edge_attr', true);
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

% Ablation variant:
% results = optimize_PPGN(..., 'disable_first_skip', true);
```

## Notes

The BO objective is the smoothed minimum validation loss by default. Final
reported model performance should still be computed from independent final
training runs over the intended seed set, not from the single-seed BO trials.
