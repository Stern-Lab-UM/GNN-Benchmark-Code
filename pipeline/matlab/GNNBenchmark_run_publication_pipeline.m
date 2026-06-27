function manifest = GNNBenchmark_run_publication_pipeline(varargin)
% GNNBenchmark_run_publication_pipeline  MATLAB top-level GNN Benchmark publication pipeline.
%
%   MANIFEST = GNNBenchmark_run_publication_pipeline('mode','mini') runs a small but
%   trainable end-to-end check. MANIFEST =
%   GNNBenchmark_run_publication_pipeline('mode','publication') uses the same stages at
%   manuscript scale. MATLAB is the orchestrator; tissue generation, BO,
%   training, prediction, analysis, and plotting remain in their existing
%   modules.
%
%   Important options:
%     mode             'mini' or 'publication' (default 'mini')
%     output_root      run folder; default <repo>/pipeline_runs/<timestamp>
%     stages           {'all'} or a subset of install_check, data_generation,
%                      bayesopt, final_training, prediction, embedding,
%                      analysis, figures
%     datasets         dataset keys; mini default {'standard_16'}
%     models           default all five models
%     weights          default {'W','UW'} for standard_16; revision conditions
%                      are W-only unless run_unweighted_revision_conditions=true
%     seeds            mini default 0; publication default 0:4
%     prompt_cache     ask at startup whether to reuse long-stage outputs
%     cache_policy     struct fields: data_generation, bayesopt,
%                      final_training, prediction, embedding
%     mini_graphs_per_dataset / mini_split_counts / mini_simulation_times
%                      make a fast train/val/test subset from publication order
%     n_trials, bo_max_epochs, final_epochs
%                      shrink or expand runtime without changing code paths
%
%   Outputs are kept outside source directories:
%     generated_data/vertex_model, bo_runs, best_hps, staged_inputs,
%     final_models, predictions/raw, predictions/consolidated,
%     analysis_tables, figures, logs, manifests.

opts = parse_inputs(varargin{:});
paths = make_paths(opts);
ensure_dir(paths.output_root);
ensure_dir(paths.manifests);
ensure_dir(paths.logs);
add_repo_paths(paths.repo_root);

manifest = struct();
manifest.started_at = stamp();
manifest.completed = false;
manifest.mode = opts.mode;
manifest.options = opts;
manifest.paths = paths;
manifest.git = git_snapshot(paths.repo_root);
manifest.stage_history = struct('stage', {}, 'status', {}, 'seconds', {}, 'message', {}, 'time', {});
manifest = save_manifest(manifest, paths, 'initialized');

fprintf('\n============================================================\n');
fprintf('[GNNBenchmark pipeline] mode:        %s\n', opts.mode);
fprintf('[GNNBenchmark pipeline] output root: %s\n', paths.output_root);
fprintf('[GNNBenchmark pipeline] repo root:   %s\n', paths.repo_root);
fprintf('============================================================\n\n');

cache_policy = resolve_cache_policy(opts);
manifest.cache_policy = cache_policy;
manifest = save_manifest(manifest, paths, 'cache_policy');

stages = expand_stages(opts.stages);
for i = 1:numel(stages)
    stage = stages{i};
    t0 = tic;
    fprintf('\n[%s] Stage %d/%d: %s\n', stamp(), i, numel(stages), stage);
    try
        switch stage
            case 'install_check'
                manifest.install_check = stage_install_check(opts, paths);
            case 'data_generation'
                manifest.data_generation = stage_data_generation(opts, paths, cache_policy.data_generation);
            case 'bayesopt'
                manifest.bayesopt = stage_bayesopt(opts, paths, cache_policy.bayesopt);
            case 'final_training'
                manifest.final_training = stage_final_training(opts, paths, cache_policy.final_training);
            case 'prediction'
                manifest.prediction = stage_prediction(opts, paths, cache_policy.prediction);
            case 'embedding'
                manifest.embedding = stage_embedding(opts, paths);
            case 'analysis'
                manifest.analysis = stage_analysis(opts, paths);
            case 'figures'
                manifest.figures = stage_figures(opts, paths);
            otherwise
                error('GNNBenchmark:pipeline:badStage', 'Unknown stage "%s".', stage);
        end
        secs = toc(t0);
        manifest.stage_history(end+1) = struct('stage', stage, 'status', 'OK', ...
            'seconds', secs, 'message', '', 'time', stamp()); %#ok<AGROW>
        fprintf('[%s] Stage complete: %s (%.1f s)\n', stamp(), stage, secs);
    catch ME
        secs = toc(t0);
        manifest.stage_history(end+1) = struct('stage', stage, 'status', 'FAILED', ...
            'seconds', secs, 'message', ME.message, 'time', stamp()); %#ok<AGROW>
        manifest = save_manifest(manifest, paths, ['failed_', stage]);
        fprintf(2, '[GNNBenchmark pipeline] FAILED in %s after %.1f s:\n%s\n', ...
            stage, secs, getReport(ME, 'extended', 'hyperlinks', 'off'));
        rethrow(ME);
    end
    manifest = save_manifest(manifest, paths, ['completed_', stage]);
end

manifest.completed = true;
manifest.completed_at = stamp();
manifest = save_manifest(manifest, paths, 'completed');
fprintf('\n[GNNBenchmark pipeline] DONE. Manifest:\n  %s\n', fullfile(paths.manifests, 'pipeline_manifest.json'));
end

function opts = parse_inputs(varargin)
p = inputParser;
p.FunctionName = 'GNNBenchmark_run_publication_pipeline';
p.addParameter('mode', 'mini', @(x) any(strcmpi(char(x), {'mini','publication'})));
p.addParameter('output_root', '', @(x) ischar(x) || isstring(x));
p.addParameter('stages', {'all'}, @(x) iscell(x) || isstring(x) || ischar(x));
p.addParameter('datasets', {}, @(x) iscell(x) || isstring(x) || ischar(x));
p.addParameter('models', {}, @(x) iscell(x) || isstring(x) || ischar(x));
p.addParameter('weights', {'W','UW'}, @(x) iscell(x) || isstring(x) || ischar(x));
p.addParameter('seeds', [], @isnumeric);
p.addParameter('workers', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('cuda', -1, @(x) isnumeric(x) && isscalar(x));
p.addParameter('python', getenv_or('GNN_BENCHMARK_PYTHON', 'python'), @(x) ischar(x) || isstring(x));
p.addParameter('prompt_cache', [], @(x) isempty(x) || (islogical(x) && isscalar(x)));
p.addParameter('cache_policy', struct(), @isstruct);
p.addParameter('run_unweighted_revision_conditions', false, @(x) islogical(x) && isscalar(x));
p.addParameter('mini_graphs_per_dataset', 6, @(x) isnumeric(x) && isscalar(x) && x >= 3);
p.addParameter('mini_split_counts', [4 1 1], @(x) isnumeric(x) && numel(x) == 3);
p.addParameter('mini_simulation_times', [1 7 2], @(x) isempty(x) || (isnumeric(x) && numel(x) == 3 && all(x >= 0)));
p.addParameter('n_trials', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
p.addParameter('num_seed_points', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
p.addParameter('bo_max_epochs', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
p.addParameter('final_epochs', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
p.addParameter('run_bayesopt', true, @(x) islogical(x) && isscalar(x));
p.addParameter('reuse_best_hps', true, @(x) islogical(x) && isscalar(x));
p.addParameter('overwrite_generation', false, @(x) islogical(x) && isscalar(x));
p.addParameter('counterfactual', false, @(x) islogical(x) && isscalar(x));
p.addParameter('counterfactual_h_min', 14, @(x) isnumeric(x) && isscalar(x));
p.addParameter('counterfactual_delta', 0.05, @(x) isnumeric(x) && isscalar(x));
p.addParameter('counterfactual_seed', 20260616, @(x) isnumeric(x) && isscalar(x));
p.addParameter('embedding_root', '', @(x) ischar(x) || isstring(x));
p.addParameter('data_root_for_figures', '', @(x) ischar(x) || isstring(x));
p.addParameter('include_ppgn', true, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});
opts = p.Results;
opts.mode = lower(char(opts.mode));
opts.stages = cellstr(opts.stages);
opts.weights = cellstr(opts.weights);
opts.python = char(opts.python);
opts.embedding_root = char(opts.embedding_root);
opts.data_root_for_figures = char(opts.data_root_for_figures);
if isempty(opts.datasets)
    if strcmp(opts.mode, 'mini')
        opts.datasets = {'standard_16'};
    else
        opts.datasets = {'standard_16','kA_10','kA_1','shear_1_2','shear_1_5','tissue_484','tissue_784','flip_two'};
    end
else
    opts.datasets = cellstr(opts.datasets);
end
if isempty(opts.models)
    opts.models = {'GraphSAGE','GAT','GIN','PNA','PPGN'};
else
    opts.models = cellstr(opts.models);
end
if ~opts.include_ppgn
    opts.models = setdiff(opts.models, {'PPGN'}, 'stable');
end
if isempty(opts.seeds)
    opts.seeds = ternary(strcmp(opts.mode, 'mini'), 0, 0:4);
end
if isempty(opts.prompt_cache)
    opts.prompt_cache = strcmp(opts.mode, 'publication');
end
if isempty(opts.n_trials)
    opts.n_trials = ternary(strcmp(opts.mode, 'mini'), 1, NaN);
end
if isempty(opts.num_seed_points)
    opts.num_seed_points = ternary(strcmp(opts.mode, 'mini'), 1, NaN);
end
if isempty(opts.bo_max_epochs)
    opts.bo_max_epochs = ternary(strcmp(opts.mode, 'mini'), 3, NaN);
end
if isempty(opts.final_epochs)
    opts.final_epochs = ternary(strcmp(opts.mode, 'mini'), 3, 120);
end
end

function paths = make_paths(opts)
paths = struct();
paths.this_dir = fileparts(mfilename('fullpath'));
paths.repo_root = fileparts(fileparts(paths.this_dir));
if isempty(opts.output_root)
    paths.output_root = fullfile(paths.repo_root, 'pipeline_runs', [datestr(now, 'yyyymmdd_HHMMSS'), '_', opts.mode]); %#ok<DATST>
else
    paths.output_root = char(opts.output_root);
end
paths.generated_root = fullfile(paths.output_root, 'generated_data', 'vertex_model');
paths.bo_root = fullfile(paths.output_root, 'bo_runs');
paths.best_hp_root = fullfile(paths.output_root, 'best_hps');
paths.staged_root = fullfile(paths.output_root, 'staged_inputs');
paths.model_root = fullfile(paths.output_root, 'final_models');
paths.pred_raw_root = fullfile(paths.output_root, 'predictions', 'raw');
paths.pred_consolidated_root = fullfile(paths.output_root, 'predictions', 'consolidated');
paths.analysis_tables = fullfile(paths.output_root, 'analysis_tables');
paths.figures = fullfile(paths.output_root, 'figures');
paths.logs = fullfile(paths.output_root, 'logs');
paths.manifests = fullfile(paths.output_root, 'manifests');
end

function add_repo_paths(repo_root)
addpath(genpath(fullfile(repo_root, 'pipeline', 'matlab')));
addpath(genpath(fullfile(repo_root, 'data_generation', 'vertex_model')));
addpath(genpath(fullfile(repo_root, 'training', 'bayesopt')));
addpath(genpath(fullfile(repo_root, 'analysis', 'matlab')));
end

function cache_policy = resolve_cache_policy(opts)
names = {'data_generation','bayesopt','final_training','prediction','embedding'};
cache_policy = struct();
for i = 1:numel(names)
    nm = names{i};
    if isfield(opts.cache_policy, nm)
        cache_policy.(nm) = logical(opts.cache_policy.(nm));
    elseif opts.prompt_cache
        cache_policy.(nm) = ask_yes_no(sprintf('Reuse cached outputs for %s if present?', nm), true);
    else
        cache_policy.(nm) = false;
    end
end
end

function stages = expand_stages(stages)
stages = lower(cellstr(stages));
if any(strcmp(stages, 'all'))
    stages = {'install_check','data_generation','bayesopt','final_training','prediction','embedding','analysis','figures'};
end
end

function out = stage_install_check(opts, paths)
log_file = fullfile(paths.logs, 'install_check.log');
component = install_component_for_models(opts.models);
cmd = sprintf('"%s" "%s" --component %s', opts.python, fullfile(paths.repo_root, 'scripts', 'check_install.py'), component);
[status, output] = run_logged(cmd, log_file, paths.repo_root);
out = struct('status', status, 'log_file', log_file, 'command', cmd, 'component', component);
if status ~= 0
    error('GNNBenchmark:pipeline:installCheck', 'Install check failed. Log: %s\n%s', log_file, output);
end
end

function component = install_component_for_models(models)
models = cellstr(models);
has_ppgn = any(strcmp(models, 'PPGN'));
has_mpnn = any(~strcmp(models, 'PPGN'));
if has_ppgn && has_mpnn
    component = 'all';
elseif has_ppgn
    component = 'ppgn';
else
    component = 'mpnn';
end
end

function out = stage_data_generation(opts, paths, reuse_cache)
summary_file = fullfile(paths.generated_root, 'generation_summary.txt');
if reuse_cache && isfile(summary_file)
    fprintf('[data_generation] Reusing %s\n', paths.generated_root);
    out = struct('reused', true, 'output_root', paths.generated_root, 'summary', summary_file);
    return
end
extra = {};
if strcmp(opts.mode, 'mini')
    extra = {'max_graphs_per_dataset', opts.mini_graphs_per_dataset, 'split_counts', opts.mini_split_counts, 'simulation_times', opts.mini_simulation_times};
end
report = GNNBenchmark_generate_vertex_model_datasets('mode', 'publication', ...
    'output_root', paths.generated_root, 'workers', opts.workers, ...
    'overwrite', opts.overwrite_generation, 'datasets', opts.datasets, ...
    'counterfactual', opts.counterfactual, ...
    'counterfactual_h_min', opts.counterfactual_h_min, ...
    'counterfactual_delta', opts.counterfactual_delta, ...
    'counterfactual_seed', opts.counterfactual_seed, extra{:});
out = struct('reused', false, 'output_root', paths.generated_root, 'summary', summary_file, 'report', report);
end

function out = stage_bayesopt(opts, paths, reuse_cache)
ensure_dir(paths.bo_root);
ensure_dir(paths.best_hp_root);
jobs = enumerate_jobs(opts, paths);
spaces = GNNBenchmark_bayesopt_search_spaces();
rows = struct('job_id', {}, 'status', {}, 'best_hp_file', {}, 'bo_file', {});
for j = 1:numel(jobs)
    job = jobs(j);
    fprintf('[bayesopt] %d/%d %s\n', j, numel(jobs), job.job_id);
    best_file = fullfile(paths.best_hp_root, [job.job_id, '_best_hps.mat']);
    if reuse_cache && opts.reuse_best_hps && isfile(best_file)
        rows(end+1) = struct('job_id', job.job_id, 'status', 'reused', 'best_hp_file', best_file, 'bo_file', ''); %#ok<AGROW>
        continue
    end
    bo_dir = fullfile(paths.bo_root, job.job_id);
    ensure_dir(bo_dir);
    if strcmp(job.family, 'MPNN')
        search = apply_runtime_overrides(spaces.mpnn_v1_l16, opts);
        if opts.run_bayesopt
            results = optimize_MPNN(job.dataset_file, job.split_dir, job.model, search.hp_ranges, search.n_trials, ...
                'cuda', opts.cuda, 'max_epochs', search.max_epochs, ...
                'patience', search.patience, 'early_stop_patience', search.early_stop_patience, ...
                'early_stop_min_delta', search.early_stop_min_delta, ...
                'num_seed_points', min(search.num_seed_points, search.n_trials), ...
                'output_dirname', bo_dir, 'skip_existing', reuse_cache, ...
                'smooth_val_loss', ~strcmp(opts.mode, 'mini'), 'python', opts.python);
            hp = best_hps_from_bayesopt(results, search.hp_ranges, job.family);
        else
            hp = default_hps(job.model, job.family);
        end
    else
        search = apply_runtime_overrides(spaces.ppgn_v1, opts);
        ppgn = stage_ppgn_dataset(job, paths, 'train_gnn_benchmark');
        if opts.run_bayesopt
            results = optimize_PPGN(ppgn.dataset_file, job.split_dir, search.hp_ranges, search.n_trials, ...
                'cuda', opts.cuda, 'max_epochs', search.max_epochs, ...
                'patience', search.patience, 'early_stop', search.early_stop, ...
                'threshold', search.threshold, ...
                'num_seed_points', min(search.num_seed_points, search.n_trials), ...
                'output_dirname', bo_dir, 'skip_existing', reuse_cache, ...
                'smooth_val_loss', ~strcmp(opts.mode, 'mini'), ...
                'ppgn_cmd', [opts.python, ' -m gnn_benchmark_ppgn.main']);
            hp = best_hps_from_bayesopt(results, search.hp_ranges, job.family);
        else
            hp = default_hps(job.model, job.family);
        end
    end
    save(best_file, 'hp', 'job');
    write_json(strrep(best_file, '.mat', '.json'), hp);
    rows(end+1) = struct('job_id', job.job_id, 'status', 'ok', 'best_hp_file', best_file, 'bo_file', latest_mat(bo_dir)); %#ok<AGROW>
end
out = struct('jobs', rows, 'best_hp_root', paths.best_hp_root, 'bo_root', paths.bo_root);
end

function search = apply_runtime_overrides(search, opts)
if ~isnan(opts.n_trials), search.n_trials = opts.n_trials; end
if ~isnan(opts.num_seed_points), search.num_seed_points = opts.num_seed_points; end
if ~isnan(opts.bo_max_epochs), search.max_epochs = opts.bo_max_epochs; end
end

function out = stage_final_training(opts, paths, reuse_cache)
ensure_dir(paths.model_root);
jobs = enumerate_jobs(opts, paths);
rows = struct('job_id', {}, 'seed', {}, 'status', {}, 'model_path', {}, 'log_file', {});
for j = 1:numel(jobs)
    job = jobs(j);
    hp = load_hps(job, paths);
    for s = opts.seeds
        model_dir = fullfile(paths.model_root, job.job_id, sprintf('seed_%d', s));
        ensure_dir(model_dir);
        done_file = fullfile(model_dir, '_TRAINING_DONE.txt');
        log_file = fullfile(model_dir, 'training.log');
        if reuse_cache && isfile(done_file)
            rows(end+1) = struct('job_id', job.job_id, 'seed', s, 'status', 'reused', 'model_path', model_dir, 'log_file', log_file); %#ok<AGROW>
            continue
        end
        if strcmp(job.family, 'MPNN')
            staged = stage_mpnn_dataset(job, paths, s);
            cmd = mpnn_train_cmd(opts, job, hp, staged.data_dir, s);
            [status, output] = run_logged(cmd, log_file, paths.repo_root);
            if status ~= 0, error('GNNBenchmark:pipeline:mpnnTrain', 'MPNN training failed: %s\n%s', log_file, output); end
            ckpt = find_checkpoint(staged.data_dir, job.model, job.weight, s);
            copyfile(ckpt, fullfile(model_dir, filename(ckpt)));
        else
            ppgn = stage_ppgn_dataset(job, paths, 'train_gnn_benchmark');
            cmd = ppgn_train_cmd(opts, job, hp, ppgn.dataset_file, job.split_dir, model_dir);
            [status, output] = run_logged(cmd, log_file, paths.repo_root);
            if status ~= 0, error('GNNBenchmark:pipeline:ppgnTrain', 'PPGN training failed: %s\n%s', log_file, output); end
        end
        write_text(done_file, sprintf('done: %s\n', stamp()));
        rows(end+1) = struct('job_id', job.job_id, 'seed', s, 'status', 'ok', 'model_path', model_dir, 'log_file', log_file); %#ok<AGROW>
    end
end
out = struct('jobs', rows, 'model_root', paths.model_root);
end

function out = stage_prediction(opts, paths, reuse_cache)
ensure_dir(paths.pred_raw_root);
ensure_dir(paths.pred_consolidated_root);
jobs = enumerate_jobs(opts, paths);
rows = struct('job_id', {}, 'seed', {}, 'status', {}, 'pred_file', {}, 'consolidated_file', {});
for j = 1:numel(jobs)
    job = jobs(j);
    for s = opts.seeds
        raw_out = fullfile(paths.pred_raw_root, sprintf('%s_seed_%d.pred.txt', job.job_id, s));
        con_out = fullfile(paths.pred_consolidated_root, consolidated_name(job, s));
        if reuse_cache && isfile(raw_out) && isfile(con_out)
            rows(end+1) = struct('job_id', job.job_id, 'seed', s, 'status', 'reused', 'pred_file', raw_out, 'consolidated_file', con_out); %#ok<AGROW>
            continue
        end
        log_file = fullfile(paths.logs, sprintf('predict_%s_seed_%d.log', job.job_id, s));
        if strcmp(job.family, 'MPNN')
            hp = load_hps(job, paths);
            ckpt = find_checkpoint(fullfile(paths.model_root, job.job_id, sprintf('seed_%d', s)), job.model, job.weight, s);
            cmd = mpnn_predict_cmd(opts, ckpt, job.dataset_file, raw_out, max(1, round(hp.batch_size)));
        else
            ppgn = stage_ppgn_dataset(job, paths, 'predict_gnn_benchmark');
            model_dir = fullfile(paths.model_root, job.job_id, sprintf('seed_%d', s));
            cmd = ppgn_predict_cmd(opts, model_dir, ppgn.dataset_file, raw_out);
        end
        [status, output] = run_logged(cmd, log_file, paths.repo_root);
        if status ~= 0, error('GNNBenchmark:pipeline:predict', 'Prediction failed: %s\n%s', log_file, output); end
        ensure_dir(fileparts(con_out));
        copyfile(raw_out, con_out);
        publish_split(job, paths);
        rows(end+1) = struct('job_id', job.job_id, 'seed', s, 'status', 'ok', 'pred_file', raw_out, 'consolidated_file', con_out); %#ok<AGROW>
    end
end
out = struct('jobs', rows, 'raw_root', paths.pred_raw_root, 'consolidated_root', paths.pred_consolidated_root);
end

function out = stage_embedding(opts, paths)
ensure_dir(paths.analysis_tables);
out = struct('status', 'skipped', 'embedding_root', opts.embedding_root, 'message', '');
if isempty(opts.embedding_root)
    out.message = ['No embedding_root supplied. Embedding example panels can be generated by ', ...
        'the figure scripts when GNN_BENCHMARK_EMBED_ENGINE and vt2d paths are configured.'];
    fprintf('[embedding] %s\n', out.message);
    return
end
out.result = GNNBenchmark_analyze_embedding_error_bounds('embedding_root', opts.embedding_root, ...
    'output_dir', fullfile(paths.analysis_tables, 'embedding_error_bounds'));
out.status = 'ok';
end

function out = stage_analysis(opts, paths)
ensure_dir(paths.analysis_tables);
if strcmp(opts.mode, 'mini')
    out = mini_analysis(paths);
else
    data_root = ternary(isempty(opts.data_root_for_figures), paths.pred_consolidated_root, opts.data_root_for_figures);
    assignin('base', 'data_root', data_root);
    assignin('base', 'rebuild_summaries', true);
    assignin('base', 'plot_after_summary', false);
    run(fullfile(paths.repo_root, 'analysis', 'matlab', 'GNNBenchmark_rebuild_all_summaries.m'));
    out = struct('status', 'ok', 'data_root', data_root);
end
end

function out = stage_figures(opts, paths)
ensure_dir(paths.figures);
if strcmp(opts.mode, 'mini')
    table_file = fullfile(paths.analysis_tables, 'mini_prediction_mae.csv');
    if ~isfile(table_file), mini_analysis(paths); end
    T = readtable(table_file, 'TextType', 'string');
    vars = string(T.Properties.VariableNames);
    label_var = vars(1);
    mae_var = vars(strcmpi(vars, 'mae'));
    if isempty(mae_var), error('GNNBenchmark:pipeline:miniFigureSchema', 'Mini MAE table has no mae column.'); end
    fig = figure('Color', 'w', 'Name', 'Mini pipeline MAE smoke test');
    bar(categorical(string(T.(label_var))), T.(mae_var(1)));
    ylabel('Mean absolute error');
    title('Mini pipeline prediction MAE');
    xtickangle(45);
    fig_file = fullfile(paths.figures, 'mini_prediction_mae.fig');
    png_file = fullfile(paths.figures, 'mini_prediction_mae.png');
    savefig(fig, fig_file);
    exportgraphics(fig, png_file, 'Resolution', 200);
    out = struct('status', 'ok', 'figure', fig_file, 'png', png_file);
else
    data_root = ternary(isempty(opts.data_root_for_figures), paths.pred_consolidated_root, opts.data_root_for_figures);
    assignin('base', 'data_root', data_root);
    run(fullfile(paths.repo_root, 'analysis', 'matlab', 'GNNBenchmark_plot_everything.m'));
    out = struct('status', 'ok', 'data_root', data_root);
end
end

function jobs = enumerate_jobs(opts, paths)
jobs = struct('dataset_key', {}, 'weight', {}, 'model', {}, 'family', {}, ...
    'dataset_file', {}, 'dataset_dir', {}, 'split_dir', {}, 'split_label', {}, ...
    'size_token', {}, 'analysis_prefix', {}, 'job_id', {});
for d = 1:numel(opts.datasets)
    dataset_key = opts.datasets{d};
    ds_dir = fullfile(paths.generated_root, 'model_ready', dataset_key, '2D');
    split_root = fullfile(paths.generated_root, 'model_ready', dataset_key, 'splits');
    if ~isfolder(ds_dir), error('GNNBenchmark:pipeline:missingDataset', 'Missing %s', ds_dir); end
    split_dirs = discover_splits(split_root, ds_dir);
    weights = weights_for_dataset(dataset_key, opts);
    for w = 1:numel(weights)
        weight = weights{w};
        dataset_file = dataset_file_for_weight(ds_dir, dataset_key, weight);
        for sp = 1:numel(split_dirs)
            [~, split_label] = fileparts(split_dirs{sp});
            for m = 1:numel(opts.models)
                model = opts.models{m};
                size_token = size_token_for_job(dataset_key, split_label);
                prefix = analysis_prefix(dataset_key, weight, split_label, size_token);
                jobs(end+1) = struct('dataset_key', dataset_key, 'weight', weight, ...
                    'model', model, 'family', model_family(model), ...
                    'dataset_file', dataset_file, 'dataset_dir', ds_dir, ...
                    'split_dir', split_dirs{sp}, 'split_label', split_label, ...
                    'size_token', size_token, 'analysis_prefix', prefix, ...
                    'job_id', clean_id(sprintf('%s_%s_%s_%s', dataset_key, split_label, weight, model))); %#ok<AGROW>
            end
        end
    end
end
end

function split_dirs = discover_splits(split_root, ds_dir)
split_dirs = {};
if isfolder(split_root)
    d = dir(split_root);
    d = d([d.isdir] & ~ismember({d.name}, {'.','..'}));
    for i = 1:numel(d)
        candidate = fullfile(d(i).folder, d(i).name);
        if all(isfile(fullfile(candidate, {'train.inds','val.inds','test.inds'})))
            split_dirs{end+1} = candidate; %#ok<AGROW>
        end
    end
end
if isempty(split_dirs) && all(isfile(fullfile(ds_dir, {'train.inds','val.inds','test.inds'})))
    split_dirs = {ds_dir};
end
if isempty(split_dirs), error('GNNBenchmark:pipeline:noSplits', 'No split folders for %s', ds_dir); end
end

function weights = weights_for_dataset(dataset_key, opts)
if strcmp(dataset_key, 'standard_16') || opts.run_unweighted_revision_conditions
    weights = opts.weights;
else
    weights = intersect(opts.weights, {'W'}, 'stable');
    if isempty(weights), weights = {'W'}; end
end
end

function file = dataset_file_for_weight(ds_dir, dataset_key, weight)
suffix = ternary(strcmp(weight, 'W'), '_weighted.txt', '_unweighted.txt');
file = fullfile(ds_dir, [dataset_key, suffix]);
if ~isfile(file), error('GNNBenchmark:pipeline:missingDataFile', 'Missing %s', file); end
end

function staged = stage_mpnn_dataset(job, paths, seed)
staged.data_dir = fullfile(paths.staged_root, 'mpnn', job.job_id, sprintf('seed_%d', seed), '2D');
ensure_dir(staged.data_dir);
copyfile(job.dataset_file, fullfile(staged.data_dir, filename(job.dataset_file)));
copy_split(job.split_dir, staged.data_dir);
end

function ppgn = stage_ppgn_dataset(job, paths, tree)
parent = ternary(strcmp(job.weight, 'W'), 'Training set lengths_to_lengths', 'Training set none_to_lengths');
ppgn.root = fullfile(paths.staged_root, 'ppgn', tree, job.job_id, parent);
ensure_dir(ppgn.root);
ppgn.dataset_file = fullfile(ppgn.root, filename(job.dataset_file));
if ~isfile(ppgn.dataset_file), copyfile(job.dataset_file, ppgn.dataset_file); end
end

function copy_split(src, dst)
for nm = {'train.inds','val.inds','test.inds'}
    copyfile(fullfile(src, nm{1}), fullfile(dst, nm{1}));
end
end

function cmd = mpnn_train_cmd(opts, job, hp, data_dir, seed)
trainer = fullfile(repo_root(), 'models', 'mpnn', 'trainer_final.py');
cmd = sprintf('"%s" "%s" --data_dir "%s" --model %s --weighted %s --use_node_feats True --cuda %d --seed %d --head regressor --norm_mode per_graph --num_layers 16 --epochs %d --hidden_channels %d --dropout %.8g --lr %.12g --batch_size %d --factor %.8g --patience 20 --early_stop_patience 40 --early_stop_min_delta 1e-4', ...
    opts.python, trainer, data_dir, job.model, boolstr(strcmp(job.weight, 'W')), opts.cuda, seed, opts.final_epochs, ...
    hp.hidden_channels, hp.dropout, hp.lr, hp.batch_size, hp.factor);
if isfield(hp, 'ablate_head_edge_attr') && hp.ablate_head_edge_attr
    cmd = [cmd, ' --ablate_head_edge_attr'];
end
end

function cmd = mpnn_predict_cmd(opts, ckpt, data_file, out_file, batch_size)
predictor = fullfile(repo_root(), 'models', 'mpnn', 'predict_final.py');
cmd = sprintf('"%s" "%s" --model-path "%s" --data-file "%s" --out "%s" --batch-size %d --cuda %d', ...
    opts.python, predictor, ckpt, data_file, out_file, batch_size, opts.cuda);
end

function cmd = ppgn_train_cmd(opts, job, hp, data_file, split_dir, out_dir)
src = fullfile(repo_root(), 'models', 'ppgn', 'train_gnn_benchmark');
args = ppgn_args(job, hp, opts.final_epochs);
core = sprintf('"%s" -m gnn_benchmark_ppgn.main train --training-data "%s" --out-dir "%s" --inds-dir "%s" --no-evaluation --args "%s"', ...
    opts.python, data_file, out_dir, split_dir, args);
cmd = with_pythonpath(core, src, opts.cuda);
end

function cmd = ppgn_predict_cmd(opts, model_dir, data_file, out_file)
src = fullfile(repo_root(), 'models', 'ppgn', 'predict_gnn_benchmark');
core = sprintf('"%s" -m gnn_benchmark_ppgn.main predict --model "%s" --input "%s" --output "%s"', opts.python, model_dir, data_file, out_file);
cmd = with_pythonpath(core, src, opts.cuda);
end

function args = ppgn_args(job, hp, epochs)
eigs = arrayfun(@(k) sprintf('eig%d', k), 1:30, 'UniformOutput', false);
node_feats = ['degree,n_min_degree,n_max_degree,n_mean_degree,n_sd_degree,' strjoin(eigs, ',')];
node_core = 'degree,n_min_degree,n_max_degree,n_mean_degree,n_sd_degree';
if strcmp(job.weight, 'W')
    input_features = ['[in_preferred_length,in_was_flipped,' node_feats ']'];
    normalize = ['[in_preferred_length,out_preferred_length,' node_core ']'];
else
    input_features = ['[' node_feats ']'];
    normalize = ['[out_preferred_length,' node_core ']'];
end
parts = {sprintf('epochs=%d', epochs), 'patience=20', 'early_stop=40', 'threshold=1e-4', ...
    'block_features=[400,400,400]', 'depth_of_mlp=2', ...
    sprintf('learning_rate=%.12g', hp.learning_rate), sprintf('batch_size=%d', hp.batch_size), ...
    sprintf('factor=%.8g', hp.factor), sprintf('gradient_clipping=%.8g', hp.gradient_clipping), ...
    sprintf('input_features=%s', input_features), 'target_features=[out_preferred_length]', ...
    sprintf('normalize=%s', normalize), 'disable_first_skip=False'};
args = strjoin(parts, ';');
end

function hp = load_hps(job, paths)
file = fullfile(paths.best_hp_root, [job.job_id, '_best_hps.mat']);
if isfile(file)
    S = load(file, 'hp');
    hp = S.hp;
else
    hp = default_hps(job.model, job.family);
end
end

function hp = best_hps_from_bayesopt(results, hp_ranges, family)
if isprop(results, 'XAtMinObjective') && ~isempty(results.XAtMinObjective)
    x = results.XAtMinObjective;
else
    [~, ix] = min(results.ObjectiveTrace);
    x = results.XTrace(ix, :);
end
hp = default_hps('', family);
names = fieldnames(hp_ranges);
for i = 1:numel(names)
    nm = names{i};
    raw = x.(nm);
    vals = hp_ranges.(nm);
    if iscell(vals)
        nums = str2double(vals);
        if all(isfinite(nums))
            [~, order] = sort(nums);
            vals = vals(order);
            if isnumeric(raw)
                val = str2double(vals{max(1, min(numel(vals), round(double(raw))))});
            elseif iscategorical(raw)
                val = str2double(char(raw));
            else
                val = str2double(char(raw));
            end
        else
            val = char(raw);
        end
    else
        val = double(raw);
    end
    if strcmp(family, 'PPGN') && strcmp(nm, 'lr'), nm = 'learning_rate'; end
    if strcmp(family, 'MPNN') && strcmp(nm, 'learning_rate'), nm = 'lr'; end
    hp.(nm) = val;
end
end

function hp = default_hps(model, family)
if strcmp(family, 'PPGN')
    hp = struct('learning_rate', 5.676e-4, 'batch_size', 8, 'factor', 0.6, 'gradient_clipping', 0.1);
else
    switch model
        case 'GraphSAGE', hp = struct('lr', 6.356e-4, 'hidden_channels', 128, 'dropout', 0, 'batch_size', 2, 'factor', 0.75);
        case 'GAT',       hp = struct('lr', 5.002e-4, 'hidden_channels', 64,  'dropout', 0, 'batch_size', 1, 'factor', 0.75);
        case 'GIN',       hp = struct('lr', 3.460e-4, 'hidden_channels', 128, 'dropout', 0, 'batch_size', 2, 'factor', 0.75);
        case 'PNA',       hp = struct('lr', 4.722e-4, 'hidden_channels', 64,  'dropout', 0, 'batch_size', 1, 'factor', 0.75);
        otherwise,        hp = struct('lr', 1e-3,    'hidden_channels', 64,  'dropout', 0, 'batch_size', 1, 'factor', 0.75);
    end
    hp.ablate_head_edge_attr = false;
end
end

function prefix = analysis_prefix(dataset_key, weight, split_label, size_token)
switch dataset_key
    case 'standard_16', prefix = sprintf('v1_%s_%s', size_token, weight);
    case 'kA_10', prefix = 'rev_kA_10';
    case 'kA_1', prefix = 'rev_kA_1';
    case 'shear_1_2', prefix = 'rev_Shear_1_2';
    case 'shear_1_5', prefix = 'rev_Shear_1_5';
    case 'tissue_484', prefix = 'rev_Tissue_484';
    case 'tissue_784', prefix = 'rev_Tissue_784';
    case 'flip_two', prefix = 'rev_Flip_two';
    otherwise, prefix = sprintf('mini_%s_%s_%s', dataset_key, split_label, weight);
end
end

function name = consolidated_name(job, seed)
switch job.dataset_key
    case 'standard_16', task = 'standard-flip'; sz = job.size_token;
    case {'kA_10','kA_1'}, task = 'kA'; sz = char(extractAfter(job.dataset_key, 'kA_'));
    case {'shear_1_2','shear_1_5'}, task = 'Shear'; sz = char(extractAfter(job.dataset_key, 'shear_'));
    case {'tissue_484','tissue_784'}, task = 'Tissue'; sz = char(extractAfter(job.dataset_key, 'tissue_'));
    case 'flip_two', task = 'Flip-two'; sz = 'na';
    otherwise, task = ['mini-' job.dataset_key]; sz = job.size_token;
end
name = sprintf('%s_%s_%s_%s_s%d.pred.txt', task, job.model, job.weight, sz, seed);
end

function token = size_token_for_job(dataset_key, split_label)
if strcmp(dataset_key, 'standard_16')
    tok = regexp(split_label, 'standard_(\d+_\d+)', 'tokens', 'once');
    if ~isempty(tok), token = tok{1}; return; end
end
token = clean_id(split_label);
end

function publish_split(job, paths)
dst = fullfile(paths.pred_consolidated_root, 'splits', clean_id(job.analysis_prefix));
ensure_dir(dst);
copy_split(job.split_dir, dst);
write_text(fullfile(dst, '_applies_to.txt'), sprintf('applies to datasets: %s\n', job.analysis_prefix));
end

function family = model_family(model)
family = ternary(strcmp(model, 'PPGN'), 'PPGN', 'MPNN');
end

function out = mini_analysis(paths)
files = dir(fullfile(paths.pred_raw_root, '*.pred.txt'));
if isempty(files), error('GNNBenchmark:pipeline:noPredictionRecords', 'No analyzable prediction records under %s', paths.pred_raw_root); end
job_id = strings(numel(files), 1);
mae = nan(numel(files), 1);
n_edges = zeros(numel(files), 1);
for i = 1:numel(files)
    f = fullfile(files(i).folder, files(i).name);
    [mae(i), n_edges(i)] = mae_file(f);
    job_id(i) = erase(files(i).name, '.pred.txt');
end
T = table(job_id, mae, n_edges);
out_file = fullfile(paths.analysis_tables, 'mini_prediction_mae.csv');
writetable(T, out_file);
out = struct('status', 'ok', 'table', out_file, 'n_files', numel(files));
fprintf('[mini analysis] Wrote %s\n', out_file);
end

function [mae, n] = mae_file(file)
fid = fopen(file, 'rt');
if fid < 0, error('Could not open %s', file); end
c = onCleanup(@() fclose(fid));
err = [];
while true
    line = fgetl(fid);
    if ~ischar(line), break; end
    nums = sscanf(line, '%f').';
    if numel(nums) >= 4
        err(end+1, 1) = abs(nums(end) - nums(end-1)); %#ok<AGROW>
    end
end
mae = mean(err, 'omitnan');
n = numel(err);
end

function ckpt = find_checkpoint(root, model, weight, seed)
weighted = boolstr(strcmp(weight, 'W'));
patterns = {sprintf('gnn_%s_weighted_%s_*_seed_%d.pth', model, weighted, seed), sprintf('*%s*seed_%d*.pth', model, seed), '*.pth'};
for i = 1:numel(patterns)
    d = dir(fullfile(root, patterns{i}));
    if ~isempty(d)
        [~, ix] = max([d.datenum]);
        ckpt = fullfile(d(ix).folder, d(ix).name);
        return
    end
end
error('GNNBenchmark:pipeline:noCheckpoint', 'No checkpoint found under %s', root);
end

function file = latest_mat(root)
d = dir(fullfile(root, '**', '*.mat'));
if isempty(d), file = ''; else, [~, ix] = max([d.datenum]); file = fullfile(d(ix).folder, d(ix).name); end
end

function [status, output] = run_logged(cmd, log_file, workdir)
ensure_dir(fileparts(log_file));
write_text(log_file, sprintf('[%s] COMMAND\n%s\n\n', stamp(), cmd));
old = pwd;
c = onCleanup(@() cd(old));
cd(workdir);
[status, output] = system(cmd);
append_text(log_file, output);
end

function cmd = with_pythonpath(core, source_dir, cuda)
if ispc
    cuda_part = ternary(cuda >= 0, sprintf('set CUDA_VISIBLE_DEVICES=%d & ', cuda), 'set CUDA_VISIBLE_DEVICES= & ');
    cmd = sprintf('set "PYTHONPATH=%s;%%PYTHONPATH%%" & %s%s', source_dir, cuda_part, core);
else
    cuda_part = ternary(cuda >= 0, sprintf('export CUDA_VISIBLE_DEVICES=%d; ', cuda), 'export CUDA_VISIBLE_DEVICES=; ');
    cmd = sprintf('export PYTHONPATH="%s:$PYTHONPATH"; %s%s', source_dir, cuda_part, core);
end
end

function root = repo_root()
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end

function manifest = save_manifest(manifest, paths, label)
manifest.last_update = stamp();
manifest.last_label = label;
write_json(fullfile(paths.manifests, 'pipeline_manifest.json'), manifest);
save(fullfile(paths.manifests, 'pipeline_manifest.mat'), 'manifest');
end

function write_json(file, data)
ensure_dir(fileparts(file));
write_text(file, jsonencode(data, 'PrettyPrint', true));
end

function write_text(file, txt)
ensure_dir(fileparts(file));
fid = fopen(file, 'wt');
if fid < 0, error('Could not open %s', file); end
fprintf(fid, '%s', txt);
fclose(fid);
end

function append_text(file, txt)
fid = fopen(file, 'at');
if fid < 0, error('Could not open %s', file); end
fprintf(fid, '%s', txt);
fclose(fid);
end

function ensure_dir(d)
if ~isempty(d) && ~isfolder(d), mkdir(d); end
end

function answer = ask_yes_no(question, default_yes)
reply = input(sprintf('%s [%s]: ', question, ternary(default_yes, 'Y/n', 'y/N')), 's');
if isempty(strtrim(reply)), answer = default_yes; else, answer = any(strcmpi(strtrim(reply), {'y','yes'})); end
end

function snap = git_snapshot(repo_root)
snap = struct('commit', '', 'status', '', 'available', false);
[st, commit] = system(sprintf('git -C "%s" rev-parse HEAD', repo_root));
if st == 0
    snap.available = true;
    snap.commit = strtrim(commit);
    [~, status] = system(sprintf('git -C "%s" status --short', repo_root));
    snap.status = strtrim(status);
end
end

function value = getenv_or(name, fallback)
value = getenv(name);
if isempty(value), value = fallback; end
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

function s = boolstr(tf)
s = ternary(tf, 'True', 'False');
end

function s = clean_id(s)
s = regexprep(char(s), '[^A-Za-z0-9_\-]+', '_');
s = regexprep(s, '_+', '_');
s = regexprep(s, '^_|_$', '');
end

function name = filename(path)
[~, b, e] = fileparts(path);
name = [b, e];
end

function t = stamp()
t = datestr(now, 'yyyy-mm-dd HH:MM:SS'); %#ok<DATST>
end
