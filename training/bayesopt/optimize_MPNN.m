function results = optimize_MPNN(dataset_filename, inds_dirname, model_name, hp_ranges, n_trials, varargin)
% optimize_MPNN  Implement optimize mpnn for this MATLAB workflow.
% Inputs: dataset_filename, inds_dirname, model_name, hp_ranges, n_trials, varargin
% Outputs: results
%OPTIMIZE_MPNN  Bayesian hyper-parameter optimization for any MPNN model.
%
%   RESULTS = OPTIMIZE_MPNN(DATASET_FILENAME, INDS_DIRNAME, MODEL_NAME,
%   HP_RANGES, N_TRIALS) runs Bayesian optimization (MATLAB's BAYESOPT)
%   over the hyper-parameters of the MPNN trainer pipeline, using
%   the dataset at DATASET_FILENAME and the train/val/test index files
%   in INDS_DIRNAME. The objective minimized is the smoothed minimum of
%   the validation loss curve written by trainer.py for each trial.
%
%   Required inputs
%   ---------------
%   DATASET_FILENAME : char/string. Full path to a single Nano .txt
%                      dataset. The basename must end in either
%                      '_weighted.txt' or '_unweighted.txt' - the
%                      weighted flag passed to trainer.py is inferred
%                      from this suffix.
%   INDS_DIRNAME     : char/string. Folder containing the three split
%                      files (train.inds, val.inds, test.inds). One
%                      graph index per line.
%   MODEL_NAME       : char/string. Any MPNN model accepted by
%                      trainer.py --model (e.g. 'GraphSAGE', 'GAT',
%                      'GIN', 'GCN', 'PNA').
%   HP_RANGES        : struct. Each field specifies one trainer.py
%                      hyper-parameter to sweep. Field names must match
%                      trainer.py CLI flag names, e.g.
%                        hp_ranges.hidden_channels = {'64','128'};
%                        hp_ranges.dropout         = {'0','0.1','0.2'};
%                        hp_ranges.lr              = [1e-4, 1e-2];
%                        hp_ranges.batch_size      = {'1','2','4'};
%                        hp_ranges.factor          = {'0.75'};  % fixed final setting
%                      Values:
%                        * cell array of char/string - categorical var.
%                        * 2-element numeric [lo hi]  - real var.
%                          Log transform is applied automatically when
%                          the field name is 'lr' (to mirror
%                          optimize_PNA.m); all other real vars are
%                          linear-scale.
%                      HPs not listed here keep trainer.py's defaults.
%   N_TRIALS         : positive integer. Total BAYESOPT evaluations
%                      (initial random seed points + GP-guided trials).
%
%   Optional name/value pairs
%   -------------------------
%   'cuda'            : GPU id for trainer.py (default 0; -1 = CPU).
%   'max_epochs'      : --epochs passed to every trial (default 120).
%   'patience'        : --patience passed to every trial (default 20).
%   'early_stop_patience' : --early_stop_patience passed to every trial
%                       (default 40).
%   'early_stop_min_delta': --early_stop_min_delta passed to every trial
%                       (default 1e-4).
%   'seed'            : --seed passed to every trial (default 0). A
%                       single seed per trial means HP-vs-noise isn't
%                       disentangled; the paper-final training should
%                       average over several seeds outside this script.
%   'use_node_feats'  : 'True'|'False' for the 35-dim node features
%                       (default 'True').
%   'ablate_head_edge_attr' : logical ablation switch. false (default)
%                       trains the length-informed architecture. true withholds
%                       raw edge attributes from only the final regression head,
%                       leaving the message-passing backbone unchanged.
%   'num_seed_points' : BAYESOPT NumSeedPoints (default 6).
%   'acquisition_fn'  : BAYESOPT AcquisitionFunctionName
%                       (default 'expected-improvement-plus').
%   'output_dirname'  : Where to save the BAYESOPT .mat result.
%                       Default: <dataset parent>/Bayesian_optimization_results/
%   'skip_existing'   : If true and the target .mat already exists,
%                       skip the run and load/return the saved result
%                       (default false).
%   'smooth_val_loss' : If true (default), each trial's objective is the
%                       min of the smoothed val_loss curve (medfilt1 w=5,
%                       moving-average span 100, trimmed to [50:end-10]).
%                       Short runs with too few logged epochs for that
%                       smoothing window automatically fall back to
%                       objective = min(val_loss), so reduced integration
%                       tests still produce valid BO observations.
%                       If false, always skip smoothing and use
%                       objective = min(val_loss) over the whole curve.
%                       Use false for smoke tests where you only want to
%                       confirm each trial runs, not how well it trains.
%
%   Output
%   ------
%   RESULTS : the BayesianOptimization object returned by BAYESOPT.
%             Also persisted to disk as a .mat file in output_dirname.
%
%   Saved files and restart behavior
%   --------------------------------
%   The final result is written to:
%       <output_dirname>/<run_name>.mat
%   A text diary is written beside it as:
%       <output_dirname>/<run_name>.log
%   A trial-level checkpoint is written after each completed BAYESOPT
%   objective evaluation as:
%       <output_dirname>/<run_name>.partial.mat
%   On restart, RUN_OR_RESUME_BAYESOPT loads the partial checkpoint and
%   re-seeds a fresh BAYESOPT call from completed finite observations.
%   This preserves finished trials after a wall-time kill or crash while
%   avoiding MATLAB resume() closures that may contain stale runtime
%   options. A crash during an individual trainer run can still lose that
%   one in-progress trial, because BAYESOPT checkpoints only after the
%   objective function returns.
%
%   How it works
%   ------------
%   * A shared working directory is created under output_dirname with
%     '2D' in its name (required by Nano's dim-inference assertion).
%     The dataset .txt and the three .inds files are symlinked into
%     it, so trainer.py's processed-cache is reused across all trials
%     - the expensive per-graph transforms run exactly once.
%   * BAYESOPT samples HPs according to HP_RANGES. Each trial launches
%     the configured trainer with --data_dir pointing at the shared
%     working directory. Unswept HPs fall back to the trainer defaults.
%   * After each trial, the function locates the freshly-created log
%     directory under <workdir>/logs/ (mtime-based, disambiguated by
%     a regex on the dim/weighted/model/hc/nl prefix), reads
%     csv/val_loss.csv, and returns the smoothed minimum validation loss
%     for full-length BO trials. When the trial has too few epochs to
%     support the smoothing window, it returns the raw minimum validation
%     loss instead. NaN is returned for failed or degenerate trials
%     (trainer nonzero exit, missing validation curve, all-zero curve).
%
%   See also gnn_benchmark_pipeline_2D_revision, BAYESOPT, OPTIMIZABLEVARIABLE.

    %% ---- parse inputs --------------------------------------------------
    p = inputParser;
    p.FunctionName = 'optimize_MPNN';
    addRequired(p, 'dataset_filename', @(x) (ischar(x) || isstring(x)) && isfile(char(x)));
    addRequired(p, 'inds_dirname',     @(x) (ischar(x) || isstring(x)) && isfolder(char(x)));
    addRequired(p, 'model_name',       @(x) ischar(x) || isstring(x));
    addRequired(p, 'hp_ranges',        @isstruct);
    addRequired(p, 'n_trials',         @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'cuda',            0,                           @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'max_epochs',      120,                         @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'patience',        20,                          @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'early_stop_patience', 40,                      @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'early_stop_min_delta', 1e-4,                   @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'seed',            0,                           @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'use_node_feats',  'True',                      @(x) any(strcmpi(char(x), {'True','False'})));
    addParameter(p, 'ablate_head_edge_attr', false,                 @(x) islogical(x) && isscalar(x));
    addParameter(p, 'num_seed_points', 6,                          @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'acquisition_fn',  'expected-improvement-plus', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dirname',  '',                          @(x) ischar(x) || isstring(x));
    addParameter(p, 'skip_existing',   false,                       @(x) islogical(x) && isscalar(x));
    default_trainer = getenv('GNN_BENCHMARK_MPNN_TRAINER');
    if isempty(default_trainer)
        repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
        default_trainer = fullfile(repo_root, 'models', 'mpnn', 'trainer_final.py');
    end
    default_python = getenv('GNN_BENCHMARK_PYTHON');
    if isempty(default_python), default_python = 'python'; end
    default_module_prefix = getenv('GNN_BENCHMARK_MPNN_MODULE_PREFIX');
    if isempty(default_module_prefix)
        default_module_prefix = ['export TERM=xterm; ', ...
            'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4; '];
    end

    addParameter(p, 'smooth_val_loss', true,                        @(x) islogical(x) && isscalar(x));
    addParameter(p, 'trainer_py',      default_trainer,             @(x) ischar(x) || isstring(x));
    addParameter(p, 'python',          default_python,              @(x) ischar(x) || isstring(x));
    % prior_log: path to a previous bayesopt .log file whose iteration
    % table should seed this run as InitialX/InitialObjective. Use this
    % to continue a BO session whose .partial.mat was never saved.
    addParameter(p, 'prior_log',       '',                          @(x) ischar(x) || isstring(x));
    % Shell prefix for the trainer subprocess. Use this to activate a
    % virtual environment, load modules, or set CUDA/CPU-thread limits.
    % By default only conservative CPU-thread caps are exported; set
    % GNN_BENCHMARK_MPNN_MODULE_PREFIX or pass 'module_prefix' for cluster-specific
    % activation commands.
    addParameter(p, 'module_prefix', default_module_prefix, @(x) ischar(x) || isstring(x));
    parse(p, dataset_filename, inds_dirname, model_name, hp_ranges, n_trials, varargin{:});
    opts = p.Results;

    dataset_filename = char(opts.dataset_filename);
    inds_dirname     = char(opts.inds_dirname);
    model_name       = char(opts.model_name);
    use_node_feats   = char(opts.use_node_feats);
    hp_ranges        = opts.hp_ranges;
    n_trials         = opts.n_trials;

    % Snapshot the prior_log immediately - later in this function the
    % diary re-initialisation deletes the existing log_filename, which
    % in a restart is the very file we want to parse.
    prior_log_src = char(opts.prior_log);
    prior_log_snapshot = '';
    if ~isempty(prior_log_src)
        if ~isfile(prior_log_src)
            error('optimize_MPNN:priorLogMissing', ...
                'prior_log file not found: %s', prior_log_src);
        end
        prior_log_snapshot = [tempname() '.log'];
        copyfile(prior_log_src, prior_log_snapshot);
    end

    %% ---- infer weighted flag from filename suffix ----------------------
    [ds_parent, base, ext] = fileparts(dataset_filename);
    if ~strcmp(ext, '.txt')
        error('optimize_MPNN:badFile', ...
            'Dataset file must have a .txt extension (got "%s").', [base ext]);
    end
    if endsWith(base, '_weighted')
        weighted = 'True';
    elseif endsWith(base, '_unweighted')
        weighted = 'False';
    else
        error('optimize_MPNN:weightedSuffix', ...
            ['Dataset filename must end in "_weighted.txt" or "_unweighted.txt" ' ...
             '(got "%s"). The weighted flag is inferred from this suffix.'], [base ext]);
    end

    %% ---- output + working directories ----------------------------------
    output_dirname = char(opts.output_dirname);
    if isempty(output_dirname)
        output_dirname = fullfile(ds_parent, 'Bayesian_optimization_results');
    end
    if ~isfolder(output_dirname)
        mkdir(output_dirname);
    end

    % Working dir name must contain '2D' to satisfy Nano's dim inference.
    work_dirname = fullfile(output_dirname, ...
        sprintf('workdir_2D_%s_weighted_%s_%s', model_name, weighted, base));
    if ~isfolder(work_dirname)
        mkdir(work_dirname);
    end

    % Symlink inputs in. Overwrites stale links; does not copy bytes.
    stage_symlink(dataset_filename, work_dirname);
    inds_files = dir(fullfile(inds_dirname, '*.inds'));
    if numel(inds_files) < 3
        warning('optimize_MPNN:inds', ...
            'Expected 3 .inds files in "%s", found %d. Proceeding anyway.', ...
            inds_dirname, numel(inds_files));
    end
    for k = 1:numel(inds_files)
        stage_symlink(fullfile(inds_dirname, inds_files(k).name), work_dirname);
    end

    result_filename = fullfile(output_dirname, ...
        sprintf('%s_2D_weighted_%s_%s.mat', model_name, weighted, base));
    if opts.skip_existing && isfile(result_filename)
        fprintf('[optimize_MPNN] %s already exists; skip_existing=true - loading and returning.\n', ...
            result_filename);
        S = load(result_filename, 'results');
        results = S.results;
        return;
    end

    %% ---- build optimizableVariable list --------------------------------
    % When a hp_ranges entry is a cell array of numeric-valued strings,
    % model it as an ORDINAL integer index into the sorted value list
    % rather than as a categorical. That way BAYESOPT's GP treats
    % neighbouring grid points (e.g. batch_size 4 vs 8) as close, instead
    % of as unrelated labels. HPs supplied as a single-value cell array are
    % folded into fixed CLI args instead of being optimized.
    fn_in = fieldnames(hp_ranges);
    optVars = {};
    fn = {};                  % fields actually swept by BO
    ordinal_map = struct();   % ordinal_map.(name) = sorted cell of strings
    frozen = struct();        % single-value HPs passed to every trial
    for k = 1:numel(fn_in)
        name = fn_in{k};
        v    = hp_ranges.(name);
        if iscell(v)
            cats = cellfun(@char, v, 'UniformOutput', false);
            nums = str2double(cats);
            if numel(cats) == 1
                frozen.(name) = cats{1};
                continue;
            end
            if all(~isnan(nums)) && ~isempty(cats)
                % Numeric cell -> ordinal integer-indexed lookup.
                [~, order] = sort(nums);
                sorted_cats = cats(order);
                ordinal_map.(name) = sorted_cats;
                optVars{end+1} = optimizableVariable(name, ... %#ok<AGROW>
                    [1, numel(sorted_cats)], 'Type', 'integer');
            else
                % Genuinely unordered strings -> stay categorical.
                optVars{end+1} = optimizableVariable(name, cats, 'Type', 'categorical'); %#ok<AGROW>
            end
            fn{end+1} = name; %#ok<AGROW>
        elseif isnumeric(v) && numel(v) == 2 && v(2) > v(1)
            if strcmpi(name, 'lr') || strcmpi(name, 'learningRate')
                optVars{end+1} = optimizableVariable(name, v(:)', 'Type', 'real', 'Transform', 'log'); %#ok<AGROW>
            else
                optVars{end+1} = optimizableVariable(name, v(:)', 'Type', 'real'); %#ok<AGROW>
            end
            fn{end+1} = name; %#ok<AGROW>
        elseif isnumeric(v) && isscalar(v)
            frozen.(name) = num2str(v, '%g');
        else
            error('optimize_MPNN:badHP', ...
                ['hp_ranges.%s must be a cell array of strings, a scalar fixed value, ' ...
                 'or a 2-element increasing numeric vector [lo hi] (real).'], name);
        end
    end
    if isempty(optVars)
        error('optimize_MPNN:noSweptHPs', 'At least one hp_ranges field must have more than one value.');
    end
    optVars = [optVars{:}];
    opts.fixed_hps = frozen;

    %% ---- run BAYESOPT --------------------------------------------------
    % Capture bayesopt's iteration table (and anything else printed
    % during the run) into a plain-text log alongside the .mat, so the
    % saturation / convergence of each HP can be reviewed afterwards
    % without reopening the BayesianOptimization object.
    log_filename = regexprep(result_filename, '\.mat$', '.log');
    if isfile(log_filename)
        delete(log_filename);
    end
    diary(log_filename);
    diary_guard = onCleanup(@() diary('off'));

    fprintf('=== optimize_MPNN ===\n');
    fprintf('timestamp       : %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS')); %#ok<DATST>
    fprintf('model           : %s\n', model_name);
    fprintf('weighted        : %s\n', weighted);
    fprintf('dataset_file    : %s\n', dataset_filename);
    fprintf('inds_dir        : %s\n', inds_dirname);
    fprintf('work_dir        : %s\n', work_dirname);
    fprintf('n_trials        : %d\n', n_trials);
    fprintf('num_seed_points : %d\n', opts.num_seed_points);
    fprintf('max_epochs      : %d\n', opts.max_epochs);
    fprintf('patience        : %d\n', opts.patience);
    fprintf('early_stop_pat  : %d\n', opts.early_stop_patience);
    fprintf('early_stop_delta: %.6g\n', opts.early_stop_min_delta);
    fprintf('ablate_head_attr: %d\n', opts.ablate_head_edge_attr);
    fprintf('cuda            : %d\n', opts.cuda);
    fprintf('acquisition_fn  : %s\n', char(opts.acquisition_fn));
    fprintf('hp_ranges       :\n');
    disp(hp_ranges);
    fprintf('fixed_hps       :\n');
    disp(opts.fixed_hps);
    fprintf('---\n');

    fun = @(x) eval_trial(x, fn, ordinal_map, work_dirname, weighted, model_name, use_node_feats, opts);

    % Per-iteration checkpoint: bayesopt writes the BO object to this file
    % after every trial. If the run is killed mid-sweep (e.g. wall-clock
    % limit), restarting reloads it and calls resume() to continue.
    partial_filename = regexprep(result_filename, '\.mat$', '.partial.mat');
    bo_args = { ...
        'MaxObjectiveEvaluations',  n_trials, ...
        'AcquisitionFunctionName',  char(opts.acquisition_fn), ...
        'NumSeedPoints',            opts.num_seed_points, ...
        'IsObjectiveDeterministic', false, ...
        'Verbose',                  1, ...
        'UseParallel',              false, ...
        'OutputFcn',                @saveToFile, ...
        'PlotFcn',                  [], ...
        'SaveFileName',             partial_filename};

    if ~isempty(prior_log_snapshot)
        [prior_X, prior_obj, prior_time] = parse_bayesopt_log( ...
            prior_log_snapshot, fn, ordinal_map);
        fprintf('[optimize_MPNN] prior_log: seeding bayesopt with %d prior trials from %s\n', ...
            height(prior_X), prior_log_src);
        prior_display = prior_X;
        prior_display.objective   = prior_obj;
        prior_display.runtime_sec = prior_time;
        disp(prior_display);
        delete(prior_log_snapshot);
        ix = find(strcmpi(bo_args(1:2:end), 'NumSeedPoints'), 1);
        if ~isempty(ix)
            bo_args{2*ix} = max(height(prior_X), 1);
        end
        bo_args = [bo_args, {'InitialX',                        prior_X, ...
                             'InitialObjective',                prior_obj, ...
                             'InitialObjectiveEvaluationTimes', prior_time}];
        results = bayesopt(fun, optVars, bo_args{:});
    else
        results = run_or_resume_bayesopt(fun, optVars, partial_filename, n_trials, bo_args);
    end

    fprintf('---\n');
    fprintf('Done. MinObjective=%.6g  MinEstObjective=%.6g\n', ...
        results.MinObjective, results.MinEstimatedObjective);

    diary('off');
    clear diary_guard;

    save(result_filename, 'results', 'opts', 'hp_ranges', 'dataset_filename', ...
         'inds_dirname', 'model_name', 'n_trials', 'weighted', 'work_dirname', ...
         'log_filename', 'ordinal_map', 'frozen');
    fprintf('[optimize_MPNN] Done. Results saved to %s\n', result_filename);
    fprintf('[optimize_MPNN] Log saved to %s\n', log_filename);
    close all;
end


% =====================================================================
% Per-trial objective: run trainer.py once and return smoothed val loss.
% =====================================================================
function objective = eval_trial(x, hp_field_names, ordinal_map, work_dirname, weighted, ...
                                model_name, use_node_feats, opts)
% eval_trial  Implement eval trial for this MATLAB workflow.
% Inputs: x, hp_field_names, ordinal_map, work_dirname, weighted, model_name, use_node_feats, opts
% Outputs: objective

    % Build the CLI argument block from fixed HPs plus sampled HPs.
    cli_block = '';
    fixed_names = fieldnames(opts.fixed_hps);
    for fk = 1:numel(fixed_names)
        fname = fixed_names{fk};
        cli_block = sprintf('%s --%s %s', cli_block, fname, char(opts.fixed_hps.(fname)));
    end
    if opts.ablate_head_edge_attr
        cli_block = sprintf('%s --ablate_head_edge_attr', cli_block);
    end

    for k = 1:numel(hp_field_names)
        name = hp_field_names{k};
        val  = x.(name);
        if isfield(ordinal_map, name)
            % Ordinal HP: BO sampled an integer index; look up the
            % corresponding string value from the sorted list.
            val_str = ordinal_map.(name){double(val)};
        elseif iscategorical(val)
            val_str = char(val);
        elseif strcmpi(name, 'lr') || strcmpi(name, 'learningRate')
            val_str = num2str(val, '%.8g');
        else
            val_str = num2str(val);
        end
        cli_block = sprintf('%s --%s %s', cli_block, name, val_str);
    end

    trainer_py = char(opts.trainer_py);
    if ~isfile(trainer_py)
        error('optimize_MPNN:trainerMissing', 'trainer_py not found: %s', trainer_py);
    end
    python_cmd = char(opts.python);
    com = ['"', python_cmd, '" "', trainer_py, '"', ...
        ' --dim 2D', ...
        ' --weighted ',       weighted, ...
        ' --use_node_feats ', use_node_feats, ...
        ' --model ',          model_name, ...
        ' --cuda ',           num2str(opts.cuda), ...
        ' --seed ',           num2str(opts.seed), ...
        ' --epochs ',         num2str(opts.max_epochs), ...
        ' --patience ',       num2str(opts.patience), ...
        ' --early_stop_patience ', num2str(opts.early_stop_patience), ...
        ' --early_stop_min_delta ', num2str(opts.early_stop_min_delta, '%g'), ...
        cli_block, ...
        ' --data_dir "',      work_dirname, '"'];

    % Two ways to interpret module_prefix:
    %  * default: a shell-env prefix that leaves the configured Python
    %    interpreter and trainer path usable in the subprocess environment.
    %  * '@cmdenv:<cmd>': place the full trainer invocation in env var
    %    GNN_BENCHMARK_MPNN_TRAINER_CMD and run <cmd> alone. This is useful for
    %    site-specific wrappers that dispatch the training command remotely.
    prefix_str = char(opts.module_prefix);
    if startsWith(prefix_str, '@cmdenv:')
        setenv('GNN_BENCHMARK_MPNN_TRAINER_CMD', com);
        curr_command = [extractAfter(prefix_str, '@cmdenv:'), ' 2>&1'];
    else
        curr_command = [prefix_str, com, ' 2>&1'];
    end

    % Timestamp marker to disambiguate which log dir this trial created.
    t_before = now - 1/86400;  % 1-second safety margin against clock skew

    [ret, out] = system(curr_command);
    if ret ~= 0
        % Surface the subprocess output so trainer/module failures aren't
        % hidden. Save the full transcript next to the per-trial workdir
        % and echo the tail into the MATLAB diary.
        stderr_file = fullfile(work_dirname, sprintf( ...
            'trial_stderr_%s.log', datestr(now, 'yyyymmdd_HHMMSS_FFF'))); %#ok<DATST>
        try
            fid = fopen(stderr_file, 'wt');
            if fid > 0
                fprintf(fid, '%s\n', curr_command);
                fprintf(fid, '--- subprocess output (ret=%d) ---\n%s\n', ret, out);
                fclose(fid);
            end
        catch
        end
        fprintf('[eval_trial] subprocess failed (ret=%d). Tail:\n', ret);
        lines = regexp(out, '\r?\n', 'split');
        lines = lines(max(1,numel(lines)-30):end);
        fprintf('%s\n', strjoin(lines, newline));
        fprintf('[eval_trial] full output: %s\n', stderr_file);
        objective = nan;
        return;
    end

    % Locate this trial's log dir. trainer.py always writes to
    % <data_dir>/logs/<timestamp>_dim_..._dropout_..._ls_..._wd_... .
    logs_dir = fullfile(work_dirname, 'logs');
    if ~isfolder(logs_dir)
        objective = nan;
        return;
    end
    dd = dir(logs_dir);
    dd = dd([dd.isdir] & ~ismember({dd.name}, {'.', '..'}));
    if isempty(dd)
        objective = nan;
        return;
    end
    % Keep only dirs whose basename matches the model/dim/weighted/nodefeats
    % configuration of this trial (this filter is stricter than mtime alone
    % and avoids picking up a parallel run if one ever slips in).
    rx = sprintf(['_dim_2D_weighted_%s_nodefeats_%s_model_%s_.*_epochs_%d_' ...
                  'hiddenChannels_\\d+_numLayers_\\d+_dropout_[\\d\\.e\\-]+' ...
                  '_ls_[\\d\\.e\\-]+_wd_[\\d\\.e\\-]+'], ...
        weighted, use_node_feats, model_name, opts.max_epochs);
    matches = ~cellfun(@isempty, regexp({dd.name}', rx, 'once'));
    dd = dd(matches);
    if isempty(dd)
        objective = nan;
        return;
    end
    dd = dd([dd.datenum] >= t_before);
    if isempty(dd)
        % Fall back to the most recent match if none post-date t_before
        % (can happen on clock jitter over NFS).
        dd = dir(logs_dir);
        dd = dd([dd.isdir] & ~ismember({dd.name}, {'.', '..'}));
        matches = ~cellfun(@isempty, regexp({dd.name}', rx, 'once'));
        dd = dd(matches);
    end
    [~, idx] = max([dd.datenum]);
    run_dir = fullfile(logs_dir, dd(idx).name);

    csv_file = fullfile(run_dir, 'csv', 'val_loss.csv');
    if ~isfile(csv_file)
        objective = nan;
        return;
    end

    fid = fopen(csv_file, 'rt');
    raw = fread(fid, inf, '*char')';
    fclose(fid);

    tok = regexp(raw, '(\d+)\,([\d\.\e\-]+)', 'tokens')';
    if isempty(tok)
        objective = nan;
        return;
    end
    tok = [tok{:}]';
    tok = reshape(tok, 2, numel(tok)/2)';
    val_loss = cellfun(@str2double, tok);

    if opts.smooth_val_loss && size(val_loss, 1) > 60
        smoothed = smooth(medfilt1(val_loss(:,2), 5), 100);
        smoothed = smoothed(50:end-10);
        if isempty(smoothed) || all(isnan(smoothed))
            objective = min(val_loss(:,2), [], 'omitnan');
        else
            objective = min(smoothed);
        end
    else
        if size(val_loss, 1) == 0
            objective = nan;
            return;
        end
        objective = min(val_loss(:,2), [], 'omitnan');
    end
    if objective == 0 || ~isfinite(objective)
        objective = nan;
    end
end


% =====================================================================
% Helpers
% =====================================================================
function stage_symlink(src_path, work_dirname)
% stage_symlink  Implement stage symlink for this MATLAB workflow.
% Inputs: src_path, work_dirname
% Outputs: none; performs side effects or updates the caller workflow.
% Symlink a file into work_dirname, overwriting any stale link/file.
    [~, name, ext] = fileparts(src_path);
    dst = fullfile(work_dirname, [name ext]);
    system(sprintf('ln -sfn "%s" "%s"', src_path, dst));
end
