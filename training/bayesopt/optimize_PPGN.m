function results = optimize_PPGN(dataset_filename, inds_dirname, hp_ranges, n_trials, varargin)
% optimize_PPGN  Implement optimize ppgn for this MATLAB workflow.
% Inputs: dataset_filename, inds_dirname, hp_ranges, n_trials, varargin
% Outputs: results
%OPTIMIZE_PPGN  Bayesian hyper-parameter optimization for PPGN.
%
%   RESULTS = OPTIMIZE_PPGN(DATASET_FILENAME, INDS_DIRNAME, HP_RANGES,
%   N_TRIALS) runs Bayesian optimization (MATLAB's BAYESOPT) over the
%   hyper-parameters of the deepcellgraph/0.1 `dcg train` CLI. The
%   objective minimized is the smoothed minimum of the validation loss
%   curve that `dcg train` writes to <out-dir>/metrics.csv for each trial.
%
%   Required inputs
%   ---------------
%   DATASET_FILENAME : char/string. Full path to a PPGN-format training
%                      file produced by generate_subsets(..., add_node_features=1).
%                      The regime is inferred from the parent directory
%                      ('Training set lengths_to_lengths' -> weighted W,
%                      'Training set none_to_lengths' -> unweighted UW).
%   INDS_DIRNAME     : char/string. Folder containing the three split
%                      files (train.inds, val.inds, test.inds).
%   HP_RANGES        : struct. Each field specifies one PPGN hyper-
%                      parameter to sweep. Field names must match the
%                      keys that appear in the `--args` string that
%                      generate_training_scripts_PPGN builds, e.g.
%                        hp_ranges.learning_rate    = [1e-5, 1e-2];
%                        hp_ranges.batch_size       = {'2','4','8','16','32'};
%                        hp_ranges.factor           = {'0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8'};
%                        hp_ranges.gradient_clipping= {'0.001','0.01','0.1','1'};
%                      Categorical: cell array of chars.
%                      Real:        2-element increasing numeric vector.
%                                   Log-scale is auto-applied to
%                                   'learning_rate'.
%   N_TRIALS         : positive integer. Total BAYESOPT evaluations.
%
%   Optional name/value pairs
%   -------------------------
%     'cuda'           : GPU id (default 0; -1 = CPU). Not a `dcg train`
%                        flag -- exported as CUDA_VISIBLE_DEVICES before
%                        each trial so PyTorch picks the right device.
%     'max_epochs'     : --args epochs=<val> (default 120 for BO).
%     'patience'       : fixed --args patience (default 20).
%     'early_stop'     : fixed --args early_stop (default 40).
%     'threshold'      : fixed --args threshold (default 1e-4).
%     'laplacian_k'    : number of Laplacian PE columns baked into the
%                        feature strings (default 30 -- matches the paper).
%     'block_features' : fixed PPGN block widths (default '[400,400,400]').
%     'num_seed_points': BAYESOPT NumSeedPoints (default 6).
%     'acquisition_fn' : BAYESOPT AcquisitionFunctionName
%                        (default 'expected-improvement-plus').
%     'output_dirname' : Where BO results + per-trial out-dirs live.
%                        Default: <dataset parent>/Bayesian_optimization_results/
%     'skip_existing'  : If true and the result .mat already exists,
%                        skip the run and load/return (default false).
%     'smooth_val_loss': If true (default), each trial's objective is the
%                        min of the smoothed val_loss curve (medfilt1 w=5,
%                        moving-average span 100, trimmed to [50:end-10]),
%                        and trials with <=60 logged epochs return NaN.
%                        If false, skip both: objective = min(val_loss)
%                        over the whole curve, and no minimum-length gate.
%                        Use false for smoke tests where you only want to
%                        confirm each trial runs, not how well it trains.
%     'keep_trial_dirs': If true, keep every trial's `dcg train --out-dir`
%                        (default false -- only the final BayesianOptimization
%                        object and each trial's metrics.csv are retained).
%     'module_prefix'  : Shell prefix that brings `dcg` onto PATH. Default
%                        activates the environment that provides the `dcg` command.
%
%   Output
%   ------
%   RESULTS : the BayesianOptimization object. Also persisted to disk as
%             a .mat file in output_dirname.
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
%   PPGN trial outputs are staged under:
%       <output_dirname>/trials_PPGN_<weighted>_<dataset>/
%   On restart, RUN_OR_RESUME_BAYESOPT loads the partial checkpoint and
%   re-seeds a fresh BAYESOPT call from completed finite observations.
%   This preserves finished trials after a wall-time kill or crash while
%   avoiding MATLAB resume() closures that may contain stale runtime
%   options. A crash during an individual dcg train run can still lose
%   that one in-progress trial, because BAYESOPT checkpoints only after
%   the objective function returns.
%
%   How it works
%   ------------
%   * BAYESOPT samples HPs according to HP_RANGES. Numeric cell arrays
%     such as batch_size and factor are represented as ordinal integer
%     variables, so the Gaussian-process surrogate treats neighboring
%     numeric choices as nearby rather than unrelated categories.
%   * Each trial builds a `dcg train` command with fixed architecture
%     arguments, the sampled optimization arguments, the requested split,
%     and the configured CUDA device/environment prefix.
%   * The objective reads the trial's metrics.csv, extracts the validation
%     loss curve, and returns the minimum of the smoothed validation loss
%     curve by default. Failed or degenerate trials return NaN and are not
%     used as warm-start observations when resuming.
%   * If n_parallel >= 2, a local MATLAB parpool is used and BAYESOPT
%     dispatches multiple trials concurrently. Parallel batches are less
%     sequentially adaptive than fully serial BO but can reduce wall time.
%
%   See also OPTIMIZE_MPNN, RUN_OR_RESUME_BAYESOPT, BAYESOPT.

    %% ---- parse inputs --------------------------------------------------
    p = inputParser;
    p.FunctionName = 'optimize_PPGN';
    addRequired(p, 'dataset_filename', @(x) (ischar(x) || isstring(x)) && isfile(char(x)));
    addRequired(p, 'inds_dirname',     @(x) (ischar(x) || isstring(x)) && isfolder(char(x)));
    addRequired(p, 'hp_ranges',        @isstruct);
    addRequired(p, 'n_trials',         @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'cuda',            0);
    % email_on_iter: empty string (default) = no email. Otherwise an
    % address and bayesopt's OutputFcn list
    % gets an extra hook that mails an iteration summary after every
    % completed iter (per-trial sequentially; per-parallel-batch under
    % UseParallel). Body has obj, best-so-far, runtime, current HPs,
    % best HPs. Send is best-effort and won't stop the BO on failure.
    addParameter(p, 'email_on_iter',   '');
    % n_parallel: bayesopt UseParallel knob.
    %   0 or 1 -> sequential (current default)
    %   N >= 2 -> ensure a local parpool of N workers is up, then run
    %            bayesopt with UseParallel=true so up to N trials evaluate
    %            concurrently. Per-trial out-dirs already include
    %            millisecond-precision timestamps + worker pid, so name
    %            collisions are not a concern. All workers inherit
    %            CUDA_VISIBLE_DEVICES from this MATLAB process and share
    %            the single GPU; PPGN trials use ~1-2 GB VRAM each so 3
    %            co-residents fit comfortably on an 80 GB A100. Caveat:
    %            bayesopt updates its surrogate after each batch of
    %            parallel trials returns (not after each one individually),
    %            so the 2nd and 3rd dispatches in a batch are slightly
    %            less surrogate-informed -- a small sample-efficiency hit
    %            offset by the wall-clock speedup.
    addParameter(p, 'n_parallel',      1);
    addParameter(p, 'max_epochs',      120);
    addParameter(p, 'patience',        20);
    addParameter(p, 'early_stop',      40);
    addParameter(p, 'threshold',       1e-4);
    addParameter(p, 'laplacian_k',     30);
    addParameter(p, 'block_features',  '[400,400,400]');
    addParameter(p, 'depth_of_mlp',    2);   % fixed in paper (Table S1); not optimised
    addParameter(p, 'disable_first_skip', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'num_seed_points', 6);
    addParameter(p, 'acquisition_fn',  'expected-improvement-plus');
    addParameter(p, 'output_dirname',  '');
    addParameter(p, 'skip_existing',   false);
    addParameter(p, 'smooth_val_loss', true);
    addParameter(p, 'keep_trial_dirs', false);
    % BO objective:
    %   'val_loss'         - default; smoothed min val_loss (legacy behaviour)
    %   'val_macro_f1'     - maximise mean over targets of val_f1_<target>
    %                        (F1 at threshold logit>0); requires the patched
    %                        dcg trainer.
    %   'val_min_f1'       - same, but min over targets (weakest-link).
    %   'val_min_f1_tuned' - same min-over-targets, but each epoch's per-
    %                        target val F1 is computed at the threshold
    %                        that maximises THAT target's F1 on val. The
    %                        operating point is logged per epoch as
    %                        val_thr_tuned_<target>; on the BO winner,
    %                        downstream test eval should apply that
    %                        threshold (NEVER tune on test). Right metric
    %                        when high pos_weight shifts the natural cut
    %                        far from logit=0 -- ranking can be excellent
    %                        while threshold-zero F1 underrates the model.
    default_module_prefix = getenv('DCG_PPGN_MODULE_PREFIX');
    if isempty(default_module_prefix)
        default_module_prefix = ['export TERM=xterm; ', ...
            'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; '];
    end

    addParameter(p, 'bo_objective',    'val_loss');
    % prior_log: path to a previous bayesopt .log file whose iteration
    % table should seed this run as InitialX/InitialObjective. Use this
    % to continue a BO session whose .partial.mat was never saved (e.g.
    % the pre-fix invocation that lacked 'OutputFcn', @saveToFile). The
    % source file is copied up-front because diary restart would delete
    % it otherwise.
    addParameter(p, 'prior_log',       '');
    % Shell prefix for the `dcg train` subprocess. Use this to activate a
    % virtual environment, load modules, or set CUDA/CPU-thread limits.
    % By default only conservative CPU-thread caps are exported; set
    % DCG_PPGN_MODULE_PREFIX or pass 'module_prefix' for cluster-specific
    % activation commands.
    addParameter(p, 'module_prefix', default_module_prefix, @(x) ischar(x) || isstring(x));
    parse(p, dataset_filename, inds_dirname, hp_ranges, n_trials, varargin{:});
    opts = p.Results;

    dataset_filename = char(opts.dataset_filename);
    inds_dirname     = char(opts.inds_dirname);
    hp_ranges        = opts.hp_ranges;
    n_trials         = opts.n_trials;

    % Snapshot the prior_log immediately -- later in this function the
    % diary re-initialisation deletes the existing log_filename, which
    % in a restart is the very file we want to parse.
    prior_log_src = char(opts.prior_log);
    prior_log_snapshot = '';
    if ~isempty(prior_log_src)
        if ~isfile(prior_log_src)
            error('optimize_PPGN:priorLogMissing', ...
                'prior_log file not found: %s', prior_log_src);
        end
        prior_log_snapshot = [tempname() '.log'];
        copyfile(prior_log_src, prior_log_snapshot);
    end

    %% ---- infer regime (W/UW) from the parent-dir name ------------------
    [ds_parent, base, ext] = fileparts(dataset_filename);
    if ~strcmp(ext, '.txt')
        error('optimize_PPGN:badFile', ...
            'Dataset file must have a .txt extension (got "%s").', [base ext]);
    end
    [~, parent_name] = fileparts(ds_parent);
    if strcmp(parent_name, 'Training set lengths_to_lengths')
        weighted = 'W';
    elseif strcmp(parent_name, 'Training set none_to_lengths')
        weighted = 'UW';
    else
        error('optimize_PPGN:regime', ...
            ['Dataset parent dir must be "Training set lengths_to_lengths" or ' ...
             '"Training set none_to_lengths" (got "%s"). The regime is inferred ' ...
             'from this name.'], parent_name);
    end

    %% ---- feature strings (mirror get_DCG_parameters, add_node_features=1)
    eigs_str = strjoin(arrayfun(@(k) sprintf('eig%d', k), 1:opts.laplacian_k, ...
        'UniformOutput', false), ',');
    node_feats = ['degree,n_min_degree,n_max_degree,n_mean_degree,n_sd_degree,' eigs_str];
    node_feats_core = 'degree,n_min_degree,n_max_degree,n_mean_degree,n_sd_degree';

    if strcmp(weighted, 'W')
        input_features = ['[in_preferred_length,in_was_flipped,' node_feats ']'];
        normalize_str  = ['[in_preferred_length,out_preferred_length,' node_feats_core ']'];
    else
        input_features = ['[' node_feats ']'];
        normalize_str  = ['[out_preferred_length,' node_feats_core ']'];
    end
    target_features = '[out_preferred_length]';

    %% ---- output dirs ---------------------------------------------------
    output_dirname = char(opts.output_dirname);
    if isempty(output_dirname)
        output_dirname = fullfile(ds_parent, 'Bayesian_optimization_results');
    end
    if ~isfolder(output_dirname)
        mkdir(output_dirname);
    end

    result_filename = fullfile(output_dirname, ...
        sprintf('PPGN_2D_weighted_%s_%s.mat', weighted, base));
    if opts.skip_existing && isfile(result_filename)
        fprintf('[optimize_PPGN] %s already exists; skip_existing=true -> loading and returning.\n', ...
            result_filename);
        S = load(result_filename, 'results');
        results = S.results;
        return;
    end

    % Per-trial out-dirs live under this folder.
    trials_root = fullfile(output_dirname, ...
        sprintf('trials_PPGN_%s_%s', weighted, base));
    if ~isfolder(trials_root)
        mkdir(trials_root);
    end

    %% ---- optimizableVariables ------------------------------------------
    % Numeric cell arrays are modelled as ORDINAL integer-indexed lookups
    % (not categoricals), so the GP surrogate sees neighbouring grid
    % points as close in HP space -- e.g. batch_size 4 vs 8 is a small
    % step, not an unrelated label switch.
    %
    % HPs that collapse to a single value (e.g. batch_size when the
    % VRAM clamp leaves only one grid point) are NOT optimised -- bayesopt's
    % optimizableVariable rejects degenerate ranges. They get folded into
    % fixed_args below instead.
    fn_in = fieldnames(hp_ranges);
    optVars = {};
    fn = {};                  % fields actually swept by BO
    frozen = struct();        % single-value HPs to fold into fixed_args
    ordinal_map = struct();   % ordinal_map.(name) = sorted cell of strings
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
                optVars{end+1} = optimizableVariable(name, ...
                    [1, numel(sorted_cats)], 'Type', 'integer'); %#ok<AGROW>
            else
                % Genuinely unordered strings -> stay categorical.
                optVars{end+1} = optimizableVariable(name, cats, 'Type', 'categorical'); %#ok<AGROW>
            end
            fn{end+1} = name; %#ok<AGROW>
        elseif isnumeric(v) && numel(v) == 2 && v(2) > v(1)
            if any(strcmpi(name, {'learning_rate', 'lr'}))
                optVars{end+1} = optimizableVariable(name, v(:)', 'Type', 'real', 'Transform', 'log'); %#ok<AGROW>
            else
                optVars{end+1} = optimizableVariable(name, v(:)', 'Type', 'real'); %#ok<AGROW>
            end
            fn{end+1} = name; %#ok<AGROW>
        elseif isnumeric(v) && isscalar(v)
            frozen.(name) = num2str(v, '%g');
        else
            error('optimize_PPGN:badHP', ...
                ['hp_ranges.%s must be a cell array of strings (categorical) ' ...
                 'or a 2-element increasing numeric vector [lo hi] (real).'], name);
        end
    end
    optVars = [optVars{:}];

    %% ---- run BAYESOPT --------------------------------------------------
    % Fixed args shared by every trial. Swept HPs are merged in eval_trial.
    fixed_args = struct();
    fixed_args.patience          = num2str(opts.patience);
    fixed_args.early_stop        = num2str(opts.early_stop);
    fixed_args.threshold         = num2str(opts.threshold, '%g');
    fixed_args.epochs            = num2str(opts.max_epochs);
    fixed_args.block_features    = opts.block_features;
    fixed_args.depth_of_mlp      = num2str(opts.depth_of_mlp);
    fixed_args.disable_first_skip = matlab_bool(opts.disable_first_skip);
    fixed_args.input_features    = input_features;
    fixed_args.target_features   = target_features;
    fixed_args.normalize         = normalize_str;

    % Fold single-value (collapsed) HPs from hp_ranges into the fixed args.
    frozen_names = fieldnames(frozen);
    for k = 1:numel(frozen_names)
        fixed_args.(frozen_names{k}) = frozen.(frozen_names{k});
    end

    % Capture bayesopt's iteration table into a .log next to the .mat
    % so the saturation / convergence of each HP can be reviewed later.
    log_filename = regexprep(result_filename, '\.mat$', '.log');
    if isfile(log_filename)
        delete(log_filename);
    end
    diary(log_filename);
    diary_guard = onCleanup(@() diary('off'));

    fprintf('=== optimize_PPGN ===\n');
    fprintf('timestamp       : %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS')); %#ok<DATST>
    fprintf('weighted        : %s\n', weighted);
    fprintf('dataset_file    : %s\n', dataset_filename);
    fprintf('inds_dir        : %s\n', inds_dirname);
    fprintf('trials_root     : %s\n', trials_root);
    fprintf('n_trials        : %d\n', n_trials);
    fprintf('num_seed_points : %d\n', opts.num_seed_points);
    fprintf('max_epochs      : %d\n', opts.max_epochs);
    fprintf('patience        : %d\n', opts.patience);
    fprintf('early_stop      : %d\n', opts.early_stop);
    fprintf('disable_first_skip: %d\n', opts.disable_first_skip);
    fprintf('cuda            : %d\n', opts.cuda);
    fprintf('acquisition_fn  : %s\n', char(opts.acquisition_fn));
    fprintf('hp_ranges       :\n');
    disp(hp_ranges);
    fprintf('fixed_args      :\n');
    disp(fixed_args);
    fprintf('---\n');

    fun = @(x) eval_trial(x, fn, ordinal_map, fixed_args, dataset_filename, inds_dirname, ...
                          trials_root, opts);

    % Per-iteration checkpoint: bayesopt writes the BO object to this file
    % after every trial. If the run is killed mid-sweep, restarting
    % reloads it and calls resume() to continue from where it left off.
    % SaveFileName alone is a silent no-op -- saving requires @saveToFile
    % in OutputFcn.
    partial_filename = regexprep(result_filename, '\.mat$', '.partial.mat');

    % Spin up a parpool when the caller requested N>=2 workers. Reuse an
    % existing pool if its size matches; otherwise tear it down and start
    % fresh so we don't end up with a stale pool from a previous run.
    use_parallel = (opts.n_parallel >= 2);
    if use_parallel
        existing = gcp('nocreate');
        if isempty(existing) || existing.NumWorkers ~= opts.n_parallel
            if ~isempty(existing)
                delete(existing);
            end
            parpool('local', opts.n_parallel);
        end
        fprintf('[optimize_PPGN] parallel BO: %d workers (UseParallel=true)\n', ...
            opts.n_parallel);
    end

    % OutputFcn list: always include @saveToFile (per-iter checkpoint).
    % Layer the email hook on top when an address was supplied. Run
    % label = .mat basename so the inbox subject line tells multiple
    % concurrent BO runs apart at a glance.
    output_fcns = {@saveToFile};
    email_addr  = char(opts.email_on_iter);
    if ~isempty(email_addr)
        [~, run_label] = fileparts(result_filename);
        output_fcns{end+1} = @(BO, State) email_bayesopt_outputfcn( ...
            BO, State, email_addr, run_label);
        fprintf('[optimize_PPGN] email_on_iter: per-iter summary -> %s\n', email_addr);
    end

    bo_args = { ...
        'MaxObjectiveEvaluations',  n_trials, ...
        'AcquisitionFunctionName',  char(opts.acquisition_fn), ...
        'NumSeedPoints',            opts.num_seed_points, ...
        'IsObjectiveDeterministic', false, ...
        'Verbose',                  1, ...
        'UseParallel',              use_parallel, ...
        'PlotFcn',                  [], ...
        'OutputFcn',                output_fcns, ...
        'SaveFileName',             partial_filename};

    if ~isempty(prior_log_snapshot)
        [prior_X, prior_obj, prior_time] = parse_bayesopt_log( ...
            prior_log_snapshot, fn, ordinal_map);
        fprintf('[optimize_PPGN] prior_log: seeding bayesopt with %d prior trials from %s\n', ...
            height(prior_X), prior_log_src);
        prior_display = prior_X;
        prior_display.objective   = prior_obj;
        prior_display.runtime_sec = prior_time;
        disp(prior_display);
        delete(prior_log_snapshot);
        % Clamp NumSeedPoints to height(prior_X) so bayesopt doesn't
        % sample additional random seeds on top of the ones we already
        % have -- the whole seed budget is the prior table.
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
         'inds_dirname', 'n_trials', 'weighted', 'fixed_args', 'log_filename', ...
         'ordinal_map');
    fprintf('[optimize_PPGN] Done. Results saved to %s\n', result_filename);
    fprintf('[optimize_PPGN] Log saved to %s\n', log_filename);
    close all;
end


% =====================================================================
% Per-trial objective: run `dcg train` once and return smoothed val loss.
% =====================================================================
function objective = eval_trial(x, hp_field_names, ordinal_map, fixed_args, ...
                                dataset_filename, inds_dirname, trials_root, opts)
% eval_trial  Implement eval trial for this MATLAB workflow.
% Inputs: x, hp_field_names, ordinal_map, fixed_args, dataset_filename, inds_dirname, trials_root, opts
% Outputs: objective

    % Build the --args string. Start from the fixed args, then overlay
    % any hp sampled by BAYESOPT (so a swept HP overrides the fixed one
    % if they ever collide).
    args_struct = fixed_args;
    for k = 1:numel(hp_field_names)
        name = hp_field_names{k};
        val  = x.(name);
        if isfield(ordinal_map, name)
            % Ordinal HP: BO sampled an integer index; look up the
            % corresponding string value from the sorted list.
            args_struct.(name) = ordinal_map.(name){double(val)};
        elseif iscategorical(val)
            args_struct.(name) = char(val);
        elseif any(strcmpi(name, {'learning_rate', 'lr'}))
            args_struct.(name) = num2str(val, '%.8g');
        else
            args_struct.(name) = num2str(val);
        end
    end

    kv = fieldnames(args_struct);
    parts = cell(1, numel(kv));
    for k = 1:numel(kv)
        parts{k} = sprintf('%s=%s', kv{k}, args_struct.(kv{k}));
    end
    args_string = strjoin(parts, ';');

    % Unique per-trial out-dir. Millisecond precision + pid covers
    % simultaneous trials on parallel workers.
    stamp = datestr(now, 'yyyymmdd_HHMMSS_FFF'); %#ok<DATST>
    out_dir = fullfile(trials_root, sprintf('trial_%s_pid%d', stamp, feature('getpid')));
    if ~isfolder(out_dir)
        mkdir(out_dir);
    end

    com = ['dcg train --training-data "', dataset_filename, ...
           '" --out-dir "', out_dir, '"', ...
           ' --no-evaluation ', ...
           ' --args "', args_string, '"', ...
           ' --inds-dir "', inds_dirname, '"'];

    % CUDA_VISIBLE_DEVICES picks the GPU for torch inside dcg. opts.cuda<0
    % pins to CPU.
    if opts.cuda >= 0
        cuda_prefix = sprintf('export CUDA_VISIBLE_DEVICES=%d; ', opts.cuda);
    else
        cuda_prefix = 'export CUDA_VISIBLE_DEVICES=; ';
    end

    % Two ways to interpret module_prefix (mirrors optimize_MPNN):
    %  * default: shell-env prefix that brings `dcg` onto PATH (HPC venv).
    %  * '@cmdenv:<cmd>': place the trainer invocation in env NANO_TRAINER_CMD
    %    and run <cmd> alone. Used by runpod_ppgn_trainer.sh: it reads
    %    NANO_TRAINER_CMD, rewrites --training-data/--inds-dir/--out-dir to
    %    pod paths, SSHs to the pod, and scp's the trial outputs back.
    prefix_str = char(opts.module_prefix);
    if startsWith(prefix_str, '@cmdenv:')
        setenv('NANO_TRAINER_CMD', [cuda_prefix, com]);
        curr_command = [extractAfter(prefix_str, '@cmdenv:'), ' 2>&1'];
    else
        curr_command = [cuda_prefix, prefix_str, com];
    end

    % --- Live-tail support ------------------------------------------------
    % Redirect dcg's stdout/stderr to a per-trial log file, and update a
    % "current.log" symlink in the trials_root so a separate terminal can
    % `tail -F <trials_root>/current.log` to watch epochs of the live trial.
    % MATLAB still reads metrics.csv afterward for the BO objective; the
    % captured stdout was already discarded (`~`), so silencing it here
    % doesn't lose any information BO needs.
    log_file = fullfile(out_dir, 'training.log');
    [~, ~] = system(sprintf('ln -sfn "%s" "%s"', ...
                            log_file, fullfile(trials_root, 'current.log')));
    curr_command = [curr_command, ' > "', log_file, '" 2>&1'];

    [ret, ~] = system(curr_command);
    objective = nan;
    if ret ~= 0
        maybe_cleanup(out_dir, opts.keep_trial_dirs);
        return;
    end

    metrics_file = fullfile(out_dir, 'metrics.csv');
    if ~isfile(metrics_file)
        maybe_cleanup(out_dir, opts.keep_trial_dirs);
        return;
    end

    % dcg train writes a CSV with header `epochs,train_loss,val_loss,...`.
    T = readtable(metrics_file);
    if ~ismember('val_loss', T.Properties.VariableNames) || height(T) < 5
        maybe_cleanup(out_dir, opts.keep_trial_dirs);
        return;
    end
    if any(strcmpi(opts.bo_objective, {'val_macro_f1', 'val_min_f1', 'val_min_f1_tuned'}))
        % Aggregate per-target val F1 columns per epoch, then take max
        % across epochs. bayesopt minimises, so return 1 - max(agg_f1).
        % NaN for trials missing the relevant F1 columns (older runs,
        % non-BCE losses, or unpatched dcg).
        %
        % Column source depends on the objective:
        %   'val_min_f1' / 'val_macro_f1' -> val_f1_<target>      (threshold logit>0)
        %   'val_min_f1_tuned'            -> val_f1_tuned_<target> (per-epoch
        %                                    sweep: F1 at the operating point
        %                                    that maxes each target's val F1)
        if strcmpi(opts.bo_objective, 'val_min_f1_tuned')
            col_prefix = 'val_f1_tuned_';
        else
            col_prefix = 'val_f1_';
        end
        % Avoid prefix collision: 'val_f1_' would also match
        % 'val_f1_tuned_*', so filter those out for the untuned path.
        all_cols = T.Properties.VariableNames;
        f1_cols = all_cols(startsWith(all_cols, col_prefix));
        if strcmpi(col_prefix, 'val_f1_')
            f1_cols = f1_cols(~startsWith(f1_cols, 'val_f1_tuned_'));
        end
        if isempty(f1_cols)
            warn_fmt = ['bo_objective=%s but no %s columns in %s. ', ...
                        'Falling back to NaN -- likely the trial used the ', ...
                        'unpatched dcg or a non-BCE loss.'];
            warning('optimize_PPGN:noF1', warn_fmt, ...
                opts.bo_objective, [col_prefix '*'], metrics_file);
            maybe_cleanup(out_dir, opts.keep_trial_dirs);
            return;
        end
        f1_mat = table2array(T(:, f1_cols));
        if any(strcmpi(opts.bo_objective, {'val_min_f1', 'val_min_f1_tuned'}))
            agg_f1 = min(f1_mat, [], 2);     % weakest-link across targets
        else
            agg_f1 = mean(f1_mat, 2, 'omitnan');
        end
        if all(isnan(agg_f1))
            maybe_cleanup(out_dir, opts.keep_trial_dirs);
            return;
        end
        best_agg = max(agg_f1, [], 'omitnan');
        if ~isfinite(best_agg)
            maybe_cleanup(out_dir, opts.keep_trial_dirs);
            return;
        end
        objective = 1 - best_agg;
        maybe_cleanup(out_dir, opts.keep_trial_dirs);
        return;
    end

    val_loss = T.val_loss;

    % Smoothing requires ~60 epochs of data; fall back to raw min for shorter
    % runs (e.g. those that early-stop fast under low patience/early_stop).
    if opts.smooth_val_loss && height(T) >= 60
        smoothed = smooth(medfilt1(val_loss, 5), 100);
        smoothed = smoothed(50:end-10);
        if isempty(smoothed)
            maybe_cleanup(out_dir, opts.keep_trial_dirs);
            return;
        end
        obj = min(smoothed);
    else
        obj = min(val_loss, [], 'omitnan');
    end
    if obj == 0 || ~isfinite(obj)
        maybe_cleanup(out_dir, opts.keep_trial_dirs);
        return;
    end
    objective = obj;

    maybe_cleanup(out_dir, opts.keep_trial_dirs);
end


function maybe_cleanup(out_dir, keep)
% maybe_cleanup  Implement maybe cleanup for this MATLAB workflow.
% Inputs: out_dir, keep
% Outputs: none; performs side effects or updates the caller workflow.
% Delete big artefacts (model.tar etc.), keep metrics.csv for forensics.
    if keep || ~isfolder(out_dir)
        return;
    end
    big = {'model.tar', 'predicted_graph.txt', 'evaluation.html', 'evaluation.ipynb'};
    for k = 1:numel(big)
        f = fullfile(out_dir, big{k});
        if isfile(f)
            delete(f);
        end
    end
end


function s = matlab_bool(x)
% matlab_bool  Convert a MATLAB logical scalar to PPGN's CLI boolean spelling.
    if x
        s = 'true';
    else
        s = 'false';
    end
end
