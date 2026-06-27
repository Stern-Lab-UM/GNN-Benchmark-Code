function results = run_or_resume_bayesopt(fun, optVars, partial_filename, n_trials, bo_args, initial_x)
% run_or_resume_bayesopt  Implement run or resume bayesopt for this MATLAB workflow.
% Inputs: fun, optVars, partial_filename, n_trials, bo_args, initial_x
% Outputs: results
%RUN_OR_RESUME_BAYESOPT  Fresh bayesopt, or resume from a partial checkpoint.
%
%   If PARTIAL_FILENAME exists and contains a BayesianOptimization object
%   (written by bayesopt's SaveFileName each iteration), load it and:
%     - if it already has >= N_TRIALS evaluations, return it as final,
%     - else re-seed a fresh bayesopt via InitialX/InitialObjective from
%       the stored observations so the CURRENT ObjectiveFcn (FUN) is used.
%   Otherwise run bayesopt fresh; if INITIAL_X is a non-empty table, it
%   is used as the warm-start InitialX for that fresh run.
%
%   Why not resume()' resume() reuses the saved bo's stored ObjectiveFcn
%   closure, which captured the options (e.g. cuda GPU id) from the
%   session that created the checkpoint. Starting fresh with the prior
%   observations as InitialX/InitialObjective lets the caller's current
%   FUN (and thus current opts) drive any new evaluations.
%
%   BO_ARGS is the full name/value list to pass (includes SaveFileName).
%   INITIAL_X is optional (default: empty table).
%
%   Checkpoint contents
%   -------------------
%   MATLAB's @saveToFile callback stores the object as variable
%   BayesoptResults inside PARTIAL_FILENAME. The helper uses XTrace,
%   ObjectiveTrace, and ObjectiveEvaluationTimeTrace from that object.
%   Failed trials with non-finite objectives are intentionally omitted
%   from restart InitialObjective entries, because BAYESOPT requires
%   finite warm-start objective values.
%
%   Restart semantics
%   -----------------
%   The helper does not itself write the final .mat file; callers save
%   the returned BayesianOptimization object after BAYESOPT completes.
%   Restarting from a partial file is safe across changes to run-time
%   execution options because only the observations are reused, not the
%   old objective function closure. INITIAL_X is for deliberate manual
%   warm starts; those rows consume part of NumSeedPoints so the total
%   number of non-GP-guided seed evaluations remains the requested value.

if nargin < 6; initial_x = table(); end

if isfile(partial_filename)
    try
        S = load(partial_filename, 'BayesoptResults');
        if isfield(S, 'BayesoptResults') && isa(S.BayesoptResults, 'BayesianOptimization')
            bo = S.BayesoptResults;
            n_done = numel(bo.ObjectiveTrace);
            if n_done >= n_trials
                fprintf(['[run_or_resume_bayesopt] %s already has %d/%d ' ...
                         'trials; using as final.\n'], partial_filename, n_done, n_trials);
                results = bo;
                return;
            end
            % Drop failed trials (NaN objective) - bayesopt rejects them
            % as InitialObjective entries.
            prior_X    = bo.XTrace;
            prior_obj  = bo.ObjectiveTrace;
            prior_time = bo.ObjectiveEvaluationTimeTrace;
            keep       = isfinite(prior_obj);
            prior_X    = prior_X(keep, :);
            prior_obj  = prior_obj(keep);
            prior_time = prior_time(keep);

            fprintf(['[run_or_resume_bayesopt] re-seeding fresh bayesopt ' ...
                     'from %d/%d prior trials in %s\n'], ...
                height(prior_X), n_trials, partial_filename);
            ix = find(strcmpi(bo_args(1:2:end), 'NumSeedPoints'), 1);
            if ~isempty(ix)
                bo_args{2*ix} = max(height(prior_X), 1);
            end
            bo_args = [bo_args, {'InitialX',                        prior_X, ...
                                 'InitialObjective',                prior_obj, ...
                                 'InitialObjectiveEvaluationTimes', prior_time}];
            results = bayesopt(fun, optVars, bo_args{:});
            return;
        end
        warning('run_or_resume_bayesopt:badFile', ...
            'Partial file %s has no BayesoptResults; starting fresh.', partial_filename);
    catch ME
        warning('run_or_resume_bayesopt:loadFailed', ...
            'Failed to load %s (%s); starting fresh.', partial_filename, ME.message);
    end
end

if ~isempty(initial_x) && istable(initial_x) && height(initial_x) > 0
    % InitialX rows count as seed evaluations in bayesopt: the first
    % NumSeedPoints trials are non-GP-guided, and InitialX fills that
    % quota before random sampling. So decrement NumSeedPoints by
    % height(initial_x) to keep the *total* non-GP-guided trial count
    % equal to the user's requested seed budget - the anchor replaces
    % one random probe rather than adding to it. Floor at 1 to avoid
    % fitting the GP from a single point.
    n_init = height(initial_x);
    ix = find(strcmpi(bo_args(1:2:end), 'NumSeedPoints'), 1);
    if ~isempty(ix)
        vix = 2*ix;
        old_nsp = bo_args{vix};
        new_nsp = max(old_nsp - n_init, 1);
        bo_args{vix} = new_nsp;
        fprintf(['[run_or_resume_bayesopt] NumSeedPoints %d -> %d ' ...
                 '(InitialX adds %d seed row(s)).\n'], old_nsp, new_nsp, n_init);
    end
    bo_args = [bo_args, {'InitialX', initial_x}];
    fprintf('[run_or_resume_bayesopt] warm-starting bayesopt at InitialX:\n');
    disp(initial_x);
end

results = bayesopt(fun, optVars, bo_args{:});
end
