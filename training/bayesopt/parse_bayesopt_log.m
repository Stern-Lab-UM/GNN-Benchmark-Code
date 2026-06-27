function [X, objective, time] = parse_bayesopt_log(log_file, hp_names, ordinal_map)
% parse_bayesopt_log  Implement parse bayesopt log for this MATLAB workflow.
% Inputs: log_file, hp_names, ordinal_map
% Outputs: X, objective, time
%PARSE_BAYESOPT_LOG  Extract InitialX / InitialObjective from a bayesopt diary.
%
%   [X, OBJECTIVE, TIME] = PARSE_BAYESOPT_LOG(LOG_FILE, HP_NAMES, ORDINAL_MAP)
%   reads the iteration table that `bayesopt` prints to the MATLAB diary
%   (and which OPTIMIZE_PPGN / OPTIMIZE_MPNN persist to a .log file) and
%   returns a MATLAB table X of prior HP samples, a column vector
%   OBJECTIVE of the observed values, and a column vector TIME of per-
%   trial runtimes in seconds.
%
%   The parser is intentionally positional: it assumes the standard
%   bayesopt header (Iter, Eval result, Objective, Objective runtime,
%   BestSoFar observed, BestSoFar estim., <HP columns...>). Column names
%   are taken from HP_NAMES (same order as the optimizableVariable list
%   passed to bayesopt) - the header's own names can wrap across lines
%   for long HP names, so they aren't reliable.
%
%   For any HP listed in ORDINAL_MAP (struct where fields are HP names
%   that were modelled as ordinal integer indices over a sorted grid),
%   the returned X column is INT64 so bayesopt's integer-type
%   optimizableVariable will accept it. Other HPs are DOUBLE.
%
%   Error/NaN trials are dropped (can't seed the GP with no objective).
%
%   Intended use: seed a fresh bayesopt with InitialX/InitialObjective
%   when the run was interrupted and no .partial.mat checkpoint exists
%   (which happens if bayesopt was invoked without 'OutputFcn',
%   @saveToFile - specifying only SaveFileName is a silent no-op).

    narginchk(2, 3);
    if nargin < 3; ordinal_map = struct(); end
    log_file = char(log_file);
    if ~isfile(log_file)
        error('parse_bayesopt_log:notFound', 'Log file not found: %s', log_file);
    end

    text  = fileread(log_file);
    lines = strsplit(text, '\n');

    % Data rows start with '|', then an integer (iter number), then '|'.
    % That distinguishes them from header rows ('| Iter', '|      ') and
    % separator rows ('|====...').
    is_data = ~cellfun('isempty', regexp(lines, '^\|\s*\d+\s*\|', 'once'));
    data_lines = lines(is_data);
    if isempty(data_lines)
        error('parse_bayesopt_log:empty', 'No iteration rows parsed from %s', log_file);
    end

    n_hp    = numel(hp_names);
    n_rows  = numel(data_lines);
    X_raw   = nan(n_rows, n_hp);
    obj_raw = nan(n_rows, 1);
    time_raw = nan(n_rows, 1);

    for i = 1:n_rows
        parts = strtrim(strsplit(data_lines{i}, '|'));
        % Leading/trailing '|' produce empty first/last cells - drop them.
        data = parts(2:end-1);
        if numel(data) ~= 6 + n_hp
            warning('parse_bayesopt_log:badRow', ...
                'Row %d has %d fields, expected %d. Skipping.', i, numel(data), 6 + n_hp);
            continue;
        end
        obj_raw(i)  = str2double(data{3});
        time_raw(i) = str2double(data{4});
        for k = 1:n_hp
            X_raw(i, k) = str2double(data{6 + k});
        end
    end

    % Drop rows with non-finite objective (error trials can't seed the GP
    % - bayesopt would refuse them as InitialObjective entries anyway).
    keep     = isfinite(obj_raw);
    X_raw    = X_raw(keep, :);
    obj_raw  = obj_raw(keep);
    time_raw = time_raw(keep);

    cols = cell(1, n_hp);
    for k = 1:n_hp
        nm = hp_names{k};
        if isfield(ordinal_map, nm)
            cols{k} = int64(round(X_raw(:, k)));
        else
            cols{k} = double(X_raw(:, k));
        end
    end
    X = table(cols{:}, 'VariableNames', hp_names);
    objective = obj_raw;
    time = time_raw;
end
