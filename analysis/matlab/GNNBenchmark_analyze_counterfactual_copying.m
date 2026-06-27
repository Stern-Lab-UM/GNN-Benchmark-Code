function outputs = GNNBenchmark_analyze_counterfactual_copying(varargin)
%GNNBenchmark_ANALYZE_COUNTERFACTUAL_COPYING  Quantify distal input-copy behavior.
%
%   OUTPUTS = GNNBenchmark_ANALYZE_COUNTERFACTUAL_COPYING(...) compares regular
%   predictions with predictions made on counterfactually perturbed weighted
%   inputs. It is designed for the distal fallback test in which pre-T1 input
%   lengths at edge-hop distance h >= H were shifted by +/-DELTA, without
%   retraining the models or changing the post-T1 targets.
%
%   The function computes, per model, the paired prediction response
%       Delta l_hat = l_hat_perturbed - l_hat_regular
%   relative to the injected input shift
%       Delta l_in  = l_pre_perturbed - l_pre_regular,
%   and the three distal MAEs used as a copy diagnostic:
%       MAE_post_regular    = mean(abs(l_hat_regular   - l_post))
%       MAE_post_perturbed  = mean(abs(l_hat_perturbed - l_post))
%       MAE_copy_perturbed  = mean(abs(l_hat_perturbed - l_pre_perturbed))
%
%   Name/value options
%   ------------------
%   'regular_pred_root'          Root containing the unperturbed .pred.txt files.
%   'counterfactual_pred_root'   Root containing perturbed .pred.txt files.
%   'inds_dir'                   Folder containing test.inds. If empty, all
%                                graphs in each file are analyzed with a warning.
%   'output_dir'                 Destination for CSVs/figures. Default is a
%                                counterfactual_copying_outputs folder beside
%                                counterfactual_pred_root.
%   'models'                     Models to analyze.
%   'seeds'                      Seed ids to pair.
%   'regular_include_token'      Optional substring required in regular files.
%   'regular_exclude_token'      Substring excluded from regular files. Default
%                                'counterfactual'.
%   'counterfactual_include_token' Substring required in perturbed files.
%                                Default 'counterfactual'.
%   'file_glob'                  Recursive file glob (default '*.pred.txt').
%   'h_min'                      Minimum edge-hop distance analyzed (default 14).
%   'delta'                      Expected absolute perturbation (default 0.05).
%   'collapse_directions'        If true, average the two directed rows of each
%                                interface before analysis (default true).
%   'pre_col','flag_col','post_col','pred_col'
%                                Weighted prediction-file columns. Defaults are
%                                [u v pre flag post pred] = [1 2 3 4 5 6].
%   'make_figure','save_figure','close_figure'
%                                Figure controls.
%
%   This function never treats l_pre_perturbed as physical ground truth. It is
%   only the counterfactual copy target expected under perfect input copying.

p = inputParser;
p.FunctionName = 'GNNBenchmark_analyze_counterfactual_copying';
default_models = {'PPGN','GraphSAGE','GAT','GIN','PNA'};
cfg = GNNBenchmark_publication_config();
default_root = '';
if isfield(cfg, 'data_root') && ~isempty(cfg.data_root)
    default_root = cfg.data_root;
end
addParameter(p, 'regular_pred_root', default_root, @(x) ischar(x) || isstring(x));
addParameter(p, 'counterfactual_pred_root', default_root, @(x) ischar(x) || isstring(x));
addParameter(p, 'inds_dir', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'models', default_models, @(x) iscell(x) || isstring(x));
addParameter(p, 'seeds', 0:4, @(x) isnumeric(x));
addParameter(p, 'regular_include_token', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'regular_exclude_token', 'counterfactual', @(x) ischar(x) || isstring(x));
addParameter(p, 'counterfactual_include_token', 'counterfactual', @(x) ischar(x) || isstring(x));
addParameter(p, 'file_glob', '*.pred.txt', @(x) ischar(x) || isstring(x));
addParameter(p, 'h_min', 14, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'delta', 0.05, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'collapse_directions', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'pre_col', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'flag_col', 4, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'post_col', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'pred_col', 6, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'make_figure', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'save_figure', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'close_figure', false, @(x) islogical(x) && isscalar(x));
parse(p, varargin{:});
opts = p.Results;
opts.regular_pred_root = char(opts.regular_pred_root);
opts.counterfactual_pred_root = char(opts.counterfactual_pred_root);
opts.inds_dir = char(opts.inds_dir);
opts.output_dir = char(opts.output_dir);
opts.models = cellstr(opts.models);
opts.regular_include_token = char(opts.regular_include_token);
opts.regular_exclude_token = char(opts.regular_exclude_token);
opts.counterfactual_include_token = char(opts.counterfactual_include_token);
opts.file_glob = char(opts.file_glob);
opts.h_min = floor(opts.h_min);

if isempty(opts.regular_pred_root) || ~isfolder(opts.regular_pred_root)
    error('GNNBenchmark:missingRegularRoot', 'regular_pred_root is missing or not a folder.');
end
if isempty(opts.counterfactual_pred_root) || ~isfolder(opts.counterfactual_pred_root)
    error('GNNBenchmark:missingCounterfactualRoot', 'counterfactual_pred_root is missing or not a folder.');
end
if isempty(opts.output_dir)
    opts.output_dir = fullfile(opts.counterfactual_pred_root, 'counterfactual_copying_outputs');
end
if ~isfolder(opts.output_dir)
    mkdir(opts.output_dir);
end

test_indices = load_test_indices(opts.inds_dir);
if isempty(test_indices)
    warning('GNNBenchmark:noTestInds', 'No inds_dir/test.inds supplied; analyzing all graphs in each prediction file.');
end

regular_files = filter_prediction_files(recursive_files(opts.regular_pred_root, opts.file_glob), ...
    opts.regular_include_token, opts.regular_exclude_token);
cf_files = filter_prediction_files(recursive_files(opts.counterfactual_pred_root, opts.file_glob), ...
    opts.counterfactual_include_token, '');
pairs = pair_prediction_files(regular_files, cf_files, opts.models, opts.seeds);
if isempty(pairs)
    warning('GNNBenchmark:noPairedRecords', 'No paired records were selected for this diagnostic.');
end

acc = initialize_accumulators(opts.models);
per_graph_rows = {};
for pidx = 1:height(pairs)
    model = pairs.model{pidx};
    seed = pairs.seed(pidx);
    [reg_names, ~, ~, ~, reg_vals] = load_dataset(pairs.regular_file{pidx}, 0);
    [cf_names, ~, ~, ~, cf_vals] = load_dataset(pairs.counterfactual_file{pidx}, 0);
    graph_ids = resolve_graph_indices(reg_names, cf_names, test_indices, pairs.regular_file{pidx});
    for gi = reshape(graph_ids, 1, [])
        gname = reg_names{gi};
        cf_gi = find(strcmp(cf_names, gname), 1);
        if isempty(cf_gi), continue; end
        [R, C] = align_prediction_rows(reg_vals{gi}, cf_vals{cf_gi}, opts.collapse_directions);
        if isempty(R) || size(R,2) < opts.pred_col || size(C,2) < opts.pred_col, continue; end
        dist = edge_hops_from_t1(R, opts.pre_col, opts.flag_col);
        d_in = C(:, opts.pre_col) - R(:, opts.pre_col);
        d_hat = C(:, opts.pred_col) - R(:, opts.pred_col);
        mask = isfinite(dist) & dist >= opts.h_min & abs(d_in) > max(1e-12, opts.delta * 1e-6);
        acc.(model).n_instances = acc.(model).n_instances + 1;
        if ~any(mask)
            per_graph_rows(end+1,:) = {model, seed, gname, 0, NaN, NaN, NaN, NaN}; %#ok<AGROW>
            continue;
        end
        acc.(model).d_in = [acc.(model).d_in; d_in(mask)];
        acc.(model).d_hat = [acc.(model).d_hat; d_hat(mask)];
        acc.(model).post_regular = [acc.(model).post_regular; abs(R(mask, opts.pred_col) - R(mask, opts.post_col))];
        acc.(model).post_perturbed = [acc.(model).post_perturbed; abs(C(mask, opts.pred_col) - C(mask, opts.post_col))];
        acc.(model).copy_perturbed = [acc.(model).copy_perturbed; abs(C(mask, opts.pred_col) - C(mask, opts.pre_col))];
        per_graph_rows(end+1,:) = {model, seed, gname, nnz(mask), ...
            mean(d_in(mask)), mean(d_hat(mask)), ...
            mean(abs(d_hat(mask) - d_in(mask))), mean(abs(C(mask, opts.pred_col) - C(mask, opts.pre_col)))}; %#ok<AGROW>
    end
end

summary = summarize_accumulators(acc, opts.models);
if isempty(per_graph_rows)
    per_graph = cell2table(cell(0,8), 'VariableNames', per_graph_varnames());
else
    per_graph = cell2table(per_graph_rows, 'VariableNames', per_graph_varnames());
end

summary_csv = fullfile(opts.output_dir, 'counterfactual_copying_summary.csv');
per_graph_csv = fullfile(opts.output_dir, 'counterfactual_copying_per_graph.csv');
writetable(summary, summary_csv);
writetable(per_graph, per_graph_csv);

fig_files = {};
if opts.make_figure
    fig = plot_copying_maes(summary, opts.models);
    if opts.save_figure
        fig_base = fullfile(opts.output_dir, 'counterfactual_copying_mae_bars');
        savefig(fig, [fig_base '.fig']);
        print(fig, [fig_base '.png'], '-dpng', '-r300');
        fig_files = {[fig_base '.fig']; [fig_base '.png']};
    end
    if opts.close_figure
        close(fig);
    end
end

outputs = struct();
outputs.summary = summary;
outputs.per_graph = per_graph;
outputs.pairs = pairs;
outputs.files = struct('summary_csv', summary_csv, 'per_graph_csv', per_graph_csv, 'figures', {fig_files});

fprintf('[GNNBenchmark_analyze_counterfactual_copying] paired files: %d\n', height(pairs));
fprintf('[GNNBenchmark_analyze_counterfactual_copying] wrote %s\n', opts.output_dir);
end


function names = per_graph_varnames()
names = {'model','seed','graph_id','n_perturbed_edges','mean_delta_l_in', ...
    'mean_delta_l_hat','mean_abs_delta_mismatch','mae_copy_perturbed'};
end


function files = recursive_files(root_dir, pattern)
d = dir(fullfile(root_dir, '**', pattern));
files = arrayfun(@(x) fullfile(x.folder, x.name), d(:), 'UniformOutput', false);
end


function files = filter_prediction_files(files, include_token, exclude_token)
keep = true(size(files));
for i = 1:numel(files)
    txt = files{i};
    if ~isempty(include_token) && isempty(regexpi(txt, regexptranslate('escape', include_token), 'once'))
        keep(i) = false;
    end
    if ~isempty(exclude_token) && ~isempty(regexpi(txt, regexptranslate('escape', exclude_token), 'once'))
        keep(i) = false;
    end
end
files = files(keep);
end


function pairs = pair_prediction_files(regular_files, cf_files, models, seeds)
rows = {};
for m = 1:numel(models)
    model = models{m};
    for seed = reshape(seeds, 1, [])
        reg = matching_files(regular_files, model, seed);
        cf = matching_files(cf_files, model, seed);
        if isempty(reg) || isempty(cf)
            continue;
        end
        if numel(reg) > 1 || numel(cf) > 1
            error('GNNBenchmark:ambiguousPredictionPair', ['Ambiguous files for %s seed %d. ', ...
                'Use *_include_token options or separate roots.\nRegular:\n%s\nCounterfactual:\n%s'], ...
                model, seed, strjoin(reg, '\n'), strjoin(cf, '\n'));
        end
        rows(end+1,:) = {model, seed, reg{1}, cf{1}}; %#ok<AGROW>
    end
end
if isempty(rows)
    pairs = cell2table(cell(0,4), 'VariableNames', {'model','seed','regular_file','counterfactual_file'});
else
    pairs = cell2table(rows, 'VariableNames', {'model','seed','regular_file','counterfactual_file'});
end
end


function out = matching_files(files, model, seed)
out = {};
seed_pat = sprintf('(^|[^A-Za-z0-9])s%d([^A-Za-z0-9]|$)', seed);
model_pat = ['(^|[^A-Za-z])', regexptranslate('escape', model), '([^A-Za-z]|$)'];
for i = 1:numel(files)
    if ~isempty(regexpi(files{i}, model_pat, 'once')) && ~isempty(regexpi(files{i}, seed_pat, 'once'))
        out{end+1} = files{i}; %#ok<AGROW>
    end
end
end


function test_indices = load_test_indices(inds_dir)
test_indices = [];
if isempty(inds_dir)
    return;
end
f = fullfile(inds_dir, 'test.inds');
if ~isfile(f)
    error('GNNBenchmark:testIndsMissing', 'test.inds not found in %s.', inds_dir);
end
x = readmatrix(f, 'FileType', 'text');
x = x(:);
if isempty(x)
    return;
end
if min(x) == 0
    x = x + 1;
end
test_indices = x;
end


function graph_ids = resolve_graph_indices(reg_names, cf_names, test_indices, filename) %#ok<INUSD>
% resolve_graph_indices  Select graph ids present in both paired graph lists.
% Inputs: reg_names, cf_names, test_indices, filename
% Outputs: graph_ids
if isempty(test_indices)
    graph_ids = 1:numel(reg_names);
else
    graph_ids = reshape(test_indices(test_indices >= 1 & test_indices <= numel(reg_names)), 1, []);
end
if isempty(graph_ids)
    return;
end
shared = ismember(reg_names(graph_ids), cf_names);
graph_ids = graph_ids(shared);
end

function [R, C] = align_prediction_rows(R0, C0, collapse_directions)
if collapse_directions
    R0 = collapse_directed_rows(R0);
    C0 = collapse_directed_rows(C0);
end
[~, iR, iC] = intersect(R0(:,1:2), C0(:,1:2), 'rows');
R = R0(iR,:);
C = C0(iC,:);
[R, ord] = sortrows(R, [1 2]);
C = C(ord,:);
end


function M2 = collapse_directed_rows(M)
pairs = sort(M(:,1:2), 2);
[uniq_pairs, ~, ic] = unique(pairs, 'rows');
M2 = zeros(size(uniq_pairs,1), size(M,2));
for k = 1:size(uniq_pairs,1)
    idx = (ic == k);
    M2(k,:) = mean(M(idx,:), 1, 'omitnan');
    M2(k,1:2) = uniq_pairs(k,:);
end
end


function dist = edge_hops_from_t1(M, pre_col, flag_col)
pairs = sort(M(:,1:2), 2);
root_rows = find(abs(M(:, pre_col)) < 1e-12);
if isempty(root_rows) && flag_col <= size(M,2)
    root_rows = find(abs(M(:, flag_col)) > 0.5);
end
if isempty(root_rows)
    root_rows = 1;
end
n_edges = size(pairs, 1);
A = false(n_edges, n_edges);
for e = 1:n_edges
    shared = pairs(:,1) == pairs(e,1) | pairs(:,1) == pairs(e,2) | ...
             pairs(:,2) == pairs(e,1) | pairs(:,2) == pairs(e,2);
    A(e, shared) = true;
end
A(1:n_edges+1:end) = false;
G = graph(A | A');
D = distances(G, root_rows(:));
if size(D,1) > 1
    dist = min(D, [], 1)';
else
    dist = D(:);
end
end


function acc = initialize_accumulators(models)
acc = struct();
for m = 1:numel(models)
    acc.(models{m}) = struct('n_instances', 0, 'd_in', [], 'd_hat', [], ...
        'post_regular', [], 'post_perturbed', [], 'copy_perturbed', []);
end
end


function summary = summarize_accumulators(acc, models)
rows = cell(numel(models), 14);
for m = 1:numel(models)
    model = models{m};
    a = acc.(model);
    pos = a.d_in > 0;
    neg = a.d_in < 0;
    slope = sum(a.d_in .* a.d_hat) ./ sum(a.d_in .^ 2);
    mae_post_regular = mean(a.post_regular, 'omitnan');
    mae_post_perturbed = mean(a.post_perturbed, 'omitnan');
    mae_copy_perturbed = mean(a.copy_perturbed, 'omitnan');
    rows(m,:) = {model, a.n_instances, numel(a.d_in), ...
        mean(a.d_in(pos), 'omitnan'), mean(a.d_hat(pos), 'omitnan'), ...
        mean(a.d_in(neg), 'omitnan'), mean(a.d_hat(neg), 'omitnan'), ...
        mean(abs(a.d_hat - a.d_in), 'omitnan'), slope, ...
        mae_post_regular, mae_post_perturbed, mae_copy_perturbed, ...
        mae_post_perturbed ./ mae_post_regular, mae_copy_perturbed ./ mae_post_perturbed};
end
summary = cell2table(rows, 'VariableNames', { ...
    'model','n_test_graph_seed_instances','n_perturbed_edges', ...
    'mean_delta_l_in_pos','mean_delta_l_hat_pos', ...
    'mean_delta_l_in_neg','mean_delta_l_hat_neg', ...
    'mean_abs_delta_hat_minus_delta_in','slope_delta_hat_vs_delta_in', ...
    'MAE_post_regular','MAE_post_perturbed','MAE_copy_perturbed', ...
    'ratio_MAE_post_perturbed_to_regular','ratio_MAE_copy_to_post_perturbed'});
end


function fig = plot_copying_maes(summary, models)
fig = figure('Color', 'w', 'Name', 'Counterfactual distal copying MAEs');
mae = [summary.MAE_post_regular, summary.MAE_post_perturbed, summary.MAE_copy_perturbed];
bar(mae);
set(gca, 'XTick', 1:numel(models), 'XTickLabel', models, 'XTickLabelRotation', 30);
ylabel('MAE over perturbed distal interfaces');
legend({'regular pred vs post-T1','perturbed pred vs post-T1','perturbed pred vs perturbed input'}, ...
    'Location', 'best');
title('Counterfactual copy diagnostic');
grid on;
end