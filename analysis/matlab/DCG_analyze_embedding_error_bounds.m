function outputs = DCG_analyze_embedding_error_bounds(varargin)
%DCG_ANALYZE_EMBEDDING_ERROR_BOUNDS  Relate graph prediction error to embedding error.
%
%   OUTPUTS = DCG_ANALYZE_EMBEDDING_ERROR_BOUNDS(...) scans saved per-graph
%   spring-embedding output files, computes graph-level prediction MAE and
%   graph-level embedding MAE, and writes publication-ready diagnostic tables
%   and figures. The function is deliberately independent of the older plotting
%   scripts so it can be used as a direct audit of the embedding-error claim.
%
%   Name/value options
%   ------------------
%   'embedding_root'       Root folder containing out_graph_*.txt files. By
%                          default the function uses
%                          <DCG_DATA_ROOT>/embeddings/per_graph when
%                          DCG_DATA_ROOT is configured.
%   'output_dir'           Destination for CSVs and figures. Default:
%                          <embedding_root>/embedding_error_bounds_outputs.
%   'file_glob'            File pattern to scan recursively (default
%                          'out_graph_*.txt').
%   'models'               Cell array of model names to parse/color.
%   'cohorts'              Numeric cohort sizes expected in paths/names.
%   'ground_truth_col'     Column containing post-T1/target lengths (default 3).
%   'prediction_col'       Column containing predicted lengths (default 4).
%   'embedding_col'        Column containing embedded lengths (default 5).
%   'make_figures'         If true, make diagnostic figures (default true).
%   'save_figures'         If true, save .fig and .png (default true).
%   'close_figures'        If true, close figures after saving (default false).
%
%   Required file layout
%   --------------------
%   Each embedding output must contain one numeric row per interface/edge. The
%   default columns assume [edge_id, pre_or_aux, ground_truth, prediction,
%   embedding]. If your embedding executable writes a different layout, pass the
%   three *_col options explicitly. Metadata such as model/cohort/seed is parsed
%   from folder/file names when possible; unparseable fields are left as NaN or
%   'unknown' but still included in the raw table.
%
%   Outputs
%   -------
%   outputs.per_graph      Table with one row per embedding file.
%   outputs.summary        Mean/SE ratio summaries by model and cohort.
%   outputs.fit            Strict log-log upper-envelope fit table.
%   outputs.files          Paths to written CSV/figure files.
%
%   The upper envelope is fit in log2 coordinates as y <= a + b*x, where
%   x=log2(MAE_model) and y=log2(MAE_emb_to_prediction). The slope b is first
%   estimated by ordinary least squares; the intercept a is then lifted to the
%   smallest value that covers 100% of finite points. This matches the intended
%   manuscript interpretation: the embedding error is bounded by a function of
%   graph-level prediction error, not explained by an average regression line.

p = inputParser;
p.FunctionName = 'DCG_analyze_embedding_error_bounds';
default_models = {'PPGN','GraphSAGE','GAT','GIN','PNA'};
default_cohorts = [1 2 4 8 16 32];
cfg = DCG_publication_config();
default_root = '';
if isfield(cfg, 'data_root') && ~isempty(cfg.data_root)
    default_root = fullfile(cfg.data_root, 'embeddings', 'per_graph');
end
addParameter(p, 'embedding_root', default_root, @(x) ischar(x) || isstring(x));
addParameter(p, 'output_dir', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'file_glob', 'out_graph_*.txt', @(x) ischar(x) || isstring(x));
addParameter(p, 'models', default_models, @(x) iscell(x) || isstring(x));
addParameter(p, 'cohorts', default_cohorts, @(x) isnumeric(x));
addParameter(p, 'ground_truth_col', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'prediction_col', 4, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'embedding_col', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'make_figures', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'save_figures', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'close_figures', false, @(x) islogical(x) && isscalar(x));
parse(p, varargin{:});
opts = p.Results;
opts.embedding_root = char(opts.embedding_root);
opts.output_dir = char(opts.output_dir);
opts.file_glob = char(opts.file_glob);
opts.models = cellstr(opts.models);

if isempty(opts.embedding_root) || ~isfolder(opts.embedding_root)
    error('DCG:embeddingRootMissing', ['Set embedding_root or DCG_DATA_ROOT. ', ...
        'Expected a folder containing saved per-graph embedding outputs.']);
end
if isempty(opts.output_dir)
    opts.output_dir = fullfile(opts.embedding_root, 'embedding_error_bounds_outputs');
end
if ~isfolder(opts.output_dir)
    mkdir(opts.output_dir);
end

files = recursive_files(opts.embedding_root, opts.file_glob);
if isempty(files)
    error('DCG:noEmbeddingFiles', 'No files matching %s under %s.', opts.file_glob, opts.embedding_root);
end

rows = cell(numel(files), 1);
n = 0;
for i = 1:numel(files)
    M = read_numeric_matrix(files{i});
    max_col = max([opts.ground_truth_col, opts.prediction_col, opts.embedding_col]);
    if isempty(M) || size(M, 2) < max_col
        warning('DCG:badEmbeddingFile', 'Skipping unreadable/short embedding file: %s', files{i});
        continue;
    end
    gt = M(:, opts.ground_truth_col);
    pred = M(:, opts.prediction_col);
    emb = M(:, opts.embedding_col);
    ok = isfinite(gt) & isfinite(pred) & isfinite(emb);
    if ~any(ok)
        warning('DCG:emptyEmbeddingFile', 'Skipping file with no finite length triplets: %s', files{i});
        continue;
    end
    ctx = parse_embedding_context(files{i}, opts.models, opts.cohorts);
    n = n + 1;
    rows{n} = {files{i}, ctx.model, ctx.cohort, ctx.seed, ctx.weighting, ctx.graph_id, ...
        nnz(ok), mean(abs(pred(ok) - gt(ok))), ...
        mean(abs(emb(ok) - pred(ok))), mean(abs(emb(ok) - gt(ok)))};
end
rows = rows(1:n);
if isempty(rows)
    error('DCG:noUsableEmbeddingFiles', 'No usable embedding files were found under %s.', opts.embedding_root);
end

per_graph = cell2table(vertcat(rows{:}), 'VariableNames', { ...
    'file','model','cohort','seed','weighting','graph_id','n_edges', ...
    'mae_model','mae_embedding_to_prediction','mae_embedding_to_ground_truth'});
per_graph.ratio_emb_to_model = per_graph.mae_embedding_to_prediction ./ per_graph.mae_model;

summary = summarize_embedding_ratios(per_graph, opts.models, opts.cohorts);
fit = fit_strict_loglog_envelope(per_graph);

csv1 = fullfile(opts.output_dir, 'embedding_error_per_graph.csv');
csv2 = fullfile(opts.output_dir, 'embedding_error_ratio_summary.csv');
csv3 = fullfile(opts.output_dir, 'embedding_error_strict_loglog_envelope.csv');
writetable(per_graph, csv1);
writetable(summary, csv2);
writetable(fit, csv3);

fig_files = {};
if opts.make_figures
    fig = plot_embedding_bounds(per_graph, summary, fit, opts);
    if opts.save_figures
        fig_base = fullfile(opts.output_dir, 'embedding_error_bounds_and_ratios');
        savefig(fig, [fig_base '.fig']);
        print(fig, [fig_base '.png'], '-dpng', '-r300');
        fig_files = {[fig_base '.fig']; [fig_base '.png']};
    end
    if opts.close_figures
        close(fig);
    end
end

outputs = struct();
outputs.per_graph = per_graph;
outputs.summary = summary;
outputs.fit = fit;
outputs.files = struct('per_graph_csv', csv1, 'summary_csv', csv2, ...
    'fit_csv', csv3, 'figures', {fig_files});

fprintf('[DCG_analyze_embedding_error_bounds] files read: %d usable / %d scanned\n', height(per_graph), numel(files));
fprintf('[DCG_analyze_embedding_error_bounds] wrote %s\n', opts.output_dir);
end


function files = recursive_files(root_dir, pattern)
% recursive_files  Return full paths matching PATTERN under ROOT_DIR.
d = dir(fullfile(root_dir, '**', pattern));
files = arrayfun(@(x) fullfile(x.folder, x.name), d(:), 'UniformOutput', false);
end


function M = read_numeric_matrix(filename)
% read_numeric_matrix  Read numeric whitespace/comma-delimited matrix robustly.
try
    M = readmatrix(filename, 'FileType', 'text');
catch
    try
        imported = importdata(filename);
        if isstruct(imported)
            M = imported.data;
        else
            M = imported;
        end
    catch
        M = [];
    end
end
if iscell(M)
    M = cell2mat(M);
end
M = double(M);
end


function ctx = parse_embedding_context(filename, models, cohorts)
% parse_embedding_context  Infer model/cohort/seed/weighting/graph id from path text.
path_txt = char(filename);
ctx = struct('model', 'unknown', 'cohort', NaN, 'seed', NaN, ...
    'weighting', 'unknown', 'graph_id', 'unknown');
for m = 1:numel(models)
    if ~isempty(regexpi(path_txt, ['(^|[^A-Za-z])', regexptranslate('escape', models{m}), '([^A-Za-z]|$)'], 'once'))
        ctx.model = models{m};
        break;
    end
end
for c = reshape(cohorts, 1, [])
    patterns = {sprintf('2_%d', c), sprintf('1_%d', c), sprintf('%d_cohort', c), sprintf('%dcohort', c), sprintf('cohorts?[_ -]?%d', c)};
    for k = 1:numel(patterns)
        if ~isempty(regexpi(path_txt, patterns{k}, 'once'))
            ctx.cohort = c;
            break;
        end
    end
    if ~isnan(ctx.cohort)
        break;
    end
end
seed_tok = regexp(path_txt, '(^|[^A-Za-z0-9])s(\d+)([^A-Za-z0-9]|$)', 'tokens', 'once');
if ~isempty(seed_tok)
    ctx.seed = str2double(seed_tok{2});
end
w_tok = regexp(path_txt, '(^|[^A-Za-z0-9])(UW|W)([^A-Za-z0-9]|$)', 'tokens', 'once');
if ~isempty(w_tok)
    ctx.weighting = w_tok{2};
end
[~, stem] = fileparts(filename);
g_tok = regexp(stem, '(graph[^\s\/]*)', 'tokens', 'once');
if ~isempty(g_tok)
    ctx.graph_id = g_tok{1};
else
    ctx.graph_id = stem;
end
end


function summary = summarize_embedding_ratios(per_graph, models, cohorts)
% summarize_embedding_ratios  Mean/SD/SE of embedding-to-prediction ratio.
rows = {};
for m = 1:numel(models)
    for c = reshape(cohorts, 1, [])
        idx = strcmp(per_graph.model, models{m}) & per_graph.cohort == c & isfinite(per_graph.ratio_emb_to_model);
        if ~any(idx)
            continue;
        end
        x = per_graph.ratio_emb_to_model(idx);
        rows(end+1,:) = {models{m}, c, numel(x), mean(x), std(x), std(x) ./ sqrt(numel(x))}; %#ok<AGROW>
    end
end
if isempty(rows)
    summary = table(strings(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
        'VariableNames', {'model','cohort','n_graphs','mean_ratio','sd_ratio','se_ratio'});
else
    summary = cell2table(rows, 'VariableNames', {'model','cohort','n_graphs','mean_ratio','sd_ratio','se_ratio'});
end
end


function fit = fit_strict_loglog_envelope(per_graph)
% fit_strict_loglog_envelope  Fit y <= intercept+slope*x in log2 space.
x = log2(per_graph.mae_model);
y = log2(per_graph.mae_embedding_to_prediction);
ok = isfinite(x) & isfinite(y);
if nnz(ok) < 2
    fit = table(NaN, NaN, nnz(ok), NaN, NaN, NaN, ...
        'VariableNames', {'slope','intercept','n_points','r2_ols','max_margin','coverage_fraction'});
    return;
end
p = polyfit(x(ok), y(ok), 1);
slope = p(1);
ols_intercept = p(2);
y_ols = polyval(p, x(ok));
ss_res = sum((y(ok) - y_ols).^2);
ss_tot = sum((y(ok) - mean(y(ok))).^2);
r2 = 1 - ss_res ./ ss_tot;
intercept = max(y(ok) - slope .* x(ok));
y_bound = intercept + slope .* x(ok);
margin = y_bound - y(ok);
fit = table(slope, intercept, nnz(ok), r2, min(margin), mean(margin >= -1e-12), ...
    'VariableNames', {'slope','intercept','n_points','r2_ols','min_margin','coverage_fraction'});
fit.ols_intercept = ols_intercept;
fit.formula = {sprintf('log2(MAE_emb) <= %.6g + %.6g*log2(MAE_model)', intercept, slope)};
end


function fig = plot_embedding_bounds(per_graph, summary, fit, opts)
% plot_embedding_bounds  Make compact scatter/envelope and ratio-by-cohort panels.
colors = model_colors(opts.models);
fig = figure('Color', 'w', 'Name', 'Embedding error bounds and ratios');
tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
for m = 1:numel(opts.models)
    idx = strcmp(per_graph.model, opts.models{m});
    scatter(log2(per_graph.mae_model(idx)), log2(per_graph.mae_embedding_to_prediction(idx)), ...
        18, colors(m,:), 'filled', 'MarkerFaceAlpha', 0.35, 'DisplayName', opts.models{m});
end
if isfinite(fit.slope(1))
    x = log2(per_graph.mae_model);
    x = x(isfinite(x));
    xs = linspace(min(x), max(x), 100);
    ys = fit.intercept(1) + fit.slope(1) .* xs;
    plot(xs, ys, 'k-', 'LineWidth', 2, 'DisplayName', 'strict upper envelope');
end
xlabel('log_2(graph MAE_{model})');
ylabel('log_2(MAE_{emb,pred})');
title('Embedding error vs. prediction error');
grid on;
legend('Location', 'best');

nexttile;
hold on;
for m = 1:numel(opts.models)
    idx = strcmp(summary.model, opts.models{m});
    if ~any(idx), continue; end
    [c, order] = sort(summary.cohort(idx));
    mu = summary.mean_ratio(idx); mu = mu(order);
    se = summary.se_ratio(idx); se = se(order);
    errorbar(c, mu, se, '-o', 'Color', colors(m,:), 'MarkerFaceColor', colors(m,:), ...
        'LineWidth', 1.5, 'DisplayName', opts.models{m});
end
xlabel('# cohorts');
ylabel('MAE_{emb,pred} / MAE_{model}');
title('Embedding-to-prediction error ratio');
grid on;
legend('Location', 'best');
end


function colors = model_colors(models)
% model_colors  Stable colors used across the MATLAB revision figures.
base = containers.Map();
base('PPGN') = [0.0000 0.4470 0.7410];
base('GraphSAGE') = [0.8500 0.3250 0.0980];
base('GAT') = [0.4660 0.6740 0.1880];
base('GIN') = [0.4940 0.1840 0.5560];
base('PNA') = [0.9290 0.6940 0.1250];
colors = zeros(numel(models), 3);
for i = 1:numel(models)
    if isKey(base, models{i})
        colors(i,:) = base(models{i});
    else
        colors(i,:) = lines(1);
    end
end
end