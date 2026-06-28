function report = GNNBenchmark_run_from_data_package(package_root, varargin)
%GNNBENCHMARK_RUN_FROM_DATA_PACKAGE  Reproduce analyses from a downloaded data package.
%
%   REPORT = GNNBENCHMARK_RUN_FROM_DATA_PACKAGE(PACKAGE_ROOT) runs the
%   post-prediction manuscript analyses using an external data package laid out
%   as documented in docs/DATA_PACKAGE.md. The function assumes predictions,
%   split indices, and optional saved spring-embedding outputs already exist; it
%   does not generate tissues, run Bayesian optimization, train models, or make
%   new predictions.
%
%   The input package is treated as read-only. Rebuilt summaries, regenerated
%   figures, embedding-bound tables, and the run report are written under a
%   separate output_root. By default this is
%       <PACKAGE_ROOT>/reanalysis_outputs/run_<timestamp>
%   with an automatic fallback to the current working folder if the package is
%   not writable.
%
%   Required package layout
%   -----------------------
%     PACKAGE_ROOT/
%       predictions/consolidated/*.pred.txt
%       predictions/consolidated/splits/*/{train.inds,val.inds,test.inds}
%
%   Optional package layout used when present
%   -----------------------------------------
%       embeddings/per_graph/out_graph_*.txt
%       manuscript_analyses/<additional diagnostic folders>
%
%   Name/value options
%   ------------------
%     output_root              Destination for regenerated outputs.
%     datasets                 Dataset keys to analyze. Default is the full
%                              manuscript/revision set supported by the MATLAB
%                              analyzer.
%     rebuild_summaries        Reparse prediction files into fresh summary .mat
%                              files (default true). This is the safest mode.
%     plot_figures             Regenerate main and revision figures (default true).
%     run_embedding_bounds     Analyze saved per-graph embeddings (default true).
%     run_counterfactual_copying
%                              Run the distal-copy diagnostic if a matching
%                              counterfactual prediction folder is present
%                              (default true; skipped if absent).
%     plot_embedding_examples  Recreate spring-embedding example panels inside
%                              GNNBenchmark_plot_results (default false because
%                              this requires vt2d geometry + spring engine,
%                              whereas the public package normally contains
%                              saved per-graph embedding outputs for the bound
%                              analysis).
%     save_png                 Save png sidecars in functions that support it
%                              (default true).
%     show_figures             Leave MATLAB figures visible while running
%                              (default false).
%     close_figures            Close figures after each plotting block
%                              (default true).
%     stop_on_error            Stop on required-stage errors (default true).
%
%   Output tree
%   -----------
%     output_root/analysis_tables/analyzer_cache/revision_2026/
%     output_root/analysis_tables/embedding_error_bounds/
%     output_root/analysis_tables/counterfactual_copying/        if available
%     output_root/figures/main/
%     output_root/figures/revision_2026/
%     output_root/manifests/data_package_analysis_report.{json,mat}
%
%   Example
%   -------
%     package_root = '/path/to/gnn_benchmark_public_data_20260627';
%     report = GNNBenchmark_run_from_data_package(package_root);

p = inputParser;
p.FunctionName = 'GNNBenchmark_run_from_data_package';
addRequired(p, 'package_root', @(x) ischar(x) || isstring(x));
addParameter(p, 'output_root', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'datasets', default_dataset_list(), @(x) iscell(x) || isstring(x) || ischar(x));
addParameter(p, 'rebuild_summaries', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'plot_figures', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'run_embedding_bounds', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'run_counterfactual_copying', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'plot_embedding_examples', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'save_png', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'show_figures', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'close_figures', true, @(x) islogical(x) && isscalar(x));
addParameter(p, 'stop_on_error', true, @(x) islogical(x) && isscalar(x));
parse(p, package_root, varargin{:});
opts = p.Results;
opts.package_root = char(opts.package_root);
opts.output_root = char(opts.output_root);
opts.datasets = cellstr(opts.datasets);

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(repo_root, 'analysis', 'matlab')));

input_paths = GNNBenchmark_data_package_paths(opts.package_root);
paths = input_paths;
paths.counterfactual_root = find_counterfactual_root(paths.manuscript_analyses_root);
if isempty(opts.output_root)
    default_parent = opts.package_root;
    if paths.is_public_package
        default_parent = paths.package_root;
    end
    opts.output_root = choose_default_output_root(default_parent);
end
paths.output_root = opts.output_root;
paths.analysis_tables = fullfile(paths.output_root, 'analysis_tables');
paths.output_analysis_cache_root = fullfile(paths.analysis_tables, 'analyzer_cache');
paths.output_revision_cache_root = fullfile(paths.output_analysis_cache_root, 'revision_2026');
paths.input_analysis_cache_root = input_paths.analysis_cache_root;
paths.input_revision_cache_root = input_paths.revision_cache_root;
if opts.rebuild_summaries || ~isfolder(paths.input_revision_cache_root)
    paths.analysis_cache_root = paths.output_analysis_cache_root;
    paths.revision_cache_root = paths.output_revision_cache_root;
else
    paths.analysis_cache_root = paths.input_analysis_cache_root;
    paths.revision_cache_root = paths.input_revision_cache_root;
end
paths.embedding_bounds_output = fullfile(paths.analysis_tables, 'embedding_error_bounds');
paths.counterfactual_output = fullfile(paths.analysis_tables, 'counterfactual_copying');
paths.figures_root = fullfile(paths.output_root, 'figures');
paths.main_figures_root = fullfile(paths.figures_root, 'main');
paths.revision_figures_root = fullfile(paths.figures_root, 'revision_2026');
paths.manifests = fullfile(paths.output_root, 'manifests');
ensure_dir(paths.analysis_tables);
ensure_dir(paths.output_analysis_cache_root);
if opts.rebuild_summaries || strcmp(paths.revision_cache_root, paths.output_revision_cache_root)
    ensure_dir(paths.output_revision_cache_root);
end
ensure_dir(paths.figures_root);
ensure_dir(paths.main_figures_root);
ensure_dir(paths.revision_figures_root);
ensure_dir(paths.manifests);

old_visibility = get(0, 'DefaultFigureVisible');
cleanup_visibility = onCleanup(@() set(0, 'DefaultFigureVisible', old_visibility)); %#ok<NASGU>
if ~opts.show_figures
    set(0, 'DefaultFigureVisible', 'off');
end

report = struct();
report.started_at = timestamp();
report.completed = false;
report.package_root = opts.package_root;
report.options = opts;
report.paths = paths;
report.steps = struct('name', {}, 'status', {}, 'seconds', {}, 'message', {}, 'details', {});
write_report(report, paths);

fprintf('\n============================================================\n');
fprintf('[GNNBenchmark data package] package: %s\n', opts.package_root);
fprintf('[GNNBenchmark data package] data:    %s\n', paths.data_root);
fprintf('[GNNBenchmark data package] output:  %s\n', paths.output_root);
fprintf('============================================================\n\n');

if opts.rebuild_summaries
    run_step('rebuild prediction summaries', true, @run_rebuild_summaries);
else
    fprintf('[GNNBenchmark data package] rebuild_summaries=false; using summaries in %s\n', paths.revision_cache_root);
end

if opts.plot_figures
    run_step('plot main and revision figures', true, @run_main_figure_plots);
end

if opts.run_embedding_bounds
    if ~isempty(paths.embedding_root)
        run_step('analyze saved embedding-error bounds', false, @run_embedding_bounds_step);
    else
        add_skipped_step('analyze saved embedding-error bounds', 'No embeddings/per_graph folder found in the data package.');
    end
end

if opts.run_counterfactual_copying
    if ~isempty(paths.counterfactual_root)
        run_step('analyze counterfactual distal copying', false, @run_counterfactual_step);
    else
        add_skipped_step('analyze counterfactual distal copying', 'No counterfactual prediction folder was detected under manuscript_analyses/.');
    end
end

report.optional_inputs = inventory_optional_inputs(paths);
report.completed = true;
report.completed_at = timestamp();
write_report(report, paths);
fprintf('\n[GNNBenchmark data package] DONE. Report:\n  %s\n', fullfile(paths.manifests, 'data_package_analysis_report.json'));

    function details = run_rebuild_summaries()
        code_dir = fileparts(mfilename('fullpath'));
        revision_cfg = struct();
        revision_cfg.data_root = paths.data_root;
        revision_cfg.datasets = opts.datasets;
        revision_cfg.rebuild_summaries = true;
        revision_cfg.plot_after_summary = false;
        revision_cfg.analysis_cache_root = paths.analysis_cache_root;
        revision_cfg.figures_root_override = paths.revision_figures_root;
        revision_cfg.models_to_exclude = {};
        revision_cfg.save_png = opts.save_png;
        revision_cfg.embed_examples = false;
        run_revision_analysis_script(fullfile(code_dir, 'GNNBenchmark_run_revision_analyses.m'), revision_cfg);
        details = struct('cache_root', paths.revision_cache_root, 'datasets', {opts.datasets});
    end

    function details = run_main_figure_plots()
        plotter_script = fullfile(fileparts(mfilename('fullpath')), 'GNNBenchmark_plot_results.m');
        revision_script = fullfile(fileparts(mfilename('fullpath')), 'GNNBenchmark_run_revision_analyses.m');
        requested_main_sets = intersect(opts.datasets, {'v1_W','v1_UW','hex'}, 'stable');
        main_sets = intersect(requested_main_sets, summaries_available(), 'stable');
        for ii = 1:numel(main_sets)
            plot_one_main_dataset(plotter_script, main_sets{ii});
        end

        revision_sets = intersect(opts.datasets, {'v1_2_16_W','kA_10','kA_1','Shear_1_2','Shear_1_5','Flip_two','Tissue_484','Tissue_784'}, 'stable');
        if ~isempty(revision_sets)
            revision_cfg = struct();
            revision_cfg.data_root = paths.data_root;
            revision_cfg.datasets = revision_sets;
            revision_cfg.rebuild_summaries = false;
            revision_cfg.plot_after_summary = true;
            revision_cfg.analysis_cache_root = paths.analysis_cache_root;
            revision_cfg.figures_root_override = paths.revision_figures_root;
            revision_cfg.calibrate_y_ranges = true;
            revision_cfg.save_png = opts.save_png;
            revision_cfg.make_composite_figures = true;
            revision_cfg.make_flip_two_interaction_figures = true;
            revision_cfg.verify_flip_two_interaction_figures = true;
            revision_cfg.models_to_exclude = {};
            revision_cfg.embed_examples = opts.plot_embedding_examples;
            run_revision_analysis_script(revision_script, revision_cfg);
        end
        details = struct('main_figures_root', paths.main_figures_root, ...
            'revision_figures_root', paths.revision_figures_root, ...
            'main_datasets', {main_sets}, 'revision_datasets', {revision_sets});
    end

    function details = run_embedding_bounds_step()
        outputs = GNNBenchmark_analyze_embedding_error_bounds( ...
            'embedding_root', paths.embedding_root, ...
            'output_dir', paths.embedding_bounds_output, ...
            'make_figures', true, ...
            'save_figures', true, ...
            'close_figures', opts.close_figures);
        details = struct('output_dir', paths.embedding_bounds_output, ...
            'n_graphs', height(outputs.per_graph));
    end

    function details = run_counterfactual_step()
        inds_dir = GNNBenchmark_consolidated_paths('inds_dir', paths.data_root, 'v1_2_16_W');
        outputs = GNNBenchmark_analyze_counterfactual_copying( ...
            'regular_pred_root', paths.data_root, ...
            'counterfactual_pred_root', paths.counterfactual_root, ...
            'inds_dir', inds_dir, ...
            'output_dir', paths.counterfactual_output, ...
            'close_figure', opts.close_figures);
        details = struct('output_dir', paths.counterfactual_output, ...
            'n_models', height(outputs.summary));
    end

    function plot_one_main_dataset(plotter_script, dataset_name)
        plot_cfg = struct();
        plot_cfg.dataset = dataset_name;
        plot_cfg.data_root = paths.data_root;
        plot_cfg.cache_dir = paths.revision_cache_root;
        plot_cfg.results_summary_filename = fullfile(paths.revision_cache_root, [dataset_name, ' - results_summary.mat']);
        plot_cfg.figures_root = paths.main_figures_root;
        plot_cfg.figures_output_dir = fullfile(paths.main_figures_root, dataset_name);
        plot_cfg.skip_cache_guard = false;
        plot_cfg.figure_panel_size = [340, 300];
        plot_cfg.scatter_marker_size = 9;
        plot_cfg.models_to_exclude = {};
        plot_cfg.embed_examples = opts.plot_embedding_examples;
        if strcmp(dataset_name, 'hex')
            plot_cfg.plot_only_hexagonality_manuscript = true;
            plot_cfg.plot_hex_hop_diagnostics = true;
            plot_cfg.plot_fallback_analysis = false;
            plot_cfg.hex_paper_uncertainty = 'sd';
        end
        run_plot_results_script(plotter_script, plot_cfg, opts.close_figures);
    end

    function names = summaries_available()
        candidates = {'v1_W','v1_UW','hex'};
        keep = false(size(candidates));
        for kk = 1:numel(candidates)
            keep(kk) = isfile(fullfile(paths.revision_cache_root, [candidates{kk}, ' - results_summary.mat']));
        end
        names = candidates(keep);
    end

    function run_step(name, required, fn)
        fprintf('\n[%s] %s\n', timestamp(), name);
        t0 = tic;
        details = struct();
        status = 'OK';
        message = '';
        caught = [];
        try
            details = fn();
        catch ME
            status = 'FAILED';
            message = ME.message;
            caught = ME;
            fprintf(2, '[GNNBenchmark data package] %s failed: %s\n', name, ME.message);
        end
        report.steps(end+1) = struct('name', name, 'status', status, ...
            'seconds', toc(t0), 'message', message, 'details', details); %#ok<AGROW>
        write_report(report, paths);
        if strcmp(status, 'FAILED') && required && opts.stop_on_error
            rethrow(caught);
        end
    end

    function add_skipped_step(name, message)
        fprintf('[GNNBenchmark data package] SKIP %s: %s\n', name, message);
        report.steps(end+1) = struct('name', name, 'status', 'SKIPPED', ...
            'seconds', 0, 'message', message, 'details', struct()); %#ok<AGROW>
        write_report(report, paths);
    end
end

function run_revision_analysis_script(script_path, cfg)
data_root = cfg.data_root; %#ok<NASGU>
datasets = cfg.datasets; %#ok<NASGU>
rebuild_summaries = cfg.rebuild_summaries; %#ok<NASGU>
plot_after_summary = cfg.plot_after_summary; %#ok<NASGU>
analysis_cache_root = cfg.analysis_cache_root; %#ok<NASGU>
figures_root_override = cfg.figures_root_override; %#ok<NASGU>
models_to_exclude = cfg.models_to_exclude; %#ok<NASGU>
save_png = cfg.save_png; %#ok<NASGU>
embed_examples = cfg.embed_examples; %#ok<NASGU>
if isfield(cfg, 'calibrate_y_ranges')
    calibrate_y_ranges = cfg.calibrate_y_ranges; %#ok<NASGU>
end
if isfield(cfg, 'make_composite_figures')
    make_composite_figures = cfg.make_composite_figures; %#ok<NASGU>
end
if isfield(cfg, 'make_flip_two_interaction_figures')
    make_flip_two_interaction_figures = cfg.make_flip_two_interaction_figures; %#ok<NASGU>
end
if isfield(cfg, 'verify_flip_two_interaction_figures')
    verify_flip_two_interaction_figures = cfg.verify_flip_two_interaction_figures; %#ok<NASGU>
end
run(script_path);
end

function run_plot_results_script(plotter_script, plot_cfg, close_figures)
GNNBenchmark_CONFIG = plot_cfg; %#ok<NASGU>
run(plotter_script);
clear GNNBenchmark_CONFIG;
if close_figures
    close all;
end
end
function datasets = default_dataset_list()
datasets = {'v1_W', 'v1_UW', 'hex', 'v1_2_16_W', 'Shear_1_2', 'Shear_1_5', ...
            'kA_1', 'kA_10', 'Flip_two', 'Tissue_484', 'Tissue_784'};
end

function root = find_counterfactual_root(parent)
root = '';
if ~isfolder(parent)
    return;
end
entries = dir(parent);
for ii = 1:numel(entries)
    if ~entries(ii).isdir || startsWith(entries(ii).name, '.')
        continue;
    end
    lname = lower(entries(ii).name);
    if contains(lname, 'counterfactual') || contains(lname, 'fallback_fingerprint') || contains(lname, 'edgehop14')
        candidate = fullfile(entries(ii).folder, entries(ii).name);
        if ~isempty(dir(fullfile(candidate, '**', '*.pred.txt')))
            root = candidate;
            return;
        end
    end
end
end

function out = inventory_optional_inputs(paths)
out = struct();
out.embedding_root_present = ~isempty(paths.embedding_root) && isfolder(paths.embedding_root);
out.counterfactual_root_present = ~isempty(paths.counterfactual_root) && isfolder(paths.counterfactual_root);
out.manuscript_analyses_root = paths.manuscript_analyses_root;
if isfolder(paths.manuscript_analyses_root)
    d = dir(paths.manuscript_analyses_root);
    names = {d([d.isdir]).name};
    names = names(~ismember(names, {'.','..'}));
    out.manuscript_analyses_dirs = names;
else
    out.manuscript_analyses_dirs = {};
end
end

function output_root = choose_default_output_root(package_root)
stamp = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<DATST>
candidate = fullfile(package_root, 'reanalysis_outputs', ['run_', stamp]);
try
    ensure_dir(candidate);
    output_root = candidate;
catch
    candidate = fullfile(pwd, ['gnn_benchmark_data_package_outputs_', stamp]);
    ensure_dir(candidate);
    output_root = candidate;
end
end

function ensure_dir(path_in)
if ~isfolder(path_in)
    mkdir(path_in);
end
end

function write_report(report, paths)
ensure_dir(paths.manifests);
mat_file = fullfile(paths.manifests, 'data_package_analysis_report.mat');
json_file = fullfile(paths.manifests, 'data_package_analysis_report.json');
save(mat_file, 'report');
write_text(json_file, jsonencode(report, 'PrettyPrint', true));
end

function write_text(filename, txt)
fid = fopen(filename, 'w');
if fid < 0
    error('GNNBenchmark:writeFailed', 'Could not write %s', filename);
end
cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, txt, 'char');
end

function s = timestamp()
s = datestr(now, 'yyyy-mm-dd HH:MM:SS'); %#ok<DATST>
end
