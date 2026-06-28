%==========================================================================
% GNNBenchmark_run_revision_analyses
%
% Batch driver for the revision datasets and the v1 weighted _2_16_
% reference panel:
%   v1_2_16_W, Shear_1_2, Shear_1_5, kA_1, kA_10, Flip_two,
%   Tissue_484, Tissue_784.
%
% This intentionally excludes:
%   - accuracy vs. dataset size (revision has one training-set size);
%   - unweighted analyses (revision data here are weighted only);
%   - manuscript hexagonality panels (those belong to the hex dataset only).
%
% By default, the driver reuses existing result summaries and regenerates the
% figures only. Callers may set `analysis_cache_root` and
% `figures_root_override` before running this script to redirect generated
% summaries and figures away from the input prediction folder; this is how the
% public data-package runner keeps downloaded data read-only.
% The generic per-dataset plotter still skips Flip_two's
% single-root MAE-vs-hop panel, because there is no unique T1 root. The driver
% then runs GNNBenchmark_plot_Flip_two_interaction, which produces the
% Flip_two-specific h_min nearest-T1 profiles, single-T1 vs two-T1 comparison
% plots, inter-flip-distance curves, interaction-zone bars, and two-distance
% heatmaps.
%==========================================================================

close all;
clc;

if ~exist('datasets', 'var') || isempty(datasets)
    datasets = {'v1_2_16_W', 'Shear_1_2', 'Shear_1_5', 'kA_1', 'kA_10', ...
                'Flip_two', 'Tissue_484', 'Tissue_784'};
end

% Rebuild summaries from cached analyses_data.mat only when explicitly set.
if ~exist('rebuild_summaries', 'var') || isempty(rebuild_summaries)
    rebuild_summaries = false;
end
if ~exist('plot_after_summary', 'var') || isempty(plot_after_summary)
    plot_after_summary = true;
end
% PPGN data is now validated/ready, so the default is to exclude NOTHING. Use
% exist() only (NOT isempty) so a caller can pass {} to mean "exclude nothing"
% without it being silently overridden back to {'PPGN'}.
if ~exist('models_to_exclude', 'var')
    models_to_exclude = {};
end
if ~exist('calibrate_y_ranges', 'var') || isempty(calibrate_y_ranges)
    calibrate_y_ranges = true;
end
if ~exist('save_png', 'var') || isempty(save_png)
    save_png = false;
end
if ~exist('figure_panel_size', 'var') || isempty(figure_panel_size)
    figure_panel_size = [340, 300];
end
if ~exist('scatter_marker_size', 'var') || isempty(scatter_marker_size)
    scatter_marker_size = 9;
end
if ~exist('make_composite_figures', 'var') || isempty(make_composite_figures)
    make_composite_figures = true;
end
if ~exist('make_flip_two_interaction_figures', 'var') || isempty(make_flip_two_interaction_figures)
    make_flip_two_interaction_figures = true;
end
if ~exist('verify_flip_two_interaction_figures', 'var') || isempty(verify_flip_two_interaction_figures)
    verify_flip_two_interaction_figures = true;
end
if ~exist('embed_examples', 'var') || isempty(embed_examples)
    embed_examples = true;   % 2D embedding example panels for each revision dataset
end
if ~exist('analysis_cache_root', 'var') || isempty(analysis_cache_root)
    analysis_cache_root = '';
end
if ~exist('revision_cache_root_override', 'var') || isempty(revision_cache_root_override)
    revision_cache_root_override = '';
end
if ~exist('figures_root_override', 'var') || isempty(figures_root_override)
    figures_root_override = '';
end

code_dir = fileparts(mfilename('fullpath'));
analyzer_script = fullfile(code_dir, 'GNNBenchmark_analyze_results.m');
plotter_script = fullfile(code_dir, 'GNNBenchmark_plot_results.m');
calibrator_function = fullfile(code_dir, 'GNNBenchmark_calibrate_saved_fig_y_ranges.m');
composite_function = fullfile(code_dir, 'GNNBenchmark_make_revision_transverse_dist_composites.m');
flip_two_interaction_function = fullfile(code_dir, 'GNNBenchmark_plot_Flip_two_interaction.m');
flip_two_verify_function = fullfile(code_dir, 'GNNBenchmark_verify_Flip_two_figures.m');

if ~exist('data_root', 'var') || isempty(data_root)
    path_cfg = GNNBenchmark_publication_config();
    data_root = path_cfg.data_root;
end
if isempty(data_root)
    error('GNNBenchmark:missingDataRoot', ['Set the consolidated prediction snapshot path ', ...
        'using the data_root variable, GNN_BENCHMARK_DATA_ROOT, or GNNBenchmark_local_config.m.']);
end

path_layout = [];
try
    path_layout = GNNBenchmark_data_package_paths(data_root);
    data_root = path_layout.data_root;
catch
    path_layout = [];
end

is_consolidated_root = GNNBenchmark_consolidated_paths('is_consolidated', data_root);
if is_consolidated_root
    % Snapshot has no legacy analyses_data.mat; parse it fresh into a revision
    % cache + figures tree kept inside the snapshot folder. Public packages
    % keep these durable outputs under analysis_tables/ and figures/ instead.
    if isstruct(path_layout) && isfield(path_layout, 'is_public_package') && path_layout.is_public_package
        source_cache_root = path_layout.analysis_cache_root;
        revision_cache_root = path_layout.revision_cache_root;
        figures_root = path_layout.figures_root;
    else
        source_cache_root = fullfile(data_root, '_analyzer_cache');
        revision_cache_root  = fullfile(source_cache_root, 'revision_2026');
        figures_root      = fullfile(data_root, '_figures');
    end
elseif endsWith(data_root, 'GNN benchmark results')
    source_cache_root = fullfile(data_root, '_analyzer_cache');
    revision_cache_root = source_cache_root;
    figures_root = fullfile(data_root, '_figures');
else
    source_cache_root = fullfile(data_root, 'Analyzer cache');
    revision_cache_root = fullfile(source_cache_root, 'revision_2026');
    figures_root = fullfile(data_root, 'Output figures');
end

if ~isempty(analysis_cache_root)
    source_cache_root = analysis_cache_root;
    revision_cache_root = fullfile(source_cache_root, 'revision_codex_2026');
    if ~rebuild_summaries
        existing_revision_cache = first_existing({
            fullfile(source_cache_root, 'revision_codex_2026')
            fullfile(source_cache_root, 'revision_2026')
            });
        if ~isempty(existing_revision_cache)
            revision_cache_root = existing_revision_cache;
        end
    end
end
if ~isempty(revision_cache_root_override)
    revision_cache_root = revision_cache_root_override;
end
if ~isempty(figures_root_override)
    figures_root = figures_root_override;
end

if ~isfolder(revision_cache_root), mkdir(revision_cache_root); end
if ~isfolder(figures_root), mkdir(figures_root); end

fprintf('[GNNBenchmark_run_revision_analyses] data root:    %s\n', data_root);
fprintf('[GNNBenchmark_run_revision_analyses] source cache: %s\n', source_cache_root);
fprintf('[GNNBenchmark_run_revision_analyses] revision cache:  %s\n', revision_cache_root);
fprintf('[GNNBenchmark_run_revision_analyses] figures:      %s\n', figures_root);
fprintf('[GNNBenchmark_run_revision_analyses] excluded models: %s\n', strjoin(models_to_exclude, ', '));

% Per-dataset try/catch so one bad/truncated file can't abort the whole batch.
batch_status = repmat({'-'}, numel(datasets), 1);
batch_msg    = repmat({''},  numel(datasets), 1);
batch_secs   = zeros(numel(datasets), 1);
batch_tic    = tic;

for batch_dataset_index_2026 = 1 : numel(datasets)
    dataset = datasets{batch_dataset_index_2026};
    fprintf('\n=== %s (%d/%d) ===\n', dataset, batch_dataset_index_2026, numel(datasets));
    ds_tic = tic;
    try
    source_dataset = dataset;
    if strcmp(dataset, 'v1_2_16_W')
        source_dataset = 'v1_W';
    end

    revision_summary = fullfile(revision_cache_root, [source_dataset, ' - results_summary.mat']);
    need_summary = rebuild_summaries || ~isfile(revision_summary);

    if is_consolidated_root
        % Consolidated snapshot: the analyzer parses prediction files straight
        % from data_root (no pre-existing analyses_data.mat to reuse).
        if need_summary
            revision_analysis = fullfile(revision_cache_root, [source_dataset, ' - analyses data.mat']);
            fprintf('Building summary from snapshot: %s\n', data_root);
            GNNBenchmark_CONFIG = struct();
            GNNBenchmark_CONFIG.dataset = source_dataset;
            GNNBenchmark_CONFIG.data_root = data_root;
            GNNBenchmark_CONFIG.load_precomputed_data = ~rebuild_summaries && isfile(revision_analysis);
            GNNBenchmark_CONFIG.output_filename = revision_analysis;
            GNNBenchmark_CONFIG.results_summary_filename = revision_summary;
            GNNBenchmark_CONFIG.cache_dir = revision_cache_root;
            if exist('analysis_seeds', 'var') && ~isempty(analysis_seeds)
                GNNBenchmark_CONFIG.seeds = analysis_seeds;
            end
            run(analyzer_script);
            clear GNNBenchmark_CONFIG;
        else
            fprintf('Using existing revision summary: %s\n', revision_summary);
        end
    else
        source_analysis = first_existing({
            fullfile(source_cache_root, source_dataset, 'analyses_data.mat')
            fullfile(source_cache_root, [source_dataset, ' - analyses data.mat'])
            });
        source_summary = first_existing({
            fullfile(source_cache_root, source_dataset, 'results_summary.mat')
            fullfile(source_cache_root, [source_dataset, ' - results_summary.mat'])
            });

        if need_summary
            if isempty(source_analysis)
                if isempty(source_summary)
                    warning('GNNBenchmark:revisionMissingCache', ...
                        'No analyses_data.mat or results_summary.mat found for %s; skipping.', dataset);
                    continue;
                end
                warning('GNNBenchmark:revisionNoAnalysisCache', ...
                    'No analyses_data.mat found for %s; plotting existing summary only: %s', dataset, source_summary);
                revision_summary = source_summary;
            else
                fprintf('Building summary from: %s\n', source_analysis);
                GNNBenchmark_CONFIG = struct();
                GNNBenchmark_CONFIG.dataset = source_dataset;
                GNNBenchmark_CONFIG.data_root = data_root;
                GNNBenchmark_CONFIG.load_precomputed_data = 1;
                GNNBenchmark_CONFIG.output_filename = source_analysis;
                GNNBenchmark_CONFIG.results_summary_filename = revision_summary;
                GNNBenchmark_CONFIG.cache_dir = revision_cache_root;
                if exist('analysis_seeds', 'var') && ~isempty(analysis_seeds)
                    GNNBenchmark_CONFIG.seeds = analysis_seeds;
                end
                run(analyzer_script);
                clear GNNBenchmark_CONFIG;
            end
        elseif isfile(revision_summary)
            fprintf('Using existing revision summary: %s\n', revision_summary);
        elseif ~isempty(source_summary)
            revision_summary = source_summary;
            fprintf('Using existing source summary: %s\n', revision_summary);
        else
            warning('GNNBenchmark:revisionMissingSummary', 'No result summary available for %s; skipping plot.', dataset);
            continue;
        end
    end

    if plot_after_summary
        figures_output_dir = GNNBenchmark_figure_paths('dataset_dir', figures_root, dataset);
        if ~isfolder(figures_output_dir), mkdir(figures_output_dir); end

        fprintf('Plotting from summary: %s\n', revision_summary);
        GNNBenchmark_CONFIG = struct();
        GNNBenchmark_CONFIG.dataset = dataset;
        GNNBenchmark_CONFIG.data_root = data_root;
        if exist('skip_cache_guard', 'var') && ~isempty(skip_cache_guard)
            GNNBenchmark_CONFIG.skip_cache_guard = skip_cache_guard;   % honor master's interim-preview override
        end
        GNNBenchmark_CONFIG.results_summary_filename = revision_summary;
        GNNBenchmark_CONFIG.figures_root = figures_root;
        GNNBenchmark_CONFIG.figures_output_dir = figures_output_dir;
        GNNBenchmark_CONFIG.models_to_exclude = models_to_exclude;
        GNNBenchmark_CONFIG.plot_hop_demos = strcmp(dataset, 'v1_W');
        GNNBenchmark_CONFIG.figure_panel_size = figure_panel_size;
        GNNBenchmark_CONFIG.scatter_marker_size = scatter_marker_size;
        GNNBenchmark_CONFIG.embed_examples = embed_examples;
        if exist('analysis_seeds', 'var') && ~isempty(analysis_seeds)
            GNNBenchmark_CONFIG.seeds = analysis_seeds;
        end
        if exist('embed_engine', 'var') && ~isempty(embed_engine), GNNBenchmark_CONFIG.embed_engine = embed_engine; end
        if exist('embed_workdir', 'var') && ~isempty(embed_workdir), GNNBenchmark_CONFIG.embed_workdir = embed_workdir; end
        if exist('embed_vt2d_std', 'var') && ~isempty(embed_vt2d_std), GNNBenchmark_CONFIG.embed_vt2d_std = embed_vt2d_std; end
        if exist('embed_vt2d_rev', 'var') && ~isempty(embed_vt2d_rev), GNNBenchmark_CONFIG.embed_vt2d_rev = embed_vt2d_rev; end
        run(plotter_script);
        clear GNNBenchmark_CONFIG;
    end
    batch_status{batch_dataset_index_2026} = 'OK';
    catch ME_ds
        batch_status{batch_dataset_index_2026} = 'FAILED';
        batch_msg{batch_dataset_index_2026}    = ME_ds.message;
        fprintf(2, '*** FAILED %s: %s\n%s\n', dataset, ME_ds.message, ...
            getReport(ME_ds, 'extended', 'hyperlinks', 'off'));
    end
    batch_secs(batch_dataset_index_2026) = toc(ds_tic);
    fprintf('-------- %s : %s  (%.1f s) --------\n', dataset, ...
        batch_status{batch_dataset_index_2026}, batch_secs(batch_dataset_index_2026));
end

fprintf('\n================ ANALYSIS BATCH SUMMARY ================\n');
for batch_dataset_index_2026 = 1 : numel(datasets)
    fprintf('  %-12s  %-7s  %8.1f s   %s\n', ...
        datasets{batch_dataset_index_2026}, ...
        batch_status{batch_dataset_index_2026}, ...
        batch_secs(batch_dataset_index_2026), ...
        batch_msg{batch_dataset_index_2026});
end
fprintf('-------------------------------------------------------\n');
fprintf('  %d/%d OK   |   total %.1f min\n', ...
    sum(strcmp(batch_status, 'OK')), numel(datasets), toc(batch_tic)/60);
fprintf('=======================================================\n');

if plot_after_summary && calibrate_y_ranges && isfile(calibrator_function)
    GNNBenchmark_calibrate_saved_fig_y_ranges(figures_root, datasets, [], save_png);
end

if plot_after_summary && make_composite_figures && isfile(composite_function)
    GNNBenchmark_make_revision_transverse_dist_composites(figures_root, save_png);
end

if plot_after_summary && make_flip_two_interaction_figures && any(strcmp(datasets, 'Flip_two')) && isfile(flip_two_interaction_function)
    % Prefer the FRESH analyses data the driver just rebuilt into revision_cache_root
    % (e.g. revision_2026); only fall back to source_cache_root for legacy
    % layouts. Reading source_cache_root FIRST risked loading a alternate-folder copy
    % of "Flip_two - analyses data.mat" (caught 2026-06-01).
    flip_two_analysis = first_existing({
        fullfile(revision_cache_root, 'Flip_two - analyses data.mat')
        fullfile(revision_cache_root, 'Flip_two', 'analyses_data.mat')
        fullfile(source_cache_root, 'Flip_two - analyses data.mat')
        fullfile(source_cache_root, 'Flip_two', 'analyses_data.mat')
        });
    if isempty(flip_two_analysis)
        warning('GNNBenchmark:revisionMissingFlipTwoAnalysis', ...
            'No Flip_two analyses_data.mat found; skipping two-source Flip_two diagnostics.');
    else
        % Save the dedicated Flip_two diagnostics directly beside the standard
        % Flip_two figures, so the dataset has one canonical figure folder.
        flip_two_output_dir = GNNBenchmark_figure_paths('dataset_dir', figures_root, 'Flip_two');
        GNNBenchmark_plot_Flip_two_interaction(flip_two_analysis, flip_two_output_dir, save_png);
        if verify_flip_two_interaction_figures && isfile(flip_two_verify_function)
            GNNBenchmark_verify_Flip_two_figures(flip_two_output_dir);
        end
    end
end

fprintf('\nRevision analysis batch complete.\n');


function path_out = first_existing(candidates)
% first_existing  Implement first existing for this MATLAB workflow.
% Inputs: candidates
% Outputs: path_out
%FIRST_EXISTING  First path in a candidate list that exists on disk.
%
%   PURPOSE   Tolerate the two historical cache-filename conventions by trying
%             several full paths in priority order and returning the first that
%             exists (e.g. '<ds>\analyses_data.mat' vs '<ds> - analyses data.mat').
%   INPUT     candidates  cell array of candidate full paths, highest priority first.
%   OUTPUT    path_out    the first candidate satisfying isfile(); '' if none exist.
%   ALGORITHM Linear scan; returns on the first hit.

path_out = '';
for i = 1 : numel(candidates)
    if isfile(candidates{i})
        path_out = candidates{i};
        return;
    end
end

end
