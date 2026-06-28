%==========================================================================
% GNNBenchmark_plot_everything
%
% One-stop plotting wrapper for the current paper figure set.
%
% Runs:
%   1. Original v1 weighted data.
%   2. Original v1 unweighted data.
%   3. Manuscript hexagonality panels from the designated hex data.
%   4. Revision datasets plus v1 weighted _2_16_ reference panel.
%
% Notes:
%   - All manuscript plots are guarded to use the test split.
%   - PPGN is now included (data regenerated + validated 2026-06-01). The two
%     Tissue datasets (484/784) have no PPGN data, so they show the other models.
%   - Revision datasets are single-size, so dataset-size plots are skipped.
%   - Flip_two still skips the generic single-root hop-distance panel inside
%     GNNBenchmark_plot_results, but the revision runner adds the dedicated
%     two-source diagnostics: h_min nearest-T1 profiles, single-T1 vs two-T1
%     comparison plots, inter-flip-distance curves, zone bars, and heatmaps.
%
% CANONICAL WORKFLOW (one source folder, no stale cache):
%   1. Put the final dataset (pred + .pth + splits) in ONE folder and rebuild
%      EVERY summary from it: run GNNBenchmark_rebuild_all_summaries.m
%      (rebuild_summaries=true) -> writes stamped summaries to that folder's cache.
%   2. Then run THIS script to plot from those summaries.
%   The plotter carries a STALE-CACHE GUARD (2026-06-01): each summary is stamped
%   with a fingerprint of the prediction files it was built from, and the plotter
%   ERRORS if that no longer matches the current data_root -- so a stale summary
%   can never silently produce wrong figures (override: GNNBenchmark_CONFIG.skip_cache_guard).
%==========================================================================

close all;
clc;

code_dir = fileparts(mfilename('fullpath'));
plotter_script = fullfile(code_dir, 'GNNBenchmark_plot_results.m');
revision_script = fullfile(code_dir, 'GNNBenchmark_run_revision_analyses.m');

figure_panel_size = [340, 300];
scatter_marker_size = 9;
models_to_exclude = {};   % PPGN data validated/ready 2026-06-01 -> include all models

% ---- DATA ROOT for the whole plotting run (single source of truth) ----------
% Set data_root in the workspace, set GNN_BENCHMARK_DATA_ROOT, or create an untracked
% GNNBenchmark_local_config.m from GNNBenchmark_local_config_template.m.
if ~exist('data_root', 'var') || isempty(data_root)
    path_cfg = GNNBenchmark_publication_config();
    data_root = path_cfg.data_root;
end
if isempty(data_root)
    error('GNNBenchmark:missingDataRoot', ['Set the consolidated prediction snapshot path ', ...
        'using the data_root variable, GNN_BENCHMARK_DATA_ROOT, or GNNBenchmark_local_config.m.']);
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
main_figures_root_override = '';
try
    path_layout = GNNBenchmark_data_package_paths(data_root);
    data_root = path_layout.data_root;
    if path_layout.is_public_package
        if isempty(analysis_cache_root)
            analysis_cache_root = path_layout.analysis_cache_root;
        end
        if isempty(revision_cache_root_override)
            revision_cache_root_override = path_layout.revision_cache_root;
        end
        if isempty(figures_root_override)
            figures_root_override = path_layout.revision_figures_root;
        end
        main_figures_root_override = path_layout.main_figures_root;
    end
catch
    % Legacy flat layouts remain supported by the underlying plotter.
end
skip_cache_guard = false;
embed_examples   = true;    % 2D spring-embedding panels (v1 only); ON for finals (2026-06-02)
% -----------------------------------------------------------------------------

original_datasets = {'v1_W', 'v1_UW'};
for d = 1 : numel(original_datasets)
    GNNBenchmark_CONFIG = struct();
    GNNBenchmark_CONFIG.dataset = original_datasets{d};
    GNNBenchmark_CONFIG.data_root = data_root;
    GNNBenchmark_CONFIG.skip_cache_guard = skip_cache_guard;
    GNNBenchmark_CONFIG.figure_panel_size = figure_panel_size;
    GNNBenchmark_CONFIG.scatter_marker_size = scatter_marker_size;
    GNNBenchmark_CONFIG.models_to_exclude = models_to_exclude;
    if ~isempty(revision_cache_root_override)
        GNNBenchmark_CONFIG.cache_dir = revision_cache_root_override;
        GNNBenchmark_CONFIG.results_summary_filename = fullfile(revision_cache_root_override, [original_datasets{d}, ' - results_summary.mat']);
    end
    if ~isempty(main_figures_root_override)
        GNNBenchmark_CONFIG.figures_root = main_figures_root_override;
        GNNBenchmark_CONFIG.figures_output_dir = fullfile(main_figures_root_override, original_datasets{d});
    end
    GNNBenchmark_CONFIG.embed_examples = embed_examples;   % 2D embedding panels (v1_W / v1_UW) -- top toggle
    run(plotter_script);
    clear GNNBenchmark_CONFIG;
    close all;
end

GNNBenchmark_CONFIG = struct();
GNNBenchmark_CONFIG.dataset = 'hex';
GNNBenchmark_CONFIG.data_root = data_root;
GNNBenchmark_CONFIG.skip_cache_guard = skip_cache_guard;
GNNBenchmark_CONFIG.plot_only_hexagonality_manuscript = true;
GNNBenchmark_CONFIG.plot_hex_hop_diagnostics = true;
GNNBenchmark_CONFIG.plot_fallback_analysis = false;
GNNBenchmark_CONFIG.hex_paper_uncertainty = 'sd';
GNNBenchmark_CONFIG.figure_panel_size = figure_panel_size;
GNNBenchmark_CONFIG.scatter_marker_size = scatter_marker_size;
GNNBenchmark_CONFIG.models_to_exclude = models_to_exclude;
if ~isempty(revision_cache_root_override)
    GNNBenchmark_CONFIG.cache_dir = revision_cache_root_override;
    GNNBenchmark_CONFIG.results_summary_filename = fullfile(revision_cache_root_override, 'hex - results_summary.mat');
    GNNBenchmark_CONFIG.hex_analyses_filename = fullfile(revision_cache_root_override, 'hex - analyses data.mat');
    GNNBenchmark_CONFIG.hex_left_analyses_filename = fullfile(revision_cache_root_override, 'v1_W - analyses data.mat');
end
if ~isempty(main_figures_root_override)
    GNNBenchmark_CONFIG.figures_root = main_figures_root_override;
    GNNBenchmark_CONFIG.figures_output_dir = fullfile(main_figures_root_override, 'hex');
end
run(plotter_script);
clear GNNBenchmark_CONFIG;
close all;

datasets = {'v1_2_16_W', 'kA_10', 'kA_1', 'Shear_1_2', 'Shear_1_5', ...
            'Flip_two', 'Tissue_484', 'Tissue_784'};

rebuild_summaries = false;
plot_after_summary = true;
calibrate_y_ranges = true;
save_png = false;
make_composite_figures = true;

run(revision_script);

fprintf('\nAll current plotting complete.\n');
