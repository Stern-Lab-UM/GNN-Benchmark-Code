%==========================================================================
%  DCG_plot_results
%
%  Plotting entry point for the consolidated DCG benchmark analyses.
%  Section 1 picks `dataset` (default v1_W; pre-set in the workspace to
%  override). Section 2's if/elseif sets all per-dataset parameters and
%  per-plot skip flags. Sections 3+ run the existing plotting logic with
%  the right gates applied.
%
%  Publication version. The core MAE extraction follows the canonical script, with
%  explicit test-split gating, graph-weighted hop summaries, and
%  graph-weighted baseline normalization for normalized plots.
%
%  The single label change: section 5's last-subplot title was hardcoded
%  '22' (a 256-cell assumption); now derived from max_cell_dist so it is
%  correct for Tissue_484 / Tissue_784.
%==========================================================================
%
%  EXTENDED OVERVIEW (added as comment-only documentation)
%  -------------------------------------------------------
%  WHAT THIS SCRIPT DOES
%    Loads a precomputed results summary `S` for one benchmark "dataset"
%    (cell tissue + weighting), reconstructs per-graph / per-hop mean
%    absolute errors (MAE) for every model, and renders the manuscript /
%    revision figures (MAE-vs-#cohorts, MAE-vs-hop-distance,
%    MAE-vs-hexagonality, identity-fallback analysis, scatter examples and
%    2D spring-embedding examples). It also writes the CSV / TXT side-cars
%    that record the exact assumptions behind each figure.
%
%  KEY DATA STRUCTURE  `S` (a struct of nested cells)
%    Indexing convention used throughout is S.<field>.<task>{seed_row, size_bin}
%    then `.(dataset_to_analyze){graph}{hop}` = a matrix with one row per
%    retained interface (edge) and one column per model.
%      * S.prediction_errors.(task) : |prediction - post_T1| per edge, by model
%      * S.predictions.(task)       : raw predicted lengths per edge, by model;
%                                     the LAST model column is the pre-T1
%                                     ("Baseline"/"no_learning") length.
%      * S.ground_truth.(task)      : true post-T1 lengths (vector per graph,
%                                     or per-hop cells once split).
%      * S.ground_truth_dist        : post-T1 lengths re-split into per-hop
%                                     cells; CREATED by
%                                     calculate_normalization_factors.
%      * S.hexagonality{size_bin}.(split) : per-graph fraction of degree-6 cells.
%      * S.disorder{size_bin}.(split)     : per-graph active-noise parameter.
%      * S.normalization.(split)    : per_size scalar / per_dist vector /
%                                     per_hexagonality vector baseline MAEs.
%
%  TERMINOLOGY
%    * "task"  : 'lengths_to_lengths' (weighted, W) or 'none_to_lengths'
%                (unweighted, UW). Datasets declare which they carry.
%    * "size_bin" / "# cohorts" : column index `siz`; number of training
%                cohorts = 2^(siz-1) for v1 (and the per-size token in
%                emb_sizetok). Single-cohort revision datasets populate
%                exactly one bin.
%    * "hop"   : graph distance (in cells) from the T1 interface. Row index
%                `h`/`c`; hop 0 is the interface itself.
%    * "seed_row" / "repetition" : training-seed replicate; index `r`.
%
%  WEIGHTING CONVENTIONS (these matter for the std / SD differences below)
%    Model curves are GRAPH-weighted: average edges within a graph first,
%    then average graphs. The baseline normalization denominators use the
%    same graph-weighting, so numerator and denominator are commensurate.
%    Normalized panels plot log2(model_MAE / baseline_MAE); the Baseline
%    reference column is forced to exactly 0 (see
%    force_baseline_reference_to_zero).
%
%  DEAD / UNUSED FUNCTIONS (kept for reference; see each function's block)
%    plot_hexagonality_distribution, plot_MAE_vs_hexagonality and
%    perform_MAE_normalization are not called by the live pipeline. In
%    particular perform_MAE_normalization hard-codes a 24-hop count that is
%    WRONG for the 484/784-cell tissues (their max hop exceeds 24).
%==========================================================================

close all;
clc;

%==========================================================================
% 1) DATASET SELECTION  --  set `dataset` (or pre-set in the workspace).
%==========================================================================
% Supported values match the analyzer's selector:
%   'v1_W'   'v1_UW'           -- v1 standard benchmark (weighted / unweighted)
%   'v1_2_16_W'                -- v1 weighted _2_16_ / 16-cohort reference
%   'hex'                      -- v1 uniform-hexagonality experiment
%   'Shear_1_2'  'Shear_1_5'   -- revision: sheared tissues
%   'kA_1'       'kA_10'       -- revision: alternative K_A
%   'Flip_two'                 -- revision: two concurrent T1 transitions
%   'Tissue_484' 'Tissue_784'  -- revision: large tissues (no PPGN)
if exist('DCG_CONFIG', 'var') && isstruct(DCG_CONFIG) && isfield(DCG_CONFIG, 'dataset') && ~isempty(DCG_CONFIG.dataset)
    dataset = DCG_CONFIG.dataset;
elseif ~exist('dataset', 'var') || isempty(dataset)
    dataset = 'v1_W';
end

% Per-dataset parameters set by the if/elseif below:
%   n_cells              cells in the tissue (256 / 484 / 784)
%   max_cell_dist_seed   starting max hop (raised from data later)
%   tasks_to_plot        which tasks the analyzer saved: {W} or {W, UW}
%   multi_cohort         true => MAE-vs-dataset-size plot is meaningful
%   multi_T1             true => skip per-distance plot (distance ambiguous)
%   has_uniform_hex      true => skip hex distribution + MAE-vs-hex plots
%   is_hex_overlay       true => hex case; flag misleading x-axis on size plot
%   has_ppgn             per-dataset PPGN capability; cosmetic only since
%                        plot_PPGN_fallback now auto-detects PPGN at runtime
%                        via ismember('PPGN', all_models). True for the 8
%                        datasets where PPGN was actually trained; false for
%                        the 2 Tissue sets (VRAM-prohibitive, never run).
%   plot_title           human-readable title prefix for figures
%   figures_subdir       per-dataset folder under figures_root

summary_dataset    = dataset;
size_bins_to_keep  = [];

if strcmp(dataset, 'v1_W')
    n_cells            = 256;
    max_cell_dist_seed = 24;
    tasks_to_plot      = {'lengths_to_lengths'};
    multi_cohort       = true;
    multi_T1           = false;
    has_uniform_hex    = false;
    is_hex_overlay     = false;
    has_ppgn           = true;
    plot_title         = 'v1 standard benchmark (weighted)';
    figures_subdir     = 'v1_W';

elseif strcmp(dataset, 'v1_2_16_W')
    n_cells            = 256;
    max_cell_dist_seed = 24;
    tasks_to_plot      = {'lengths_to_lengths'};
    multi_cohort       = false;      % v1 reference panel: use only _2_16_ / 16-cohort bin
    multi_T1           = false;
    has_uniform_hex    = false;
    is_hex_overlay     = false;
    has_ppgn           = true;
    plot_title         = 'v1 standard benchmark, 16 cohorts (weighted)';
    figures_subdir     = 'v1_2_16_W';
    summary_dataset    = 'v1_W';
    size_bins_to_keep  = 5;          % # cohorts = 2^(5-1) = 16

elseif strcmp(dataset, 'v1_UW')
    n_cells            = 256;
    max_cell_dist_seed = 24;
    tasks_to_plot      = {'none_to_lengths'};
    multi_cohort       = true;
    multi_T1           = false;
    has_uniform_hex    = false;
    is_hex_overlay     = false;
    has_ppgn           = true;
    plot_title         = 'v1 standard benchmark (unweighted)';
    figures_subdir     = 'v1_UW';

elseif strcmp(dataset, 'hex')
    n_cells            = 256;
    max_cell_dist_seed = 24;
    tasks_to_plot      = {'lengths_to_lengths'};
    multi_cohort       = false;      % FIX 2026-05-23: hex no longer carries
                                     % v1 graft -- it's a single cohort now
    multi_T1           = false;
    has_uniform_hex    = true;       % skip hex-distribution + MAE-vs-hex
                                     % unless plot_MAE_vs_hexagonality is
                                     % uncommented and the v1+hex comparison
                                     % is wired plotter-side
    is_hex_overlay     = false;      % no graft -> no position-4 confusion
    has_ppgn           = true;
    plot_title         = 'Uniform-hexagonality tissue';
    figures_subdir     = 'hex';

elseif strcmp(dataset, 'Shear_1_2')
    n_cells            = 256;
    max_cell_dist_seed = 24;
    tasks_to_plot      = {'lengths_to_lengths'};
    multi_cohort       = false;
    multi_T1           = false;
    has_uniform_hex    = false;
    is_hex_overlay     = false;
    has_ppgn           = true;
    plot_title         = 'Sheared tissue, \lambda = 1.2';
    figures_subdir     = 'Shear_1_2';

elseif strcmp(dataset, 'Shear_1_5')
    n_cells            = 256;
    max_cell_dist_seed = 24;
    tasks_to_plot      = {'lengths_to_lengths'};
    multi_cohort       = false;
    multi_T1           = false;
    has_uniform_hex    = false;
    is_hex_overlay     = false;
    has_ppgn           = true;
    plot_title         = 'Sheared tissue, \lambda = 1.5';
    figures_subdir     = 'Shear_1_5';

elseif strcmp(dataset, 'kA_1')
    n_cells            = 256;
    max_cell_dist_seed = 24;
    tasks_to_plot      = {'lengths_to_lengths'};
    multi_cohort       = false;
    multi_T1           = false;
    has_uniform_hex    = false;
    is_hex_overlay     = false;
    has_ppgn           = true;
    plot_title         = 'Soft area modulus, kA = 1';
    figures_subdir     = 'kA_1';

elseif strcmp(dataset, 'kA_10')
    n_cells            = 256;
    max_cell_dist_seed = 24;
    tasks_to_plot      = {'lengths_to_lengths'};
    multi_cohort       = false;
    multi_T1           = false;
    has_uniform_hex    = false;
    is_hex_overlay     = false;
    has_ppgn           = true;
    plot_title         = 'Stiff area modulus, kA = 10';
    figures_subdir     = 'kA_10';

elseif strcmp(dataset, 'Flip_two')
    n_cells            = 256;
    max_cell_dist_seed = 24;
    tasks_to_plot      = {'lengths_to_lengths'};
    multi_cohort       = false;
    multi_T1           = true;        % => skip per-distance plot
    has_uniform_hex    = false;
    is_hex_overlay     = false;
    has_ppgn           = true;
    plot_title         = 'Two concurrent T1 transitions';
    figures_subdir     = 'Flip_two';

elseif strcmp(dataset, 'Tissue_484')
    n_cells            = 484;
    max_cell_dist_seed = 40;
    tasks_to_plot      = {'lengths_to_lengths'};
    multi_cohort       = false;
    multi_T1           = false;
    has_uniform_hex    = false;
    is_hex_overlay     = false;
    has_ppgn           = false;       % VRAM-prohibitive for PPGN
    plot_title         = '484-cell tissue';
    figures_subdir     = 'Tissue_484';

elseif strcmp(dataset, 'Tissue_784')
    n_cells            = 784;
    max_cell_dist_seed = 50;
    tasks_to_plot      = {'lengths_to_lengths'};
    multi_cohort       = false;
    multi_T1           = false;
    has_uniform_hex    = false;
    is_hex_overlay     = false;
    has_ppgn           = false;
    plot_title         = '784-cell tissue';
    figures_subdir     = 'Tissue_784';

else
    error('DCG:unknownDataset', ['Unknown dataset "%s". Valid: v1_W, v1_UW, ', ...
        'v1_2_16_W, hex, Shear_1_2, Shear_1_5, kA_1, kA_10, Flip_two, Tissue_484, Tissue_784.'], ...
        dataset);
end

%==========================================================================
% 2) DATASET-INDEPENDENT CONFIGURATION
%==========================================================================
path_cfg                 = DCG_publication_config();
data_root                = path_cfg.data_root;
cache_dir                = '';
figures_root             = '';
results_summary_filename = '';
figures_output_dir       = '';
hex_analyses_filename    = '';
hex_left_analyses_filename = '';

n_edges_per_graph                 = 3 * n_cells;
max_cell_dist                     = max_cell_dist_seed;
dataset_to_analyze                = 'test';
seeds                             = 1:5;
normalize_MAE                     = 1;
use_log                           = 1;
h_bins_for_quality_analysis       = 0.4 : 0.1 : 1;
h_bins_for_initial_histogram      = 0.4 : 0.02 : 1;
hex_paper_bins                    = 0.4 : 0.03 : 1;    % 20 bins over [0.4,1] to match the manuscript (was 0.025 = 24 bins; 2026-06-01)
active_noise_paper_bins           = 0 : 0.01 : 0.5;
hex_paper_size_bin                = [];
hex_paper_uncertainty             = 'sd';
hex_paper_split                   = 'test';
hex_save_separate_panels          = false;
hex_left_size_bin                 = 6;       % v1_W 32-cohort dataset
hex_left_seed_col                 = 1;       % descriptive graph topology/disorder
hex_left_model                    = 'PNA';   % topology is model-independent
hex_left_task                     = 'lengths_to_lengths';
hex_left_splits                   = {'train','val','test'};
plot_only_hexagonality_manuscript = false;
plot_hex_hop_diagnostics         = false;
plot_fallback_analysis           = true;
skip_cache_guard                 = false;   % bypass the stale-summary guard (NOT recommended)
expected_analysis_algorithm_version = '2026-06-06_triple_vertex_hops_v1';
plot_hop_demos                   = strcmp(dataset, 'v1_W');
models_to_exclude                = {};
figure_panel_size                = [340, 300];
scatter_marker_size              = 9;
scores_to_show                    = [99, 50, 0];
embed_color_percentiles           = [0.5 99];  % Fig B per-panel color clipping; [] = full min/max
embed_color_scale                 = 'linear';  % Fig B edge colors: raw |l-Lpred| on a linear color ramp
ratio_threshold_identity_function = log2(2);

colors = [
    0.0000, 0.4470, 0.7410;  % blue
    0.8500, 0.3250, 0.0980;  % red
    0.4660, 0.6740, 0.1880;  % green
    0.4940, 0.1840, 0.5560;  % purple
    0.9290, 0.6940, 0.1250;  % orange
    0, 0, 0                  % black
    ];

% Optional caller override. Use a DCG_CONFIG struct in the base workspace to
% redirect input/output paths or tweak plotting parameters without editing
% this file.
if exist('DCG_CONFIG', 'var') && isstruct(DCG_CONFIG)
    if isfield(DCG_CONFIG, 'data_root'), data_root = DCG_CONFIG.data_root; end
    if isfield(DCG_CONFIG, 'cache_dir'), cache_dir = DCG_CONFIG.cache_dir; end
    if isfield(DCG_CONFIG, 'figures_root'), figures_root = DCG_CONFIG.figures_root; end
    if isfield(DCG_CONFIG, 'figures_subdir'), figures_subdir = DCG_CONFIG.figures_subdir; end
    if isfield(DCG_CONFIG, 'results_summary_filename'), results_summary_filename = DCG_CONFIG.results_summary_filename; end
    if isfield(DCG_CONFIG, 'figures_output_dir'), figures_output_dir = DCG_CONFIG.figures_output_dir; end
    if isfield(DCG_CONFIG, 'hex_analyses_filename'), hex_analyses_filename = DCG_CONFIG.hex_analyses_filename; end
    if isfield(DCG_CONFIG, 'hex_left_analyses_filename'), hex_left_analyses_filename = DCG_CONFIG.hex_left_analyses_filename; end
    if isfield(DCG_CONFIG, 'dataset_to_analyze'), dataset_to_analyze = DCG_CONFIG.dataset_to_analyze; end
    if isfield(DCG_CONFIG, 'seeds'), seeds = DCG_CONFIG.seeds; end
    if isfield(DCG_CONFIG, 'normalize_MAE'), normalize_MAE = DCG_CONFIG.normalize_MAE; end
    if isfield(DCG_CONFIG, 'use_log'), use_log = DCG_CONFIG.use_log; end
    if isfield(DCG_CONFIG, 'h_bins_for_quality_analysis'), h_bins_for_quality_analysis = DCG_CONFIG.h_bins_for_quality_analysis; end
    if isfield(DCG_CONFIG, 'h_bins_for_initial_histogram'), h_bins_for_initial_histogram = DCG_CONFIG.h_bins_for_initial_histogram; end
    if isfield(DCG_CONFIG, 'hex_paper_bins'), hex_paper_bins = DCG_CONFIG.hex_paper_bins; end
    if isfield(DCG_CONFIG, 'active_noise_paper_bins'), active_noise_paper_bins = DCG_CONFIG.active_noise_paper_bins; end
    if isfield(DCG_CONFIG, 'hex_paper_size_bin'), hex_paper_size_bin = DCG_CONFIG.hex_paper_size_bin; end
    if isfield(DCG_CONFIG, 'hex_paper_uncertainty'), hex_paper_uncertainty = DCG_CONFIG.hex_paper_uncertainty; end
    if isfield(DCG_CONFIG, 'hex_paper_split'), hex_paper_split = DCG_CONFIG.hex_paper_split; end
    if isfield(DCG_CONFIG, 'hex_save_separate_panels'), hex_save_separate_panels = DCG_CONFIG.hex_save_separate_panels; end
    if isfield(DCG_CONFIG, 'skip_cache_guard'), skip_cache_guard = DCG_CONFIG.skip_cache_guard; end
    if isfield(DCG_CONFIG, 'hex_left_size_bin'), hex_left_size_bin = DCG_CONFIG.hex_left_size_bin; end
    if isfield(DCG_CONFIG, 'hex_left_seed_col'), hex_left_seed_col = DCG_CONFIG.hex_left_seed_col; end
    if isfield(DCG_CONFIG, 'hex_left_model'), hex_left_model = DCG_CONFIG.hex_left_model; end
    if isfield(DCG_CONFIG, 'hex_left_task'), hex_left_task = DCG_CONFIG.hex_left_task; end
    if isfield(DCG_CONFIG, 'hex_left_splits'), hex_left_splits = DCG_CONFIG.hex_left_splits; end
    if isfield(DCG_CONFIG, 'plot_only_hexagonality_manuscript'), plot_only_hexagonality_manuscript = DCG_CONFIG.plot_only_hexagonality_manuscript; end
    if isfield(DCG_CONFIG, 'plot_hex_hop_diagnostics'), plot_hex_hop_diagnostics = DCG_CONFIG.plot_hex_hop_diagnostics; end
    if isfield(DCG_CONFIG, 'plot_fallback_analysis'), plot_fallback_analysis = DCG_CONFIG.plot_fallback_analysis; end
    if isfield(DCG_CONFIG, 'plot_hop_demos'), plot_hop_demos = DCG_CONFIG.plot_hop_demos; end
    if isfield(DCG_CONFIG, 'models_to_exclude'), models_to_exclude = DCG_CONFIG.models_to_exclude; end
    if isfield(DCG_CONFIG, 'figure_panel_size'), figure_panel_size = DCG_CONFIG.figure_panel_size; end
    if isfield(DCG_CONFIG, 'scatter_marker_size'), scatter_marker_size = DCG_CONFIG.scatter_marker_size; end
    if isfield(DCG_CONFIG, 'scores_to_show'), scores_to_show = DCG_CONFIG.scores_to_show; end
    if isfield(DCG_CONFIG, 'embed_color_percentiles'), embed_color_percentiles = DCG_CONFIG.embed_color_percentiles; end
    if isfield(DCG_CONFIG, 'embed_color_scale'), embed_color_scale = DCG_CONFIG.embed_color_scale; end
    if isfield(DCG_CONFIG, 'ratio_threshold_identity_function'), ratio_threshold_identity_function = DCG_CONFIG.ratio_threshold_identity_function; end
    if isfield(DCG_CONFIG, 'summary_dataset'), summary_dataset = DCG_CONFIG.summary_dataset; end
    if isfield(DCG_CONFIG, 'size_bins_to_keep'), size_bins_to_keep = DCG_CONFIG.size_bins_to_keep; end
    if isfield(DCG_CONFIG, 'colors'), colors = DCG_CONFIG.colors; end
end

% Direct-script runs do not necessarily define DCG_CONFIG. Keep the driver
% override mechanism, but provide embedding defaults here so running this file
% alone still produces the cached/recomputed embedding panels.
if ~exist('DCG_CONFIG', 'var') || ~isstruct(DCG_CONFIG)
    DCG_CONFIG = struct();
end
if ~isfield(DCG_CONFIG, 'embed_examples')
    DCG_CONFIG.embed_examples = true;
end
if ~isfield(DCG_CONFIG, 'embed_recompute')
    DCG_CONFIG.embed_recompute = false;
end
if ~isfield(DCG_CONFIG, 'embed_color_percentiles')
    DCG_CONFIG.embed_color_percentiles = embed_color_percentiles;
end
if ~isfield(DCG_CONFIG, 'embed_color_scale')
    DCG_CONFIG.embed_color_scale = embed_color_scale;
end
if ~isfield(DCG_CONFIG, 'embed_engine') || isempty(DCG_CONFIG.embed_engine)
    DCG_CONFIG.embed_engine = path_cfg.embed_engine;
end
if ~isfield(DCG_CONFIG, 'embed_workdir') || isempty(DCG_CONFIG.embed_workdir)
    DCG_CONFIG.embed_workdir = path_cfg.embed_workdir;
end
if ~isfield(DCG_CONFIG, 'embed_vt2d_std') || isempty(DCG_CONFIG.embed_vt2d_std)
    DCG_CONFIG.embed_vt2d_std = path_cfg.embed_vt2d_std;
end
if ~isfield(DCG_CONFIG, 'embed_vt2d_rev') || isempty(DCG_CONFIG.embed_vt2d_rev)
    DCG_CONFIG.embed_vt2d_rev = path_cfg.embed_vt2d_rev;
end

if isempty(data_root)
    error('DCG:missingDataRoot', ['Set the consolidated prediction snapshot path ', ...
        'using DCG_CONFIG.data_root, DCG_DATA_ROOT, or DCG_local_config.m.']);
end

if isempty(cache_dir)
    cache_dir = fullfile(data_root, '_analyzer_cache');
    % Consolidated snapshot keeps the revision summaries/analyses in a
    % 'revision_2026' subfolder (matches DCG_run_revision_analyses
    % and DCG_rebuild_all_summaries). Prefer it when present so a
    % DIRECT plotter call -- e.g. the v1_W / v1_UW / hex calls from
    % DCG_plot_everything that don't pass results_summary_filename -- finds the
    % summaries instead of looking one level too high in _analyzer_cache\.
    revision_cache = fullfile(cache_dir, 'revision_2026');
    if isfolder(revision_cache)
        cache_dir = revision_cache;
    end
end
if isempty(figures_root)
    figures_root = fullfile(data_root, '_figures');
    revision_fig_datasets = {'v1_2_16_W', 'Shear_1_2', 'Shear_1_5', ...
        'kA_1', 'kA_10', 'Flip_two', 'Tissue_484', 'Tissue_784'};
    if ismember(dataset, revision_fig_datasets) && ...
            DCG_consolidated_paths('is_consolidated', data_root)
        % Direct runs should land in the same canonical revision figure tree
        % used by DCG_run_revision_analyses.
        figures_root = fullfile(figures_root, 'revision_2026');
    end
end
if isempty(results_summary_filename)
    results_summary_filename = fullfile(cache_dir, [summary_dataset, ' - results_summary.mat']);
end
if isempty(hex_analyses_filename)
    hex_analyses_filename = fullfile(cache_dir, [summary_dataset, ' - analyses data.mat']);
end
if isempty(hex_left_analyses_filename)
    hex_left_analyses_filename = fullfile(cache_dir, 'v1_W - analyses data.mat');
end
if isempty(figures_output_dir)
    figures_output_dir = fullfile(figures_root, figures_subdir);
end

hex_left_panel_cfg = struct( ...
    'analyses_filename', hex_left_analyses_filename, ...
    'size_bin', hex_left_size_bin, ...
    'seed_col', hex_left_seed_col, ...
    'model', hex_left_model, ...
    'task', hex_left_task, ...
    'splits', {hex_left_splits});

if ~strcmp(dataset_to_analyze, 'test')
    error('DCG:testOnlyRequired', ...
        'Manuscript plots must use dataset_to_analyze=''test''. Current value: %s', dataset_to_analyze);
end
if ~strcmp(hex_paper_split, 'test')
    error('DCG:testOnlyRequired', ...
        'Manuscript hexagonality panels must use hex_paper_split=''test''. Current value: %s', hex_paper_split);
end

if ~isfolder(figures_output_dir)
    mkdir(figures_output_dir);
end
fprintf('[DCG_plot_results] figures output: %s\n', figures_output_dir);
setappdata(0, 'DCG_CURRENT_DATASET_LABEL', sprintf('%s (%s)', plot_title, dataset));

% Opt-in compositor for the revision "extreme task family" figures. This is
% gated so ordinary single-dataset plotting is unchanged unless the caller
% explicitly requests the new composites.
if isfield(DCG_CONFIG, 'make_extreme_task_composites_only') && ...
        isequal(DCG_CONFIG.make_extreme_task_composites_only, true)
    DCG_make_extreme_task_composites_internal(data_root, cache_dir, figures_root, ...
        dataset_to_analyze, h_bins_for_quality_analysis, DCG_CONFIG);
    return;
end

remove_stale_fallback_outputs(figures_output_dir);
if strcmp(dataset, 'hex')
    remove_stale_non_hex_outputs_from_hex_folder(figures_output_dir);
end

fprintf('[DCG_plot_results] dataset=%s | tasks=%s | n_cells=%d | multi_cohort=%d | multi_T1=%d | hex_overlay=%d\n', ...
    dataset, strjoin(tasks_to_plot, ','), n_cells, multi_cohort, multi_T1, is_hex_overlay);

%==========================================================================
% 3) LOAD SUMMARY + AUTO-RAISE max_cell_dist FROM DATA
%==========================================================================
loaded_summary = load(results_summary_filename, 'S', 'all_models', 'tasks', 'data_sets');
S = loaded_summary.S;
all_models = loaded_summary.all_models;
loaded_tasks = loaded_summary.tasks;
data_sets = loaded_summary.data_sets;

% --- CACHE-STALENESS GUARD (2026-06-01) --------------------------------------
% Refuse to plot a summary that no longer matches its source folder. The analyzer
% stamped 'source_manifest' (dcg_source_manifest); recompute it from
% the CURRENT data_root and compare. ERROR on mismatch (folder changed since the
% summary was built -> would plot stale data); WARN if the summary predates the
% guard (no stamp). Override with DCG_CONFIG.skip_cache_guard = true.
if ~skip_cache_guard
    stamp_loaded = load(results_summary_filename, 'source_manifest', 'analysis_algorithm_version');
    if ~isfield(stamp_loaded, 'source_manifest')
        warning('DCG:noCacheStamp', ['Summary "%s" has NO provenance stamp (built by an ' ...
            'older analyzer). Cannot verify it is current -- rebuild it ' ...
            '(rebuild_summaries=true) to be safe.'], results_summary_filename);
    elseif ~strcmp(stamp_loaded.source_manifest, dcg_source_manifest(data_root))
        error('DCG:staleCache', ['STALE SUMMARY -- refusing to plot.\n  %s\nwas built from a ' ...
            'DIFFERENT set of prediction files than currently sit in\n  %s\nRebuild it ' ...
            '(rebuild_summaries=true) before plotting. (Override: ' ...
            'DCG_CONFIG.skip_cache_guard = true.)'], results_summary_filename, data_root);
    end
    if ~isfield(stamp_loaded, 'analysis_algorithm_version')
        error('DCG:staleAnalysisAlgorithm', ['STALE SUMMARY -- refusing to plot.\n  %s\nwas built ' ...
            'without an analysis-algorithm stamp. Rebuild it with the current analyzer so hop ' ...
            'distances use the row-preserving historical vertex-line definition.'], results_summary_filename);
    elseif ~strcmp(stamp_loaded.analysis_algorithm_version, expected_analysis_algorithm_version)
        error('DCG:staleAnalysisAlgorithm', ['STALE SUMMARY -- refusing to plot.\n  %s\nwas built ' ...
            'with analysis_algorithm_version="%s", but the plotter expects "%s". Rebuild it ' ...
            '(rebuild_summaries=true) before plotting.'], results_summary_filename, ...
            stamp_loaded.analysis_algorithm_version, expected_analysis_algorithm_version);
    end
end
tasks = tasks_to_plot;
if ~isequal(data_sets, {'test'})
    error('DCG:testOnlyRequired', ...
        'Loaded results_summary was not generated from the test split only. data_sets=[%s].', strjoin(data_sets, ', '));
end
if ~all(ismember(tasks, loaded_tasks))
    error('DCG:taskMismatch', ...
        'Dataset selector requested tasks [%s], but %s contains [%s].', ...
        strjoin(tasks, ', '), results_summary_filename, strjoin(loaded_tasks, ', '));
end
if ~all(isfield(S.prediction_errors, tasks))
    error('DCG:missingTaskField', ...
        'Loaded S.prediction_errors does not contain all selected tasks: [%s].', ...
        strjoin(tasks, ', '));
end
if ~isequal(tasks, loaded_tasks)
    fprintf('[DCG_plot_results] Using selector tasks [%s]; loaded summary also contains [%s].\n', ...
        strjoin(tasks, ', '), strjoin(loaded_tasks, ', '));
end
all_models{strcmp(all_models, 'no_learning')} = 'Baseline';
if ~isempty(models_to_exclude)
    [S, all_models, colors] = drop_models_from_summary(S, all_models, colors, models_to_exclude, tasks, dataset_to_analyze);
end
colors = paper_model_colors(all_models);
S.hexagonality = S.hexagonality(1,:)';
if isfield(S, 'disorder')
    S.disorder = S.disorder(1,:)';
end
if ~isempty(size_bins_to_keep)
    S = keep_summary_size_bins(S, tasks, size_bins_to_keep);
    fprintf('[DCG_plot_results] keeping size bin(s): %s (# cohorts: %s)\n', ...
        mat2str(size_bins_to_keep), mat2str(2.^(size_bins_to_keep-1)));
end

% Auto-raise max_cell_dist if data has deeper hops than the seed value
% (Tissue_484 / Tissue_784 reach ~33 / ~42).
observed_max_hop = 0;
for t_chk = 1 : length(tasks)
    pe_chk = S.prediction_errors.(tasks{t_chk});
    for ii_chk = 1 : numel(pe_chk)
        if isempty(pe_chk{ii_chk}), continue; end
        ds_chk = pe_chk{ii_chk}.(dataset_to_analyze);
        observed_max_hop = max(observed_max_hop, max(cellfun(@numel, ds_chk)));
    end
end
if observed_max_hop > max_cell_dist
    fprintf('[DCG_plot_results] max_cell_dist raised %d -> %d (data has deeper hops).\n', ...
        max_cell_dist, observed_max_hop);
    max_cell_dist = observed_max_hop;
end

manuscript_hex_panels_already_done = false;
if plot_only_hexagonality_manuscript
    if strcmp(dataset, 'hex') && isfile(hex_analyses_filename)
        plot_manuscript_hexagonality_panels_from_cache(hex_analyses_filename, figures_output_dir, ...
            colors, hex_paper_bins, hex_paper_uncertainty, n_cells, hex_paper_split, hex_save_separate_panels, hex_left_panel_cfg, models_to_exclude);
    else
        plot_manuscript_hexagonality_panels(S, all_models, dataset_to_analyze, figures_output_dir, colors, ...
            hex_paper_bins, active_noise_paper_bins, hex_paper_size_bin, hex_paper_uncertainty, hex_save_separate_panels);
    end
    manuscript_hex_panels_already_done = true;
    if ~plot_hex_hop_diagnostics
        fprintf('[DCG_plot_results] plot_only_hexagonality_manuscript=true -- generated manuscript hexagonality panels only.\n');
        return;
    end
    fprintf('[DCG_plot_results] plot_only_hexagonality_manuscript=true with plot_hex_hop_diagnostics=true -- continuing to hex-only hop diagnostics.\n');
end

%==========================================================================
% 4) SUMMARY STATS + MAE EXTRACTION  (calculation -- unchanged)
%==========================================================================
calc_total_number_of_graphs(all_models, tasks, S, dataset_to_analyze, n_edges_per_graph);
% plot_hexagonality_distribution(S, h_bins_for_initial_histogram, figures_output_dir);

[MAE_individuals, ~, ~, ~, ~, ~, ~, S] = extract_MAEs(tasks, S, all_models, dataset_to_analyze, max_cell_dist, h_bins_for_quality_analysis);

% we calculate normalization factors even when we don't use normalize_MAE since we report means via text:
S = calculate_normalization_factors(S, dataset_to_analyze, h_bins_for_quality_analysis, max_cell_dist);
write_normalization_assumptions(figures_output_dir, dataset_to_analyze);

%==========================================================================
% 5) HOP DEMOS  (pre-T1 vs post-T1 scatter at hops 0, mid, last)
%==========================================================================
% Calculation unchanged from the canonical plotter. The only label change:
% the last-hop subplot title was hardcoded '22' (256-cell assumption);
% now derived from max_cell_dist so it is correct for Tissue_484 / 784.
if plot_hop_demos
h = 8;
max_mid_dist = 0;
max_last_dist = 0;
hop_to_take = 8;
first_hop = nan(0,2);
last_hop = nan(0,2);
mid_hop = nan(0,2);
mid_hop_to_take = h;
idx = 0;
for j = 1 : size(S.ground_truth_dist{1,end}.(dataset_to_analyze),1)
    try
        % Multi-T1 robust: canonical plotter used `first_hop(end+1,:) = rhs`
        % which assumes hop-1 has exactly one row per graph (single T1 root
        % edge). For Flip_two there are two T1 root edges per graph, so the
        % RHS is 2x2 and the row-assignment throws. Use vertical concat so
        % both rows are kept (identical behaviour for single-T1 datasets).
        rhs_hop1 = [S.predictions.lengths_to_lengths{1,end}.(dataset_to_analyze){j}{1}(:,end), S.ground_truth_dist{1,end}.(dataset_to_analyze){j}{1}];
        first_hop = [first_hop; rhs_hop1]; %#ok<AGROW>

        last_used_idx = find(~cellfun(@isempty, S.predictions.lengths_to_lengths{1,end}.(dataset_to_analyze){j}), 1, 'last');
        curr_max_dist = max(abs(S.predictions.lengths_to_lengths{1,end}.(dataset_to_analyze){j}{last_used_idx}(:,end) - S.ground_truth_dist{1,end}.(dataset_to_analyze){j}{last_used_idx}));
        if curr_max_dist > max_last_dist
            max_last_dist = curr_max_dist;
            last_hop = [S.predictions.lengths_to_lengths{1,end}.(dataset_to_analyze){j}{last_used_idx}(:,end), S.ground_truth_dist{1,end}.(dataset_to_analyze){j}{last_used_idx}];
        end

        curr_max_dist = max(abs(S.predictions.lengths_to_lengths{1,end}.(dataset_to_analyze){j}{mid_hop_to_take}(:,end) - S.ground_truth_dist{1,end}.(dataset_to_analyze){j}{mid_hop_to_take}));
        if curr_max_dist > max_mid_dist
            max_mid_dist = curr_max_dist;
            mid_hop = [S.predictions.lengths_to_lengths{1,end}.(dataset_to_analyze){j}{mid_hop_to_take}(:,end), S.ground_truth_dist{1,end}.(dataset_to_analyze){j}{mid_hop_to_take}];
        end
    catch
        continue;
    end
end

figure;
for i = 1 : 3
    subplot(1, 3, i);
    if i == 1
        a = first_hop;
        tit = '0';
    elseif i == 2
        a = mid_hop;
        tit = num2str(hop_to_take-1);
    else
        a = last_hop;
        tit = num2str(max_cell_dist - 1);   % was hardcoded '22'
    end
    plot(a(:,2), a(:,1), '.b', 'MarkerSize', 5);
    lims = [max(0, min(a(:))), max(a(:))];
    lims = [lims(1) - range(lims)*0.05, lims(2) + range(lims)*0.05];
    xlim(lims);
    ylim(lims);
    axis image;
    xlabel('Post-T1 length (ground truth)');
    ylabel('Predicted length');
    title(['h = ', tit, ', max relative error (%) = ', num2str(max(abs((a(:,2) - a(:,1)) ./ a(:,2))) * 100)]);
    hold on;
    plot(lims, lims, 'Color', [0.5 0.5 0.5]);
end

dcg_savefig_visible(fullfile(figures_output_dir, 'hop demos.fig'));
else
    fprintf('[DCG_plot_results] skipping hop demos for this dataset/task selection.\n');
end

%==========================================================================
% 6) SCATTER-PLOT EXAMPLES
%==========================================================================
% The scatter examples are selected from S/MAE_individuals directly, across
% every non-baseline model, seed/repetition, cohort size, and graph. No raw
% prediction file paths are needed. UW (none_to_lengths) is included so its
% scatter examples mirror the embedding examples, which use the same percentile
% selection over MAE_individuals. Flip_two is single-cohort but still needs the
% same 2x3 scatter panel to accompany its embedding examples.
do_scatter_examples = ~strcmp(dataset, 'hex') && ...
    (multi_cohort || strcmp(dataset, 'Flip_two'));
if do_scatter_examples
    pre_figs = findall(groot, 'Type', 'figure');
    try
        plot_scatter_plot_examples(scores_to_show, tasks, MAE_individuals, S, figures_output_dir, all_models, dataset_to_analyze, figure_panel_size, scatter_marker_size);
    catch ME
        % Close any figure(s) the failed call opened (the function calls
        % figure(...) before the fopen on the cluster path, so a blank
        % maximized window is left behind otherwise).
        new_figs = setdiff(findall(groot, 'Type', 'figure'), pre_figs);
        if ~isempty(new_figs), close(new_figs); end
        warning('DCG:scatterFailed', ...
            'plot_scatter_plot_examples failed: %s', ME.message);
    end
else
    fprintf('[DCG_plot_results] skipping plot_scatter_plot_examples for this dataset/task selection.\n');
    remove_stale_scatter_example_files(figures_output_dir);
end

% 2D-embedding versions of the example-percentile graphs (default ON; heavy
% only for missing cache entries). Set DCG_CONFIG.embed_examples = false to skip.
% Mirrors the scatter-example selection above, then re-reads each chosen graph's
% raw prediction file for cell-pair topology + the three target-length columns
% (S carries lengths but no cell ids), guarding every panel with an S-vs-file
% MAE check so a wrong graph can never be embedded silently.
% default ON: embed unless DCG_CONFIG.embed_examples is explicitly set to false.
% DCG_CONFIG.embed_recompute defaults false, so existing out_<sim_id> cache
% files are reused and only missing relaxations are calculated/saved.
do_embed = ~(exist('DCG_CONFIG', 'var') && isstruct(DCG_CONFIG) && ...
    isfield(DCG_CONFIG, 'embed_examples') && isequal(DCG_CONFIG.embed_examples, false));
% 2026-06-02: previously this ALSO excluded the none_to_lengths-only selection
% (which is what v1_UW uses), silently dropping every v1_UW embedding. That task
% embeds fine -- it reads the UW pred files for the target lengths and only borrows
% the W partner for the flip map -- so it's now allowed. (hex stays excluded:
% manuscript hexagonality panels only.)
if do_embed && ~strcmp(dataset, 'hex')
    pre_figs_e = findall(groot, 'Type', 'figure');
    try
        plot_embedding_examples(scores_to_show, tasks, MAE_individuals, S, figures_output_dir, all_models, dataset_to_analyze, dataset, data_root, DCG_CONFIG);
    catch ME
        new_figs_e = setdiff(findall(groot, 'Type', 'figure'), pre_figs_e);
        if ~isempty(new_figs_e), close(new_figs_e); end
        warning('DCG:embedFailed', 'plot_embedding_examples failed: %s', ME.message);
    end
end

%==========================================================================
% 7) MAE EXTRACTION -- THREE PASSES (each run produces all three)
%==========================================================================
% Each run now produces THREE sets of MAE results so you don't have to flip
% global flags and re-run:
%   1. Raw linear MAE   (no log, no normalization)        [pass 1]
%   2. Raw log2(MAE)    (log only, no normalization)      [pass 2]
%   3. log2(nMAE)       (log + per-size/per-dist norm)    [pass 3]
% Calculation logic is unchanged from the canonical -- this section just
% back-to-backs three extract_MAEs calls with the right use_log + S state.

% --- Pass 1: linear MAE (use_log = 0, no normalization).
% Also capture MAE_hexagonality_avg / _sd (slots 4, 5) for the per-hexagonality
% plot block in section 8. The hex plot helper applies log internally based on
% its `use_log` flag, so we deliberately pass it raw values from a use_log=0
% extract.
[~, MAE_size_avg_lin, MAE_size_sd_lin, MAE_hex_avg_raw, MAE_hex_sd_raw, MAE_dists_avg_lin, MAE_dists_sd_lin, ~] = ...
    extract_MAEs(tasks, S, all_models, dataset_to_analyze, max_cell_dist, h_bins_for_quality_analysis, 0);

% --- Pass 2: log2(MAE) (use_log = 1, no normalization).
[~, MAE_size_avg_raw, MAE_size_sd_raw, ~, ~, MAE_dists_avg_raw, MAE_dists_sd_raw, ~] = ...
    extract_MAEs(tasks, S, all_models, dataset_to_analyze, max_cell_dist, h_bins_for_quality_analysis, use_log);

% --- Normalized pass: log2(nMAE). Same per-size then per-dist normalization
%     the canonical's if(normalize_MAE) branch did, restoring S between.
original_predictions_error = S.prediction_errors;

% normalizing per size:
for t = 1 : length(tasks)
    for s = 1 : length(seeds)
        for ss = 1 : size(S.prediction_errors.(tasks{t}),2)
            if isempty(S.prediction_errors.(tasks{t}){s,ss})
                continue;
            end
            for g = 1 : length(S.prediction_errors.(tasks{t}){s,ss}.(dataset_to_analyze))
                for h = 1 : length(S.prediction_errors.(tasks{t}){s,ss}.(dataset_to_analyze){g})

                    S.prediction_errors.(tasks{t}){s,ss}.(dataset_to_analyze){g}{h} = ...
                        S.prediction_errors.(tasks{t}){s,ss}.(dataset_to_analyze){g}{h} ./ S.normalization.(dataset_to_analyze).per_size(ss);

                end
            end
        end
    end
end

[~, MAE_size_avg_norm, MAE_size_sd_norm, ~, ~, ~, ~, ~] = extract_MAEs(tasks, S, all_models, dataset_to_analyze, max_cell_dist, h_bins_for_quality_analysis, use_log, 1);
S.prediction_errors = original_predictions_error;
baseline_idx_for_norm = find(strcmp(all_models, 'Baseline') | strcmp(all_models, 'no_learning'), 1);
if isempty(baseline_idx_for_norm)
    baseline_idx_for_norm = length(all_models);
end
MAE_size_avg_norm = force_baseline_reference_to_zero(MAE_size_avg_norm, tasks, baseline_idx_for_norm);
MAE_size_sd_norm = force_baseline_reference_to_zero(MAE_size_sd_norm, tasks, baseline_idx_for_norm);

% normalizing per dist:
for t = 1 : length(tasks)
    for s = 1 : length(seeds)
        for ss = 1 : size(S.prediction_errors.(tasks{t}),2)
            if isempty(S.prediction_errors.(tasks{t}){s,ss})
                continue;
            end
            for g = 1 : length(S.prediction_errors.(tasks{t}){s,ss}.(dataset_to_analyze))
                for h = 1 : length(S.prediction_errors.(tasks{t}){s,ss}.(dataset_to_analyze){g})

                    S.prediction_errors.(tasks{t}){s,ss}.(dataset_to_analyze){g}{h} = ...
                        S.prediction_errors.(tasks{t}){s,ss}.(dataset_to_analyze){g}{h} ./ S.normalization.(dataset_to_analyze).per_dist{ss}(h);

                end
            end
        end
    end
end

[~, ~, ~, ~, ~, MAE_dists_avg_norm, MAE_dists_sd_norm, ~] = extract_MAEs(tasks, S, all_models, dataset_to_analyze, max_cell_dist, h_bins_for_quality_analysis, use_log, 1);
S.prediction_errors = original_predictions_error;
MAE_dists_avg_norm = force_baseline_reference_to_zero(MAE_dists_avg_norm, tasks, baseline_idx_for_norm);
MAE_dists_sd_norm = force_baseline_reference_to_zero(MAE_dists_sd_norm, tasks, baseline_idx_for_norm);

%==========================================================================
% 8) PLOTS (gated per-dataset; MAE-vs-size and MAE-vs-dist each run THREE
%    times -- linear MAE, log2(MAE), log2(nMAE); .fig files are renamed
%    after each call so all three surface side-by-side in figures_output_dir).
%==========================================================================

% MAE vs dataset size: meaningful only when we have multiple cohorts.
% Three passes: linear MAE, log2(MAE), log2(nMAE).
if multi_cohort
    % Pass 1: linear MAE (use_log = 0; y_min forced to 0 inside the function).
    plot_MAE_vs_dataset_size(MAE_size_avg_lin, MAE_size_sd_lin, figures_output_dir, tasks, 0, colors, all_models, 'Mean graph MAE', figure_panel_size);
    movefile(fullfile(figures_output_dir, 'MAE vs dataset size (normalized log scale).fig'), ...
             fullfile(figures_output_dir, 'MAE vs dataset size (raw MAE).fig'));

    % Pass 2: log2(MAE)
    plot_MAE_vs_dataset_size(MAE_size_avg_raw, MAE_size_sd_raw, figures_output_dir, tasks, use_log, colors, all_models, 'log2(mean graph MAE)', figure_panel_size);
    movefile(fullfile(figures_output_dir, 'MAE vs dataset size (normalized log scale).fig'), ...
             fullfile(figures_output_dir, 'MAE vs dataset size (log2 MAE).fig'));

    % Pass 3: log2(nMAE)
    plot_MAE_vs_dataset_size(MAE_size_avg_norm, MAE_size_sd_norm, figures_output_dir, tasks, use_log, colors, all_models, 'log2(normalized mean graph MAE)', figure_panel_size);
    movefile(fullfile(figures_output_dir, 'MAE vs dataset size (normalized log scale).fig'), ...
             fullfile(figures_output_dir, 'MAE vs dataset size (log2 nMAE).fig'));

    if is_hex_overlay
        fprintf('[DCG_plot_results] NOTE on the MAE-vs-size figures: this is the hex run, so x-axis position 4 (cohort size 8) is the HEX tissue, NOT v1 size-8. Standard v1 cohorts are at positions 1, 2, 3, 5, 6.\n');
    end
else
    fprintf('[DCG_plot_results] single-cohort dataset -- skipping MAE-vs-dataset-size plots.\n');
end

% MAE vs distance in the GENERIC plotter assumes a single T1 root. That is
% ambiguous for multi-T1 graphs (Flip_two), so this panel is skipped here.
% Flip_two-specific distance structure is plotted later by
% DCG_plot_Flip_two_interaction using h_min=min(d1,d2), inter-flip
% distance, zones, and two-distance heatmaps.
% Same three passes (linear / log2(MAE) / log2(nMAE)) as the size plot.
if ~multi_T1
    % Pass 1: linear MAE (use_log = 0; y_min forced to 0 inside the function).
    plot_MAE_vs_dist(MAE_dists_avg_lin, MAE_dists_sd_lin, figures_output_dir, tasks, 0, colors, all_models, 'Mean graph MAE', figure_panel_size);
    movefile(fullfile(figures_output_dir, 'MAE vs traverse dist.fig'), ...
             fullfile(figures_output_dir, 'MAE vs traverse dist (raw MAE).fig'));

    % Pass 2: log2(MAE)
    plot_MAE_vs_dist(MAE_dists_avg_raw, MAE_dists_sd_raw, figures_output_dir, tasks, use_log, colors, all_models, 'log2(mean graph MAE)', figure_panel_size);
    movefile(fullfile(figures_output_dir, 'MAE vs traverse dist.fig'), ...
             fullfile(figures_output_dir, 'MAE vs traverse dist (log2 MAE).fig'));

    % Pass 3: log2(nMAE)
    plot_MAE_vs_dist(MAE_dists_avg_norm, MAE_dists_sd_norm, figures_output_dir, tasks, use_log, colors, all_models, 'log2(normalized graph MAE)', figure_panel_size);
    movefile(fullfile(figures_output_dir, 'MAE vs traverse dist.fig'), ...
             fullfile(figures_output_dir, 'MAE vs traverse dist (log2 nMAE).fig'));
else
    fprintf('[DCG_plot_results] multi-T1 dataset -- skipping generic single-root MAE-vs-distance plot; Flip_two-specific nearest/inter-flip diagnostics are produced by DCG_plot_Flip_two_interaction.\n');
end

% Manuscript hexagonality figures must come from the designated hex dataset,
% not from post-hoc hexagonality bins in the standard/revision datasets.
fprintf('[DCG_plot_results] skipping generic MAE-vs-hexagonality plots; manuscript hexagonality uses dataset=''hex'' only.\n');
remove_stale_plot_file(figures_output_dir, 'MAE vs hexagonality.fig');

if ~manuscript_hex_panels_already_done && strcmp(dataset, 'hex') && isfile(hex_analyses_filename)
    try
        plot_manuscript_hexagonality_panels_from_cache(hex_analyses_filename, figures_output_dir, ...
            colors, hex_paper_bins, hex_paper_uncertainty, n_cells, hex_paper_split, hex_save_separate_panels, hex_left_panel_cfg, models_to_exclude);
    catch ME
        warning('DCG:hexPaperPanelsFailed', ...
            'plot_manuscript_hexagonality_panels_from_cache errored: %s', ME.message);
    end
elseif ~manuscript_hex_panels_already_done && strcmp(dataset, 'hex') && ismember('lengths_to_lengths', tasks) && isfield(S, 'hexagonality')
    try
        plot_manuscript_hexagonality_panels(S, all_models, dataset_to_analyze, figures_output_dir, colors, ...
            hex_paper_bins, active_noise_paper_bins, hex_paper_size_bin, hex_paper_uncertainty, hex_save_separate_panels);
    catch ME
        warning('DCG:hexPaperPanelsFailed', ...
            'plot_manuscript_hexagonality_panels errored: %s', ME.message);
    end
end

% Fallback analysis (refactored 2026-05-22 -- takes a focus_model arg now):
% per-graph per-hop log-ratio + change-point detection covers EVERY
% non-baseline model. The histogram + example panels focus on one model
% picked here -- PPGN if loaded, otherwise PNA. has_ppgn no longer gates
% the call.
if plot_fallback_analysis
    if ismember('PPGN', all_models)
        fallback_focus_model = 'PPGN';
    else
        fallback_focus_model = 'PNA';
    end
    pre_figs = findall(groot, 'Type', 'figure');
    try
        plot_PPGN_fallback(all_models, S, dataset_to_analyze, ratio_threshold_identity_function, figures_output_dir, fallback_focus_model, figure_panel_size, scatter_marker_size);
    catch ME
        new_figs = setdiff(findall(groot, 'Type', 'figure'), pre_figs);
        if ~isempty(new_figs), close(new_figs); end
        warning('DCG:fallbackFailed', ...
            'plot_PPGN_fallback errored (%s focus): %s', fallback_focus_model, ME.message);
    end
else
    fprintf('[DCG_plot_results] skipping fallback analysis by configuration.\n');
end


function A = force_baseline_reference_to_zero(A, tasks, baseline_idx)
% force_baseline_reference_to_zero  Zero out the Baseline column in normalized data.
%
% PURPOSE
%   On normalized (log2-ratio) panels the Baseline model is the reference, so
%   its value is log2(baseline/baseline) = log2(1) = 0 by definition. This
%   helper overwrites the Baseline column with an exact 0 to remove any
%   floating-point residue (and any artificial SD band) before plotting.
%
% INPUTS
%   A           : results container. Either a struct with one field per task,
%                 or already a per-task value. Each task value is either a cell
%                 array of [graphs x models] matrices or a single numeric
%                 [rows x models] matrix.
%   tasks       : cellstr of task names to process (e.g. {'lengths_to_lengths'}).
%   baseline_idx: column index of the Baseline / no_learning model.
%
% OUTPUT
%   A           : same container with column baseline_idx set to 0 wherever it
%                 exists and is wide enough.
%
% ALGORITHM
%   For each task present in A: if the value is a cell, set column baseline_idx
%   to 0 in every non-empty matrix that has at least baseline_idx columns; if
%   it is numeric, set that column to 0 directly.
%
% EDGE CASES
%   Missing task fields and matrices narrower than baseline_idx are skipped, so
%   the function never errors on partially-populated datasets.

for t = 1 : numel(tasks)
    task = tasks{t};
    if ~isfield(A, task)
        continue;
    end
    if iscell(A.(task))
        for ii = 1 : numel(A.(task))
            if ~isempty(A.(task){ii}) && size(A.(task){ii}, 2) >= baseline_idx
                A.(task){ii}(:, baseline_idx) = 0;
            end
        end
    elseif isnumeric(A.(task)) && size(A.(task), 2) >= baseline_idx
        A.(task)(:, baseline_idx) = 0;
    end
end

end


function dcg_savefig_visible(varargin)
% dcg_savefig_visible  Save a figure as .fig with a dataset-tagged window name.
%
% PURPOSE
%   Thin wrapper around MATLAB savefig that (1) forces the figure visible so
%   the saved .fig reopens shown, and (2) stamps the window Name with the
%   current dataset label (if one was published via setappdata) for easier
%   identification when many figures are open.
%
% INPUTS (variadic)
%   dcg_savefig_visible(filename)            : saves the current figure (gcf).
%   dcg_savefig_visible(fig_handle, filename): saves the given figure handle.
%
% OUTPUTS
%   none (writes the .fig file to disk as a side effect).
%
% ALGORITHM
%   Resolve handle/filename from nargin. Set 'Visible','on'. If appdata key
%   DCG_CURRENT_DATASET_LABEL on root (0) is set and non-empty, set the figure
%   Name to "<label> | <file stem>" with NumberTitle off. Then savefig().
%
% EDGE CASES
%   If no dataset label is registered the Name is left untouched.

if nargin == 1
    fig_handle = gcf;
    filename = varargin{1};
else
    fig_handle = varargin{1};
    filename = varargin{2};
end

set(fig_handle, 'Visible', 'on');
if isappdata(0, 'DCG_CURRENT_DATASET_LABEL')
    dataset_label = getappdata(0, 'DCG_CURRENT_DATASET_LABEL');
else
    dataset_label = '';
end
if ~isempty(dataset_label)
    [~, file_stem] = fileparts(char(filename));
    set(fig_handle, 'Name', sprintf('%s | %s', dataset_label, file_stem), ...
        'NumberTitle', 'off');
end
savefig(fig_handle, filename);

end


function S = keep_summary_size_bins(S, tasks, size_bins_to_keep)
% keep_summary_size_bins  Restrict a summary S to a chosen set of size bins.
%
% PURPOSE
%   Some analyses want only specific size_bin columns (e.g. the 16-cohort
%   reference panel keeps bin 5 only). This blanks out all other size-bin
%   columns across the prediction/ground-truth fields and the per-size
%   metadata, so downstream code sees data only in the retained bins.
%
% INPUTS
%   S                : results summary struct.
%   tasks            : cellstr of task names to trim.
%   size_bins_to_keep: vector of column indices (size bins) to retain.
%
% OUTPUT
%   S                : same struct with non-kept size bins emptied.
%
% ALGORITHM
%   For each task, run keep_cell_columns on prediction_errors / predictions /
%   ground_truth (these are {seed x size} cells -> drop unwanted columns).
%   For hexagonality / disorder (1-D {size} cells) run keep_cell_entries.
%
% EDGE CASES
%   Fields that are absent (hexagonality / disorder) are simply skipped.

for t = 1 : numel(tasks)
    task = tasks{t};
    S.prediction_errors.(task) = keep_cell_columns(S.prediction_errors.(task), size_bins_to_keep);
    S.predictions.(task) = keep_cell_columns(S.predictions.(task), size_bins_to_keep);
    S.ground_truth.(task) = keep_cell_columns(S.ground_truth.(task), size_bins_to_keep);
end

if isfield(S, 'hexagonality')
    S.hexagonality = keep_cell_entries(S.hexagonality, size_bins_to_keep);
end
if isfield(S, 'disorder')
    S.disorder = keep_cell_entries(S.disorder, size_bins_to_keep);
end

end


function C = keep_cell_columns(C, size_bins_to_keep)
% keep_cell_columns  Blank all columns of a 2-D cell array except the kept ones.
%
% PURPOSE
%   Helper for keep_summary_size_bins on {seed_row x size_bin} cell arrays:
%   keeps the requested size-bin columns and empties the rest (rather than
%   deleting columns, so the column geometry / size-bin indexing is preserved).
%
% INPUTS
%   C                : 2-D cell array (rows = seeds, cols = size bins).
%   size_bins_to_keep: column indices to keep.
%
% OUTPUT
%   C                : same shape; non-kept columns set to {[]} entries.
%
% ALGORITHM
%   Clamp size_bins_to_keep to valid in-range columns, compute the complement,
%   and assign {[]} to every cell of each dropped column.
%
% EDGE CASES
%   Empty C is returned unchanged. Out-of-range indices are ignored.

if isempty(C)
    return;
end

valid_keep = size_bins_to_keep(size_bins_to_keep >= 1 & size_bins_to_keep <= size(C, 2));
drop_cols = setdiff(1:size(C, 2), valid_keep);
for col = drop_cols
    C(:, col) = {[]};
end

end


function C = keep_cell_entries(C, size_bins_to_keep)
% keep_cell_entries  Blank all entries of a 1-D cell array except the kept ones.
%
% PURPOSE
%   Same idea as keep_cell_columns but for 1-D {size_bin} cell arrays
%   (hexagonality, disorder): keep the requested linear indices, empty the rest.
%
% INPUTS
%   C                : 1-D cell array indexed by size bin.
%   size_bins_to_keep: linear indices to keep.
%
% OUTPUT
%   C                : same length; non-kept entries set to [].
%
% ALGORITHM
%   Clamp the keep list to 1..numel(C), drop the complement by assigning [].
%
% EDGE CASES
%   Empty C is returned unchanged; out-of-range indices are ignored.

if isempty(C)
    return;
end

valid_keep = size_bins_to_keep(size_bins_to_keep >= 1 & size_bins_to_keep <= numel(C));
drop_idx = setdiff(1:numel(C), valid_keep);
for ii = drop_idx
    C{ii} = [];
end

end


function remove_stale_plot_file(figures_output_dir, filename)
% remove_stale_plot_file  Delete one named output file if it exists.
%
% PURPOSE
%   Housekeeping: remove a single stale figure/output so a skipped plot does
%   not leave an out-of-date file behind in the dataset's figures folder.
%
% INPUTS
%   figures_output_dir : folder holding the dataset's figures.
%   filename           : bare file name to remove inside that folder.
%
% OUTPUTS
%   none (deletes the file and prints a notice if it was present).
%
% ALGORITHM
%   Build the full path; if isfile, delete it and log the removal.

target = fullfile(figures_output_dir, filename);
if isfile(target)
    delete(target);
    fprintf('[DCG_plot_results] removed stale plot file: %s\n', target);
end

end


function remove_stale_scatter_example_files(figures_output_dir)
% remove_stale_scatter_example_files  Delete old scatter-example .fig files.
%
% PURPOSE
%   Clear previously generated "Scatter plot examples (...).fig" files so a new
%   run does not mix fresh and stale scatter panels.
%
% INPUTS
%   figures_output_dir : folder to scan/clean.
%
% OUTPUTS
%   none (deletes matching files, logging each).
%
% ALGORITHM
%   dir() for the glob 'Scatter plot examples (*.fig', delete every match.

stale_files = dir(fullfile(figures_output_dir, 'Scatter plot examples (*.fig'));
for i = 1 : numel(stale_files)
    target = fullfile(stale_files(i).folder, stale_files(i).name);
    delete(target);
    fprintf('[DCG_plot_results] removed stale scatter example file: %s\n', target);
end

end

function remove_stale_fallback_outputs(figures_output_dir)
% remove_stale_fallback_outputs  Delete old fallback-analysis figures/tables.
%
% PURPOSE
%   Remove every artifact produced by plot_PPGN_fallback (hop-distribution and
%   example figures, the CSV table, the assumptions TXT) so a re-run starts
%   clean and never shows a previous model's fallback figures.
%
% INPUTS
%   figures_output_dir : folder to scan/clean.
%
% OUTPUTS
%   none (deletes matching files, logging each).
%
% ALGORITHM
%   Iterate a list of filename globs (model-prefixed .fig names plus the fixed
%   table/assumptions names); dir() each pattern and delete existing matches.

stale_patterns = { ...
    '* fallback hop distribution.fig', ...
    '* fallback example change point.fig', ...
    '* fallback example scatter plot.fig', ...
    'fallback_analysis_table.csv', ...
    'fallback_analysis_assumptions.txt'};

for p = 1 : numel(stale_patterns)
    stale_files = dir(fullfile(figures_output_dir, stale_patterns{p}));
    for i = 1 : numel(stale_files)
        target = fullfile(stale_files(i).folder, stale_files(i).name);
        if isfile(target)
            delete(target);
            fprintf('[DCG_plot_results] removed stale fallback output: %s\n', target);
        end
    end
end

end

function remove_stale_non_hex_outputs_from_hex_folder(figures_output_dir)
% remove_stale_non_hex_outputs_from_hex_folder  Prune non-manuscript files in hex dir.
%
% PURPOSE
%   For the uniform-hexagonality dataset only the manuscript hexagonality
%   panels are meaningful. This removes leftover non-hex figures (MAE-vs-size,
%   scatter examples, hop demos, plain MAE-vs-hex, hexagonality distribution)
%   from the hex output folder so the hex run does not retain irrelevant plots.
%
% INPUTS
%   figures_output_dir : the hex dataset's figures folder.
%
% OUTPUTS
%   none (deletes matching files, logging each).
%
% ALGORITHM
%   Iterate a fixed list of filename globs; dir() each and delete existing
%   matches.

stale_patterns = { ...
    'MAE vs dataset size*.fig', ...
    'Scatter plot examples (*.fig', ...
    'hop demos.fig', ...
    'MAE vs hexagonality.fig', ...
    'Hexagonality distribution.fig'};

for p = 1 : numel(stale_patterns)
    stale_files = dir(fullfile(figures_output_dir, stale_patterns{p}));
    for i = 1 : numel(stale_files)
        target = fullfile(stale_files(i).folder, stale_files(i).name);
        if isfile(target)
            delete(target);
            fprintf('[DCG_plot_results] removed stale non-hex-manuscript output from hex folder: %s\n', target);
        end
    end
end

end

function colors = paper_model_colors(model_names)
% paper_model_colors  Map model names to fixed manuscript RGB colors.
%
% PURPOSE
%   Return the canonical per-model plotting colors used throughout the paper so
%   every figure colors a given model identically.
%
% INPUTS
%   model_names : char/cellstr of model names (case-insensitive).
%
% OUTPUT
%   colors      : numel(model_names) x 3 RGB matrix (rows aligned to input).
%
% ALGORITHM / DECISIONS
%   switch on lower(name): PPGN=blue [0 .447 .741], GraphSAGE=red/orange
%   [.85 .325 .098], GAT=green [.466 .674 .188], GIN=purple [.494 .184 .556],
%   PNA=yellow/orange [.929 .694 .125], Baseline/no_learning=black [0 0 0].
%   Any unrecognized model falls back to mid-grey [.35 .35 .35].

model_names = cellstr(model_names);
colors = zeros(numel(model_names), 3);
for i = 1 : numel(model_names)
    switch lower(model_names{i})
        case 'ppgn'
            colors(i,:) = [0.0000, 0.4470, 0.7410];  % paper blue
        case 'graphsage'
            colors(i,:) = [0.8500, 0.3250, 0.0980];  % paper red/orange
        case 'gat'
            colors(i,:) = [0.4660, 0.6740, 0.1880];  % paper green
        case 'gin'
            colors(i,:) = [0.4940, 0.1840, 0.5560];  % paper purple
        case 'pna'
            colors(i,:) = [0.9290, 0.6940, 0.1250];  % paper yellow/orange
        case {'baseline', 'no_learning'}
            colors(i,:) = [0, 0, 0];                  % paper black
        otherwise
            colors(i,:) = [0.35, 0.35, 0.35];
    end
end

end

function ordered_idx = paper_model_plot_order(model_names, baseline_first)
% paper_model_plot_order  Indices that reorder models into manuscript order.
%
% PURPOSE
%   Produce a permutation of column indices so curves/legends are drawn in the
%   canonical paper order rather than the arbitrary order models appear in S.
%
% INPUTS
%   model_names   : char/cellstr of model names.
%   baseline_first: (optional, default true) place 'Baseline' first; if false,
%                   Baseline is omitted from the leading order list.
%
% OUTPUT
%   ordered_idx   : row vector of indices into model_names, in paper order,
%                   with any names not in the canonical list appended (stable).
%
% ALGORITHM
%   Canonical order = {Baseline, PPGN, GraphSAGE, GAT, GIN, PNA}. Drop the
%   leading 'Baseline' if baseline_first is false. For each name in order,
%   find its first case-insensitive match and append its index; finally append
%   the indices of any leftover models in their original order.

if nargin < 2 || isempty(baseline_first)
    baseline_first = true;
end
paper_order = {'Baseline', 'PPGN', 'GraphSAGE', 'GAT', 'GIN', 'PNA'};
model_names = cellstr(model_names);
ordered_idx = [];
if ~baseline_first
    paper_order = paper_order(2:end);
end
for i = 1 : numel(paper_order)
    idx = find(strcmpi(model_names, paper_order{i}), 1);
    if ~isempty(idx)
        ordered_idx(end+1) = idx; %#ok<AGROW>
    end
end
extra_idx = setdiff(1:numel(model_names), ordered_idx, 'stable');
ordered_idx = [ordered_idx, extra_idx];

end

function ordered_names = order_models_for_paper(model_names, include_baseline)
% order_models_for_paper  Reorder model NAMES into manuscript order.
%
% PURPOSE
%   Like paper_model_plot_order but returns the reordered name list itself
%   (used when selecting/ordering cached model fields).
%
% INPUTS
%   model_names     : char/cellstr of model names.
%   include_baseline: (optional, default false) if true, bracket the GNN order
%                     with {'Baseline'} ... {'no_learning'}.
%
% OUTPUT
%   ordered_names   : cellstr of names in paper order, with any unrecognized
%                     names appended at the end.
%
% ALGORITHM
%   Base order = {PPGN, GraphSAGE, GAT, GIN, PNA} (optionally wrapped with
%   Baseline / no_learning). For each ordered name take its first
%   case-insensitive match from model_names; then append any names not yet
%   used (preserving their order).

if nargin < 2 || isempty(include_baseline)
    include_baseline = false;
end
model_names = cellstr(model_names);
paper_order = {'PPGN', 'GraphSAGE', 'GAT', 'GIN', 'PNA'};
if include_baseline
    paper_order = [{'Baseline'}, paper_order, {'no_learning'}];
end
ordered_names = {};
for i = 1 : numel(paper_order)
    idx = find(strcmpi(model_names, paper_order{i}), 1);
    if ~isempty(idx)
        ordered_names{end+1} = model_names{idx}; %#ok<AGROW>
    end
end
extra_names = model_names(~ismember(lower(model_names), lower(ordered_names)));
ordered_names = [ordered_names, extra_names];

end


function write_normalization_assumptions(figures_output_dir, dataset_to_analyze)
% write_normalization_assumptions  Write the normalization_assumptions.txt side-car.
%
% PURPOSE
%   Record, as a human-readable TXT next to the figures, exactly how the
%   baseline-normalized (log2-ratio) plots were built for this split, so the
%   figures are reproducible/auditable.
%
% INPUTS
%   figures_output_dir : destination folder.
%   dataset_to_analyze : split name (e.g. 'test') named in the file.
%
% OUTPUTS
%   none (writes 'normalization_assumptions.txt').
%
% ALGORITHM / DECISIONS DOCUMENTED IN THE FILE
%   States that raw model MAE = mean |pred - post_T1|; Baseline = pre-T1 length
%   column in S.predictions; size/hop/hexagonality denominators are the
%   graph-weighted Baseline MAE (within graph -> across graphs in seed ->
%   across seeds); non-finite or <=eps denominators -> NaN; normalized plots
%   show log2(model/baseline) with the Baseline reference pinned to 0.
%
% EDGE CASES
%   If fopen fails, warns and returns without writing.

fid = fopen(fullfile(figures_output_dir, 'normalization_assumptions.txt'), 'w');
if fid < 0
    warning('DCG:normalizationAssumptionsWriteFailed', 'Could not write normalization assumptions file.');
    return;
end
fprintf(fid, 'Baseline normalization assumptions for split "%s"\n', dataset_to_analyze);
fprintf(fid, '1. Raw model MAE is the mean absolute error to post-T1 length.\n');
fprintf(fid, '2. Baseline/no_learning is the pre-T1 length column in S.predictions.\n');
fprintf(fid, '3. Size-normalization denominator is Baseline MAE averaged within graph, then across graphs within seed, then across seeds.\n');
fprintf(fid, '4. Hop-normalization denominator is Baseline MAE at each hop, averaged within graph, then across graphs within seed, then across seeds.\n');
fprintf(fid, '5. Hexagonality-normalization denominator uses the same graph-weighted Baseline MAE within each hexagonality bin.\n');
fprintf(fid, '6. Denominators that are non-finite or <= eps are set to NaN to avoid artificial spikes from division by zero.\n');
fprintf(fid, '7. Normalized plots show log2(model MAE / baseline MAE). The plotted Baseline reference is explicitly set to 0 with zero band.\n');
fclose(fid);

end


function [S, all_models, colors] = drop_models_from_summary(S, all_models, colors, models_to_exclude, tasks, dataset_to_analyze)
% drop_models_from_summary  Remove model columns from S and the model list.
%
% PURPOSE
%   Physically drop the requested models from the per-edge data so all
%   downstream MAE math and plotting see only the kept models (used e.g. to
%   exclude PPGN on datasets where it was never trained).
%
% INPUTS
%   S                 : results summary struct.
%   all_models        : current cellstr of model names (column order).
%   colors            : current color matrix aligned to all_models.
%   models_to_exclude : char/cellstr of model names to remove.
%   tasks             : cellstr of task names to trim.
%   dataset_to_analyze: split name whose per-graph cells are trimmed.
%
% OUTPUTS
%   S                 : with the model columns removed from prediction_errors
%                       and predictions (per graph, per hop).
%   all_models        : kept names only.
%   colors            : kept rows only (if it was full-width).
%
% ALGORITHM
%   Build a drop_mask = ismember(all_models, models_to_exclude). If nothing
%   matches, log and return unchanged. Otherwise, for fields
%   {prediction_errors, predictions}, walk every {seed,size}->graph->hop matrix
%   and, when its width equals numel(all_models), keep only the unmasked
%   columns. Finally trim all_models and (if wide enough) colors by keep_mask.
%
% EDGE CASES
%   Hop matrices whose width != numel(all_models) are left untouched (they were
%   already trimmed or are empty), preventing double-trimming.

models_to_exclude = cellstr(models_to_exclude);
drop_mask = ismember(all_models, models_to_exclude);
if ~any(drop_mask)
    fprintf('[DCG_plot_results] models_to_exclude requested [%s], but none were present.\n', ...
        strjoin(models_to_exclude, ', '));
    return;
end

keep_mask = ~drop_mask;
fprintf('[DCG_plot_results] excluding model columns from plots: %s\n', ...
    strjoin(all_models(drop_mask), ', '));

fields_to_trim = {'prediction_errors', 'predictions'};
for f = 1 : numel(fields_to_trim)
    field_name = fields_to_trim{f};
    for t = 1 : numel(tasks)
        task = tasks{t};
        if ~isfield(S.(field_name), task)
            continue;
        end
        C = S.(field_name).(task);
        for ii = 1 : numel(C)
            if isempty(C{ii}) || ~isfield(C{ii}, dataset_to_analyze)
                continue;
            end
            graphs = C{ii}.(dataset_to_analyze);
            for g = 1 : numel(graphs)
                hops = graphs{g};
                for h = 1 : numel(hops)
                    if ~isempty(hops{h}) && size(hops{h}, 2) == numel(all_models)
                        hops{h} = hops{h}(:, keep_mask);
                    end
                end
                graphs{g} = hops;
            end
            C{ii}.(dataset_to_analyze) = graphs;
        end
        S.(field_name).(task) = C;
    end
end

all_models = all_models(keep_mask);
if size(colors, 1) >= numel(keep_mask)
    colors = colors(keep_mask, :);
end

end


function plot_PPGN_fallback(all_models, S, dataset_to_analyze, ratio_threshold_identity_function, figures_output_dir, focus_model, figure_panel_size, scatter_marker_size)
% plot_PPGN_fallback  Detect/quantify "identity fallback" per graph and plot examples.
%
% PURPOSE
%   A model exhibits "fallback" on a graph when, beyond some hop, it stops
%   predicting the true post-T1 lengths and instead reproduces the pre-T1
%   (Baseline) lengths. This scans every non-baseline model on every graph,
%   measures a per-hop log-ratio, finds a single change-point, flags graphs
%   that switch from post-like to baseline-like, writes a CSV + assumptions
%   TXT, and renders three example figures for one focus model.
%
% INPUTS
%   all_models                        : cellstr of model names (column order).
%   S                                 : results summary (uses
%                                       S.predictions.lengths_to_lengths and
%                                       S.ground_truth_dist).
%   dataset_to_analyze                : split name.
%   ratio_threshold_identity_function : threshold T on the log2 ratio (the
%                                       code uses +/-T; T = log2(2) = 1 means a
%                                       2x MAE ratio either way).
%   figures_output_dir                : output folder.
%   focus_model                       : model used for the example figures
%                                       (default 'PPGN' if present else 'PNA').
%   figure_panel_size                 : [w h] px per scatter panel (optional).
%   scatter_marker_size               : marker size for scatter points (opt).
%
% OUTPUTS
%   none returned. Writes fallback_analysis_assumptions.txt,
%   fallback_analysis_table.csv, and three .fig files (hop distribution,
%   change-point example, before/after scatter example) for focus_model.
%
% MATH (per graph g, model m, hop h)  -- see fallback_graph_values
%   FB(h) = log2( MAE_h(pred, post_T1) / MAE_h(pred, pre_T1) ).
%     numerator   = mean|pred - true post_T1| over edges at hop h.
%     denominator = mean|pred - pre_T1 (Baseline col)| over edges at hop h.
%   FB(h) < 0  => prediction closer to TRUE post-T1 (good).
%   FB(h) > 0  => prediction closer to the pre-T1 baseline (fallback).
%   Hop 0 is forced to NaN (excluded). Hops with empty edges, non-finite
%   values, or denominator <= 0 are excluded.
%
% ALGORITHM
%   1. Identify focus_model index, baseline_idx (Baseline/no_learning, else
%      last column) and the set of non-baseline model_indices.
%   2. Write the assumptions TXT (records every rule above and the threshold).
%   3. For each size/seed/graph/non-baseline-model: get FB(h) via
%      fallback_graph_values; require >= min_valid_hops (=4) finite non-hop0
%      values; run findchangepts(FB, Statistic=mean, MaxNumChanges=1) to locate
%      the transition; split into before/after the change row; compute
%      mean_before, mean_after. fallback_flag = (mean_before <= -T) &
%      (mean_after >= +T). Record a status string per case
%      (ok/too_few_valid_hops/no_changepoint/empty_segment/(no_)fallback/
%      findchangepts_error). Accumulate one table row per graph/model.
%   4. Save the table; print the fraction of graphs flagged per model per size.
%   5. Pick the focus model's most-separated fallback example (min separation =
%      most negative mean_after - mean_before) for the example figures. If the
%      focus model has no fallback example, fall back to whichever model has the
%      most; if none at all, warn and return.
%   6. Figure 1: histogram of change hops among focus-model fallback graphs.
%      Figure 2: FB(h) vs hop with the change hop (xline), +/-T (ylines) and the
%      two piecewise-mean fits overlaid as red segments.
%      Figure 3: 2x2 scatter of predicted vs post-T1 and vs pre-T1, split into
%      hops before vs at/after the change, each titled with its MAE.
%
% DECISIONS / EDGE CASES
%   findchangepts failures are caught and recorded, not fatal. Selected example
%   with an empty before/after segment warns and returns. The focus model is
%   re-selected at runtime, so has_ppgn no longer gates this analysis.
%
% FALLBACK ANALYSIS (formerly PPGN-only).
% For each graph/model/hop h, define:
%   FB(h) = log2( MAE_h(prediction, post_T1 length)
%                / MAE_h(prediction, pre_T1 length) )
% Negative FB means predictions are closer to the true post-T1 lengths.
% Positive FB means predictions are closer to the pre-T1 baseline. A fallback
% graph is one where the changepoint splits strongly-post-like early hops from
% strongly-baseline-like later hops.

if nargin < 6 || isempty(focus_model)
    if ismember('PPGN', all_models)
        focus_model = 'PPGN';
    else
        focus_model = 'PNA';
    end
end
if nargin < 7 || isempty(figure_panel_size)
    figure_panel_size = [340, 300];
end
if nargin < 8 || isempty(scatter_marker_size)
    scatter_marker_size = 9;
end
focus_idx_mask = strcmp(all_models, focus_model);
if ~any(focus_idx_mask)
    error('DCG:focusModelNotFound', 'focus_model "%s" not in all_models. Available: %s', focus_model, strjoin(all_models, ', '));
end

baseline_idx = find(strcmp(all_models, 'Baseline') | strcmp(all_models, 'no_learning'), 1);
if isempty(baseline_idx)
    baseline_idx = length(all_models);
end
model_indices = setdiff(1:length(all_models), baseline_idx, 'stable');
min_valid_hops = 4;

assumptions_filename = fullfile(figures_output_dir, 'fallback_analysis_assumptions.txt');
fid = fopen(assumptions_filename, 'w');
fprintf(fid, 'Fallback analysis assumptions\n');
fprintf(fid, '1. Uses task lengths_to_lengths only.\n');
fprintf(fid, '2. Per graph/model/hop: FB(h)=log2(MAE(pred, post_T1)/MAE(pred, pre_T1)).\n');
fprintf(fid, '3. MAE at a hop is averaged over all retained interfaces at that hop.\n');
fprintf(fid, '4. pre_T1 is the Baseline/no_learning column in S.predictions; post_T1 is S.ground_truth_dist.\n');
fprintf(fid, '5. Hop 0 is excluded from changepoint detection and before/after means.\n');
fprintf(fid, '6. Hops with empty edge sets, non-finite values, or denominator <= 0 are excluded.\n');
fprintf(fid, '7. A graph/model requires at least %d finite non-hop0 hops.\n', min_valid_hops);
fprintf(fid, '8. Changelocation is detected with findchangepts(..., Statistic=mean, MaxNumChanges=1).\n');
fprintf(fid, '9. fallback_flag = mean_before <= -%g and mean_after >= %g.\n', ratio_threshold_identity_function, ratio_threshold_identity_function);
fprintf(fid, '10. The change-point figure overlays the two piecewise mean fits from findchangepts as red horizontal segments.\n');
fclose(fid);

row_task = {};
row_model = {};
row_status = {};
row_size = [];
row_seed = [];
row_graph = [];
row_transition_hop = [];
row_mean_before = [];
row_mean_after = [];
row_separation = [];
row_fallback = [];
row_n_valid = [];
row_first_valid_hop = [];
row_last_valid_hop = [];

for siz = 1 : size(S.predictions.lengths_to_lengths,2)
    for r = 1 : size(S.predictions.lengths_to_lengths,1)
        if isempty(S.predictions.lengths_to_lengths{r,siz})
            continue;
        end
        for g = 1 : length(S.predictions.lengths_to_lengths{r,siz}.(dataset_to_analyze))
            for m = model_indices
                [~, log_ratio] = fallback_graph_values(S, dataset_to_analyze, r, siz, g, m, baseline_idx);
                valid_rows = find(isfinite(log_ratio));
                status = 'ok';
                transition_hop = NaN;
                mean_before = NaN;
                mean_after = NaN;
                fallback_flag = false;
                first_valid_hop = NaN;
                last_valid_hop = NaN;
                if ~isempty(valid_rows)
                    first_valid_hop = valid_rows(1) - 1;
                    last_valid_hop = valid_rows(end) - 1;
                end

                if numel(valid_rows) < min_valid_hops
                    status = 'too_few_valid_hops';
                else
                    try
                        cp_local = findchangepts(log_ratio(valid_rows), ...
                            'Statistic', 'mean', 'MaxNumChanges', 1);
                        if isempty(cp_local)
                            status = 'no_changepoint';
                        else
                            transition_row = valid_rows(cp_local);
                            before_rows = valid_rows(valid_rows < transition_row);
                            after_rows = valid_rows(valid_rows >= transition_row);
                            if isempty(before_rows) || isempty(after_rows)
                                status = 'empty_segment';
                            else
                                transition_hop = transition_row - 1;
                                mean_before = mean(log_ratio(before_rows), 'omitnan');
                                mean_after = mean(log_ratio(after_rows), 'omitnan');
                                fallback_flag = mean_before <= -ratio_threshold_identity_function && ...
                                    mean_after >= ratio_threshold_identity_function;
                                if fallback_flag
                                    status = 'fallback';
                                else
                                    status = 'no_fallback';
                                end
                            end
                        end
                    catch ME
                        status = ['findchangepts_error: ', ME.identifier];
                    end
                end

                row_task{end+1,1} = 'lengths_to_lengths'; %#ok<AGROW>
                row_model{end+1,1} = all_models{m}; %#ok<AGROW>
                row_status{end+1,1} = status; %#ok<AGROW>
                row_size(end+1,1) = siz; %#ok<AGROW>
                row_seed(end+1,1) = r; %#ok<AGROW>
                row_graph(end+1,1) = g; %#ok<AGROW>
                row_transition_hop(end+1,1) = transition_hop; %#ok<AGROW>
                row_mean_before(end+1,1) = mean_before; %#ok<AGROW>
                row_mean_after(end+1,1) = mean_after; %#ok<AGROW>
                row_separation(end+1,1) = mean_after - mean_before; %#ok<AGROW>
                row_fallback(end+1,1) = fallback_flag; %#ok<AGROW>
                row_n_valid(end+1,1) = numel(valid_rows); %#ok<AGROW>
                row_first_valid_hop(end+1,1) = first_valid_hop; %#ok<AGROW>
                row_last_valid_hop(end+1,1) = last_valid_hop; %#ok<AGROW>
            end
        end
    end
end

fallback_table = table(row_task, row_model, row_size, row_seed, row_graph, ...
    row_transition_hop, row_mean_before, row_mean_after, row_separation, ...
    row_fallback, row_n_valid, row_first_valid_hop, row_last_valid_hop, ...
    row_status, ...
    'VariableNames', {'task','model','size_bin','seed_row','graph_idx', ...
    'transition_hop','mean_before','mean_after','separation', ...
    'fallback_flag','n_valid_hops','first_valid_hop','last_valid_hop', ...
    'status'});
writetable(fallback_table, fullfile(figures_output_dir, 'fallback_analysis_table.csv'));

fprintf('Fallback analysis table saved: %s\n', fullfile(figures_output_dir, 'fallback_analysis_table.csv'));
populated_size_bins = unique(row_size(isfinite(row_size)))';
for i = populated_size_bins
    to_take = fallback_table.size_bin == i & ismember(fallback_table.status, {'fallback','no_fallback'});
    fraction_of_graphs_with_fallback_behavior = nan(1, numel(model_indices));
    for mm = 1 : numel(model_indices)
        model_mask = strcmp(fallback_table.model, all_models{model_indices(mm)});
        denom = nnz(to_take & model_mask);
        if denom > 0
            fraction_of_graphs_with_fallback_behavior(mm) = nnz(to_take & model_mask & fallback_table.fallback_flag) / denom;
        end
    end
    fprintf('Fraction of graphs with fallback behavior (size_bin=%d, # cohorts=%d):\n', ...
        i, 2^(i-1));
    disp(all_models(model_indices)');
    disp(fraction_of_graphs_with_fallback_behavior);
end

% plotting the distribution of hops at which we had a clear fallback:
focus_rows = strcmp(fallback_table.model, focus_model) & fallback_table.fallback_flag;
if ~any(focus_rows)
    fallback_counts = zeros(1, numel(model_indices));
    for mm = 1 : numel(model_indices)
        fallback_counts(mm) = nnz(strcmp(fallback_table.model, all_models{model_indices(mm)}) & fallback_table.fallback_flag);
    end
    [best_count, best_mm] = max(fallback_counts);
    if best_count > 0
        old_focus_model = focus_model;
        focus_model = all_models{model_indices(best_mm)};
        focus_idx_mask = strcmp(all_models, focus_model);
        focus_rows = strcmp(fallback_table.model, focus_model) & fallback_table.fallback_flag;
        fprintf('No fallback examples found for focus model %s; using %s for example figures (%d fallback graphs).\n', ...
            old_focus_model, focus_model, best_count);
    else
        warning('DCG:noFallbackExamples', 'No fallback examples found for any model.');
        return;
    end
end
fig_hist = figure('Position', [300 300 430 430], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', [focus_model, ' fallback hop distribution']);
finite_transition_hops = fallback_table.transition_hop(focus_rows);
finite_transition_hops = finite_transition_hops(isfinite(finite_transition_hops));
finite_last_hops = fallback_table.last_valid_hop(isfinite(fallback_table.last_valid_hop));
max_hist_hop = max([finite_transition_hops; finite_last_hops; 1]);
histogram(finite_transition_hops, -0.5:1:(max_hist_hop + 0.5), 'Normalization', 'probability')
title(['Change hop in fallback graphs (n=', num2str(nnz(focus_rows)), ')']);
xlabel('Hop');
ylabel('Fraction');
title(sprintf('Distribution of fallback hops in %s predicted graphs', focus_model));
axis square;
xlim([-1 max_hist_hop + 1]);
set(gca, 'XTick', 0:2:max_hist_hop);
dcg_savefig_visible(fig_hist, fullfile(figures_output_dir, [focus_model, ' fallback hop distribution.fig']));

[~, local_example_idx] = min(fallback_table.separation(focus_rows));
focus_row_indices = find(focus_rows);
threshold_example_idx = focus_row_indices(local_example_idx);
siz = fallback_table.size_bin(threshold_example_idx);
r = fallback_table.seed_row(threshold_example_idx);
g = fallback_table.graph_idx(threshold_example_idx);
pt = fallback_table.transition_hop(threshold_example_idx) + 1;

% preparing the values of the example:
[vals, MAE_ratio] = fallback_graph_values(S, dataset_to_analyze, r, siz, g, find(focus_idx_mask), baseline_idx);

% showing the change point in the log ratio MAE:
fig_cp = figure('Position', [300 300 430 430], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', [focus_model, ' fallback example change point']);
hop_axis = (0:numel(MAE_ratio)-1)';
plot(hop_axis, MAE_ratio, '-o', 'LineWidth', 1.25);
hold on;
xline(pt - 1, '--k');
yline(-ratio_threshold_identity_function, ':');
yline(ratio_threshold_identity_function, ':');
before_fit = mean(MAE_ratio(1:pt-1), 'omitnan');
after_fit = mean(MAE_ratio(pt:end), 'omitnan');
plot(hop_axis(1:pt-1), before_fit * ones(pt-1,1), 'r-', 'LineWidth', 2);
plot(hop_axis(pt:end), after_fit * ones(numel(MAE_ratio)-pt+1,1), 'r-', 'LineWidth', 2);
hold off;
xlabel('Hop');
ylabel('FB(h) = log2(MAE(pred, post) / MAE(pred, pre))');
title({sprintf('%s fallback example: # cohorts=%d seed=%d graph=%d', focus_model, 2^(siz-1), r, g), ...
    sprintf('red fits: before = %.3g, after = %.3g', before_fit, after_fit)});
legend({'FB(h)', 'change hop', '-threshold', '+threshold', 'pre-change fit', 'post-change fit'}, ...
    'Location', 'best');
axis square;
set(gca, 'Box', 'off', 'TickDir', 'out', 'FontSize', 8, 'LineWidth', 0.75);

dcg_savefig_visible(fig_cp, fullfile(figures_output_dir, [focus_model, ' fallback example change point.fig']));

% showing the scatter plot comparing edge lengths:
fallback_fig_w = max(760, figure_panel_size(1) * 2 + 170);
fallback_fig_h = max(680, figure_panel_size(2) * 2 + 170);
fig_scatter = figure('Position', [100 100 fallback_fig_w fallback_fig_h], ...
    'DockControls', 'off', 'NumberTitle', 'off', ...
    'Name', [focus_model, ' fallback example scatter plot']);

before_cells = vals(1:pt-1);
before_cells = before_cells(~cellfun(@isempty, before_cells));
after_cells = vals(pt:end);
after_cells = after_cells(~cellfun(@isempty, after_cells));
if isempty(before_cells) || isempty(after_cells)
    warning('DCG:fallbackExampleEmpty', 'Selected fallback example has an empty before/after segment.');
    return;
end
example_before_pt = cell2mat(before_cells);

subplot(2,2,1);
plot([0;1.35], [0;1.35], 'Color', [0.5 0.5 0.5]);
hold on;
plot(example_before_pt(:,end), example_before_pt(:,focus_idx_mask), '.b', 'MarkerSize', scatter_marker_size);
hold off;
axis image;
xlim([0, 1.35]);
ylim([0, 1.35]);
xlabel('Post-T1 length');
ylabel('Predicted length');
title({['hop < ', num2str(pt)], ['avg. log ratio = ', num2str(mean(MAE_ratio(1:pt-1), 'omitnan'))], ...
    ['MAE = ', num2str(mean(abs(example_before_pt(:,focus_idx_mask) - example_before_pt(:,end))))]});
set(gca, 'FontSize', 8, 'LineWidth', 0.75, 'TickDir', 'out');

subplot(2,2,2);
plot([0;1.35], [0;1.35], 'Color', [0.5 0.5 0.5]);
hold on;
plot(example_before_pt(:,end-1), example_before_pt(:,focus_idx_mask), '.b', 'MarkerSize', scatter_marker_size);
hold off;
axis image;
xlim([0, 1.35]);
ylim([0, 1.35]);
xlabel('Pre-T1 length');
ylabel('Predicted length');
title({['hop < ', num2str(pt)], ['avg. log ratio = ', num2str(mean(MAE_ratio(1:pt-1), 'omitnan'))], ...
    ['MAE = ', num2str(mean(abs(example_before_pt(:,focus_idx_mask) - example_before_pt(:,end-1))))]});
set(gca, 'FontSize', 8, 'LineWidth', 0.75, 'TickDir', 'out');

example_after_pt = cell2mat(after_cells);

subplot(2,2,3);
plot([0;1.35], [0;1.35], 'Color', [0.5 0.5 0.5]);
hold on;
plot(example_after_pt(:,end), example_after_pt(:,focus_idx_mask), '.b', 'MarkerSize', scatter_marker_size);
hold off;
axis image;
xlim([0, 1.35]);
ylim([0, 1.35]);
xlabel('Post-T1 length');
ylabel('Predicted length');
title({['hop >= ', num2str(pt)], ['avg. log ratio = ', num2str(mean(MAE_ratio(pt:end), 'omitnan'))], ...
    ['MAE = ', num2str(mean(abs(example_after_pt(:,focus_idx_mask) - example_after_pt(:,end))))]});
set(gca, 'FontSize', 8, 'LineWidth', 0.75, 'TickDir', 'out');

subplot(2,2,4);
plot([0;1.35], [0;1.35], 'Color', [0.5 0.5 0.5]);
hold on;
plot(example_after_pt(:,end-1), example_after_pt(:,focus_idx_mask), '.b', 'MarkerSize', scatter_marker_size);
hold off;
axis image;
xlim([0, 1.35]);
ylim([0, 1.35]);
xlabel('Pre-T1 length');
ylabel('Predicted length');
title({['hop >= ', num2str(pt)], ['avg. log ratio = ', num2str(mean(MAE_ratio(pt:end), 'omitnan'))], ...
    ['MAE = ', num2str(mean(abs(example_after_pt(:,focus_idx_mask) - example_after_pt(:,end-1))))]});
set(gca, 'FontSize', 8, 'LineWidth', 0.75, 'TickDir', 'out');

dcg_savefig_visible(fig_scatter, fullfile(figures_output_dir, [focus_model, ' fallback example scatter plot.fig']));

end


function [vals_by_hop, log_ratio] = fallback_graph_values(S, dataset_to_analyze, r, siz, g, model_idx, baseline_idx)
% fallback_graph_values  Per-hop FB(h) log-ratio and stacked values for one graph.
%
% PURPOSE
%   Compute, for a single (seed r, size siz, graph g, model model_idx), the
%   per-hop fallback ratio FB(h) and the per-hop matrices used by the example
%   scatter plots. Shared by both the table-building loop and the example
%   rendering in plot_PPGN_fallback.
%
% INPUTS
%   S                 : results summary (uses S.predictions.lengths_to_lengths
%                       and S.ground_truth_dist for this r,siz).
%   dataset_to_analyze: split name.
%   r, siz, g         : seed-row, size-bin, graph indices.
%   model_idx         : column of the model whose ratio is computed.
%   baseline_idx      : column of the Baseline / pre-T1 lengths.
%
% OUTPUTS
%   vals_by_hop : n_hops x 1 cell; entry h = [pred_matrix(:, all models),
%                 post_T1(:)] for that hop (empty hops -> 0 x (n_models+1)).
%                 The scatter code reads predicted = col model_idx,
%                 post_T1 = last col, pre_T1 = second-to-last col.
%   log_ratio   : n_hops x 1 vector of FB(h); NaN where undefined.
%
% MATH
%   FB(h) = log2( mean|pred(:,model_idx) - post_T1| /
%                 mean|pred(:,model_idx) - pred(:,baseline_idx)| ),
%   evaluated only when the denominator > 0 and both means are finite.
%
% ALGORITHM
%   n_hops = max(#pred hops, #gt hops). Infer n_model_cols from the first
%   non-empty prediction cell. For each hop with non-empty pred and gt, stack
%   [pred, post] into vals_by_hop and compute FB(h). Finally set log_ratio(1)
%   (hop 0) to NaN to exclude it from change-point detection.
%
% EDGE CASES
%   Missing/empty hops yield an empty value block and leave log_ratio(h)=NaN.

pred_cells = S.predictions.lengths_to_lengths{r,siz}.(dataset_to_analyze){g};
gt_cells = S.ground_truth_dist{r,siz}.(dataset_to_analyze){g};
n_hops = max(numel(pred_cells), numel(gt_cells));
vals_by_hop = cell(n_hops, 1);
log_ratio = nan(n_hops, 1);
first_nonempty = find(~cellfun(@isempty, pred_cells), 1);
if isempty(first_nonempty)
    n_model_cols = 0;
else
    n_model_cols = size(pred_cells{first_nonempty}, 2);
end

for h = 1 : n_hops
    if h > numel(pred_cells) || h > numel(gt_cells) || isempty(pred_cells{h}) || isempty(gt_cells{h})
        vals_by_hop{h} = zeros(0, n_model_cols + 1);
        continue;
    end
    pred_mat = pred_cells{h};
    post_vals = gt_cells{h};
    vals_by_hop{h} = [pred_mat, post_vals];
    numerator = mean(abs(pred_mat(:,model_idx) - post_vals), 'omitnan');
    denominator = mean(abs(pred_mat(:,model_idx) - pred_mat(:,baseline_idx)), 'omitnan');
    if denominator > 0 && isfinite(numerator) && isfinite(denominator)
        log_ratio(h) = log2(numerator / denominator);
    end
end

if ~isempty(log_ratio)
    log_ratio(1) = NaN;
end

end


function S = calculate_normalization_factors(S, dataset_to_analyze, h_bins_for_quality_analysis, max_cell_dist)
% calculate_normalization_factors  Graph-weighted Baseline MAE denominators.
%
% PURPOSE
%   Compute the baseline (no-learning / identity) MAE used to normalize all
%   plots, AND build S.ground_truth_dist (post-T1 lengths re-split into per-hop
%   cells). The baseline MAE measures how far the pre-T1 lengths already are
%   from the true post-T1 lengths, i.e. the error of predicting "no change".
%
% INPUTS
%   S                          : results summary. Reads
%                                S.predictions.lengths_to_lengths (pre-T1 length
%                                is the LAST column, accessed as (:,end)) and
%                                S.ground_truth.lengths_to_lengths (post-T1
%                                lengths as a per-graph vector). Optionally reads
%                                S.hexagonality{siz}.(split).
%   dataset_to_analyze         : split name to normalize.
%   h_bins_for_quality_analysis: edges of the hexagonality bins (length B+1).
%   max_cell_dist              : maximum hop index to tabulate per_dist over.
%
% OUTPUTS (written into S)
%   S.normalization.(split).per_size          : 1 x nSizes scalar baseline MAE.
%   S.normalization.(split).per_dist{siz}      : max_cell_dist x 1 vector,
%                                                baseline MAE per hop.
%   S.normalization.(split).per_hexagonality{siz}: (B) x 1 vector, baseline MAE
%                                                per hexagonality bin.
%   S.ground_truth_dist{r,siz}.(split){g}      : per-hop cells of post-T1 length.
%
% MATH  (baseline MAE = mean | pre-T1  -  true post-T1 |)
%   per edge:  e = | pre_T1_length - post_T1_length |.
%   per graph (per_size): mean of e over ALL retained edges of the graph
%                         (all hops pooled) -> graph_mae_all(g).
%   per graph per hop (per_dist): mean of e over edges at that hop
%                                 -> graph_mae_by_hop(g,h).
%   AGGREGATION ORDER (graph-weighted, matching the model curves):
%     within graph (above) -> across graphs WITHIN a seed
%       per_size_seed(r) = mean_g graph_mae_all(g)
%       per_dist_seed(:,r) = mean_g graph_mae_by_hop(g,:)
%     -> across seeds
%       per_size(siz)  = mean_r per_size_seed(r)
%       per_dist{siz}  = mean_r per_dist_seed(:,r)
%   per_hexagonality: same, but graphs are grouped per hexagonality bin
%     (bin b keeps graphs with hex in (edge_b, edge_{b+1}]) before the
%     within-seed mean and the across-seed mean.
%
% ALGORITHM
%   For each size bin and seed: re-split the post-T1 vector into per-hop cells
%   matching the per-hop row counts of the prediction cells (skipping graphs
%   whose total edge count does not match), store into S.ground_truth_dist,
%   accumulate per-graph and per-hop baseline errors, then form the seed means
%   and finally the across-seed means. Scalars pass through
%   sanitize_normalization_scalar and vectors through
%   sanitize_normalization_vector (NaN-out degenerate denominators).
%
% EDGE CASES
%   Graphs with empty predictions, empty/length-mismatched post vectors, or
%   missing hexagonality are skipped (their slots become NaN / {}), so an empty
%   bin never produces a spurious finite denominator.
%

% Baseline normalization factors are computed with the same weighting used
% by the plotted MAE summaries: interface MAE is averaged within graph, then
% graphs are averaged within seed/repetition, then seed means are averaged.
% This avoids mixing an edge-weighted denominator with graph-weighted model
% curves, and avoids the older per-hop max(pre/post) denominator.
S.normalization.(dataset_to_analyze).per_size = nan(1,size(S.predictions.lengths_to_lengths,2));
S.normalization.(dataset_to_analyze).per_dist = cell(1,size(S.predictions.lengths_to_lengths,2));
S.normalization.(dataset_to_analyze).per_hexagonality = cell(1,size(S.predictions.lengths_to_lengths,2));
S.ground_truth_dist = cell(size(S.predictions.lengths_to_lengths));

for siz = 1 : size(S.predictions.lengths_to_lengths,2)

    per_size_seed = nan(size(S.predictions.lengths_to_lengths,1), 1);
    per_dist_seed = nan(max_cell_dist, size(S.predictions.lengths_to_lengths,1));
    per_hex_seed = nan(length(h_bins_for_quality_analysis)-1, size(S.predictions.lengths_to_lengths,1));

    for r = 1 : size(S.predictions.lengths_to_lengths,1)
        if isempty(S.predictions.lengths_to_lengths{r,siz})
            continue;
        end

        if isempty(S.ground_truth.lengths_to_lengths{r,siz})
            continue;
        end

        n_graphs = length(S.ground_truth.lengths_to_lengths{r,siz}.(dataset_to_analyze));
        S.ground_truth_dist{r,siz}.(dataset_to_analyze) = cell(n_graphs,1);
        graph_mae_all = nan(n_graphs, 1);
        graph_mae_by_hop = nan(n_graphs, max_cell_dist);

        if isfield(S, 'hexagonality') && numel(S.hexagonality) >= siz && ...
                ~isempty(S.hexagonality{siz}) && isfield(S.hexagonality{siz}, dataset_to_analyze)
            graph_hexagonality = S.hexagonality{siz}.(dataset_to_analyze);
        else
            graph_hexagonality = nan(n_graphs, 1);
        end

        for g = 1 : length(S.ground_truth.lengths_to_lengths{r,siz}.(dataset_to_analyze))
            pred_cells = S.predictions.lengths_to_lengths{r,siz}.(dataset_to_analyze){g};
            if isempty(pred_cells)
                S.ground_truth_dist{r,siz}.(dataset_to_analyze){g} = {};
                continue;
            end
            curr_siz = cellfun(@(x) size(x,1), pred_cells);
            post_vector = S.ground_truth.lengths_to_lengths{r,siz}.(dataset_to_analyze){g};
            if isempty(post_vector) || sum(curr_siz) ~= numel(post_vector)
                S.ground_truth_dist{r,siz}.(dataset_to_analyze){g} = {};
                continue;
            end
            gt_cells = mat2cell(post_vector, curr_siz, 1);
            S.ground_truth_dist{r,siz}.(dataset_to_analyze){g} = gt_cells;

            all_graph_baseline_errors = [];
            n_hops = min([max_cell_dist, numel(pred_cells), numel(gt_cells)]);
            for h = 1 : n_hops
                if isempty(pred_cells{h}) || isempty(gt_cells{h}) || size(pred_cells{h},2) < 1
                    continue;
                end
                pre_vals = pred_cells{h}(:,end);
                post_vals = gt_cells{h};
                edge_errors = abs(pre_vals - post_vals);
                graph_mae_by_hop(g,h) = mean(edge_errors, 'omitnan');
                all_graph_baseline_errors = [all_graph_baseline_errors; edge_errors(:)]; %#ok<AGROW>
            end
            graph_mae_all(g) = mean(all_graph_baseline_errors, 'omitnan');
        end

        per_size_seed(r) = mean(graph_mae_all, 'omitnan');
        per_dist_seed(:,r) = mean(graph_mae_by_hop, 1, 'omitnan')';

        for b = 1 : length(h_bins_for_quality_analysis)-1
            curr_graphs = graph_hexagonality > h_bins_for_quality_analysis(b) & ...
                graph_hexagonality <= h_bins_for_quality_analysis(b+1);
            per_hex_seed(b,r) = mean(graph_mae_all(curr_graphs), 'omitnan');
        end
    end

    S.normalization.(dataset_to_analyze).per_size(siz) = sanitize_normalization_scalar(mean(per_size_seed, 'omitnan'));
    S.normalization.(dataset_to_analyze).per_dist{siz} = sanitize_normalization_vector(mean(per_dist_seed, 2, 'omitnan'));
    S.normalization.(dataset_to_analyze).per_hexagonality{siz} = sanitize_normalization_vector(mean(per_hex_seed, 2, 'omitnan'));
end

end


function out = sanitize_normalization_scalar(x)
% sanitize_normalization_scalar  Guard a scalar normalization denominator.
%
% PURPOSE
%   Prevent division-by-(near)-zero spikes in normalized plots by rejecting a
%   degenerate scalar baseline MAE.
%
% INPUT
%   x   : candidate scalar denominator.
%
% OUTPUT
%   out : NaN if x is empty, non-finite, or x <= eps; otherwise x unchanged.
%
% DECISION
%   The threshold is MATLAB eps (~2.2e-16); anything at or below it (including
%   exact 0 and negatives) is treated as unusable -> NaN.

if isempty(x) || ~isfinite(x) || x <= eps
    out = NaN;
else
    out = x;
end

end


function out = sanitize_normalization_vector(x)
% sanitize_normalization_vector  Element-wise guard for a normalization vector.
%
% PURPOSE
%   Vector form of sanitize_normalization_scalar: NaN-out individual
%   per-hop / per-hexagonality denominators that would divide unsafely.
%
% INPUT
%   x   : numeric vector of candidate denominators.
%
% OUTPUT
%   out : copy of x with every element that is non-finite OR <= eps set to NaN.
%
% DECISION
%   Same eps threshold as the scalar version, applied per element so good bins
%   survive even when others are empty.

out = x;
out(~isfinite(out) | out <= eps) = NaN;

end


function calc_total_number_of_graphs(all_models, tasks, S, dataset_to_analyze, n_edges_per_graph)
% calc_total_number_of_graphs  Print per-model graph (and edge) counts.
%
% PURPOSE
%   Diagnostic: report, per model and task, how many graphs (and implied edges)
%   contributed at each (seed_row, size_bin), so coverage can be sanity-checked.
%
% INPUTS
%   all_models        : cellstr of model names.
%   tasks             : cellstr of task names.
%   S                 : results summary (reads S.prediction_errors).
%   dataset_to_analyze: split name.
%   n_edges_per_graph : assumed edges per graph (used only for the edge total).
%
% OUTPUTS
%   none (prints count matrices to the command window).
%
% ALGORITHM
%   For each model m and task: allocate [seeds x sizes] count matrices; a graph
%   counts for model m when its first hop cell has a non-NaN value in column m
%   (i.e. the model actually produced a prediction there). Multiply by
%   n_edges_per_graph for the edge total. disp() the populated tasks only.
%
% NOTE
%   total_n_edges is computed but only the graph counts are displayed.

total_n_graphs = struct;

for m = 1 : length(all_models)
    for t = 1 : length(tasks)

        total_repetitions = size(S.prediction_errors.(tasks{t}),1);
        total_sizes = size(S.prediction_errors.(tasks{t}),2);

        total_n_graphs.(tasks{t}) = zeros(total_repetitions, total_sizes);
        total_n_edges.(tasks{t}) = zeros(total_repetitions, total_sizes);

        for repetition = 1 : total_repetitions
            for siz = 1 : total_sizes
                if isempty(S.prediction_errors.(tasks{t}){repetition,siz})
                    continue;
                end
                n_graphs = sum(cellfun(@(x) ~isnan(x{1}(1,m)), S.prediction_errors.(tasks{t}){repetition,siz}.(dataset_to_analyze)));
                total_n_graphs.(tasks{t})(repetition, siz) = n_graphs;
                total_n_edges.(tasks{t})(repetition, siz) = n_graphs * n_edges_per_graph;
            end
        end
    end

    fprintf(['*** ', all_models{m}, ' ***', newline]);
    fprintf(newline);
    % Only disp tasks that were populated by the loop above (so we don't
    % crash on single-task datasets like the revision data).
    for t = 1 : length(tasks)
        disp(['Total graphs in "', tasks{t}, '":']);
        disp(total_n_graphs.(tasks{t}));
        fprintf(newline);
    end
end

end


function plot_hexagonality_distribution(S, h_bins_for_initial_histogram, figures_output_dir, dataset_to_analyze)
% plot_hexagonality_distribution  Histogram of per-graph hexagonality. [DEAD/UNUSED]
%
% *** DEAD / UNUSED ***
%   This function is NOT called by the live pipeline (the hexagonality
%   distribution plot is disabled/superseded by the manuscript hex panels).
%   Retained for reference only; do not assume it runs.
%
% PURPOSE (as written)
%   Plot the probability histogram of graph hexagonality for the last size bin
%   of a split and save it as 'Hexagonality distribution.fig'.
%
% INPUTS
%   S                            : reads S.hexagonality{end}.(split).
%   h_bins_for_initial_histogram : histogram bin edges.
%   figures_output_dir           : output folder.
%   dataset_to_analyze           : split name (defaults to 'test' if missing).
%
% OUTPUTS
%   none (saves a .fig).
%
% ALGORITHM
%   histogram(all_hexagonalities, edges, Normalization='probability'); title
%   shows n, mean and SD; square axes; save via dcg_savefig_visible.

if nargin < 4 || isempty(dataset_to_analyze)
    dataset_to_analyze = 'test';
end
all_hexagonalities = S.hexagonality{end}.(dataset_to_analyze);
figure;
histogram(all_hexagonalities, h_bins_for_initial_histogram, 'Normalization', 'probability');
title({['Graph hexagonality distribution (n = ', num2str(length(all_hexagonalities)), ')'], ...
    ['(avg. = ', num2str(mean(all_hexagonalities)), ', SD = ', num2str(std(all_hexagonalities)), ')']});
axis square;

dcg_savefig_visible(fullfile(figures_output_dir, 'Hexagonality distribution.fig'));

end


function plot_manuscript_hexagonality_panels_from_cache(analyses_filename, figures_output_dir, colors, hex_bins, uncertainty_mode, n_cells, split_name, save_separate_panels, left_panel_cfg, models_to_exclude)
% plot_manuscript_hexagonality_panels_from_cache  Two-panel hex figure from an analyses cache.
%
% PURPOSE
%   Build the manuscript hexagonality figure from a cached analyses .mat (the
%   struct I), rather than from S. Panel a: graph hexagonality vs active noise.
%   Panel b: identity-normalized error (nMAE_Id) vs hexagonality per model.
%   Also writes the assumptions TXT and two data CSVs.
%
% INPUTS
%   analyses_filename : .mat holding struct I (panel-b + default panel-a source).
%   figures_output_dir: output folder.
%   colors            : ignored/overwritten (recomputed via paper_model_colors).
%   hex_bins          : hexagonality bin edges for panel b.
%   uncertainty_mode  : 'sd' or 'sem' for the shaded bands.
%   n_cells           : cells per tissue (used to histogram cell degree).
%   split_name        : MUST be 'test' (enforced downstream).
%   save_separate_panels: if true, also save each panel as its own .fig.
%   left_panel_cfg    : struct selecting the panel-a source (file/size_bin/
%                       seed_col/model/task/splits); defaulted if empty.
%   models_to_exclude : optional model names to drop from panel b.
%
% OUTPUTS
%   none returned. Saves 'Manuscript hexagonality panels.fig' (+ optional
%   separate panels), 'manuscript_hexagonality_assumptions.txt', and two CSVs.
%
% ALGORITHM
%   Load I; list model fields (drop no_learning + excluded), order for paper,
%   append 'Baseline', recolor. Determine test graph indices via
%   hex_cache_split_indices. Panel a quantities from hex_left_panel_graph_quality
%   + summarize_by_x; panel b from hex_cache_identity_normalized_error. Write
%   assumptions TXT, draw the tiled 1x2 figure (draw_manuscript_hex_noise +
%   draw_manuscript_hex_error), and export the underlying numbers to CSV.
%
% MATH (panel b nMAE_Id) -- see hex_cache_identity_normalized_error
%   Per graph/model MAE = mean|pred post-T1 - true post-T1|; Baseline MAE from
%   the pre-T1 column; nMAE_Id = log2(model mean-graph MAE / Baseline mean-graph
%   MAE), computed per seed and bin, then averaged across seeds.
%
% EDGE CASES
%   If exclusions empty the model list, warns and skips panel b. Empty hex bins
%   stay NaN in the CSV (curves just bridge finite bins visually).

loaded = load(analyses_filename, 'I');
I = loaded.I;
model_fields = fieldnames(I)';
model_fields = model_fields(~strcmp(model_fields, 'no_learning'));
if nargin < 10 || isempty(models_to_exclude)
    models_to_exclude = {};
end
models_to_exclude = cellstr(models_to_exclude);
if ~isempty(models_to_exclude)
    model_fields = model_fields(~ismember(model_fields, models_to_exclude));
end
model_fields = order_models_for_paper(model_fields, false);
if isempty(model_fields)
    warning('DCG:hexCacheNoModelsAfterExclusion', ...
        'No cached model fields remain after applying models_to_exclude; skipping manuscript hexagonality panel b.');
    return;
end
all_models_for_plot = [model_fields, {'Baseline'}];
colors = paper_model_colors(all_models_for_plot);
if nargin < 7 || isempty(split_name)
    split_name = 'test';
end
if nargin < 8 || isempty(save_separate_panels)
    save_separate_panels = false;
end
if nargin < 9 || isempty(left_panel_cfg)
    left_panel_cfg = struct( ...
        'analyses_filename', analyses_filename, ...
        'size_bin', 1, ...
        'seed_col', 1, ...
        'model', model_fields{end}, ...
        'task', 'lengths_to_lengths', ...
        'splits', {{'all'}});
end

task = 'lengths_to_lengths';
subset_idx = 1;
n_graphs = numel(I.(model_fields{1}).(task).vals{subset_idx,1});
n_seeds = size(I.(model_fields{1}).(task).vals, 2);
test_graph_indices = hex_cache_split_indices(I, model_fields{end}, task, subset_idx, split_name);

[active_noise_all, graph_hexagonality_all, left_meta] = hex_left_panel_graph_quality(left_panel_cfg, n_cells);
[~, graph_hexagonality_test] = hex_cache_graph_quality(I, model_fields{end}, task, subset_idx, test_graph_indices, n_cells);
[x_noise, y_noise, e_noise, n_noise] = summarize_by_x(active_noise_all, graph_hexagonality_all, uncertainty_mode);
[x_hex, y_hex, e_hex, n_hex] = hex_cache_identity_normalized_error(I, model_fields, task, subset_idx, test_graph_indices, graph_hexagonality_test, hex_bins, uncertainty_mode);

assumptions_filename = fullfile(figures_output_dir, 'manuscript_hexagonality_assumptions.txt');
fid = fopen(assumptions_filename, 'w');
fprintf(fid, 'Manuscript hexagonality figure assumptions\n');
fprintf(fid, '1. Panel a source: %s\n', left_panel_cfg.analyses_filename);
fprintf(fid, '2. Panel a uses the full v1_W 32-cohort graph set: size_bin=%d (# cohorts=%d), seed_col=%d, model/task=%s/%s.\n', ...
    left_panel_cfg.size_bin, 2^(left_panel_cfg.size_bin-1), left_panel_cfg.seed_col, left_panel_cfg.model, left_panel_cfg.task);
fprintf(fid, '3. Panel a split union: %s; graph positions used=%d; train=%d, val=%d, test=%d.\n', ...
    strjoin(left_meta.splits, ', '), left_meta.n_graphs, left_meta.n_train, left_meta.n_val, left_meta.n_test);
fprintf(fid, '4. Panel a computes graph hexagonality from the raw edge list as fraction of cells with degree 6.\n');
fprintf(fid, '5. Panel a active noise comes from graph_id.disorder; values >1 are divided by 100.\n');
fprintf(fid, '6. Panel b source: %s\n', analyses_filename);
fprintf(fid, '7. Panel b uses split="%s", %d graph positions out of %d total cached hex positions, for all model-performance quantities.\n', split_name, numel(test_graph_indices), n_graphs);
fprintf(fid, '8. Panel b uses task lengths_to_lengths for %d seeds and the same split="%s" graph positions.\n', n_seeds, split_name);
fprintf(fid, '9. Per graph/model MAE is mean(abs(predicted post-T1 length - true post-T1 length)) over retained interfaces.\n');
fprintf(fid, '10. Baseline is the raw pre-T1 length from column 3 of the PNA matrix; nMAE_Id = log2(model mean graph MAE / Baseline mean graph MAE).\n');
fprintf(fid, '11. nMAE_Id is computed separately per seed and hexagonality bin before averaging seeds.\n');
fprintf(fid, '12. Shaded bands use "%s" across graph samples in panel a and across seeds in panel b.\n', uncertainty_mode);
fprintf(fid, '13. Simulation-box insets from the original manuscript are not recreated here.\n');
fprintf(fid, '14. Empty hexagonality bins remain NaN in the exported CSV; plotted curves connect adjacent finite bins for visual continuity only.\n');
fclose(fid);

if ~save_separate_panels
    remove_manuscript_hex_separate_panel_files(figures_output_dir);
end

if save_separate_panels
    fig_a = figure('Position', [300 300 430 390]);
    ax_a = axes(fig_a);
    draw_manuscript_hex_noise(ax_a, x_noise, y_noise, e_noise);
    dcg_savefig_visible(fig_a, fullfile(figures_output_dir, 'Manuscript hexagonality vs active noise.fig'));

    fig_b = figure('Position', [760 300 430 390]);
    ax_b = axes(fig_b);
    draw_manuscript_hex_error(ax_b, x_hex, y_hex, e_hex, colors, all_models_for_plot);
    dcg_savefig_visible(fig_b, fullfile(figures_output_dir, 'Manuscript identity-normalized error vs hexagonality.fig'));
end

fig = figure('Position', [200 200 900 390]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax_a = nexttile(tl, 1);
draw_manuscript_hex_noise(ax_a, x_noise, y_noise, e_noise);
ax_b = nexttile(tl, 2);
draw_manuscript_hex_error(ax_b, x_hex, y_hex, e_hex, colors, all_models_for_plot);
dcg_savefig_visible(fig, fullfile(figures_output_dir, 'Manuscript hexagonality panels.fig'));

T_noise = table(x_noise(:), y_noise(:), e_noise(:), n_noise(:), ...
    'VariableNames', {'active_noise','hexagonality_mean','hexagonality_error','n_graphs'});
writetable(T_noise, fullfile(figures_output_dir, 'manuscript_hexagonality_vs_active_noise_data.csv'));

rows_x = [];
rows_model = {};
rows_mean = [];
rows_error = [];
rows_n = [];
for b = 1 : numel(x_hex)
    for m = 1 : numel(all_models_for_plot)
        rows_x(end+1,1) = x_hex(b); %#ok<AGROW>
        rows_model{end+1,1} = all_models_for_plot{m}; %#ok<AGROW>
        rows_mean(end+1,1) = y_hex(b,m); %#ok<AGROW>
        rows_error(end+1,1) = e_hex(b,m); %#ok<AGROW>
        rows_n(end+1,1) = n_hex(b,m); %#ok<AGROW>
    end
end
T_hex = table(rows_x, rows_model, rows_mean, rows_error, rows_n, ...
    'VariableNames', {'hexagonality','model','nMAE_Id_mean','nMAE_Id_error','n_seeds'});
writetable(T_hex, fullfile(figures_output_dir, 'manuscript_identity_normalized_error_vs_hexagonality_data.csv'));

end


function graph_indices = hex_cache_split_indices(I, model_field, task, subset_idx, split_name)
% hex_cache_split_indices  Fetch the cached test-split graph indices.
%
% PURPOSE
%   Return the graph positions belonging to the test split from the analyses
%   cache I, with a hard guard that only the test split is permitted for the
%   manuscript hexagonality panels.
%
% INPUTS
%   I          : analyses cache struct.
%   model_field: model field name to read indices from.
%   task       : task field name.
%   subset_idx : size-bin / subset index into the cached cells.
%   split_name : split name; MUST equal 'test'.
%
% OUTPUT
%   graph_indices : column vector of graph indices for the test split.
%
% DECISIONS / EDGE CASES
%   Errors (DCG:testOnlyRequired) if split_name is not 'test'; errors
%   (DCG:hexSplitMissing) if the split is absent from the cache.

if ~strcmp(split_name, 'test')
    error('DCG:testOnlyRequired', ...
        'Manuscript hexagonality panels must use the test split. Current value: %s', split_name);
end
if ~isfield(I.(model_field).(task).inds{subset_idx}, split_name)
    error('DCG:hexSplitMissing', 'Split "%s" not found in the hex analyses cache.', split_name);
end
graph_indices = I.(model_field).(task).inds{subset_idx}.(split_name)(:);

end


function remove_manuscript_hex_separate_panel_files(figures_output_dir)
% remove_manuscript_hex_separate_panel_files  Delete the separate hex panel files.
%
% PURPOSE
%   When the combined (tiled) hex figure is preferred, remove any previously
%   saved standalone panel files so they do not linger.
%
% INPUTS
%   figures_output_dir : output folder.
%
% OUTPUTS
%   none (deletes the four named .fig/.png files if present).
%
% ALGORITHM
%   Iterate the fixed list of standalone panel filenames and delete each that
%   exists.

stale_files = {
    'Manuscript hexagonality vs active noise.fig'
    'Manuscript hexagonality vs active noise.png'
    'Manuscript identity-normalized error vs hexagonality.fig'
    'Manuscript identity-normalized error vs hexagonality.png'
    };

for i = 1 : numel(stale_files)
    stale_path = fullfile(figures_output_dir, stale_files{i});
    if isfile(stale_path)
        delete(stale_path);
    end
end

end


function [active_noise, graph_hexagonality, meta] = hex_left_panel_graph_quality(left_panel_cfg, n_cells)
% hex_left_panel_graph_quality  Per-graph active noise + hexagonality for panel a.
%
% PURPOSE
%   From a (possibly different) analyses cache, gather each graph's active-noise
%   parameter and its hexagonality, for the hexagonality-vs-noise panel a.
%
% INPUTS
%   left_panel_cfg : struct with fields analyses_filename, model, task,
%                    size_bin, seed_col, splits ('all' or named splits).
%   n_cells        : cells per tissue (degree histogram upper edge).
%
% OUTPUTS
%   active_noise        : Ng x 1 active-noise per selected graph.
%   graph_hexagonality  : Ng x 1 fraction of degree-6 cells per graph.
%   meta                : struct (splits used, n_graphs, n_train/n_val/n_test).
%
% MATH
%   hexagonality(g) = mean(cell_degree == 6), where cell_degree comes from
%   histcounting the cell ids in columns 1:2 of the graph's edge matrix over
%   bins 0.5:1:(n_cells+0.5). active_noise = graph_id.disorder(g), divided by
%   100 if the max exceeds 1 (percent -> fraction).
%
% ALGORITHM
%   Load I; assert the requested model/task/size_bin/seed_col exist. Select
%   graph indices = all graphs ('all') or the union of the named splits
%   (deduplicated). For each valid (non-NaN) graph matrix compute hexagonality
%   and read its disorder; rescale noise; assemble meta with per-split counts.
%
% EDGE CASES
%   Graphs with empty or NaN-sentinel matrices are left NaN.

loaded_left = load(left_panel_cfg.analyses_filename, 'I');
I_left = loaded_left.I;

model = left_panel_cfg.model;
task = left_panel_cfg.task;
size_bin = left_panel_cfg.size_bin;
seed_col = left_panel_cfg.seed_col;
splits = left_panel_cfg.splits;
if ischar(splits) || isstring(splits)
    splits = cellstr(splits);
end

assert(isfield(I_left, model), 'Panel-a source missing model "%s": %s', model, left_panel_cfg.analyses_filename);
assert(isfield(I_left.(model), task), 'Panel-a source missing task "%s" for model "%s": %s', task, model, left_panel_cfg.analyses_filename);
assert(size_bin <= size(I_left.(model).(task).vals, 1), 'Panel-a size_bin=%d exceeds cached vals rows.', size_bin);
assert(seed_col <= size(I_left.(model).(task).vals, 2), 'Panel-a seed_col=%d exceeds cached vals columns.', seed_col);

all_vals = I_left.(model).(task).vals{size_bin, seed_col};
graph_id = I_left.(model).(task).graph_id{size_bin, seed_col};
inds = I_left.(model).(task).inds{size_bin, seed_col};

graph_indices = [];
split_counts = struct('train', 0, 'val', 0, 'test', 0);
if any(strcmpi(splits, 'all'))
    graph_indices = (1:numel(all_vals))';
    meta_splits = {'all'};
else
    meta_splits = splits(:)';
    for s = 1:numel(splits)
        split_name = char(splits{s});
        assert(isfield(inds, split_name), 'Panel-a source missing split "%s".', split_name);
        curr = inds.(split_name)(:);
        graph_indices = [graph_indices; curr]; %#ok<AGROW>
        if isfield(split_counts, split_name)
            split_counts.(split_name) = numel(curr);
        end
    end
    graph_indices = unique(graph_indices);
end

active_noise = nan(numel(graph_indices), 1);
graph_hexagonality = nan(numel(graph_indices), 1);
for ii = 1:numel(graph_indices)
    g = graph_indices(ii);
    graph_mat = all_vals{g};
    if isempty(graph_mat) || isnan(graph_mat(1))
        continue;
    end
    cell_degree = histcounts(reshape(graph_mat(:,1:2), [], 1), 0.5:1:(n_cells+0.5))';
    graph_hexagonality(ii) = mean(cell_degree == 6);
    active_noise(ii) = graph_id.disorder(g);
end

if max(active_noise, [], 'omitnan') > 1
    active_noise = active_noise ./ 100;
end

meta = struct();
meta.splits = meta_splits;
meta.n_graphs = numel(graph_indices);
meta.n_train = split_counts.train;
meta.n_val = split_counts.val;
meta.n_test = split_counts.test;

end


function [active_noise, graph_hexagonality] = hex_cache_graph_quality(I, model_field, task, subset_idx, graph_indices, n_cells)
% hex_cache_graph_quality  Active noise + hexagonality for a given index set.
%
% PURPOSE
%   Like hex_left_panel_graph_quality but reads an already-loaded cache I at a
%   fixed (model, task, subset, seed=1) and a supplied list of graph indices.
%   Used to characterize the test-split graphs for panel b.
%
% INPUTS
%   I            : analyses cache struct (already in memory).
%   model_field  : model field to read from.
%   task         : task field.
%   subset_idx   : size-bin / subset index (uses seed column 1).
%   graph_indices: indices of graphs to evaluate.
%   n_cells      : cells per tissue (degree histogram upper edge).
%
% OUTPUTS
%   active_noise       : numel(graph_indices) x 1 active noise.
%   graph_hexagonality : numel(graph_indices) x 1 fraction of degree-6 cells.
%
% MATH / ALGORITHM
%   Same as hex_left_panel_graph_quality: hexagonality = mean(cell_degree==6)
%   from columns 1:2; active_noise = graph_id.disorder(g), divided by 100 if
%   max > 1. Empty/NaN graphs stay NaN.

active_noise = nan(numel(graph_indices), 1);
graph_hexagonality = nan(numel(graph_indices), 1);
for ii = 1 : numel(graph_indices)
    g = graph_indices(ii);
    graph_mat = I.(model_field).(task).vals{subset_idx,1}{g};
    if isempty(graph_mat) || isnan(graph_mat(1))
        continue;
    end
    cell_degree = histcounts(reshape(graph_mat(:,1:2), [], 1), 0.5:1:(n_cells+0.5))';
    graph_hexagonality(ii) = mean(cell_degree == 6);
    active_noise(ii) = I.(model_field).(task).graph_id{subset_idx}.disorder(g);
end
if max(active_noise, [], 'omitnan') > 1
    active_noise = active_noise ./ 100;
end

end


function [x, y, e, n] = summarize_by_x(x_raw, y_raw, uncertainty_mode)
% summarize_by_x  Group y by unique x and return mean, spread, and count.
%
% PURPOSE
%   Collapse scattered (x,y) samples onto the unique x values (rounded), giving
%   the mean y, an uncertainty band, and the sample count per x. Used to draw
%   panel a (hexagonality vs active noise) as a single aggregated curve.
%
% INPUTS
%   x_raw, y_raw     : equal-length sample vectors.
%   uncertainty_mode : 'sd' or 'sem' (passed to uncertainty_from_values).
%
% OUTPUTS
%   x : sorted unique x (rounded to 6 decimals).
%   y : mean of y within each x group (omitnan).
%   e : uncertainty (SD or SEM) within each group.
%   n : number of finite samples per group.
%
% ALGORITHM
%   Keep finite (x,y) pairs; round x to 6 dp; for each unique rounded x take the
%   matching y's and compute mean / uncertainty / count.

valid = isfinite(x_raw) & isfinite(y_raw);
x_raw = x_raw(valid);
y_raw = y_raw(valid);
x = unique(round(x_raw, 6));
y = nan(size(x));
e = nan(size(x));
n = nan(size(x));
rounded_x = round(x_raw, 6);
for i = 1 : numel(x)
    vals = y_raw(rounded_x == x(i));
    y(i) = mean(vals, 'omitnan');
    e(i) = uncertainty_from_values(vals, uncertainty_mode);
    n(i) = nnz(isfinite(vals));
end

end


function [x, y, e, n] = hex_cache_identity_normalized_error(I, model_fields, task, subset_idx, graph_indices, graph_hexagonality, hex_bins, uncertainty_mode)
% hex_cache_identity_normalized_error  nMAE_Id vs hexagonality from a cache (panel b).
%
% PURPOSE
%   Compute the identity-baseline-normalized error nMAE_Id for each model as a
%   function of graph hexagonality, from the analyses cache I, across seeds.
%
% INPUTS
%   I                 : analyses cache struct.
%   model_fields      : cellstr of model field names (Baseline added as the
%                       last, (numel+1)-th, column internally).
%   task              : task field.
%   subset_idx        : size-bin / subset index.
%   graph_indices     : graph indices corresponding to graph_hexagonality.
%   graph_hexagonality: per-graph hexagonality used to bin graphs.
%   hex_bins          : hexagonality bin edges (length nBins+1).
%   uncertainty_mode  : 'sd' or 'sem'.
%
% OUTPUTS
%   x : bin centers (nBins x 1).
%   y : nBins x (nModels+1) mean nMAE_Id (last column = Baseline ~ 0).
%   e : nBins x (nModels+1) uncertainty across seeds.
%   n : nBins x (nModels+1) finite-seed counts.
%
% MATH
%   For each seed s and hex bin b: take graphs in (hex_bins(b),hex_bins(b+1)].
%   The reference matrix is model_fields{end}'s cached vals; post_T1 = its
%   column end-1, pre_T1 = its column 3. Per graph:
%     model MAE   = mean|model_pred(:,end) - post_T1|,
%     Baseline MAE= mean|pre_T1 - post_T1|.
%   Average those graph MAEs within the bin, then
%     nMAE_Id(s,b,:) = log2( mean-graph model MAE / mean-graph Baseline MAE ),
%   only when the Baseline MAE > 0 and finite. Finally average over seeds.
%
% EDGE CASES
%   Empty bins / missing graph matrices contribute NaN; bins with no valid
%   Baseline MAE stay NaN for that seed.

n_models = numel(model_fields) + 1;
n_seeds = size(I.(model_fields{1}).(task).vals, 2);
n_bins = numel(hex_bins) - 1;
x = (hex_bins(1:end-1) + hex_bins(2:end))' / 2;
seed_log = nan(n_seeds, n_bins, n_models);

for s = 1 : n_seeds
    for b = 1 : n_bins
        local_graph_idx = find(graph_hexagonality > hex_bins(b) & graph_hexagonality <= hex_bins(b+1));
        if isempty(local_graph_idx)
            continue;
        end

        graph_mae = nan(numel(local_graph_idx), n_models);
        for gg = 1 : numel(local_graph_idx)
            g = graph_indices(local_graph_idx(gg));
            ref_mat = I.(model_fields{end}).(task).vals{subset_idx,s}{g};
            if isempty(ref_mat) || isnan(ref_mat(1))
                continue;
            end
            post_t1 = ref_mat(:,end-1);
            pre_t1 = ref_mat(:,3);
            for m = 1 : numel(model_fields)
                curr_mat = I.(model_fields{m}).(task).vals{subset_idx,s}{g};
                if isempty(curr_mat) || isnan(curr_mat(1))
                    continue;
                end
                graph_mae(gg,m) = mean(abs(curr_mat(:,end) - post_t1), 'omitnan');
            end
            graph_mae(gg,end) = mean(abs(pre_t1 - post_t1), 'omitnan');
        end

        baseline_mae = mean(graph_mae(:,end), 'omitnan');
        model_mae = mean(graph_mae, 1, 'omitnan');
        if baseline_mae > 0 && isfinite(baseline_mae)
            seed_log(s,b,:) = log2(model_mae ./ baseline_mae);
        end
    end
end

y = nan(n_bins, n_models);
e = nan(n_bins, n_models);
n = zeros(n_bins, n_models);
for b = 1 : n_bins
    for m = 1 : n_models
        vals = seed_log(:,b,m);
        y(b,m) = mean(vals, 'omitnan');
        e(b,m) = uncertainty_from_values(vals, uncertainty_mode);
        n(b,m) = nnz(isfinite(vals));
    end
end

end


function plot_manuscript_hexagonality_panels(S, all_models, dataset_to_analyze, figures_output_dir, colors, hex_bins, active_noise_bins, size_bin, uncertainty_mode, save_separate_panels)
% plot_manuscript_hexagonality_panels  Two-panel hex figure computed from S.
%
% PURPOSE
%   S-based counterpart of plot_manuscript_hexagonality_panels_from_cache.
%   Panel a: hexagonality vs active noise (from S.disorder / S.hexagonality).
%   Panel b: nMAE_Id vs hexagonality per model (from S.prediction_errors).
%   Writes the assumptions TXT and the two data CSVs.
%
% INPUTS
%   S                 : results summary.
%   all_models        : cellstr of model names.
%   dataset_to_analyze: split name.
%   figures_output_dir: output folder.
%   colors            : per-model colors (recomputed inside the draw helper).
%   hex_bins          : hexagonality bin edges for panel b.
%   active_noise_bins : noise bin edges for panel a (only used if noise levels
%                       are too numerous to plot one point per level).
%   size_bin          : size bin to use; auto-selected via
%                       select_hex_paper_size_bin when empty.
%   uncertainty_mode  : 'sd' (default) or 'sem'.
%   save_separate_panels: also save each panel standalone when true.
%
% OUTPUTS
%   none returned. Saves the tiled figure (when both panels have data),
%   'manuscript_hexagonality_assumptions.txt', and up to two CSVs.
%
% ALGORITHM
%   Resolve size_bin and uncertainty_mode defaults; write the assumptions TXT;
%   compute panel-a numbers via manuscript_hex_noise_curve and panel-b numbers
%   via manuscript_hex_error_curve; draw/save figures guarded on data presence;
%   export the underlying numbers to CSV.
%
% MATH (panel b)  -- see manuscript_hex_error_curve
%   nMAE_Id = log2(model mean-graph MAE / Baseline mean-graph MAE), per seed and
%   hex bin, then averaged across seeds.

if nargin < 8 || isempty(size_bin)
    size_bin = select_hex_paper_size_bin(S, all_models, dataset_to_analyze);
    if isempty(size_bin)
        warning('DCG:hexPaperNoSize', 'No populated lengths_to_lengths size bins; skipping manuscript hexagonality panels.');
        return;
    end
end
if nargin < 9 || isempty(uncertainty_mode)
    uncertainty_mode = 'sd';
end
if nargin < 10 || isempty(save_separate_panels)
    save_separate_panels = false;
end

assumptions_filename = fullfile(figures_output_dir, 'manuscript_hexagonality_assumptions.txt');
fid = fopen(assumptions_filename, 'w');
fprintf(fid, 'Manuscript hexagonality figure assumptions\n');
fprintf(fid, '1. Panel a uses S.disorder and S.hexagonality from split "%s" at size_bin=%d (# cohorts=%d).\n', dataset_to_analyze, size_bin, 2^(size_bin-1));
fprintf(fid, '2. Panel b uses task lengths_to_lengths and dataset split "%s" at size_bin=%d (# cohorts=%d).\n', dataset_to_analyze, size_bin, 2^(size_bin-1));
fprintf(fid, '3. Panel b graph MAE is averaged over retained interfaces within graph, then graphs are averaged within each hexagonality bin.\n');
fprintf(fid, '4. nMAE_Id = log2(model mean graph MAE / Baseline mean graph MAE), computed separately per seed and hexagonality bin before averaging seeds.\n');
fprintf(fid, '5. Shaded bands use "%s" across graph samples in panel a and across seeds in panel b.\n', uncertainty_mode);
fprintf(fid, '6. Simulation-box insets from the original manuscript are not recreated here.\n');
fprintf(fid, '7. Empty hexagonality bins remain NaN in the exported CSV; plotted curves connect adjacent finite bins for visual continuity only.\n');
fclose(fid);

if ~save_separate_panels
    remove_manuscript_hex_separate_panel_files(figures_output_dir);
end

[x_noise, y_noise, e_noise, n_noise] = manuscript_hex_noise_curve(S, dataset_to_analyze, size_bin, active_noise_bins, uncertainty_mode);
[x_hex, y_hex, e_hex, n_hex] = manuscript_hex_error_curve(S, all_models, dataset_to_analyze, size_bin, hex_bins, uncertainty_mode);

if save_separate_panels && any(isfinite(y_noise))
    fig_a = figure('Position', [300 300 430 390]);
    ax_a = axes(fig_a);
    draw_manuscript_hex_noise(ax_a, x_noise, y_noise, e_noise);
    dcg_savefig_visible(fig_a, fullfile(figures_output_dir, 'Manuscript hexagonality vs active noise.fig'));
end

if save_separate_panels && any(isfinite(y_hex(:)))
    fig_b = figure('Position', [760 300 430 390]);
    ax_b = axes(fig_b);
    draw_manuscript_hex_error(ax_b, x_hex, y_hex, e_hex, colors, all_models);
    dcg_savefig_visible(fig_b, fullfile(figures_output_dir, 'Manuscript identity-normalized error vs hexagonality.fig'));
end

if any(isfinite(y_noise)) && any(isfinite(y_hex(:)))
    fig = figure('Position', [200 200 900 390]);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax_a = nexttile(tl, 1);
    draw_manuscript_hex_noise(ax_a, x_noise, y_noise, e_noise);
    title(ax_a, 'Hexagonality vs. active noise');
    ax_b = nexttile(tl, 2);
    draw_manuscript_hex_error(ax_b, x_hex, y_hex, e_hex, colors, all_models);
    title(ax_b, {'Identity-baseline-normalized', 'error vs. hexagonality'});
    dcg_savefig_visible(fig, fullfile(figures_output_dir, 'Manuscript hexagonality panels.fig'));
end

if any(isfinite(y_noise))
    T_noise = table(x_noise(:), y_noise(:), e_noise(:), n_noise(:), ...
        'VariableNames', {'active_noise','hexagonality_mean','hexagonality_error','n_graphs'});
    writetable(T_noise, fullfile(figures_output_dir, 'manuscript_hexagonality_vs_active_noise_data.csv'));
end

if any(isfinite(y_hex(:)))
    rows_x = [];
    rows_model = {};
    rows_mean = [];
    rows_error = [];
    rows_n = [];
    for b = 1 : numel(x_hex)
        for m = 1 : numel(all_models)
            rows_x(end+1,1) = x_hex(b); %#ok<AGROW>
            rows_model{end+1,1} = all_models{m}; %#ok<AGROW>
            rows_mean(end+1,1) = y_hex(b,m); %#ok<AGROW>
            rows_error(end+1,1) = e_hex(b,m); %#ok<AGROW>
            rows_n(end+1,1) = n_hex(b,m); %#ok<AGROW>
        end
    end
    T_hex = table(rows_x, rows_model, rows_mean, rows_error, rows_n, ...
        'VariableNames', {'hexagonality','model','nMAE_Id_mean','nMAE_Id_error','n_seeds'});
    writetable(T_hex, fullfile(figures_output_dir, 'manuscript_identity_normalized_error_vs_hexagonality_data.csv'));
end

end


function size_bin = select_hex_paper_size_bin(S, all_models, dataset_to_analyze)
% select_hex_paper_size_bin  Pick the largest size bin with all models present.
%
% PURPOSE
%   Choose which size bin the manuscript hexagonality panels should use:
%   prefer the largest populated bin in which every model has finite data.
%
% INPUTS
%   S                 : results summary (reads S.prediction_errors.lengths_to_lengths).
%   all_models        : cellstr of model names that must all be present.
%   dataset_to_analyze: split name.
%
% OUTPUT
%   size_bin : selected size-bin index, or [] if none qualifies.
%
% ALGORITHM
%   Scan populated size bins from largest to smallest; accumulate which models
%   show any finite value across that bin's graphs/hops; return the first
%   (largest) bin where all models are covered.
%
% EDGE CASES
%   Returns [] when no bin satisfies the all-models condition (caller warns and
%   skips the panels).

size_bin = [];
populated_sizes = find(any(~cellfun(@isempty, S.prediction_errors.lengths_to_lengths), 1));
for siz = fliplr(populated_sizes)
    has_model = false(1, numel(all_models));
    for r = 1 : size(S.prediction_errors.lengths_to_lengths, 1)
        if isempty(S.prediction_errors.lengths_to_lengths{r,siz})
            continue;
        end
        graphs = S.prediction_errors.lengths_to_lengths{r,siz}.(dataset_to_analyze);
        for g = 1 : numel(graphs)
            if isempty(graphs{g})
                continue;
            end
            hop_cells = graphs{g};
            nonempty = ~cellfun(@isempty, hop_cells);
            if ~any(nonempty)
                continue;
            end
            vals = cell2mat(hop_cells(nonempty));
            has_model = has_model | any(isfinite(vals), 1);
            if all(has_model)
                size_bin = siz;
                return;
            end
        end
    end
end

end


function [x, y, e, n] = manuscript_hex_noise_curve(S, dataset_to_analyze, size_bin, active_noise_bins, uncertainty_mode)
% manuscript_hex_noise_curve  Hexagonality vs active noise (panel a) from S.
%
% PURPOSE
%   Aggregate per-graph (active noise, hexagonality) from S into a curve of
%   mean hexagonality vs noise, either one point per distinct noise level or,
%   if there are too many levels, binned by active_noise_bins.
%
% INPUTS
%   S                 : results summary (reads S.disorder{size_bin} and
%                       S.hexagonality{size_bin}).
%   dataset_to_analyze: split name.
%   size_bin          : size bin to read.
%   active_noise_bins : noise bin edges (used only in the >80-levels branch).
%   uncertainty_mode  : 'sd' or 'sem'.
%
% OUTPUTS
%   x : noise levels or bin centers.
%   y : mean hexagonality at each x.
%   e : SD/SEM of hexagonality at each x.
%   n : finite-sample count at each x.
%
% ALGORITHM
%   Gather noise/hex vectors for the split (rescale noise /100 if max>1); keep
%   finite pairs. If <=80 unique noise levels, average hexagonality per exact
%   level; else bin by active_noise_bins (last bin right-closed) and average
%   within each bin.
%
% EDGE CASES
%   Returns empty arrays when the size bin, disorder, or hexagonality data are
%   missing or all non-finite.

x = [];
y = [];
e = [];
n = [];
if ~isfield(S, 'disorder') || numel(S.disorder) < size_bin || isempty(S.disorder{size_bin}) || ...
        numel(S.hexagonality) < size_bin || isempty(S.hexagonality{size_bin})
    return;
end

noise_struct = S.disorder{size_bin};
hex_struct = S.hexagonality{size_bin};
split_names = {dataset_to_analyze};
all_noise = [];
all_hex = [];
for s = 1 : numel(split_names)
    split = split_names{s};
    if isfield(noise_struct, split) && isfield(hex_struct, split)
        all_noise = [all_noise; noise_struct.(split)(:)]; %#ok<AGROW>
        all_hex = [all_hex; hex_struct.(split)(:)]; %#ok<AGROW>
    end
end
if isempty(all_noise) && isfield(noise_struct, dataset_to_analyze) && isfield(hex_struct, dataset_to_analyze)
    all_noise = noise_struct.(dataset_to_analyze)(:);
    all_hex = hex_struct.(dataset_to_analyze)(:);
end

valid = isfinite(all_noise) & isfinite(all_hex);
all_noise = all_noise(valid);
all_hex = all_hex(valid);
if isempty(all_noise)
    return;
end
if max(all_noise) > 1
    all_noise = all_noise ./ 100;
end

rounded_noise = round(all_noise, 6);
noise_levels = unique(rounded_noise);
if numel(noise_levels) <= 80
    x = noise_levels(:);
    y = nan(size(x));
    e = nan(size(x));
    n = nan(size(x));
    for i = 1 : numel(x)
        vals = all_hex(rounded_noise == x(i));
        y(i) = mean(vals, 'omitnan');
        e(i) = uncertainty_from_values(vals, uncertainty_mode);
        n(i) = nnz(isfinite(vals));
    end
else
    x = (active_noise_bins(1:end-1) + active_noise_bins(2:end))' / 2;
    y = nan(size(x));
    e = nan(size(x));
    n = nan(size(x));
    for i = 1 : numel(x)
        if i == numel(x)
            take = all_noise >= active_noise_bins(i) & all_noise <= active_noise_bins(i+1);
        else
            take = all_noise >= active_noise_bins(i) & all_noise < active_noise_bins(i+1);
        end
        vals = all_hex(take);
        y(i) = mean(vals, 'omitnan');
        e(i) = uncertainty_from_values(vals, uncertainty_mode);
        n(i) = nnz(isfinite(vals));
    end
end

end


function [x, y, e, n] = manuscript_hex_error_curve(S, all_models, dataset_to_analyze, size_bin, hex_bins, uncertainty_mode)
% manuscript_hex_error_curve  nMAE_Id vs hexagonality (panel b) from S.
%
% PURPOSE
%   S-based version of hex_cache_identity_normalized_error: identity-normalized
%   error per model vs hexagonality, using the per-edge prediction errors in S
%   rather than a cache.
%
% INPUTS
%   S                 : results summary (reads S.prediction_errors.lengths_to_lengths
%                       and S.hexagonality{size_bin}.(split)).
%   all_models        : cellstr of model names; Baseline column located by name
%                       (else last column).
%   dataset_to_analyze: split name.
%   size_bin          : size bin to read.
%   hex_bins          : hexagonality bin edges (length nBins+1).
%   uncertainty_mode  : 'sd' or 'sem'.
%
% OUTPUTS
%   x : hex bin centers (nBins x 1).
%   y : nBins x nModels mean nMAE_Id.
%   e : nBins x nModels uncertainty across seeds.
%   n : nBins x nModels finite-seed counts.
%
% MATH
%   Per seed r and bin b: for each graph in the bin, graph MAE per model =
%   mean over all (pooled) hop rows of S.prediction_errors (mean|pred-post_T1|).
%   Average graph MAEs within the bin to get model_mae and baseline_mae (the
%   Baseline column). nMAE_Id(r,b,:) = log2(model_mae ./ baseline_mae) when
%   baseline_mae > 0 and finite. Then average across seeds.
%
% EDGE CASES
%   Missing hexagonality data returns all-NaN outputs of the right shape.

baseline_idx = find(strcmp(all_models, 'Baseline') | strcmp(all_models, 'no_learning'), 1);
if isempty(baseline_idx)
    baseline_idx = numel(all_models);
end

n_bins = numel(hex_bins) - 1;
n_models = numel(all_models);
n_reps = size(S.prediction_errors.lengths_to_lengths, 1);
x = (hex_bins(1:end-1) + hex_bins(2:end))' / 2;
seed_log = nan(n_reps, n_bins, n_models);

if numel(S.hexagonality) < size_bin || isempty(S.hexagonality{size_bin}) || ...
        ~isfield(S.hexagonality{size_bin}, dataset_to_analyze)
    y = nan(n_bins, n_models);
    e = nan(n_bins, n_models);
    n = zeros(n_bins, n_models);
    return;
end

hex_vals = S.hexagonality{size_bin}.(dataset_to_analyze)(:);

for r = 1 : n_reps
    if isempty(S.prediction_errors.lengths_to_lengths{r,size_bin})
        continue;
    end

    graphs = S.prediction_errors.lengths_to_lengths{r,size_bin}.(dataset_to_analyze);
    for b = 1 : n_bins
        graph_idx = find(hex_vals > hex_bins(b) & hex_vals <= hex_bins(b+1));
        if isempty(graph_idx)
            continue;
        end

        graph_mae = nan(numel(graph_idx), n_models);
        for gg = 1 : numel(graph_idx)
            g = graph_idx(gg);
            if g > numel(graphs) || isempty(graphs{g})
                continue;
            end
            hop_cells = graphs{g};
            nonempty = ~cellfun(@isempty, hop_cells);
            if ~any(nonempty)
                continue;
            end
            vals = cell2mat(hop_cells(nonempty));
            graph_mae(gg,:) = mean(vals, 1, 'omitnan');
        end

        baseline_mae = mean(graph_mae(:,baseline_idx), 'omitnan');
        model_mae = mean(graph_mae, 1, 'omitnan');
        if baseline_mae > 0 && isfinite(baseline_mae)
            seed_log(r,b,:) = log2(model_mae ./ baseline_mae);
        end
    end
end

y = nan(n_bins, n_models);
e = nan(n_bins, n_models);
n = zeros(n_bins, n_models);
for b = 1 : n_bins
    for m = 1 : n_models
        vals = seed_log(:,b,m);
        y(b,m) = mean(vals, 'omitnan');
        e(b,m) = uncertainty_from_values(vals, uncertainty_mode);
        n(b,m) = nnz(isfinite(vals));
    end
end

end


function draw_manuscript_hex_noise(ax, x, y, e)
% draw_manuscript_hex_noise  Render panel a (hexagonality vs active noise).
%
% PURPOSE
%   Draw the single grey shaded-error curve of hexagonality vs active noise
%   into the given axes with the manuscript styling.
%
% INPUTS
%   ax      : target axes handle.
%   x, y, e : noise levels, mean hexagonality, and its uncertainty band.
%
% OUTPUTS
%   none (draws into ax).
%
% ALGORITHM / DECISIONS
%   plot_shaded_line in grey; box off, ticks out; xlim [0 0.5], ylim [0 1];
%   square axes; labeled "Active noise (sigma)" vs "Hexagonality".

plot_shaded_line(ax, x, y, e, [0.35 0.35 0.35], '-');
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 8, 'LineWidth', 0.75);
xlabel(ax, 'Active noise (\sigma)', 'FontSize', 9);
ylabel(ax, 'Hexagonality', 'FontSize', 9);
title(ax, 'Hexagonality vs. active noise', 'FontSize', 10, 'FontWeight', 'bold');
xlim(ax, [0, 0.5]);
ylim(ax, [0, 1]);
axis(ax, 'square');

end


function draw_manuscript_hex_error(ax, x, y, e, colors, all_models)
% draw_manuscript_hex_error  Render panel b (nMAE_Id vs hexagonality, all models).
%
% PURPOSE
%   Draw one shaded-error curve per model of nMAE_Id vs hexagonality, in paper
%   color and order, with a legend, into the given axes.
%
% INPUTS
%   ax         : target axes handle.
%   x          : hexagonality bin centers.
%   y, e       : nBins x nModels means and uncertainty bands.
%   colors     : ignored (recomputed via paper_model_colors(all_models)).
%   all_models : cellstr of model names (drives color, order, legend).
%
% OUTPUTS
%   none (draws into ax).
%
% ALGORITHM / DECISIONS
%   Recompute colors; locate baseline_idx (unused beyond color); order curves
%   via paper_model_plot_order(...,true); plot each via plot_shaded_line; add a
%   legend (underscores -> spaces). Style: box off, ticks out, xlim [0.4 1];
%   ylim auto-fit but at least [-2.2, 4]; square axes.

colors = paper_model_colors(all_models);
baseline_idx = find(strcmp(all_models, 'Baseline') | strcmp(all_models, 'no_learning'), 1);
if isempty(baseline_idx)
    baseline_idx = numel(all_models);
end
plot_order = paper_model_plot_order(all_models, true);
hold(ax, 'on');
line_handles = gobjects(numel(plot_order), 1);
for ii = 1 : numel(plot_order)
    m = plot_order(ii);
    line_handles(ii) = plot_shaded_line(ax, x, y(:,m), e(:,m), colors(m,:), '-');
end
hold(ax, 'off');
legend(ax, line_handles, regexprep(all_models(plot_order), '_', ' '), 'Location', 'northeast', 'FontSize', 8);
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 8, 'LineWidth', 0.75);
xlabel(ax, 'Hexagonality', 'FontSize', 9);
ylabel(ax, 'nMAE_{Id}', 'FontSize', 9);
title(ax, {'Identity-baseline-normalized', 'error vs. hexagonality'}, 'FontSize', 10, 'FontWeight', 'bold');
xlim(ax, [0.4, 1]);
valid = isfinite(y) & isfinite(e);
if any(valid(:))
    y_low = min(y(valid) - e(valid));
    y_high = max(y(valid) + e(valid));
    ylim(ax, [min(-2.2, y_low - 0.1), max(4, y_high + 0.1)]);
end
axis(ax, 'square');

end


function h_line = plot_shaded_line(ax, x, y, e, color, line_style)
% plot_shaded_line  Line plot with a translucent +/- e shaded band.
%
% PURPOSE
%   Draw y vs x as a colored line with a semi-transparent band spanning
%   [y-e, y+e], using only the finite samples.
%
% INPUTS
%   ax        : target axes.
%   x, y, e   : equal-length vectors (band half-width e).
%   color     : 1x3 RGB for line and band.
%   line_style: optional line spec (default '-').
%
% OUTPUT
%   h_line : handle to the drawn line (for legends).
%
% ALGORITHM / EDGE CASES
%   Keep finite (x,y,e); if any, fill the band (FaceAlpha 0.18, no edge,
%   HandleVisibility off) then plot the line on top. If nothing is finite, plot
%   a NaN placeholder so a legend entry still exists.

if nargin < 6 || isempty(line_style)
    line_style = '-';
end
x = x(:);
y = y(:);
e = e(:);
valid = isfinite(x) & isfinite(y) & isfinite(e);
if any(valid)
    xv = x(valid);
    yv = y(valid);
    ev = e(valid);
    fill(ax, [xv; flipud(xv)], [yv - ev; flipud(yv + ev)], color, ...
        'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    hold(ax, 'on');
    h_line = plot(ax, xv, yv, line_style, 'Color', color, 'LineWidth', 1.35);
else
    h_line = plot(ax, NaN, NaN, line_style, 'Color', color, 'LineWidth', 1.35);
end

end


function e = uncertainty_from_values(vals, uncertainty_mode)
% uncertainty_from_values  SD or SEM of a sample vector (ignoring NaNs).
%
% PURPOSE
%   Central helper that turns a set of values into the error-band half-width
%   used by the hexagonality curves.
%
% INPUTS
%   vals             : numeric vector (NaNs dropped).
%   uncertainty_mode : 'sd' -> standard deviation; 'sem' -> SD/sqrt(N).
%
% OUTPUT
%   e : the requested spread, or NaN if vals has no finite entries.
%
% MATH / DECISIONS
%   sd = std(vals, 0, 'omitnan') (sample SD, normalized by N-1). For 'sem',
%   e = sd / sqrt(numel finite vals). Unknown modes warn and fall back to SD.

vals = vals(isfinite(vals));
if isempty(vals)
    e = NaN;
    return;
end
sd = std(vals, 0, 'omitnan');
switch lower(char(uncertainty_mode))
    case 'sem'
        e = sd / sqrt(numel(vals));
    case 'sd'
        e = sd;
    otherwise
        warning('DCG:unknownUncertainty', 'Unknown uncertainty "%s"; using SD.', char(uncertainty_mode));
        e = sd;
end

end


function plot_scatter_plot_examples(scores_to_show, tasks, MAE_individuals, S, figures_output_dir, all_models, dataset_to_analyze, figure_panel_size, scatter_marker_size)
% plot_scatter_plot_examples  Predicted-vs-true scatter examples at chosen percentiles.
%
% PURPOSE
%   For each task, pick example graphs at requested error percentiles and show,
%   for each, a 2-row panel: top = a model's predicted vs true post-T1 lengths,
%   bottom = the Baseline (pre-T1) vs true post-T1 lengths, each titled with its
%   MAE. Lets readers see what low/median/high-error predictions look like.
%
% INPUTS
%   scores_to_show     : percentiles (0-100) of the per-graph/model MAE pool to
%                        illustrate (e.g. [10 50 90]).
%   tasks              : cellstr of task names.
%   MAE_individuals    : per-(seed,size) {graphs x models} MAE cells (from
%                        extract_MAEs) used to build and rank the score pool.
%   S                  : results summary (for the actual length values).
%   figures_output_dir : output folder.
%   all_models         : cellstr of model names.
%   dataset_to_analyze : split name.
%   figure_panel_size  : [w h] px per panel (optional).
%   scatter_marker_size: marker size (optional).
%
% OUTPUTS
%   none returned. Saves 'Scatter plot examples (<task>).fig' per task.
%
% ALGORITHM
%   Locate baseline_idx and the non-baseline model_indices. Pool every finite
%   per-graph/model MAE (excluding Baseline) with its (r,siz,g,m) provenance and
%   sort ascending. For each requested percentile, map to a rank index, recover
%   (r,siz,g,m), gather the predicted/baseline lengths from S over the valid
%   hops, and draw the two scatter subplots against the y=x reference line.
%   Logs each example's percentile/score/model/cohorts/seed/graph/disorder.
%
% DECISIONS / EDGE CASES
%   Percentile->rank uses round(pct*N/100) clamped to [1,N]. Examples missing
%   the needed model column, or with no valid hops, are skipped with a warning.
%   Tasks with no finite scores are skipped.

if nargin < 8 || isempty(figure_panel_size)
    figure_panel_size = [340, 300];
end
if nargin < 9 || isempty(scatter_marker_size)
    scatter_marker_size = 9;
end

baseline_idx = find(strcmp(all_models, 'Baseline') | strcmp(all_models, 'no_learning'), 1);
if isempty(baseline_idx)
    baseline_idx = length(all_models);
end
model_indices = setdiff(1:length(all_models), baseline_idx, 'stable');

for t = 1 : length(tasks)
    task = tasks{t};
    scores = [];
    reps = [];
    sizes = [];
    graphs = [];
    models = [];

    for r = 1 : size(MAE_individuals.(task), 1)
        for siz = 1 : size(MAE_individuals.(task), 2)
            if isempty(MAE_individuals.(task){r,siz})
                continue;
            end
            curr_mae = MAE_individuals.(task){r,siz};
            curr_model_indices = model_indices(model_indices <= size(curr_mae, 2));
            for m = curr_model_indices
                valid_graphs = find(isfinite(curr_mae(:,m)));
                scores = [scores; curr_mae(valid_graphs,m)]; %#ok<AGROW>
                reps = [reps; repmat(r, numel(valid_graphs), 1)]; %#ok<AGROW>
                sizes = [sizes; repmat(siz, numel(valid_graphs), 1)]; %#ok<AGROW>
                graphs = [graphs; valid_graphs(:)]; %#ok<AGROW>
                models = [models; repmat(m, numel(valid_graphs), 1)]; %#ok<AGROW>
            end
        end
    end

    if isempty(scores)
        warning('DCG:scatterNoScores', 'No finite graph/model MAE scores available for task %s.', task);
        continue;
    end

    [scores_sorted, order] = sort(scores, 'ascend');
    fig_w = max(760, figure_panel_size(1) * numel(scores_to_show) + 220);
    fig_h = max(620, figure_panel_size(2) * 2 + 170);
    fid = figure('Position', [100 100 fig_w fig_h], 'DockControls', 'off', ...
        'NumberTitle', 'off', 'Name', ['Scatter examples ', task]);

    for i = 1 : length(scores_to_show)
        rank_i = max(1, min(numel(scores_sorted), round(scores_to_show(i) * numel(scores_sorted) / 100)));
        rec_i = order(rank_i);
        r = reps(rec_i);
        siz = sizes(rec_i);
        g = graphs(rec_i);
        m = models(rec_i);

        pred_cells = S.predictions.(task){r,siz}.(dataset_to_analyze){g};
        gt_cells = S.ground_truth.(task){r,siz}.(dataset_to_analyze){g};
        if ~iscell(gt_cells)
            gt_cells = mat2cell(gt_cells, cellfun(@(x) size(x,1), pred_cells), 1);
        end
        n_hops = min(numel(pred_cells), numel(gt_cells));
        pred_cells = pred_cells(1:n_hops);
        gt_cells = gt_cells(1:n_hops);
        valid_curr_hops = ~cellfun(@isempty, pred_cells) & ~cellfun(@isempty, gt_cells) & ...
            cellfun(@(x) size(x,2) >= m, pred_cells);
        if ~any(valid_curr_hops)
            warning('DCG:scatterMissingModelColumn', ...
                'Skipping scatter example for task=%s model=%s: prediction column %d is unavailable.', ...
                task, all_models{m}, m);
            continue;
        end
        curr_prediction = cell2mat(cellfun(@(x) x(:,m), pred_cells(valid_curr_hops), 'UniformOutput', false));
        curr_ground_truth = cell2mat(gt_cells(valid_curr_hops));
        valid_baseline_hops = ~cellfun(@isempty, pred_cells) & ~cellfun(@isempty, gt_cells) & ...
            baseline_idx <= numel(all_models) & cellfun(@(x) size(x,2) >= baseline_idx, pred_cells);
        if any(valid_baseline_hops)
            baseline_prediction = cell2mat(cellfun(@(x) x(:,baseline_idx), pred_cells(valid_baseline_hops), 'UniformOutput', false));
            baseline_ground_truth = cell2mat(gt_cells(valid_baseline_hops));
        else
            baseline_prediction = [];
            baseline_ground_truth = [];
        end
        curr_disorder = scatter_graph_metadata_value(S, 'disorder', dataset_to_analyze, r, siz, g);

        fprintf('Scatter example %s: percentile=%g score=%g model=%s cohorts=%d seed_row=%d graph=%d disorder=%g\n', ...
            task, scores_to_show(i), scores_sorted(rank_i), all_models{m}, 2^(siz-1), r, g, curr_disorder);

        subplot(2, length(scores_to_show), i);
        plot([0,1.3], [0,1.3], 'Color', [0.5 0.5 0.5]);
        hold on;
        plot(curr_ground_truth, curr_prediction, '.b', 'MarkerSize', scatter_marker_size);
        hold off;
        axis image;
        xlim([0, 1.3]);
        ylim([0, 1.3]);
        xlabel('Post-T1 length (ground truth)');
        ylabel('Predicted Lengths');
        if scores_to_show(i) >= 90
            score_label = [num2str(scores_to_show(i)), 'th percentile (high-error) example'];
        elseif scores_to_show(i) <= 10
            score_label = [num2str(scores_to_show(i)), 'th percentile (low-error) example'];
        else
            score_label = [num2str(scores_to_show(i)), 'th percentile example'];
        end

        title({score_label, ...
            ['MAE = ', num2str(mean(abs(curr_prediction - curr_ground_truth), 'omitnan'))], ...
            ['Model = ', all_models{m}, ', # cohorts = ', num2str(2^(siz-1)), ', seed row = ', num2str(r)]});

        subplot(2, length(scores_to_show), i + length(scores_to_show));
        plot([0,1.3], [0,1.3], 'Color', [0.5 0.5 0.5]);
        hold on;
        if ~isempty(baseline_prediction)
            plot(baseline_ground_truth, baseline_prediction, '.b', 'MarkerSize', scatter_marker_size);
        end
        hold off;
        axis image;
        xlim([0, 1.3]);
        ylim([0, 1.3]);
        xlabel('Post-T1 length (ground truth)');
        ylabel('Pre-T1 Lengths (baseline)');
        if ~isempty(baseline_prediction)
            baseline_mae = mean(abs(baseline_prediction - baseline_ground_truth), 'omitnan');
            baseline_title = ['MAE = ', num2str(baseline_mae)];
        else
            baseline_title = 'Baseline column unavailable for this task';
        end
        title({strrep(score_label, 'example', 'baseline'), baseline_title, ...
            ['Model = ', all_models{baseline_idx}, ', graph = ', num2str(g), ', disorder = ', num2str(curr_disorder)]});
    end

    dcg_savefig_visible(fid, fullfile(figures_output_dir, ['Scatter plot examples (', task, ').fig']));
end

end


function val = scatter_graph_metadata_value(S, field_name, dataset_to_analyze, r, siz, g)
% scatter_graph_metadata_value  Safely fetch one per-graph metadata scalar from S.
%
% PURPOSE
%   Look up a single graph's metadata value (e.g. its 'disorder') for the
%   scatter-example titles, tolerating both the collapsed per-size layout and
%   the older per-(seed,size) layout.
%
% INPUTS
%   S                 : results summary.
%   field_name        : metadata field (e.g. 'disorder').
%   dataset_to_analyze: split name.
%   r, siz, g         : seed-row, size-bin, graph indices.
%
% OUTPUT
%   val : the requested scalar, or NaN if unavailable.
%
% ALGORITHM
%   If the field is absent, return NaN. Prefer S.field{r,siz} when both dims
%   exist; otherwise fall back to S.field{siz}. Then index .(split)(g) when the
%   entry is a struct with that split and is long enough.
%
% EDGE CASES
%   Any missing layer (field/entry/split/index) yields NaN rather than an error.

val = nan;
if ~isfield(S, field_name)
    return;
end

metadata_cells = S.(field_name);
metadata_entry = [];

% Most metadata are size-specific and were collapsed above to S.field(size).
% Older summaries may still retain repetition-by-size cells.
if size(metadata_cells, 1) >= r && size(metadata_cells, 2) >= siz
    metadata_entry = metadata_cells{r, siz};
elseif numel(metadata_cells) >= siz
    metadata_entry = metadata_cells{siz};
end

if isempty(metadata_entry) || ~isstruct(metadata_entry) || ~isfield(metadata_entry, dataset_to_analyze)
    return;
end

metadata_vals = metadata_entry.(dataset_to_analyze);
if numel(metadata_vals) >= g
    val = metadata_vals(g);
end

end


function [MAE_individuals, MAE_size_avg, MAE_size_sd, MAE_hexagonality_avg, MAE_hexagonality_sd, MAE_dists_avg, MAE_dists_sd, S] = ...
    extract_MAEs(tasks, S, all_models, dataset_to_analyze, max_cell_dist, h_bins_for_quality_analysis, use_log, normalized)
% extract_MAEs  Reduce per-edge errors in S into all the MAE summaries to plot.
%
% PURPOSE
%   The core reduction. Converts S.prediction_errors (per-edge |pred-truth| by
%   model) into: per-graph MAEs; MAE vs dataset size; MAE vs hop distance; and
%   MAE vs hexagonality, each with a matching spread (SD across seeds). All
%   model curves are GRAPH-weighted (edges averaged within a graph first).
%
% INPUTS
%   tasks                      : cellstr of task names.
%   S                          : results summary (reads/extends
%                                S.prediction_errors and S.predictions;
%                                uses S.hexagonality for the hex split).
%   all_models                 : cellstr of model names (column order).
%   dataset_to_analyze         : split name.
%   max_cell_dist              : maximum hop index; per-graph hop cells are
%                                right-padded with empty matrices to this length.
%   h_bins_for_quality_analysis: hexagonality bin edges (length B+1).
%   use_log                    : if truthy, plot log2 of the MAEs. WHERE the
%                                log2 is applied depends on `normalized` (below).
%   normalized                 : 0 (default) for the raw / log2(MAE) passes; 1 for
%                                the log2(nMAE) passes. Controls graph-level log2
%                                placement: MAE stays ARITHMETIC (log2 AFTER the
%                                graph-mean), nMAE is GEOMETRIC (log2 PER graph,
%                                then averaged). See MATH / AGGREGATION ORDER.
%
% OUTPUTS
%   MAE_individuals.(task){r,siz} : nGraphs x nModels per-graph MAE (log2 if
%                                   use_log). Edge -> graph reduction only.
%   MAE_size_avg.(task)           : nSizes x nModels mean over seeds of the
%                                   per-seed graph-mean MAE.
%   MAE_size_sd.(task)            : matching SD ACROSS SEEDS, std(...,0) i.e.
%                                   normalized by N-1 (sample SD; fixed 2026-06-01).
%   MAE_dists_avg.(task){siz}     : nHops x nModels graph-weighted MAE per hop.
%   MAE_dists_sd.(task){siz}      : matching SD ACROSS SEEDS, std(...,0,3) i.e.
%                                   normalized by N-1 (sample SD).
%   MAE_hexagonality_avg.(task){b}(siz,:) : per hex bin b, size siz, model.
%   MAE_hexagonality_sd.(task){b}(siz,:)  : matching SD across seeds, std(...,0).
%   S                             : returned with the per-hop padding applied.
%
% MATH / AGGREGATION ORDER
%   LOG-PLACEMENT RULE (2026-06-01): MAE is ALWAYS arithmetic -- even when shown
%   as log2 it is log2(arithmetic-mean MAE): graphs are averaged LINEARLY and
%   log2 is taken AFTER the graph-mean (still before the seed-mean). nMAE is
%   ALWAYS geometric -- log2 is taken PER GRAPH and then averaged (log-space).
%   The `normalized` flag selects which: 0 -> arithmetic (raw + log2(MAE)); 1 ->
%   geometric (log2(nMAE)). Both size and dist paths follow this identically.
%   per-graph MAE: mean over a graph's edges (all hops pooled) of
%     |pred-truth|, per model  -> MAE_individuals.
%   MAE vs size: within each (seed,size) average graphs (mean over graphs of
%     the per-graph MAE) -> per-seed value; then mean ACROSS SEEDS for *_avg and
%     std ACROSS SEEDS for *_sd. FIXED 2026-06-01: the size path now uses
%     std(curr_vals,0,...) = SAMPLE SD (normalized by N-1).
%   MAE vs hop: MAE_dists keeps one row per graph per hop (graph-weighted, NOT
%     edge-weighted); per (seed,size) average edges within graph at each hop,
%     then this code stacks graphs along dim 3 and takes mean/std over dim 3
%     ACROSS SEEDS, std(...,0,3,...) => normalized by N-1 (sample SD). The 5
%     seeds are a SAMPLE, so all per-seed SD paths (size / distance /
%     hexagonality) now use the unbiased N-1 estimator consistently.
%   MAE vs hexagonality: within each (seed,size,hexbin) average the per-graph
%     MAEs of graphs whose hexagonality is in (edge_b, edge_{b+1}]; then mean
%     and std(...,0) ACROSS SEEDS per (size, bin) [sample SD; fixed 2026-06-01].
%
% DECISIONS / EDGE CASES
%   use_log + normalized together place the log2 (see LOG-PLACEMENT RULE):
%   normalized=1 logs per-graph BEFORE the graph-mean (geometric nMAE);
%   normalized=0 logs AFTER the graph-mean (arithmetic MAE). Unpopulated size bins
%   (single-cohort revision datasets) emit NaN rows of the correct width so the
%   later cell2mat concatenations succeed. Each graph's hop list is padded to
%   max_cell_dist with empty 0 x nModels matrices.

if ~exist('use_log', 'var')
    use_log = 0;
end
if ~exist('normalized', 'var')
    % normalized: 0 for the raw + log2(MAE) passes, 1 for the log2(nMAE) passes.
    % Selects the graph-level log2 placement so MAE stays ARITHMETIC (log2 AFTER
    % the graph-mean) while nMAE is GEOMETRIC (log2 PER graph, then averaged).
    % See the LOG-PLACEMENT RULE in the header. Default 0 = arithmetic.
    normalized = 0;
end

% first, calculating individual graph MAEs:
for t = 1 : length(tasks)

    % allocating memory:
    total_repetitions = size(S.prediction_errors.(tasks{t}),1);
    total_sizes = size(S.prediction_errors.(tasks{t}),2);
    MAE_individuals.(tasks{t}) = cell(total_repetitions, total_sizes);

    % calculating MAEs:
    for siz = 1 : total_sizes
        for repetition = 1 : total_repetitions
            if isempty(S.prediction_errors.(tasks{t}){repetition,siz})
                continue;
            end

            n_graphs = length(S.prediction_errors.(tasks{t}){repetition,siz}.(dataset_to_analyze));
            MAE_individuals.(tasks{t}){repetition, siz} = nan(n_graphs, length(all_models));

            for g = 1 : n_graphs
                MAE_individuals.(tasks{t}){repetition, siz}(g,:) = mean(cell2mat(S.prediction_errors.(tasks{t}){repetition,siz}.(dataset_to_analyze){g}), 1, 'omitnan');

                if use_log && normalized
                    % nMAE (normalized) pass ONLY: GEOMETRIC graph-aggregation --
                    % log2 EACH graph here so graphs are later averaged in
                    % log-space. The non-normalized log2(MAE) pass keeps per-graph
                    % MAEs LINEAR and logs AFTER the graph-mean (below), so MAE
                    % stays arithmetic.
                    MAE_individuals.(tasks{t}){repetition, siz}(g,:) = log2(MAE_individuals.(tasks{t}){repetition, siz}(g,:));
                end
            end

            n_datasets = length(S.prediction_errors.(tasks{t}){repetition,siz}.(dataset_to_analyze));

            % this should have been earlier in the code...
            for d = 1 : n_datasets
                S.prediction_errors.(tasks{t}){repetition,siz}.(dataset_to_analyze){d}(end+1:max_cell_dist) = {zeros(0, length(all_models))};
                S.predictions.(tasks{t}){repetition,siz}.(dataset_to_analyze){d}(end+1:max_cell_dist) = {zeros(0, length(all_models))};
            end

        end
    end
end

for t = 1 : length(tasks)
    MAE_size_avg.(tasks{t}) = cell(1,size(MAE_individuals.(tasks{t}),2));
    MAE_size_sd.(tasks{t}) = cell(1,size(MAE_individuals.(tasks{t}),2));
    for i = 1 : size(MAE_individuals.(tasks{t}),2)
        col_i = MAE_individuals.(tasks{t})(:,i);
        nonempty = ~cellfun(@isempty, col_i);
        % Single-cohort datasets (revision data) populate only one size bin;
        % the rest are empty. Emit a NaN row of the right width for empty
        % bins so the cell2mat() below still concatenates cleanly.
        if ~any(nonempty)
            MAE_size_avg.(tasks{t}){i} = nan(1, length(all_models));
            MAE_size_sd.(tasks{t}){i}  = nan(1, length(all_models));
            continue;
        end
        curr_vals = cell2mat(cellfun(@(x) mean(x, 1, 'omitnan'), col_i(nonempty), 'UniformOutput', false));
        if use_log && ~normalized
            % ARITHMETIC log2(MAE): graphs were averaged LINEARLY above, so take
            % log2 here -- after the per-seed graph-mean, before the seed-mean.
            curr_vals = log2(curr_vals);
        end
        MAE_size_avg.(tasks{t}){i} = mean(curr_vals, 1, 'omitnan');
        MAE_size_sd.(tasks{t}){i} = std(curr_vals, 0, 'omitnan');
    end
    MAE_size_avg.(tasks{t}) = cell2mat(MAE_size_avg.(tasks{t})');
    MAE_size_sd.(tasks{t}) = cell2mat(MAE_size_sd.(tasks{t})');
end

for t = 1 : length(tasks)

    % allocating memory:
    total_repetitions = size(S.prediction_errors.(tasks{t}),1);
    total_sizes = size(S.prediction_errors.(tasks{t}),2);

    % calculating MAEs:
    for repetition = 1 : total_repetitions
        for siz = 1 : total_sizes
            if isempty(S.prediction_errors.(tasks{t}){repetition,siz})
                continue;
            end

            n_datasets = length(S.prediction_errors.(tasks{t}){repetition,siz}.(dataset_to_analyze));

            for d = 1 : n_datasets
                S.prediction_errors.(tasks{t}){repetition,siz}.(dataset_to_analyze){d}(end+1:max_cell_dist) = {zeros(0, length(all_models))};
            end

            curr_set = S.prediction_errors.(tasks{t}){repetition,siz}.(dataset_to_analyze);

            % One row per graph per hop. The older code concatenated every
            % edge at a hop across all graphs, which made the distance plot
            % edge-weighted. This keeps each graph at equal weight.
            MAE_dists.(tasks{t}){repetition, siz} = cell(1, max_cell_dist);

            for c = 1 : max_cell_dist
                per_graph_vals = nan(n_datasets, length(all_models));
                for d = 1 : n_datasets
                    hop_vals = curr_set{d}{c};
                    if ~isempty(hop_vals)
                        per_graph_vals(d,:) = mean(hop_vals, 1, 'omitnan');
                    end
                end
                MAE_dists.(tasks{t}){repetition, siz}{c} = per_graph_vals;
            end

        end
    end
end

for t = 1 : length(tasks)

    % Per (rep, size) cell of MAE_dists -> per-graph-averaged hop matrix.
    % Empty entries (unpopulated size bins for single-cohort datasets) stay
    % empty; the original cellfun assumed every entry was a cell and crashed.
    md = MAE_dists.(tasks{t});
    all_per_graph_vals = cell(size(md));
    for ii = 1 : numel(md)
        if isempty(md{ii})
            all_per_graph_vals{ii} = [];
        elseif use_log && normalized
            % nMAE (normalized) pass: GEOMETRIC -- log2 EACH graph's per-hop MAE,
            % THEN average graphs (log-space), matching the size/hex nMAE path.
            all_per_graph_vals{ii} = cell2mat(cellfun(@(x) mean(log2(x), 1, 'omitnan'), md{ii}, 'UniformOutput', false)');
        else
            % raw MAE and the non-normalized log2(MAE) pass: average graphs
            % LINEARLY here; the log2 (if any) is applied AFTER this graph-mean
            % (below), so log2(MAE) stays ARITHMETIC.
            all_per_graph_vals{ii} = cell2mat(cellfun(@(x) mean(x, 1, 'omitnan'), md{ii}, 'UniformOutput', false)');
        end
    end

    MAE_dists_avg.(tasks{t}) = cell(1,size(all_per_graph_vals,2));
    MAE_dists_sd.(tasks{t}) = cell(1,size(all_per_graph_vals,2));
    for i = 1 : size(all_per_graph_vals,2)
        curr_vals = permute(all_per_graph_vals(:,i), [3,2,1]);
        if use_log && ~normalized
            % ARITHMETIC log2(MAE): graphs were averaged LINEARLY above, so log2
            % each seed's per-hop graph-mean here (before the seed-mean). The
            % normalized nMAE pass already logged per-graph, so it is skipped.
            curr_vals = cellfun(@log2, curr_vals, 'UniformOutput', false);
        end
        MAE_dists_avg.(tasks{t}){i} = mean(cell2mat(curr_vals), 3, 'omitnan');
        MAE_dists_sd.(tasks{t}){i} = std(cell2mat(curr_vals), 0, 3, 'omitnan');
    end

end

% lastly, we calculate MAE for hexagonality levels:
for t = 1 : length(tasks)

    MAE_hexagonality_all.(tasks{t}) = cell(size(MAE_individuals.(tasks{t}),2), length(h_bins_for_quality_analysis)-1);

    for b = 1 : length(h_bins_for_quality_analysis)-1
        for c = 1 : size(MAE_individuals.(tasks{t}),2)

            for r = 1 : size(MAE_individuals.(tasks{t}),1)

                if isempty(MAE_individuals.(tasks{t}){r,c})
                    continue;
                end

                curr_graphs = S.hexagonality{c}.(dataset_to_analyze) > h_bins_for_quality_analysis(b) & S.hexagonality{c}.(dataset_to_analyze) <= h_bins_for_quality_analysis(b+1);
                if any(curr_graphs)
                    MAE_hexagonality_all.(tasks{t}){c,b}(r,:) = mean(MAE_individuals.(tasks{t}){r,c}(curr_graphs,:), 1, 'omitnan');
                end
            end

            if ~isempty(MAE_hexagonality_all.(tasks{t}){c,b})
                MAE_hexagonality_avg.(tasks{t}){b}(c,:) = mean(MAE_hexagonality_all.(tasks{t}){c,b}, 1, 'omitnan'); % "MAE_hexagonality_avg.(tasks{t}){b}(c,:)" - the 'b' is the hexagonality level, the 'c' is cohort size, and the ':' is the model
                MAE_hexagonality_sd.(tasks{t}){b}(c,:) = std(MAE_hexagonality_all.(tasks{t}){c,b}, 0, 'omitnan');
            else
                MAE_hexagonality_avg.(tasks{t}){b}(c,:) = nan(1, length(all_models));
                MAE_hexagonality_sd.(tasks{t}){b}(c,:) = nan(1, length(all_models));
            end

            % MAE_hexagonality_sd.(tasks{t}){b}(c,:) = MAE_hexagonality_sd.(tasks{t}){b}(c,:) / sqrt(length(seeds));

        end
    end
end

end


function [MAE_size_avg, MAE_size_sd, MAE_dists_avg, MAE_dists_sd, MAE_individuals, MAE_hexagonality_avg, MAE_hexagonality_sd] = ...
    perform_MAE_normalization(MAE_size_avg, MAE_size_sd, MAE_dists_avg, MAE_dists_sd, MAE_individuals, MAE_hexagonality_avg, MAE_hexagonality_sd, S, tasks, dataset_to_analyze)
% perform_MAE_normalization  Divide every MAE summary by its baseline. [DEAD/UNUSED]
%
% *** DEAD / UNUSED ***
%   This function is NOT called by the live pipeline. The current normalized
%   panels instead plot log2(model/baseline) directly (with the Baseline column
%   pinned to 0 via force_baseline_reference_to_zero), so this ratio-in-place
%   helper is superseded. Retained for reference only.
%
%   *** KNOWN BUG (another reason it is unused) ***
%   The per-hop (per_dist) reshape HARD-CODES a 24-hop count:
%       mat2cell(cell2mat(...per_dist), 24, ones(1, nSizes))
%   This assumes max_cell_dist == 24 (true for the 256-cell tissues) and is
%   WRONG for the 484-cell and 784-cell revision tissues, whose maximum hop
%   exceeds 24. Running this on Tissue_484 / Tissue_784 would mis-shape (or
%   error on) the per_dist normalization. Do NOT resurrect without replacing
%   the literal 24 with the actual max_cell_dist / per-hop length.
%
% PURPOSE (as written)
%   Convert each absolute-MAE summary into a baseline-normalized ratio
%   (model MAE / baseline MAE) for the size, distance, hexagonality and
%   per-graph quantities, forcing the Baseline column to 1 (x/x).
%
% INPUTS
%   The seven MAE summary structs from extract_MAEs, plus S (for
%   S.normalization.(split).{per_size,per_dist,per_hexagonality}), tasks, and
%   the split name.
%
% OUTPUTS
%   The same seven summaries, each divided element-wise by the corresponding
%   baseline denominator; the Baseline column set to 1 via self-division.
%
% MATH
%   size:  MAE_size_avg ./ per_size'; Baseline col -> col/col = 1.
%   dist:  each size cell ./ the (24 x 1) per_dist column for that size.
%   indiv: each {r,siz} matrix / per_size(siz); Baseline col -> 1.
%   hex:   each {b}(siz,:) ./ per_hexagonality; Baseline col -> 1.

for t = 1 : length(tasks)
    MAE_size_avg.(tasks{t}) = MAE_size_avg.(tasks{t}) ./ S.normalization.(dataset_to_analyze).per_size';
    MAE_size_avg.(tasks{t})(:,end) = MAE_size_avg.(tasks{t})(:,end) ./ MAE_size_avg.(tasks{t})(:,end);

    MAE_size_sd.(tasks{t}) = MAE_size_sd.(tasks{t}) ./ S.normalization.(dataset_to_analyze).per_size';
    MAE_size_sd.(tasks{t})(:,end) = MAE_size_sd.(tasks{t})(:,end) ./ MAE_size_sd.(tasks{t})(:,end);

    MAE_dists_avg.(tasks{t}) = cellfun(@(x,y) x ./ y, MAE_dists_avg.(tasks{t}), ...
        mat2cell(cell2mat(S.normalization.(dataset_to_analyze).per_dist), 24, ones(1,size(S.normalization.(dataset_to_analyze).per_dist,2))), ...
        'UniformOutput', false);
    % MAE_dists_avg.(tasks{t}) = cellfun(@(x) [x(1:end-1), 1], MAE_dists_avg.(tasks{t}), 'UniformOutput', false);

    MAE_dists_sd.(tasks{t}) = cellfun(@(x,y) x ./ y, MAE_dists_sd.(tasks{t}), ...
        mat2cell(cell2mat(S.normalization.(dataset_to_analyze).per_dist), 24, ones(1,size(S.normalization.(dataset_to_analyze).per_dist,2))), ...
        'UniformOutput', false);
    % MAE_dists_sd.(tasks{t}) = cellfun(@(x) [x(1:end-1), 1], MAE_dists_sd.(tasks{t}), 'UniformOutput', false);

    MAE_individuals.(tasks{t}) = cellfun(@(x,y) x / y, MAE_individuals.(tasks{t}), repmat(num2cell(S.normalization.(dataset_to_analyze).per_size), ...
        [size(MAE_individuals.(tasks{t}),1), 1]), 'UniformOutput', false);
    to_correct = ~cellfun(@isempty, MAE_individuals.(tasks{t}));
    MAE_individuals.(tasks{t})(to_correct) = cellfun(@(x) [x(:,1:end-1), x(:,end) ./ x(:,end)], MAE_individuals.(tasks{t})(to_correct), 'UniformOutput', false);

    MAE_hexagonality_avg.(tasks{t}) = cellfun(@(x,y) x ./ y, MAE_hexagonality_avg.(tasks{t}), ...
        mat2cell(cell2mat(S.normalization.(dataset_to_analyze).per_hexagonality)', size(S.normalization.(dataset_to_analyze).per_hexagonality{1},1), ones(1,length(S.normalization.(dataset_to_analyze).per_hexagonality))), ...
        'UniformOutput', false);
    MAE_hexagonality_avg.(tasks{t}) = cellfun(@(x) [x(:,1:end-1), x(:,end) ./ x(:,end)], MAE_hexagonality_avg.(tasks{t}), 'UniformOutput', false);

    MAE_hexagonality_sd.(tasks{t}) = cellfun(@(x,y) x ./ y, MAE_hexagonality_sd.(tasks{t}), ...
        mat2cell(cell2mat(S.normalization.(dataset_to_analyze).per_hexagonality)', size(S.normalization.(dataset_to_analyze).per_hexagonality{1},1), ones(1,length(S.normalization.(dataset_to_analyze).per_hexagonality))), ...
        'UniformOutput', false);
    MAE_hexagonality_sd.(tasks{t}) = cellfun(@(x) [x(:,1:end-1), x(:,end) ./ x(:,end)], MAE_hexagonality_sd.(tasks{t}), 'UniformOutput', false);

end

end


function [y_min, y_max] = padded_axis_limits(y_min, y_max, prefer_zero_for_positive_raw)
% padded_axis_limits  Pad a [min,max] range; optionally snap the floor to 0.
%
% PURPOSE
%   Compute pleasant y-axis limits: add a small margin around the data and,
%   for raw (non-log) MAE panels, prefer 0 as the lower bound when the whole
%   band sits above 0.
%
% INPUTS
%   y_min, y_max                : raw data extremes (e.g. min/max of band).
%   prefer_zero_for_positive_raw: if true and the padded floor is still >0,
%                                 set the floor to 0 (default false).
%
% OUTPUTS
%   y_min, y_max : padded limits.
%
% ALGORITHM / DECISIONS
%   pad = 5% of the range, or max(1e-6, 5% of |y|) when min==max. Expand both
%   ends by pad. If prefer_zero_for_positive_raw and y_min>0, clamp y_min to 0
%   (but a band dipping below 0 keeps its negative floor so it is not clipped).
%
% EDGE CASES
%   Empty / non-finite inputs default to [0,1].

if nargin < 3
    prefer_zero_for_positive_raw = false;
end
if isempty(y_min) || isempty(y_max) || ~isfinite(y_min) || ~isfinite(y_max)
    y_min = 0;
    y_max = 1;
    return;
end
if y_min == y_max
    pad = max(1e-6, abs(y_min) * 0.05);
else
    pad = 0.05 * (y_max - y_min);
end
y_min = y_min - pad;
y_max = y_max + pad;

% Raw MAE panels look cleaner with zero as the lower bound, but only when
% the full mean-SD band is still above zero. If the SD shade dips below
% zero, keep the negative range so the shade is not clipped.
if prefer_zero_for_positive_raw && y_min > 0
    y_min = 0;
end

end


function plot_MAE_vs_dataset_size(MAE_size_avg, MAE_size_sd, figures_output_dir, tasks, use_log, colors, all_models, y_label, figure_panel_size)
% plot_MAE_vs_dataset_size  Plot MAE vs number of training cohorts, per task.
%
% PURPOSE
%   One subplot per task showing each model's MAE (raw, log2, or normalized
%   log2 depending on the inputs/y_label) as a function of dataset size, with
%   shaded SD-across-seeds error bands. Saved as a single .fig.
%
% INPUTS
%   MAE_size_avg  : per-task nSizes x nModels means (from extract_MAEs).
%   MAE_size_sd   : matching SD-across-seeds bands.
%   figures_output_dir : output folder.
%   tasks         : cellstr of task names (one subplot each).
%   use_log       : controls only the y-axis floor choice (prefer 0 when ~use_log).
%   colors        : ignored (recomputed via paper_model_colors).
%   all_models    : cellstr of model names (color + plot order + legend).
%   y_label       : per-call y-axis text (defaults to the legacy normalized log2 label).
%   figure_panel_size : [w h] px per panel (optional).
%
% OUTPUTS
%   none returned. Saves 'MAE vs dataset size (normalized log scale).fig'.
%
% ALGORITHM / DECISIONS
%   Recompute colors and paper plot order. Per task: plot each model with
%   shadedErrorBar; x-ticks labeled 2^(0:nSizes-1) cohorts (drawn at unit
%   spacing). Show the legend on the UW panel ('none_to_lengths') or on the
%   sole panel of single-task datasets. A shared y-limit is derived from all
%   bands via padded_axis_limits (floor snapped to 0 when ~use_log).
%
% Optional final arg `y_label` (added 2026-05-23): per-call y-axis label so
% each pass (raw / log2 / log2 nMAE) carries the right text. Defaults to
% the legacy 'log2(normalized mean graph MAE)' for backward compat.
if nargin < 8 || isempty(y_label)
    y_label = 'log2(normalized mean graph MAE)';
end
if nargin < 9 || isempty(figure_panel_size)
    figure_panel_size = [340, 300];
end
colors = paper_model_colors(all_models);
plot_order = paper_model_plot_order(all_models, true);

fig_w = max(420, figure_panel_size(1) * length(tasks) + 140);
fig_h = max(360, figure_panel_size(2) + 120);
fid = figure('Position', [100 100 fig_w fig_h]);
all_y_low = [];
all_y_high = [];

for t = 1 : length(tasks)
    subplot(1, length(tasks), t);

    y = MAE_size_avg.(tasks{t});
    y_err_sd = MAE_size_sd.(tasks{t});
    band_low = y - y_err_sd;
    band_high = y + y_err_sd;
    all_y_low = [all_y_low; band_low(:)]; %#ok<AGROW>
    all_y_high = [all_y_high; band_high(:)]; %#ok<AGROW>
    x = 2.^(0:size(y,1)-1);
    for m = plot_order
        shadedErrorBar(1:size(y,1), y(:,m), y_err_sd(:,m), 'lineprops', {'-', 'Color', colors(m,:)}, 'transparent', true);
        hold on;
    end
    % plot(y, 'o', 'MarkerFaceColor', 'w', 'Color', 'k');
    hold off;
    set(gca, 'XTick', 1:length(x), 'XTickLabel', x);
    xlim([0.5, length(x)+0.5]);
    title(regexprep(tasks{t}, '_', ' '));
    axis square;

    % Show the legend on the UW panel (paper convention) or, for single-task
    % datasets (revision data), on the only panel.
    if strcmp(tasks{t}, 'none_to_lengths') || length(tasks) == 1
        legend(regexprep(all_models(plot_order), '_', ' '), 'Location', 'southwest');
    end

    xlabel('# Cohorts (~41 graphs per cohort)');
    ylabel(y_label);
end

y_min = min(all_y_low, [], 'omitnan');
y_max = max(all_y_high, [], 'omitnan');
[y_min, y_max] = padded_axis_limits(y_min, y_max, ~use_log);

ylim([y_min, y_max]);

dcg_savefig_visible(fid, fullfile(figures_output_dir, 'MAE vs dataset size (normalized log scale).fig'));

end


function plot_MAE_vs_dist(MAE_dists_avg, MAE_dists_sd, figures_output_dir, tasks, use_log, colors, all_models, y_label, figure_panel_size)
% plot_MAE_vs_dist  Plot MAE vs hop distance from the T1 interface.
%
% PURPOSE
%   A grid of subplots (rows = tasks, columns = populated size bins) showing
%   each model's MAE as a function of hop distance, with shaded SD bands. Saved
%   as a single .fig.
%
% INPUTS
%   MAE_dists_avg : per-task {siz} nHops x nModels means (from extract_MAEs).
%   MAE_dists_sd  : matching SD-across-seeds bands.
%   figures_output_dir : output folder.
%   tasks         : cellstr of task names (one row each).
%   use_log       : controls only the y-axis floor choice (prefer 0 when ~use_log).
%   colors        : recomputed via paper_model_colors.
%   all_models    : cellstr of model names (color + order; baseline located).
%   y_label       : per-call y-axis text (defaults to the legacy label).
%   figure_panel_size : [w h] px per panel (optional).
%
% OUTPUTS
%   none returned. Saves 'MAE vs traverse dist.fig'.
%
% ALGORITHM / DECISIONS
%   Only size bins with data are drawn (single-cohort datasets populate one).
%   For each (task,size): hop axis = 0..nHops-1; keep hop rows that carry real
%   (non-Baseline) model data via hop_rows_with_real_model_data; plot each model
%   with shadedErrorBar. A per-task shared y-limit (padded_axis_limits, floor 0
%   when ~use_log) is applied, and x-limits are set to the first..last finite
%   hop so sparse final hops in large tissues are still shown.
%
% EDGE CASES
%   No populated size bins -> warn and return. A subplot with no finite hop rows
%   warns and is left blank.
%
% Optional final arg `y_label` (added 2026-05-23): per-call y-axis label so
% each pass (raw / log2 / log2 nMAE) carries the right text. Defaults to
% the legacy 'log2(normalized graph MAE)' for backward compat.
if nargin < 8 || isempty(y_label)
    y_label = 'log2(normalized graph MAE)';
end
if nargin < 9 || isempty(figure_panel_size)
    figure_panel_size = [340, 300];
end
colors = paper_model_colors(all_models);
plot_order = paper_model_plot_order(all_models, true);
baseline_idx = find(strcmp(all_models, 'Baseline') | strcmp(all_models, 'no_learning'), 1);

% Only plot the size bins that actually carry data. Single-cohort datasets
% (revision data) populate exactly one bin; the v1 standard data populates
% all six. Empty bins would crash the cell2mat / (end,:) = [] below.
nonempty_siz = find(~cellfun(@isempty, MAE_dists_avg.(tasks{1})));
if isempty(nonempty_siz)
    warning('plot_MAE_vs_dist: no populated size bins, skipping.');
    return;
end
n_siz = numel(nonempty_siz);

fig_w = max(420, figure_panel_size(1) * n_siz + 120);
fig_h = max(360, figure_panel_size(2) * length(tasks) + 120);
figure('Position', [100 100 fig_w fig_h], 'DockControls', 'off', 'NumberTitle', 'off', 'Name', 'MAE vs traverse dist');

idx = 0;
for t = 1 : length(tasks)
    max_y = -inf;
    min_y = inf;
    for sp = 1 : n_siz
        siz = nonempty_siz(sp);

        curr_MAE_dist_avg = cell2mat(MAE_dists_avg.(tasks{t})(:,siz)');
        curr_MAE_dist_sd = cell2mat(MAE_dists_sd.(tasks{t})(:,siz)');

        idx = idx + 1;
        subplot(length(tasks), n_siz, idx);
        set(gca, 'ColorOrder', colors, 'NextPlot', 'replacechildren');

        % Keep all observed hops, including the final hop in large tissues.
        % Rows with no finite model values are dropped, but sparse final-hop
        % rows are retained so hop-resolved revision figures reach the true
        % graph boundary.
        hop_x = (0:size(curr_MAE_dist_avg,1)-1)';
        finite_hops = hop_rows_with_real_model_data(curr_MAE_dist_avg, baseline_idx);
        curr_MAE_dist_avg = curr_MAE_dist_avg(finite_hops,:);
        curr_MAE_dist_sd = curr_MAE_dist_sd(finite_hops,:);
        hop_x = hop_x(finite_hops);
        if isempty(hop_x)
            warning('plot_MAE_vs_dist: no finite hop rows for task=%s size_bin=%d.', tasks{t}, siz);
            continue;
        end

        for m = plot_order
            shadedErrorBar(hop_x, curr_MAE_dist_avg(:,m), curr_MAE_dist_sd(:,m), 'lineprops', {'-', 'Color', colors(m,:)}, 'transparent', true);
            hold on;
        end

        axis square;
        xlabel('Hops from T1 interface');
        ylabel(y_label);
        title({['# Cohorts = ', num2str(2^(siz-1)), ','], regexprep(tasks{t}, '_', ' ')});
        % legend(regexprep(all_models(plot_order), '_', ' '), 'Location', 'southeast');
        band_low = curr_MAE_dist_avg - curr_MAE_dist_sd;
        band_high = curr_MAE_dist_avg + curr_MAE_dist_sd;
        max_y = max(max_y, max(band_high(:), [], 'omitnan'));
        min_y = min(min_y, min(band_low(:), [], 'omitnan'));
    end

    [min_y, max_y] = padded_axis_limits(min_y, max_y, ~use_log);

    for sp = 1 : n_siz
        siz = nonempty_siz(sp);
        subplot(length(tasks), n_siz, sp + (t-1) * n_siz);
        ylim([min_y, max_y]);
        curr_avg_for_xlim = MAE_dists_avg.(tasks{t}){siz};
        finite_hops = find(hop_rows_with_real_model_data(curr_avg_for_xlim, baseline_idx));
        if isempty(finite_hops)
            xlim([-0.5, 0.5]);
        else
            first_hop = finite_hops(1) - 1;
            last_hop = finite_hops(end) - 1;
            xlim([first_hop - 0.5, last_hop + 0.5]);
        end
    end
end

dcg_savefig_visible(fullfile(figures_output_dir, 'MAE vs traverse dist.fig'));

end


function finite_hops = hop_rows_with_real_model_data(y, baseline_idx)
% hop_rows_with_real_model_data  Mask of hop rows where some non-baseline model is finite.
%
% PURPOSE
%   Identify which hop rows actually carry model predictions, so the distance
%   plots drop fully-empty hops but keep sparse real ones (e.g. the final hop in
%   large tissues).
%
% INPUTS
%   y           : nHops x nModels matrix (means for one task/size).
%   baseline_idx: column to ignore when judging "real model data".
%
% OUTPUT
%   finite_hops : nHops x 1 logical; true where any non-baseline column is finite.
%
% ALGORITHM / EDGE CASES
%   Exclude the baseline column from consideration; a row is kept if any
%   remaining column is finite. If excluding the baseline leaves no columns, all
%   columns are considered. Empty y returns an empty 0x1 logical.

if isempty(y)
    finite_hops = false(0, 1);
    return;
end

model_cols = 1:size(y, 2);
if ~isempty(baseline_idx) && baseline_idx >= 1 && baseline_idx <= size(y, 2)
    model_cols(model_cols == baseline_idx) = [];
end
if isempty(model_cols)
    model_cols = 1:size(y, 2);
end
finite_hops = any(isfinite(y(:, model_cols)), 2);

end


function plot_MAE_vs_hexagonality(MAE_hexagonality_avg, figures_output_dir, tasks, use_log, colors, all_models, h_bins_for_quality_analysis, y_label)
% plot_MAE_vs_hexagonality  Plot MAE vs hexagonality per size bin. [DEAD/UNUSED]
%
% *** DEAD / UNUSED ***
%   This plain MAE-vs-hexagonality plot is NOT called by the live pipeline; the
%   manuscript hexagonality figure is produced by the
%   plot_manuscript_hexagonality_panels[_from_cache] functions instead.
%   Retained for reference only.
%
% PURPOSE (as written)
%   Full-screen grid (rows = tasks, columns = size bins) of each model's MAE vs
%   hexagonality bin center, saved as 'MAE vs hexagonality.fig'.
%
% INPUTS
%   MAE_hexagonality_avg : per-task {b}(siz,:) means (from extract_MAEs).
%   figures_output_dir   : output folder.
%   tasks                : cellstr of task names.
%   use_log              : if truthy, plot log2(MAE).
%   colors               : recomputed via paper_model_colors.
%   all_models           : cellstr of model names (color + order).
%   h_bins_for_quality_analysis : hexagonality bin edges (x = bin centers).
%   y_label              : per-call y-axis text (default legacy label).
%
% OUTPUTS
%   none (saves a .fig).
%
% ALGORITHM / DECISIONS
%   Reshape the per-bin/per-size cells into a [bins x models x sizes] array;
%   per size subplot, plot each model (Baseline/no_learning dashed, others
%   solid). Shared per-task y-limits (5% pad, floor 0 when ~use_log); xlim
%   [0.4 1]; square axes.
%
% Optional final arg `y_label` (added 2026-05-23): per-call y-axis label so
% each pass (raw / log2 / log2 nMAE) carries the right text. Defaults to
% the legacy 'log2(normalized mean graph MAE)' for backward compat.
if nargin < 8 || isempty(y_label)
    y_label = 'log2(normalized mean graph MAE)';
end
colors = paper_model_colors(all_models);
plot_order = paper_model_plot_order(all_models, false);

% plotting MAE vs. hexagonality:
figure('WindowState', 'maximized', 'DockControls', 'off', 'NumberTitle', 'off', 'Name', 'Fullscreen Figure');

x = (h_bins_for_quality_analysis(1:end-1) + h_bins_for_quality_analysis(2:end))/2;
for t = 1 : length(tasks)

    y_data = cell2mat(permute(MAE_hexagonality_avg.(tasks{t}), [3,1,2]));
    y_data = permute(y_data, [3,2,1]);

    max_y = -inf;
    min_y = inf;

    for siz = 1 : size(y_data,3)

        % y = cell2mat(MAE_hexagonality_avg.(tasks{t})(:,siz));
        y = y_data(:,:,siz);
        if use_log
            y = log2(y);
        end
        subplot(length(tasks), size(y_data,3), siz + (t-1) * size(y_data,3));

        for m = plot_order
            if m > size(y, 2)
                continue;
            end
            if strcmpi(all_models{m}, 'Baseline') || strcmpi(all_models{m}, 'no_learning')
                line_style = '--';
            else
                line_style = '-';
            end
            plot(x, y(:,m), line_style, 'LineWidth', 1.5, 'Color', colors(m,:));
            hold on;
        end
        hold on;
        % plot(x, y, 'o', 'MarkerFaceColor', 'w', 'Color', 'k');
        hold off;
        set(gca, 'Xtick', x);
        if strcmp(tasks{t}, 'none_to_lengths')
            % legend(regexprep(all_models, '_', ' '), 'Location', 'southwest');
        end
        xlabel('Hexagonality');
        ylabel(y_label);
        title({['# Cohorts = ', num2str(2^(siz-1)), ','], regexprep(tasks{t}, '_', ' ')});

        max_y = max(max_y, max(y(:)));
        min_y = min(min_y, min(y(:)));

    end

    y_range = max_y - min_y;
    max_y = max_y + 0.05 * y_range;
    min_y = min_y - 0.05 * y_range;

    if ~use_log
        min_y = 0;
    end

    for siz = 1 : size(MAE_hexagonality_avg.(tasks{t}),2)
        subplot(length(tasks), size(y_data,3), siz + (t-1) * size(y_data,3));
        ylim([min_y, max_y]);
        xlim([0.4, 1]);
        axis square;
    end

end

dcg_savefig_visible(fullfile(figures_output_dir, 'MAE vs hexagonality.fig'));

end


% ============================================================================
% 2D-embedding example panels (ported into the revision branch 2026-05-30).
% Fig A: 2 x nS overlays (top = model-prediction embedding, bottom = pre-T1
% baseline embedding); GT black, embedded blue, new-T1 interface red (pred)/
% green (GT). Titles carry MAE(emb,pred|base) and MAE(emb,GT).
% Fig B: 1 x nS, prediction embedding only, each edge colored by |l - L_pred|.
% Selection mirrors plot_scatter_plot_examples (struct-based (r,siz,g,m)); the
% raw prediction file supplies cell-pair topology + the three target-length
% columns S does not carry. A per-panel S-vs-file MAE guard prevents embedding
% the wrong graph if the file<->S index mapping ever drifts.
% ============================================================================
function plot_embedding_examples(scores_to_show, tasks, MAE_individuals, S, figures_output_dir, all_models, dataset_to_analyze, dataset, data_root, DCG_CONFIG)
% plot_embedding_examples  2D spring-embedding example panels (Fig A overlay, Fig B per-edge).
%
% PURPOSE
%   For example graphs chosen at requested error percentiles, embed the
%   predicted, ground-truth and pre-T1-baseline edge-length sets into 2D using
%   an EXTERNAL spring relaxation engine, rigidly align them, and render:
%     Fig A: GT (black) overlaid with the embedded prediction (blue) and with
%            the baseline (bottom row); new-T1 interfaces red (pred)/green (GT);
%            titles carry MAE(emb,pred|base) and MAE(emb,GT).
%     Fig B: prediction embedding only, each edge colored by |l - L_pred|.
%
% INPUTS
%   scores_to_show     : percentiles (0-100) to illustrate.
%   tasks              : cellstr of task names (W='lengths_to_lengths',
%                        UW='none_to_lengths').
%   MAE_individuals    : per-(seed,size) {graphs x models} MAE cells, used to
%                        rank and pick examples (mirrors plot_scatter_plot_examples).
%   S                  : results summary (for the S-side MAE guard, emb_sMAE).
%   figures_output_dir : output folder for the .fig/.png.
%   all_models         : cellstr of model names.
%   dataset_to_analyze : split name (selects the .inds file).
%   dataset            : dataset key (drives file prefixes and vt2d roots).
%   data_root          : default prediction-file root (overridable via DCG_CONFIG).
%   DCG_CONFIG         : optional config struct (engine path, work/vt2d roots,
%                        recompute flag, predRoot/indsRoot overrides).
%
% OUTPUTS
%   none returned. Per task saves 'Embedding examples overlay (<task>).fig/.png'
%   and 'Embedding examples per-edge (<task>).fig/.png'.
%
% EXTERNAL ENGINE
%   spring_embed.exe (path from DCG_CONFIG.embed_engine). Each variant
%   (gt/pred/base) writes a 6-column temp prediction file and the engine relaxes
%   it into ./output polygon files; relaxations are cached on disk and run in
%   parallel (parfor). If the engine is absent the function warns and returns.
%
% DATA FLOW / ALGORITHM
%   1. Resolve config + roots; detect consolidated-snapshot layout.
%   2. Percentile selection of (r,siz,g,m) exactly as plot_scatter_plot_examples.
%   3. For each pick: read the split .inds to map graph index g -> the file's
%      simulation id; locate the raw prediction file (per dataset-prefix/model/
%      seed). Apply the S-VS-FILE MAE GUARD: recompute mean|col6-col5| from the
%      file block and compare to S's MAE; if they differ by more than
%      max(5e-3, 0.1*|sMAE|) skip this example (file<->S index drift would embed
%      the wrong graph).
%   4. Resolve the matching .vt2d (periodic box) and build the T1 flip map from
%      the W prediction partner (emb_flip): pairs with flip column > 0.5 are the
%      flipped (new-T1) interfaces.
%   5. Write per-variant temp files (source columns gt=5, pred=6, base=3) and run
%      the engine. Read back polygons (stage 1 = pre-T1 topology, stage 3 =
%      relaxed), recover cell topology, unwrap the periodic tiling, and rigidly
%      (Kabsch) align prediction/baseline onto GT.
%   6. Recompute displayed embedding MAEs from the drawn stage-3 geometry:
%      maePT=|l_drawn-L_pred|, maePG=|l_drawn-L_GT|, maeBT=|l_drawn-L_base|,
%      maeBG=|l_drawn-L_GT|. The engine out_ file still supplies the cell-pair
%      list and target columns, but its reported-length column is not trusted
%      for title MAEs. Fig B uses the same de novo |l_drawn-L_pred| values with
%      per-panel percentile-clipped turbo coloring.
%   7. Draw Fig A (overlays) and Fig B (per-edge turbo coloring); save .fig/.png.
%
% MATH / DECISIONS / EDGE CASES
%   Alignment is rotation+translation only (no scale/reflection), via SVD with a
%   reflection-correcting determinant sign (see emb_align_to). The flip
%   threshold is 0.5. The MAE guard tolerance is max(5e-3, 0.1*|sMAE|). Missing
%   .vt2d, missing engine output, or geometry errors skip that panel with a
%   warning rather than aborting the figure. nS<1 returns immediately.

engine    = emb_getcfg(DCG_CONFIG, 'embed_engine',   '');
workdir   = emb_getcfg(DCG_CONFIG, 'embed_workdir',  fullfile(tempdir, 'dcg_springs_embed'));
stdRoot   = emb_getcfg(DCG_CONFIG, 'embed_vt2d_std', '');
revRoot   = emb_getcfg(DCG_CONFIG, 'embed_vt2d_rev', '');
recompute = emb_getcfg(DCG_CONFIG, 'embed_recompute', false);
colorPct  = emb_getcfg(DCG_CONFIG, 'embed_color_percentiles', [0.5 99]);
shearVt2d = emb_getcfg(DCG_CONFIG, 'embed_shear_affine_vt2d', true);
colorScale = lower(string(emb_getcfg(DCG_CONFIG, 'embed_color_scale', 'linear')));
if ~ismember(colorScale, ["log", "linear"])
    warning('DCG:embedColorScale', 'unknown embed_color_scale=%s; using linear.', colorScale);
    colorScale = "linear";
end
useLogColor = colorScale == "log";
saveOverlay = emb_getcfg(DCG_CONFIG, 'embed_save_overlay', true);
savePerEdge = emb_getcfg(DCG_CONFIG, 'embed_save_per_edge', true);

% Prediction files use the relocated flat layout, the same tree the analyzer
% reads: <predRoot>/pred_<prefix>__<model>_s<seed>.txt with split indices in
% <indsRoot>/<prefix>/<split>.inds. <prefix> is dataset+size+weighting (see
% emb_dataset_prefix). Override the roots via DCG_CONFIG if the data moves.
% When predRoot is a consolidated snapshot the names/splits differ; the
% DCG_consolidated_paths resolver handles both (pred + inds) below.
predRoot = emb_getcfg(DCG_CONFIG, 'embed_pred_root', data_root);
indsRoot = emb_getcfg(DCG_CONFIG, 'embed_inds_root', fullfile(data_root, 'inds'));
emb_consolidated = DCG_consolidated_paths('is_consolidated', predRoot);

if exist(engine,'file') ~= 2
    warning('embed: engine not found (%s) -- skipping embedding examples.', engine);
    return;
end
nS = length(scores_to_show);
if nS < 1, return; end
variants = {'gt',5; 'pred',6; 'base',3};
fprintf('[embed-diag] ENTER: dataset=%s nS=%d tasks={%s} consolidated=%d recompute=%d\n', ...
    dataset, nS, strjoin(tasks, ','), emb_consolidated, recompute);

baseline_idx = find(strcmp(all_models, 'Baseline') | strcmp(all_models, 'no_learning'), 1);
if isempty(baseline_idx), baseline_idx = length(all_models); end
model_indices = setdiff(1:length(all_models), baseline_idx, 'stable');

for t = 1 : length(tasks)
    tk = tasks{t};
    if strcmp(tk,'lengths_to_lengths'), wstr = 'W'; else, wstr = 'UW'; end

    % ---- percentile selection (mirrors plot_scatter_plot_examples) ----
    scoresArr = []; repsArr = []; sizesArr = []; graphsArr = []; modelsArr = [];
    for r = 1 : size(MAE_individuals.(tk), 1)
        for siz = 1 : size(MAE_individuals.(tk), 2)
            cm = MAE_individuals.(tk){r,siz};
            if isempty(cm), continue; end
            cmi = model_indices(model_indices <= size(cm,2));
            for m = cmi
                vg = find(isfinite(cm(:,m)));
                scoresArr = [scoresArr; cm(vg,m)];                  %#ok<AGROW>
                repsArr   = [repsArr;   repmat(r,   numel(vg), 1)]; %#ok<AGROW>
                sizesArr  = [sizesArr;  repmat(siz, numel(vg), 1)]; %#ok<AGROW>
                graphsArr = [graphsArr; vg(:)];                     %#ok<AGROW>
                modelsArr = [modelsArr; repmat(m,   numel(vg), 1)]; %#ok<AGROW>
            end
        end
    end
    if isempty(scoresArr)
        warning('embed: no finite MAE scores for task %s -- skipping.', tk);
        continue;
    end
    [scoresSorted, order] = sort(scoresArr, 'ascend');

    sel = repmat(emb_blank(0), 1, nS);
    for i = 1 : nS
        s_i = emb_blank(scores_to_show(i));
        try
            rank_i = max(1, min(numel(scoresSorted), round(scores_to_show(i) * numel(scoresSorted) / 100)));
            rec_i  = order(rank_i);
            r = repsArr(rec_i); siz = sizesArr(rec_i); g = graphsArr(rec_i); m = modelsArr(rec_i);
            curr_model = all_models{m};

            prefix  = emb_dataset_prefix(dataset, siz, wstr);
            wprefix = emb_dataset_prefix(dataset, siz, 'W');   % W partner for the flip map

            if emb_consolidated
                inds_dir = DCG_consolidated_paths('inds_dir', predRoot, prefix);
                if isempty(inds_dir), error('no split folder for prefix %s', prefix); end
                inds_filename = fullfile(inds_dir, [dataset_to_analyze, '.inds']);
            else
                inds_filename = fullfile(indsRoot, prefix, [dataset_to_analyze, '.inds']);
            end
            cf = fopen(inds_filename, 'rt');
            if cf < 0, error('inds file not found: %s', inds_filename); end
            inds = fread(cf, inf, '*char')'; fclose(cf);
            inds = str2num(inds) + 1; %#ok<ST2NM>
            curr_graph_ind = inds(g);

            % Flat relocated layout: one file per (dataset-prefix, model, seed),
            % identical 6-col "Simulation id:" block format for every model
            % (PPGN included), so a single path + load_dataset covers all types.
            if emb_consolidated
                predf  = DCG_consolidated_paths('pred_file', predRoot, prefix,  curr_model, r - 1);
                wpredf = DCG_consolidated_paths('pred_file', predRoot, wprefix, curr_model, r - 1);
            else
                predf  = fullfile(predRoot, sprintf('pred_%s__%s_s%d.txt', prefix,  curr_model, r - 1));
                wpredf = fullfile(predRoot, sprintf('pred_%s__%s_s%d.txt', wprefix, curr_model, r - 1));
            end
            if exist(predf, 'file') ~= 2, error('pred file not found: %s', predf); end
            gn = load_dataset(predf);

            egn = gn{curr_graph_ind};

            % S-vs-file MAE guard: file row order need not match S, but the mean
            % |pred - GT| over the graph must agree; if not, we grabbed the wrong
            % graph (file<->S index drift) -> skip rather than embed a mismatch.
            sMAE = emb_sMAE(S, tk, dataset_to_analyze, r, siz, g, m);
            prows = emb_read_block(predf, [egn, '.txt']);
            fileMAE = NaN;
            if ~isempty(prows)
                % Prediction is always the LAST column and the true/target the
                % one before it -- holds for 6-col (W) and 4-col (UW) pred files.
                c_true = cellfun(@(rr) str2double(rr{end-1}), prows);
                c_pred = cellfun(@(rr) str2double(rr{end}),   prows);
                fileMAE = mean(abs(c_pred - c_true), 'omitnan');
            end
            if isfinite(sMAE) && isfinite(fileMAE) && abs(fileMAE - sMAE) > max(5e-3, 0.1*abs(sMAE))
                warning(['embed guard (%s %g%%): file MAE %.4f ~= S MAE %.4f for %s ', ...
                    '(model %s, seed %d, n=%d, g %d) -- skipping to avoid wrong-graph embedding.'], ...
                    tk, scores_to_show(i), fileMAE, sMAE, egn, curr_model, r, 2^(siz-1), g);
                s_i.ok = false;
            else
                s_i.ok          = true;
                s_i.model       = curr_model;
                s_i.subset_size = siz;
                s_i.best_r      = g;
                s_i.seedrow     = r;
                s_i.sMAE        = sMAE;
                s_i.sim_id      = [egn, '.txt'];
                s_i.pred_file   = predf;
                s_i.wpred_file  = wpredf;
                s_i.cfgkey      = matlab.lang.makeValidName(sprintf('%s_%s_%s', wstr, curr_model, egn));
                if shearVt2d && ~isempty(emb_shear_lambda(dataset))
                    % Shear examples must not reuse the old cache generated from
                    % the square, pre-stretch vt2d starting geometry.
                    s_i.cfgkey = [s_i.cfgkey, '_affineShear'];
                end
            end
        catch ME
            warning('embed select (%s, %g%%): %s', tk, scores_to_show(i), ME.message);
        end
        sel(i) = s_i;
    end
    fprintf('[embed-diag] task %s: %d/%d examples passed selection (ok=1). If 0, read the "embed select/guard" warnings just above for the reason.\n', ...
        tk, sum([sel.ok]), nS);

    % ---- write per-variant temp prediction files + collect engine runs ----
    runRD = {}; runVT = {}; runPT = {}; runSI = {};
    for i = 1 : nS
        if ~sel(i).ok, continue; end
        vt2d = emb_resolve_vt2d(sel(i).sim_id, stdRoot, revRoot, dataset);
        if isempty(vt2d)
            warning('embed: no .vt2d for %s -- skipping.', sel(i).sim_id);
            sel(i).ok = false; continue;
        end
        if shearVt2d
            [vt2d, shear_msg] = emb_prepare_shear_vt2d(vt2d, workdir, dataset);
            if ~isempty(shear_msg)
                fprintf('[embed shear vt2d] %s: %s\n', sel(i).cfgkey, shear_msg);
            end
        end
        sel(i).vt2d = vt2d;
        [fm, fp] = emb_flip(sel(i).wpred_file, sel(i).sim_id);
        if isempty(fm), [fm, fp] = emb_flip(sel(i).pred_file, sel(i).sim_id); end
        sel(i).flipMap = fm; sel(i).flipPairs = fp;
        for v = 1 : size(variants,1)
            tag = variants{v,1}; srccol = variants{v,2};
            rd = fullfile(workdir, 'emb', sel(i).cfgkey, tag);
            sel(i).rundir.(tag) = rd;
            marker = fullfile(rd, 'output', ['out_', sel(i).sim_id]);
            if recompute || exist(marker,'file') ~= 2
                if exist(fullfile(rd,'output'),'dir') ~= 7, mkdir(fullfile(rd,'output')); end
                tmp = fullfile(workdir, sprintf('%s_%s_%s', tag, sel(i).cfgkey, sel(i).sim_id));
                % Baseline (pre-T1 in_preferred_length) is model-independent and
                % only exists in the 6-col W file; the 4-col UW pred file lacks it.
                % Always source 'base' from the W partner (for W datasets
                % wpred_file == pred_file, so this changes nothing there).
                src_file = sel(i).pred_file;
                if strcmp(tag, 'base'), src_file = sel(i).wpred_file; end
                try
                    emb_write_pred(tmp, sel(i).sim_id, src_file, srccol, fm);
                    runRD{end+1} = rd; runVT{end+1} = vt2d; runPT{end+1} = tmp; runSI{end+1} = sel(i).sim_id; %#ok<AGROW>
                catch ME
                    warning('embed write_pred (%s/%s): %s', sel(i).cfgkey, tag, ME.message);
                end
            end
        end
    end

    eng = engine;
    if ~isempty(runRD)
        fprintf('embed[%s]: %d relaxations queued (~5 min each; results cached on disk)...\n', tk, numel(runRD));
        parfor q = 1 : numel(runRD)
            emb_run_engine(eng, runVT{q}, runPT{q}, runSI{q}, runRD{q});
        end
    end

    % ---- load geometry, compute MAEs, gather Fig-B per-edge values ----
    G = repmat(struct('ok',false), 1, nS);
    gmax = 0;
    for i = 1 : nS
        if ~sel(i).ok, continue; end
        ok3 = true;
        for v = 1 : size(variants,1)
            if exist(fullfile(sel(i).rundir.(variants{v,1}),'output',['out_',sel(i).sim_id]),'file') ~= 2
                ok3 = false;
            end
        end
        if ~ok3
            warning('embed: missing engine output for %s -- panel skipped.', sel(i).cfgkey);
            continue;
        end
        try
            perio = emb_read_perio(sel(i).vt2d);
            GTp3 = emb_read_polys(sel(i).rundir.gt,   3);
            GTp1 = emb_read_polys(sel(i).rundir.gt,   1);
            PRp3 = emb_read_polys(sel(i).rundir.pred, 3);
            BAp3 = emb_read_polys(sel(i).rundir.base, 3);
            Nc = numel(GTp3);

            [~, cgGT] = emb_topology(GTp3, perio);
            [~, cgPR] = emb_topology(PRp3, perio);
            [~, cgBA] = emb_topology(BAp3, perio);
            [adj1, ~] = emb_topology(GTp1, perio);

            fp = sel(i).flipPairs; nflip = size(fp,1);
            newPairs = zeros(nflip,2);
            for f = 1 : nflip
                A = fp(f,1); B = fp(f,2);
                cd = find(adj1(A,:) & adj1(B,:));
                if numel(cd) ~= 2, cd = [cd, nan(1,2)]; cd = cd(1:2); end %#ok<AGROW>
                newPairs(f,:) = cd;
            end
            if nflip == 0, seedCell = 1; else, seedCell = fp(1,1); end

            GTu = emb_unwrap_cells(GTp3, cgGT, emb_adjacency_from(cgGT,Nc), perio, seedCell);
            PRu = emb_unwrap_cells(PRp3, cgPR, emb_adjacency_from(cgPR,Nc), perio, seedCell);
            BAu = emb_unwrap_cells(BAp3, cgBA, emb_adjacency_from(cgBA,Nc), perio, seedCell);
            % Each variant is unwrapped from the torus INDEPENDENTLY, so near a
            % box edge corresponding cells can land in different periodic images
            % -- a chunk one box off that the rigid (Kabsch) fit below cannot undo
            % (shows up as a lateral "drift"). Snap each pred/base cell onto the
            % nearest periodic image of its GT counterpart so all three share one
            % frame before aligning. nsnap>0 => some cells were period-jumped.
            [PRu, nsnapP] = emb_snap_frame(PRu, GTu, perio);
            [BAu, nsnapB] = emb_snap_frame(BAu, GTu, perio);
            [PRu, rmsdP] = emb_align_to(PRu, GTu);
            [BAu, rmsdB] = emb_align_to(BAu, GTu);
            fprintf('[embed align %s] pred: %d cells snapped, rmsd=%.4g | base: %d cells snapped, rmsd=%.4g\n', ...
                sel(i).cfgkey, nsnapP, rmsdP, nsnapB, rmsdB);

            Mp = emb_read_out(sel(i).rundir.pred, sel(i).sim_id);
            Mb = emb_read_out(sel(i).rundir.base, sel(i).sim_id);

            gg = struct();
            gg.GTu = GTu; gg.PRu = PRu; gg.BAu = BAu;
            gg.GTseg = emb_edge_segs(GTu, cgGT, newPairs);
            gg.PRseg = emb_edge_segs(PRu, cgPR, newPairs);
            gg.BAseg = emb_edge_segs(BAu, cgBA, newPairs);
            [gg.segsB, gg.valsB, gg.lenB] = emb_edge_segs_error_de_novo(PRu, cgPR, Mp(:,1:2), Mp(:,4));
            [~, valsPG, ~] = emb_edge_segs_error_de_novo(PRu, cgPR, Mp(:,1:2), Mp(:,3));
            [~, valsBT, ~] = emb_edge_segs_error_de_novo(BAu, cgBA, Mb(:,1:2), Mb(:,4));
            [~, valsBG, ~] = emb_edge_segs_error_de_novo(BAu, cgBA, Mb(:,1:2), Mb(:,3));
            % Do not use out_<sim_id> column 5 for these title MAEs. Some cached
            % embedding outputs echo the requested target length there (e.g. an
            % exactly zero |col5-col4| for a visibly distorted PNA example), while
            % Fig B already measures lengths de novo from the drawn polygons. Keep
            % the overlay titles tied to the same geometry that is actually shown.
            gg.maePT = emb_mean_finite(gg.valsB);   % |drawn pred embedding - L_pred|
            gg.maePG = emb_mean_finite(valsPG);     % |drawn pred embedding - L_GT|
            gg.maeBT = emb_mean_finite(valsBT);     % |drawn baseline embedding - L_base|
            gg.maeBG = emb_mean_finite(valsBG);     % |drawn baseline embedding - L_GT|

            allpts = [vertcat(GTu{:}); vertcat(PRu{:}); vertcat(BAu{:})];
            pad = 0.03 * max(max(allpts,[],1) - min(allpts,[],1));
            gg.xl = [min(allpts(:,1))-pad, max(allpts(:,1))+pad];
            gg.yl = [min(allpts(:,2))-pad, max(allpts(:,2))+pad];
            gg.ok = true;
            % NOTE (2026-06-02): G was templated with only the 'ok' field (so
            % skipped examples still answer ~G(i).ok in the draw loops). gg
            % carries the full geometry struct, so a whole-struct "G(i) = gg"
            % throws "Subscripted assignment between dissimilar structures".
            % Copy field-by-field instead; the first assignment back-fills the
            % new fields ([]) onto every G element, keeping the array uniform.
            gg_fields = fieldnames(gg);
            for gf = 1 : numel(gg_fields)
                G(i).(gg_fields{gf}) = gg.(gg_fields{gf});
            end
            finite_valsB = gg.valsB(isfinite(gg.valsB));
            if ~isempty(finite_valsB), gmax = max(gmax, max(finite_valsB)); end
        catch ME
            warning('embed geometry (%s): %s', sel(i).cfgkey, ME.message);
        end
    end
    if gmax <= 0, gmax = 1; end

    % -------------------------------- Fig A: overlays --------------------------
    if saveOverlay
        figA = figure('WindowState','maximized','NumberTitle','off','Color','w', ...
            'Name', ['Embedding overlay (', tk, ')']);
        for i = 1 : nS
            if i > numel(G) || ~G(i).ok, continue; end
            axp = subplot(2, nS, i);
            emb_draw_overlay(axp, G(i).GTu, G(i).GTseg, G(i).PRu, G(i).PRseg, ...
                'k', [0 0.2 0.85], [0 0.6 0], [0.85 0 0], 0.6, 2.6, G(i).xl, G(i).yl);
            title(axp, {sprintf('%g%%  %s  (n=%d, seed %d)', sel(i).pct, sel(i).model, 2^(sel(i).subset_size-1), sel(i).seedrow), ...
                sprintf('MAE(emb,pred)=%s   MAE(emb,GT)=%s', emb_fmt_mae(G(i).maePT), emb_fmt_mae(G(i).maePG))}, ...
                'FontWeight','normal','FontSize',9);
            axb = subplot(2, nS, i + nS);
            emb_draw_overlay(axb, G(i).GTu, G(i).GTseg, G(i).BAu, G(i).BAseg, ...
                'k', [0 0.2 0.85], [0 0.6 0], [0.85 0 0], 0.6, 2.6, G(i).xl, G(i).yl);
            title(axb, {sprintf('%g%%  baseline (pre-T1)', sel(i).pct), ...
                sprintf('MAE(emb,base)=%s   MAE(emb,GT)=%s', emb_fmt_mae(G(i).maeBT), emb_fmt_mae(G(i).maeBG))}, ...
                'FontWeight','normal','FontSize',9);
        end
        try savefig(figA, fullfile(figures_output_dir, ['Embedding examples overlay (', tk, ').fig'])); catch ME, warning('DCG:embedSavefigA', 'embed savefig A: %s', ME.message); end
        try exportgraphics(figA, fullfile(figures_output_dir, ['Embedding examples overlay (', tk, ').png']), 'Resolution', 200); catch, end
    end

    % --------------------- Fig B: prediction per-edge distortion ---------------
    % PER-PANEL clim (2026-06-03): each panel gets its own percentile-clipped
    % clim so a few tiny/huge residuals do not dominate the full dynamic range.
    % Configure with DCG_CONFIG.embed_color_percentiles; default [0.5 99], use
    % [] for full min/max. Default color values are the raw linear errors
    % |l - L_pred|, so edge colors, colorbar values, and title clim are in the
    % same units. DCG_CONFIG.embed_color_scale='log' is retained only as an
    % explicit diagnostic option.
    if savePerEdge
        figB = figure('WindowState','maximized','NumberTitle','off','Color','w', ...
            'Name', ['Embedding per-edge (', tk, ')']);
        for i = 1 : nS
            if i > numel(G) || ~G(i).ok, continue; end
            ax = subplot(1, nS, i);
            vi  = G(i).valsB(:);
            vi  = vi(isfinite(vi));
            if useLogColor
                [rawLo, rawHi] = emb_color_limits(vi, true, colorPct);
                valsForColor = log10(max(G(i).valsB, rawLo));
                lo = log10(rawLo);
                hi = log10(rawHi);
                scaleLabel = 'log10';
                climLabel = sprintf('raw clim[%.2g, %.2g]', rawLo, rawHi);
            else
                [lo, hi] = emb_color_limits(vi, false, colorPct);
                valsForColor = G(i).valsB;
                scaleLabel = 'linear';
                climLabel = sprintf('clim[%.2g, %.2g]', lo, hi);
            end
            emb_draw_colored(ax, G(i).segsB, valsForColor, lo, hi, 1.4, false);
            axis(ax,'equal'); axis(ax,'off'); xlim(ax, G(i).xl); ylim(ax, G(i).yl);
            colormap(ax, turbo); clim(ax, [lo hi]); set(ax, 'ColorScale', 'linear');
            cb = colorbar(ax);
            emb_configure_embedding_colorbar(cb, lo, hi, false);
            if useLogColor
                cb.Label.String = 'log_{10}(|l - L_{pred}|)';
            else
                cb.Label.String = sprintf('|l - L_{pred}|  (%s)', scaleLabel);
            end
            title(ax, {sprintf('%g%%  %s', sel(i).pct, sel(i).model), ...
                climLabel}, 'FontWeight','normal','FontSize',9);
        end
        try savefig(figB, fullfile(figures_output_dir, ['Embedding examples per-edge (', tk, ').fig'])); catch ME, warning('DCG:embedSavefigB', 'embed savefig B: %s', ME.message); end
        try exportgraphics(figB, fullfile(figures_output_dir, ['Embedding examples per-edge (', tk, ').png']), 'Resolution', 200); catch, end
    end
end

end


% ----------------------- embedding: small helpers -----------------------------
function m = emb_mean_finite(x)
% emb_mean_finite  Mean over finite entries only, NaN if none exist.
%
% PURPOSE
%   Embedding geometry comparisons can legitimately contain NaNs for interfaces
%   that are absent from the drawn phase-3 topology. Centralize the finite-entry
%   mean so overlay-title MAEs and Fig-B de novo edge errors use the same
%   missing-edge convention.
x = x(:);
x = x(isfinite(x));
if isempty(x), m = nan; else, m = mean(x); end
end

function s = emb_fmt_mae(x)
% emb_fmt_mae  Format overlay-title MAEs without hiding tiny nonzero values.
%
% PURPOSE
%   Fixed %.4f labels can print a real geometric error of 1e-6 as 0.0000,
%   which looks like an impossible exact embedding. Use decimal formatting for
%   manuscript-scale values and compact significant digits for very small ones.
if ~isfinite(x)
    s = 'NaN';
elseif x == 0
    s = '0';
elseif abs(x) < 5e-4
    s = sprintf('%.2g', x);
else
    s = sprintf('%.4f', x);
end
end

function v = emb_getcfg(C, f, d)
% emb_getcfg  Read config field f from struct C, else default d.
%
% PURPOSE
%   Tiny accessor used to pull optional embedding settings out of DCG_CONFIG
%   with a fallback.
%
% INPUTS
%   C : config struct (may be missing/empty).
%   f : field name to read.
%   d : default value if C lacks a non-empty field f.
%
% OUTPUT
%   v : C.(f) when present and non-empty, otherwise d.
if exist('C','var') && ~isempty(C) && isstruct(C) && isfield(C,f) && ~isempty(C.(f))
    v = C.(f);
else
    v = d;
end
end

function tok = emb_sizetok(siz)
% emb_sizetok  Map a v1 size index to its flat-layout size token.
%
% PURPOSE
%   Translate the plotter's size-bin index (1..6) into the string token used in
%   the relocated prediction-file/folder names.
%
% INPUT
%   siz : size-bin index.
%
% OUTPUT
%   tok : token ('2_1','2_2','2_4','2_8','2_16','1_32'); defaults to '2_16' for
%         out-of-range indices.
%
% v1 size index -> flat-prefix size token (matches the relocated folder names).
switch siz
    case 1, tok = '2_1';
    case 2, tok = '2_2';
    case 3, tok = '2_4';
    case 4, tok = '2_8';
    case 5, tok = '2_16';
    case 6, tok = '1_32';
    otherwise, tok = '2_16';
end
end

function pfx = emb_dataset_prefix(dataset, siz, wstr)
% emb_dataset_prefix  Build the flat-layout file prefix for a dataset/size/weighting.
%
% PURPOSE
%   Compute the <prefix> used in pred_<prefix>__<model>_s<seed>.txt and the
%   inds/<prefix>/ folder, from the plotter dataset name, size index, and
%   weighting string.
%
% INPUTS
%   dataset : dataset key (v1_W/v1_UW/v1_2_16_W/hex/rev_* families).
%   siz     : size-bin index (v1 only; via emb_sizetok).
%   wstr    : weighting suffix 'W' or 'UW'.
%
% OUTPUT
%   pfx : e.g. 'v1_2_16_W', 'v1_2_16_W' (16-cohort ref), 'hex_2_8_W', or
%         'rev_<dataset>' for the single-size revision sets.
%
% Map a plotter dataset name + size index + weighting ('W'/'UW') to the flat
% relocated-layout prefix used in pred_<prefix>__<model>_s<seed>.txt and
% inds/<prefix>/. v1 carries a per-size token and a weighting suffix; the
% single-size revision sets are weighted-only and use a bare rev_<name>.
switch dataset
    case {'v1_W', 'v1_UW'}
        pfx = sprintf('v1_%s_%s', emb_sizetok(siz), wstr);
    case 'v1_2_16_W'
        pfx = sprintf('v1_2_16_%s', wstr);
    case 'hex'
        pfx = sprintf('hex_2_8_%s', wstr);
    otherwise
        pfx = sprintf('rev_%s', dataset);   % Shear_1_2, Shear_1_5, kA_1, kA_10, Flip_two, Tissue_484, Tissue_784
end
end

function lambda = emb_shear_lambda(dataset)
% emb_shear_lambda  Area-preserving rectangular stretch used by shear datasets.
%
% PURPOSE
%   The revision shear simulations do not apply an xy-skew. In the new C code
%   they call expand_box(shearFactor*Lx, Ly/shearFactor), and expand_box scales
%   every vertex coordinate as x <- xNEW/xOLD*x and y <- yNEW/yOLD*y. The spring
%   embedding should therefore start from the same rectangularly stretched vt2d
%   box, not from the square pre-stretch vt2d file.
%
% INPUT
%   dataset : plotter dataset key.
%
% OUTPUT
%   lambda : [] for non-shear datasets, 1.2 for Shear_1_2, 1.5 for Shear_1_5.
switch dataset
    case 'Shear_1_2'
        lambda = 1.2;
    case 'Shear_1_5'
        lambda = 1.5;
    otherwise
        lambda = [];
end
end

function [vt2d_out, msg] = emb_prepare_shear_vt2d(vt2d_in, workdir, dataset)
% emb_prepare_shear_vt2d  Materialize a shear-aware vt2d for spring embedding.
%
% PURPOSE
%   The available revision vt2d files under "New code version\All vt2d\Shear"
%   are the initial square geometries. The simulation later stretches the
%   periodic box and vertex coordinates by x*lambda and y/lambda. If the spring
%   embedding starts from the original square vt2d, the relaxed shear examples
%   keep a square-like aspect ratio and their MAE(embedding,prediction) is
%   inflated or misleading. This helper writes a transformed copy into the
%   embedding work directory and returns that path.
%
% INPUTS
%   vt2d_in : original vt2d path resolved by emb_resolve_vt2d.
%   workdir : embedding cache/work directory.
%   dataset : dataset key; only Shear_1_2 and Shear_1_5 are transformed.
%
% OUTPUTS
%   vt2d_out : original path for non-shear/already-sheared files, otherwise the
%              transformed copy in <workdir>\sheared_vt2d\<dataset>\.
%   msg      : short diagnostic printed by the caller.
%
% DECISIONS / EDGE CASES
%   If the vt2d box already has the target aspect ratio lambda^2 (within a small
%   log tolerance), we assume it has already been stretched and use it as-is.
%   Failures warn and fall back to the original path so a single malformed vt2d
%   does not abort the whole figure.
vt2d_out = vt2d_in;
msg = '';

lambda = emb_shear_lambda(dataset);
if isempty(lambda)
    return;
end

try
    [Lx, Ly] = emb_vt2d_box(vt2d_in);
    targetAspect = lambda^2;
    currentAspect = Lx / Ly;
    if isfinite(currentAspect) && currentAspect > 0 && abs(log(currentAspect / targetAspect)) < 0.03
        msg = sprintf('using already-sheared vt2d (box %.6g x %.6g)', Lx, Ly);
        return;
    end

    outdir = fullfile(workdir, 'sheared_vt2d', dataset);
    if exist(outdir, 'dir') ~= 7
        mkdir(outdir);
    end

    [~, stem, ext] = fileparts(vt2d_in);
    lamTag = strrep(sprintf('%.6g', lambda), '.', '_');
    vt2d_out = fullfile(outdir, sprintf('%s_lambda_%s%s', stem, lamTag, ext));

    if exist(vt2d_out, 'file') ~= 2
        emb_write_affine_shear_vt2d(vt2d_in, vt2d_out, lambda);
    end

    [Lx2, Ly2] = emb_vt2d_box(vt2d_out);
    msg = sprintf('affine shear lambda=%.6g box %.6g x %.6g -> %.6g x %.6g', ...
        lambda, Lx, Ly, Lx2, Ly2);
catch ME
    warning('DCG:embedShearVt2d', ...
        'could not prepare shear vt2d for %s: %s; using original vt2d.', vt2d_in, ME.message);
    vt2d_out = vt2d_in;
    msg = '';
end
end

function [Lx, Ly] = emb_vt2d_box(vt2d_file)
% emb_vt2d_box  Read the periodic-box dimensions from a vt2d file.
%
% The vt2d files used here begin with a count/header line followed by a box
% line. We intentionally parse only the first two non-empty numeric lines so
% comments/blank lines do not affect the result.
fid = fopen(vt2d_file, 'rt');
if fid < 0
    error('could not open vt2d file: %s', vt2d_file);
end
cleanupObj = onCleanup(@() fclose(fid));

numericLines = {};
while ~feof(fid) && numel(numericLines) < 2
    line = fgetl(fid);
    if ~ischar(line), break; end
    vals = sscanf(strtrim(line), '%f')';
    if ~isempty(vals)
        numericLines{end+1} = vals; %#ok<AGROW>
    end
end

if numel(numericLines) < 2 || numel(numericLines{2}) < 2
    error('could not read vt2d box from %s', vt2d_file);
end

Lx = numericLines{2}(1);
Ly = numericLines{2}(2);
end

function emb_write_affine_shear_vt2d(src, dst, lambda)
% emb_write_affine_shear_vt2d  Write vt2d copy with x*lambda and y/lambda.
%
% PURPOSE
%   Mirror the shear simulation's expand_box call for the embedding engine:
%   periodic box x-length and all vertex x coordinates are multiplied by lambda;
%   y-length and all vertex y coordinates are divided by lambda. Connectivity and
%   all non-vertex records are left untouched.
raw = fileread(src);
lines = regexp(raw, '\r\n|\n|\r', 'split');
nonempty = find(~cellfun(@(s) isempty(strtrim(s)), lines));
if numel(nonempty) < 3
    error('vt2d file has too few non-empty lines: %s', src);
end

counts = sscanf(strtrim(lines{nonempty(1)}), '%f')';
if isempty(counts) || ~isfinite(counts(1)) || counts(1) < 1
    error('could not read vertex count from %s', src);
end
nVertices = round(counts(1));
if numel(nonempty) < 2 + nVertices
    error('vt2d file ended before %d vertex lines: %s', nVertices, src);
end

boxIdx = nonempty(2);
boxVals = sscanf(strtrim(lines{boxIdx}), '%f')';
if numel(boxVals) < 2
    error('could not read box dimensions from %s', src);
end
boxVals(1) = boxVals(1) * lambda;
boxVals(2) = boxVals(2) / lambda;
lines{boxIdx} = emb_format_numeric_line(boxVals);

vertexIdx = nonempty(3 : 2 + nVertices);
for ii = 1 : numel(vertexIdx)
    idx = vertexIdx(ii);
    vals = sscanf(strtrim(lines{idx}), '%f')';
    if numel(vals) < 2
        error('could not read vertex coordinates at line %d in %s', idx, src);
    end
    vals(1) = vals(1) * lambda;
    vals(2) = vals(2) / lambda;
    lines{idx} = emb_format_numeric_line(vals);
end

fid = fopen(dst, 'wt');
if fid < 0
    error('could not create transformed vt2d file: %s', dst);
end
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, '%s', strjoin(lines, newline));
end

function line = emb_format_numeric_line(vals)
% emb_format_numeric_line  Stable compact numeric formatting for vt2d rows.
line = strtrim(sprintf('%.12g ', vals));
end

function s = emb_blank(pct)
% emb_blank  Construct an empty per-example selection struct.
%
% PURPOSE
%   Initialize the placeholder record for one embedding example (one requested
%   percentile) with ok=false and empty fields, to be filled in during
%   selection.
%
% INPUT
%   pct : the percentile this slot represents.
%
% OUTPUT
%   s : struct with fields ok, pct, model, subset_size, best_r, seedrow, sMAE,
%       sim_id, pred_file, wpred_file, cfgkey, vt2d, flipMap, flipPairs, rundir.
s = struct('ok',false, 'pct',pct, 'model','', 'subset_size',1, 'best_r',0, ...
    'seedrow',1, 'sMAE',NaN, ...
    'sim_id','', 'pred_file','', 'wpred_file','', 'cfgkey','', 'vt2d','', ...
    'flipMap',[], 'flipPairs',[], 'rundir',struct('gt','','pred','','base',''));
end

function v = emb_sMAE(S, task, ds, r, siz, g, m)
% emb_sMAE  S-side mean |prediction - ground truth| for one graph/model.
%
% PURPOSE
%   Compute the same per-graph MAE that plot_scatter_plot_examples uses, so it
%   can be compared against the value recomputed from the raw prediction file
%   (the embedding guard).
%
% INPUTS
%   S          : results summary.
%   task       : task name.
%   ds         : split name.
%   r, siz, g  : seed-row, size-bin, graph indices.
%   m          : model column.
%
% OUTPUT
%   v : mean|pred(:,m) - post_T1| over valid hops, or NaN if unavailable.
%
% ALGORITHM
%   Read S.predictions / S.ground_truth for (task,r,siz,g); split ground truth
%   into per-hop cells if needed; keep hops where both are non-empty and the
%   prediction has >= m columns; concatenate and average |pred-truth|.
%
% Mean |prediction - ground truth| for one (task, seed-row, size, graph, model),
% computed straight from S exactly as plot_scatter_plot_examples does, so it can
% be cross-checked against the same quantity recomputed from the raw file.
v = NaN;
if ~isfield(S,'predictions') || ~isfield(S,'ground_truth'), return; end
pred_cells = S.predictions.(task){r,siz}.(ds){g};
gt_cells   = S.ground_truth.(task){r,siz}.(ds){g};
if ~iscell(gt_cells)
    gt_cells = mat2cell(gt_cells, cellfun(@(x) size(x,1), pred_cells), 1);
end
n_hops = min(numel(pred_cells), numel(gt_cells));
pred_cells = pred_cells(1:n_hops); gt_cells = gt_cells(1:n_hops);
valid = ~cellfun(@isempty,pred_cells) & ~cellfun(@isempty,gt_cells) & ...
    cellfun(@(x) size(x,2) >= m, pred_cells);
if ~any(valid), return; end
cp = cell2mat(cellfun(@(x) x(:,m), pred_cells(valid), 'UniformOutput', false));
cg = cell2mat(gt_cells(valid));
v = mean(abs(cp - cg), 'omitnan');
end

function p = emb_resolve_vt2d(sim_id, stdRoot, revRoot, dataset)
% emb_resolve_vt2d  Locate the .vt2d periodic-box file for a simulation id.
%
% PURPOSE
%   Find the geometry file (.vt2d, which carries the periodic box) that matches
%   a given prediction simulation id, across the standard (v1) and revision data
%   trees.
%
% INPUTS
%   sim_id  : simulation id like 'graph_<...>.txt' (or '<...>.txt').
%   stdRoot : root for standard/v1 vt2d files (name matches sim-id stem).
%   revRoot : root for revision vt2d files (nested per family).
%   dataset : dataset key, used to choose the revision subfolder.
%
% OUTPUT
%   p : full path to the matching .vt2d, or '' if not found.
%
% ALGORITHM
%   Strip 'graph_' / '.txt' -> mid; try stdRoot/final_<mid>.vt2d first. For
%   revision datasets, map the dataset to its subfolder and glob
%   final_<ncells>_*_<rep>_<dis>.vt2d (kA varies per family, hence the wildcard).
%   Fallback: recursive exact-name search anywhere under revRoot.
%
% EDGE CASES
%   Returns '' when no candidate matches (caller skips that example).
if nargin < 4, dataset = ''; end
mid  = regexprep(sim_id, '^graph_', '');
mid  = regexprep(mid, '\.txt$', '');
name = ['final_', mid, '.vt2d'];
p = '';
% std / v1: vt2d name matches the sim-id stem directly
cand = fullfile(stdRoot, name);
if exist(cand,'file') == 2, p = cand; return; end
% revision datasets: sim-id tokens 1/2/3 = ncells/rep/dis; vt2d is
% final_<ncells>_<kA>_<rep>_<dis>.vt2d (kA varies per family, so glob it).
switch dataset
    case 'kA_1',       sub = fullfile('kA', 'kA 1');
    case 'kA_10',      sub = fullfile('kA', 'kA 10');
    case 'Shear_1_2',  sub = fullfile('Shear', 'Shear 1_2');
    case 'Shear_1_5',  sub = fullfile('Shear', 'Shear 1_5');
    case 'Flip_two',   sub = fullfile('Flip two', 'All flip two');
    case 'Tissue_484', sub = fullfile('Tissue size', 'Tissue size 484');
    case 'Tissue_784', sub = fullfile('Tissue size', 'Tissue size 784');
    otherwise,         sub = '';
end
revFolder = '';
if ~isempty(sub), revFolder = fullfile(revRoot, sub, 'initial'); end
if ~isempty(revFolder) && exist(revFolder,'dir') == 7
    tok = strsplit(mid, '_');
    if numel(tok) >= 3
        pat = sprintf('final_%s_*_%s_%s.vt2d', tok{1}, tok{2}, tok{3});
        d = dir(fullfile(revFolder, pat));
        if ~isempty(d), p = fullfile(d(1).folder, d(1).name); return; end
    end
end
% fallback: recursive exact-name search anywhere under revRoot
if exist(revRoot,'dir') == 7
    d = dir(fullfile(revRoot, '**', name));
    if ~isempty(d), p = fullfile(d(1).folder, d(1).name); end
end
end

function k = emb_pairkey(a,b), k = sprintf('%d_%d', min(a,b), max(a,b)); end
% emb_pairkey  Order-independent string key for an unordered cell pair (a,b).
%
% PURPOSE
%   Build a canonical map key for an interface between two cells, independent of
%   which cell is listed first, used to look up flip states.
%
% INPUTS
%   a, b : the two cell indices.
%
% OUTPUT
%   k : 'min_max' string, e.g. emb_pairkey(7,3) -> '3_7'.
%
% NOTE
%   The entire function body is on the signature line above; this block is
%   documentation only and adds no executable statements.

function [flipMap, flipPairs] = emb_flip(wpredfile, sim_id)
% emb_flip  Read the T1 flip state per interface from a (W) prediction file.
%
% PURPOSE
%   Extract which interfaces undergo a T1 transition ("flip") for a given
%   simulation, from the weighted prediction file's per-edge flip column.
%
% INPUTS
%   wpredfile : path to the W prediction file (preferred source of flip column).
%   sim_id    : simulation id block to read.
%
% OUTPUTS
%   flipMap   : containers.Map from emb_pairkey(c1,c2) -> the raw flip-column
%               string for that interface.
%   flipPairs : Nx2 sorted unique cell-pairs whose flip value > 0.5 (the
%               new-T1 interfaces).
%
% DECISIONS / EDGE CASES
%   Flip threshold is 0.5. If the block is missing/empty (or unreadable), both
%   outputs are empty and the caller falls back to the non-W prediction file.
flipMap = []; flipPairs = [];
try
    wrows = emb_read_block(wpredfile, sim_id);
catch
    return;
end
if isempty(wrows), return; end
flipMap = containers.Map('KeyType','char','ValueType','char');
for k = 1 : numel(wrows)
    r = wrows{k}; c1 = str2double(r{1}); c2 = str2double(r{2});
    flipMap(emb_pairkey(c1,c2)) = r{4};
    if str2double(r{4}) > 0.5, flipPairs = [flipPairs; sort([c1 c2])]; end %#ok<AGROW>
end
flipPairs = unique(flipPairs, 'rows');
end

function rows = emb_read_block(file, sim)
% emb_read_block  Extract the 6-token rows of one "Simulation id:" block.
%
% PURPOSE
%   Parse a prediction text file and return the per-edge rows belonging to a
%   single simulation block.
%
% INPUTS
%   file : path to a prediction text file.
%   sim  : simulation id to extract (matched after 'Simulation id:').
%
% OUTPUT
%   rows : cell array; each entry is a 1x6 cellstr of the whitespace-split
%          tokens for one edge row of that block.
%
% ALGORITHM
%   Stream lines; toggle "in block" when a 'Simulation id:' header matches sim
%   (stop at the next header); inside the block, keep lines that split into
%   exactly 6, 5, or 4 tokens. The 5-token case is the PPGN-UW format
%   [c1 c2 artificial_zero out_pref pred]; the main loader strips that
%   artificial zero before analysis, and the embedding writer handles it below.
%
% EDGE CASES
%   Errors if the file cannot be opened; returns {} if the sim block is absent.
rows = {};
fid = fopen(file,'r'); if fid < 0, error('cannot open %s', file); end
c = onCleanup(@() fclose(fid));
inblk = false;
while true
    ln = fgetl(fid);
    if ~ischar(ln), break; end
    if startsWith(ln,'Simulation id:')
        cur = strtrim(extractAfter(ln,'Simulation id:'));
        if strcmp(cur,sim), inblk = true; continue;
        elseif inblk, break; else, continue; end
    end
    if inblk
        tkn = strsplit(strtrim(ln));
        % 6-col: lengths_to_lengths (W)       [c1 c2 in_pref flip out_pref pred]
        % 5-col: PPGN none_to_lengths (UW)    [c1 c2 dummy0  out_pref pred]
        % 4-col: MPNN none_to_lengths (UW)    [c1 c2 out_pref pred]
        if ismember(numel(tkn), [4 5 6]), rows{end+1} = tkn; end %#ok<AGROW>
    end
end
end

function emb_write_pred(tmp, sim_id, predfile, srccol, flipMap)
% emb_write_pred  Write a temp 6-col prediction file for the embedding engine.
%
% PURPOSE
%   Emit the per-variant input the spring engine consumes: the same simulation
%   block, but with the 6th column replaced by the chosen target-length source
%   (gt/pred/base) and flip states optionally overridden from the W file.
%
% INPUTS
%   tmp      : output temp file path.
%   sim_id   : simulation id whose block to copy.
%   predfile : source prediction file.
%   srccol   : source column index supplying the engine's last column
%              (gt=5, pred=6, base=3 per the variants table).
%   flipMap  : optional emb_pairkey -> flip-string map; when present, each row's
%              flip column (4) is overwritten by the map value for that pair.
%
% OUTPUTS
%   none (writes tmp).
%
% ALGORITHM
%   Read the block; write 'Simulation id:' then, per row, columns 1-5 verbatim
%   (with col 4 possibly replaced from flipMap) followed by column srccol.
%
% EDGE CASES
%   Errors if the sim block is missing or the temp file cannot be opened.
rows = emb_read_block(predfile, sim_id);
if isempty(rows), error('sim id %s not found in %s', sim_id, predfile); end
ncol = numel(rows{1});
fid = fopen(tmp,'w'); if fid < 0, error('cannot write %s', tmp); end
c = onCleanup(@() fclose(fid));
fprintf(fid,'Simulation id: %s\n', sim_id);
for k = 1 : numel(rows)
    r = rows{k};
    if ncol >= 6
        % 6-col source [c1 c2 in_pref flip out_pref pred]: engine input = cols
        % 1-5 verbatim (flip col 4 optionally overridden) + the chosen srccol.
        if ~isempty(flipMap)
            key = emb_pairkey(str2double(r{1}), str2double(r{2}));
            if isKey(flipMap,key), r{4} = flipMap(key); end
        end
        fprintf(fid,'%s %s %s %s %s %s\n', r{1},r{2},r{3},r{4},r{5}, r{srccol});
    elseif ncol == 5
        % 5-col PPGN-UW source [c1 c2 artificial_zero true pred]. This is the
        % same semantic data as the 4-col UW branch, with one extra dummy column
        % inserted after the cell ids. Synthesize the engine's 6-col input:
        %     c1 c2 <true> <flip> <true> <target>
        switch srccol
            case 5, tgt = r{4};   % gt  = out_preferred (true post-T1 length)
            case 6, tgt = r{5};   % pred
            otherwise, error('5-col PPGN none_to_lengths file has no source column %d', srccol);
        end
        flipstr = '0';
        if ~isempty(flipMap)
            key = emb_pairkey(str2double(r{1}), str2double(r{2}));
            if isKey(flipMap,key), flipstr = flipMap(key); end
        end
        fprintf(fid,'%s %s %s %s %s %s\n', r{1},r{2}, r{4}, flipstr, r{4}, tgt);
    else
        % 4-col source [c1 c2 out_pref(true) pred] (none_to_lengths; no input
        % length / flip columns). Synthesize the engine's 6-col input:
        %     c1 c2 <true> <flip> <true> <target>
        % the engine reads col4=flip and col6=target; cols 3/5 are unused filler.
        % srccol comes from the 6-col variant table: 5(gt)->true(col3),
        % 6(pred)->pred(col4). 'base' (srccol 3) is sourced from the W partner
        % file by the caller, so it never reaches this 4-col branch.
        switch srccol
            case 5, tgt = r{3};   % gt  = out_preferred (true post-T1 length)
            case 6, tgt = r{4};   % pred
            otherwise, error('4-col none_to_lengths file has no source column %d', srccol);
        end
        flipstr = '0';
        if ~isempty(flipMap)
            key = emb_pairkey(str2double(r{1}), str2double(r{2}));
            if isKey(flipMap,key), flipstr = flipMap(key); end
        end
        fprintf(fid,'%s %s %s %s %s %s\n', r{1},r{2}, r{3}, flipstr, r{3}, tgt);
    end
end
end

function emb_run_engine(exe, vt2d, predtmp, sim_id, rundir)
% emb_run_engine  Invoke the external spring-embedding engine for one variant.
%
% PURPOSE
%   Run spring_embed.exe on a (vt2d, temp prediction, sim id) triple, producing
%   relaxed-geometry output files under rundir/output.
%
% INPUTS
%   exe     : path to the engine executable.
%   vt2d    : periodic-box geometry file.
%   predtmp : temp prediction file written by emb_write_pred.
%   sim_id  : simulation id argument for the engine.
%   rundir  : working directory the engine runs in (output goes to ./output).
%
% OUTPUTS
%   none returned (engine writes files; stdout/stderr suppressed).
%
% DECISIONS
%   Uses a shell-side `cd /d` so it is parfor-safe and never changes MATLAB's
%   own cwd; system() errors are swallowed (missing outputs are detected later).
%
% Shell-side cd (parfor-safe; never touches MATLAB's cwd). Engine writes ./output.
cmd = sprintf('cd /d "%s" && "%s" "%s" "%s" "%s" >nul 2>&1', rundir, exe, vt2d, predtmp, sim_id);
try system(cmd); catch, end
end

function perio = emb_read_perio(vt2d)
% emb_read_perio  Read the periodic box size [Lx Ly] from a .vt2d file.
%
% PURPOSE
%   Recover the periodicity vector used to unwrap the periodic tiling, from the
%   second numeric line of the geometry file.
%
% INPUT
%   vt2d : geometry file path.
%
% OUTPUT
%   perio : 1x2 [Lx Ly] periodic box lengths.
%
% ALGORITHM
%   Read the first two non-empty numeric lines; take the first two numbers of
%   the second line.
%
% EDGE CASES
%   Errors if fewer than two numeric lines exist, or if the values are not a
%   valid positive 2-vector.
fid = fopen(vt2d,'r'); c = onCleanup(@() fclose(fid));
nums = {};
while ~feof(fid)
    ln = strtrim(fgetl(fid));
    if isempty(ln), continue; end
    v = sscanf(ln,'%f')';
    if isempty(v), continue; end
    nums{end+1} = v; %#ok<AGROW>
    if numel(nums) >= 2, break; end
end
if numel(nums) < 2, error('could not read perio from %s', vt2d); end
perio = nums{2}(1:2);
if numel(perio) ~= 2 || any(perio <= 0), error('bad perio in %s', vt2d); end
end

function cells = emb_read_polys(rundir, stage)
% emb_read_polys  Read per-cell polygon vertices from engine X_/Y_ outputs.
%
% PURPOSE
%   Load the relaxed (or initial) cell polygons produced by the engine for a
%   given stage into a cell array of vertex coordinate lists.
%
% INPUTS
%   rundir : the variant's run directory (reads rundir/output/X_<stage>.txt and
%            Y_<stage>.txt).
%   stage  : engine stage (1 = pre-T1 topology, 3 = relaxed in this code).
%
% OUTPUT
%   cells : Nc x 1 cell; entry i = [x y] vertices of polygon i.
%
% ALGORITHM
%   Row i of X_/Y_ stores the vertex count in column 1 and the coordinates in
%   columns 2..n+1; assemble each polygon's [x y] from those.
M = readmatrix(fullfile(rundir,'output',sprintf('X_%d.txt',stage)),'FileType','text');
N = readmatrix(fullfile(rundir,'output',sprintf('Y_%d.txt',stage)),'FileType','text');
nc = size(M,1); cells = cell(nc,1);
for i = 1 : nc
    n = M(i,1);
    cells{i} = [M(i,2:n+1).', N(i,2:n+1).'];
end
end

function M = emb_read_out(rundir, sim_id)
% emb_read_out  Read the engine per-edge out_ table (first 5 columns).
%
% PURPOSE
%   Load the engine's per-interface output used for the embedding MAEs and the
%   Fig B per-edge values.
%
% INPUTS
%   rundir : variant run directory (reads rundir/output/out_<sim_id>).
%   sim_id : simulation id (file suffix).
%
% OUTPUT
%   M : N x 5 matrix. Columns used downstream: 1-2 = the cell pair; 3 = L_GT;
%       4 = L_pred (or L_base for the baseline variant); 5 = embedded length l.
M = readmatrix(fullfile(rundir,'output',['out_',sim_id]),'FileType','text');
M = M(:,1:5);
end

function [adj, cellG] = emb_topology(cells, perio)
% emb_topology  Recover shared-vertex ids and a cell adjacency from polygons.
%
% PURPOSE
%   Determine which polygon vertices coincide (under periodic wrapping) and,
%   from that, which cells are neighbors (share an edge).
%
% INPUTS
%   cells : Nc x 1 cell of polygon [x y] vertex lists.
%   perio : 1x2 periodic box used to wrap coordinates before matching.
%
% OUTPUTS
%   adj   : Nc x Nc logical cell adjacency (true when two cells share >= 2
%           vertices, i.e. an edge); from emb_adjacency_from.
%   cellG : Nc x 1 cell; per cell, the global vertex id of each of its vertices.
%
% ALGORITHM / DECISIONS
%   Wrap every vertex by mod(.,perio), round to tol = 1e-9, and unique-rows them
%   into global vertex ids; group ids back per cell; build adjacency.
tol = 1e-9;
allV = vertcat(cells{:});
w = mod(allV, perio);
[~,~,gid] = unique(round(w./tol), 'rows', 'stable');
cellG = cell(numel(cells),1); k = 0;
for i = 1 : numel(cells)
    n = size(cells{i},1); cellG{i} = gid(k+1:k+n); k = k + n;
end
adj = emb_adjacency_from(cellG, numel(cells));
end

function adj = emb_adjacency_from(cellG, Nc)
% emb_adjacency_from  Cell adjacency from per-cell global vertex ids.
%
% PURPOSE
%   Build the Nc x Nc neighbor matrix: two cells are adjacent when they share an
%   edge, i.e. at least two common global vertices.
%
% INPUTS
%   cellG : Nc x 1 cell of per-cell global vertex id lists.
%   Nc    : number of cells.
%
% OUTPUT
%   adj   : Nc x Nc logical, true where two cells share >= 2 vertices.
%
% ALGORITHM
%   For each global vertex, collect the cells incident to it; for every pair of
%   such cells increment a shared-vertex counter; threshold the counter at >= 2.
G = max(vertcat(cellG{:}));
incid = cell(G,1);
for i = 1 : Nc
    for g = cellG{i}.'
        incid{g}(end+1) = i; %#ok<AGROW>
    end
end
cnt = zeros(Nc,Nc);
for g = 1 : G
    cc = unique(incid{g});
    for a = 1 : numel(cc)
        for b = a+1 : numel(cc)
            cnt(cc(a),cc(b)) = cnt(cc(a),cc(b)) + 1;
            cnt(cc(b),cc(a)) = cnt(cc(b),cc(a)) + 1;
        end
    end
end
adj = cnt >= 2;
end

function placed = emb_unwrap_cells(cells, cellG, adj, perio, seed)
% emb_unwrap_cells  Unwrap periodic cell polygons into one contiguous tiling.
%
% PURPOSE
%   Remove periodic-boundary jumps so neighboring cells are placed next to each
%   other in real space, enabling a sensible 2D overlay and alignment.
%
% INPUTS
%   cells : Nc x 1 cell of (possibly wrapped) polygon [x y] lists.
%   cellG : per-cell global vertex ids (from emb_topology).
%   adj   : cell adjacency matrix.
%   perio : 1x2 periodic box.
%   seed  : index of the cell to anchor the unwrap at.
%
% OUTPUT
%   placed : Nc x 1 cell of unwrapped polygon coordinates.
%
% ALGORITHM
%   BFS from seed: for each newly placed cell i, for each unplaced neighbor j
%   sharing a vertex, shift j by the nearest periodic image
%   d = round((ci-cj)./perio).*perio so the shared vertex coincides; mark done.
%
% EDGE CASES
%   Any cell never reached (disconnected) keeps its original coordinates.
Nc = numel(cells); placed = cell(Nc,1); done = false(Nc,1);
placed{seed} = cells{seed}; done(seed) = true; q = seed;
while ~isempty(q)
    i = q(1); q(1) = [];
    for j = find(adj(i,:))
        if done(j), continue; end
        g = intersect(cellG{i}, cellG{j}); g = g(1);
        ci = placed{i}(find(cellG{i}==g,1),:);
        cj = cells{j}(find(cellG{j}==g,1),:);
        d  = round((ci-cj)./perio).*perio;
        placed{j} = cells{j} + d; done(j) = true; q(end+1) = j; %#ok<AGROW>
    end
end
for i = 1 : Nc, if ~done(i), placed{i} = cells{i}; end, end
end

function [Xa, rmsd] = emb_align_to(X, P)
% emb_align_to  Kabsch rigid alignment of point set X onto reference P.
%
% PURPOSE
%   Best-fit X onto P using rotation + translation only (no scaling, no
%   reflection), so the embedded prediction/baseline can be overlaid on GT.
%
% INPUTS
%   X : cell of point lists to be transformed (concatenated for the fit).
%   P : cell of reference point lists (same total point count/order).
%
% OUTPUTS
%   Xa   : X transformed onto P (same cell structure).
%   rmsd : root-mean-square deviation between aligned X and P.
%
% MATH (Kabsch)
%   Center both clouds; H = (X-mean)' * (P-mean); [U,~,V] = svd(H);
%   enforce a proper rotation with D = diag([1, sign(det(V*U'))]) to forbid
%   reflection; Rot = V*D*U'; translate by mr - mq*Rot'.
%
% Rigid (rotation + translation, no scaling, no reflection) best-fit of X onto P.
Q = vertcat(X{:}); R = vertcat(P{:});
mq = mean(Q,1); mr = mean(R,1);
H = (Q-mq).' * (R-mr);
[U,~,V] = svd(H);
D = eye(2); D(2,2) = sign(det(V*U.'));
Rot = V*D*U.'; tvec = mr - mq*Rot.';
Xa = cellfun(@(c) c*Rot.' + tvec, X, 'uni', 0);
Qa = Q*Rot.' + tvec; rmsd = sqrt(mean(sum((Qa-R).^2,2)));
end

function [Xs, nsnap] = emb_snap_frame(X, G, perio)
% emb_snap_frame  Put each cell of X in the same periodic image as G's cell.
%
% PURPOSE
%   Corresponding cells in two independently-unwrapped tilings should occupy the
%   same periodic image, but an independent unwrap can disagree by whole boxes
%   near the edges. Shift each X cell by the nearest periodic multiple that
%   brings its centroid onto G's, so a subsequent rigid (Kabsch) fit is not
%   defeated by a per-cell box offset.
%
% INPUTS
%   X, G  : Nc x 1 cells of unwrapped polygon coords (X is snapped onto G's frame).
%   perio : 1 x 2 periodic box.
%
% OUTPUTS
%   Xs    : X with each cell shifted onto G's periodic image.
%   nsnap : number of cells that actually moved (i.e. were period-jumped).
Xs = X; nsnap = 0;
for c = 1 : numel(X)
    if isempty(X{c}) || isempty(G{c}), continue; end
    d = round((mean(G{c},1) - mean(X{c},1)) ./ perio) .* perio;
    if any(d ~= 0), nsnap = nsnap + 1; end
    Xs{c} = X{c} + d;
end
end

function segs = emb_edge_segs(placed, cellG, pairs)
% emb_edge_segs  Build drawable line segments for given cell-cell interfaces.
%
% PURPOSE
%   For each requested cell pair, return the 2-point segment of their shared
%   edge in the unwrapped layout, for drawing interfaces / colored edges.
%
% INPUTS
%   placed : Nc x 1 cell of unwrapped polygon coordinates.
%   cellG  : per-cell global vertex ids.
%   pairs  : Np x 2 list of cell index pairs (may contain NaN / out-of-range).
%
% OUTPUT
%   segs   : Np x 1 cell; each a 2x2 [x y; x y] segment, or [] if undefined.
%
% ALGORITHM / EDGE CASES
%   Skip invalid pairs (NaN or out of range) -> []. Find shared global vertices;
%   take their positions in the first cell. Two shared vertices -> the segment;
%   one -> a degenerate point doubled; none -> [].
segs = cell(size(pairs,1),1);
nc = numel(cellG);
for f = 1 : size(pairs,1)
    C = pairs(f,1); D = pairs(f,2);
    if any(isnan([C D])) || C < 1 || D < 1 || C > nc || D > nc, segs{f} = []; continue; end
    sh = intersect(cellG{C}, cellG{D});
    if isempty(sh), segs{f} = []; continue; end
    pos = arrayfun(@(g) find(cellG{C}==g,1), sh);
    pts = placed{C}(pos,:);
    if size(pts,1) >= 2, segs{f} = pts(1:2,:);
    elseif size(pts,1) == 1, segs{f} = [pts; pts];
    else, segs{f} = []; end
end
end

function [segs, vals, lens] = emb_edge_segs_error_de_novo(placed, cellG, pairs, targets)
% emb_edge_segs_error_de_novo  Recompute Fig-B edge errors from drawn geometry.
%
% PURPOSE
%   Build the same drawable interface segments as emb_edge_segs, but compute
%   |l - L_pred| from the segment length in the plotted relaxed tissue instead
%   of trusting the engine's rounded out_ length column. This keeps Fig B tied
%   to the tissue that is actually drawn while preserving the existing layout.
%
% INPUTS
%   placed  : Nc x 1 cell of aligned/unwrapped polygon coordinates.
%   cellG   : per-cell global vertex ids.
%   pairs   : Np x 2 cell-pair list from out_<sim_id>.
%   targets : Np x 1 L_pred target column from out_<sim_id>.
%
% OUTPUTS
%   segs : Np x 1 drawable segments.
%   vals : Np x 1 abs(recomputed_segment_length - target).
%   lens : Np x 1 recomputed segment lengths.
segs = emb_edge_segs(placed, cellG, pairs);
vals = nan(size(pairs,1), 1);
lens = nan(size(pairs,1), 1);
targets = targets(:);
for f = 1 : numel(segs)
    s = segs{f};
    if isempty(s) || size(s,1) < 2 || f > numel(targets) || ~isfinite(targets(f))
        continue;
    end
    lens(f) = hypot(s(2,1) - s(1,1), s(2,2) - s(1,2));
    vals(f) = abs(lens(f) - targets(f));
end
end

function emb_draw_overlay(ax, REFu, REFseg, Xu, Xseg, cR, cX, cRhop, cXhop, lwCell, lwHop, xl, yl)
% emb_draw_overlay  Overlay a reference and a candidate embedding (Fig A).
%
% PURPOSE
%   Draw the reference (GT) cells and a candidate (prediction or baseline)
%   embedding on the same axes, plus their respective new-T1 interface segments.
%
% INPUTS
%   ax            : target axes.
%   REFu, Xu      : reference and candidate unwrapped polygons.
%   REFseg, Xseg  : reference and candidate new-T1 interface segments.
%   cR, cX        : cell outline colors for reference / candidate.
%   cRhop, cXhop  : interface (hop) colors for reference / candidate.
%   lwCell, lwHop : line widths for cells / interfaces.
%   xl, yl        : axis limits.
%
% OUTPUTS
%   none (draws into ax; equal aspect, axis off).
hold(ax,'on');
% Draw the reference as a slightly thicker underlay. When the embedded
% prediction nearly overlaps the GT, equal-width blue lines can completely
% hide the black reference even though both layers are present.
emb_draw_cells(ax, REFu, cR, max(lwCell, 1.8 * lwCell));
emb_draw_cells(ax, Xu,   cX, lwCell);
emb_draw_hops(ax, REFseg, cRhop, lwHop);
emb_draw_hops(ax, Xseg,   cXhop, lwHop);
axis(ax,'equal'); axis(ax,'off'); xlim(ax,xl); ylim(ax,yl);
end

function emb_draw_cells(ax, cells, c, lw)
% emb_draw_cells  Draw closed cell-polygon outlines.
%
% PURPOSE
%   Stroke each cell polygon as a closed loop in a single color.
%
% INPUTS
%   ax    : target axes.
%   cells : cell of polygon [x y] vertex lists.
%   c     : line color.
%   lw    : line width.
%
% OUTPUTS
%   none (draws into ax). Empty polygons are skipped; each loop is closed by
%   repeating its first vertex.
for i = 1 : numel(cells)
    P = cells{i}; if isempty(P), continue; end
    plot(ax, [P(:,1);P(1,1)], [P(:,2);P(1,2)], '-', 'Color', c, 'LineWidth', lw);
end
end

function emb_draw_hops(ax, segs, c, lw)
% emb_draw_hops  Draw interface (new-T1) segments with endpoint dots.
%
% PURPOSE
%   Highlight the new-T1 interfaces as colored line segments with marker dots at
%   their endpoints.
%
% INPUTS
%   ax   : target axes.
%   segs : cell of 2x2 segments (from emb_edge_segs).
%   c    : color.
%   lw   : line width (marker size scales with it).
%
% OUTPUTS
%   none (draws into ax). Empty segments are skipped.
for f = 1 : numel(segs)
    s = segs{f}; if isempty(s), continue; end
    plot(ax, s(:,1), s(:,2), '-', 'Color', c, 'LineWidth', lw);
    plot(ax, s(:,1), s(:,2), '.', 'Color', c, 'MarkerSize', max(8, lw*5));
end
end

function emb_configure_embedding_colorbar(cb, lo, hi, use_log)
% emb_configure_embedding_colorbar  Stable, readable ticks for Fig B colorbars.
%
% MATLAB's automatic log-colorbar ticks can be sparse (for example only one
% labelled decade inside a narrow range) or visually uneven for clipped
% percentile ranges. Use five ticks evenly spaced along the actual color ramp:
% geometric spacing for log ramps, arithmetic spacing for linear ramps.
if nargin < 4 || isempty(use_log), use_log = false; end
if ~isfinite(lo) || ~isfinite(hi) || hi <= lo
    return;
end

if use_log
    lo = max(lo, eps);
    ticks = exp(linspace(log(lo), log(hi), 5));
else
    ticks = linspace(lo, hi, 5);
end

cb.Ticks = ticks;
cb.TickLabels = arrayfun(@emb_color_tick_label, ticks, 'UniformOutput', false);
end

function label = emb_color_tick_label(v)
% emb_color_tick_label  Compact numeric tick labels for embedding error bars.
if ~isfinite(v)
    label = '';
elseif v == 0
    label = '0';
elseif abs(v) < 1e-3 || abs(v) >= 1e3
    label = sprintf('%.1e', v);
else
    label = sprintf('%.3g', v);
end
end

function [lo, hi] = emb_color_limits(vals, use_log, colorPct)
% emb_color_limits  Percentile-clipped color limits for Fig B.
vals = vals(:);
vals = vals(isfinite(vals));
if isempty(vals)
    lo = eps;
    hi = 1;
    return;
end

if use_log
    vals = vals(vals > 0);
    if isempty(vals)
        lo = eps;
        hi = 1;
        return;
    end
end

[lo, hi] = emb_percentile_limits(vals, colorPct);
if use_log
    lo = max(lo, eps);
    if hi <= lo, hi = lo * 10; end
else
    if hi <= lo, hi = lo + 1; end
end
end

function [lo, hi] = emb_percentile_limits(vals, colorPct)
vals = sort(vals(:));
vals = vals(isfinite(vals));
if isempty(vals)
    lo = eps;
    hi = 1;
    return;
end
if isempty(colorPct)
    lo = vals(1);
    hi = vals(end);
    return;
end
p = colorPct(:).';
if numel(p) ~= 2
    error('DCG:embedColorPercentiles', 'embed_color_percentiles must be [] or [low high].');
end
p = max(0, min(100, p));
if p(2) < p(1), p = fliplr(p); end
lo = emb_percentile_value(vals, p(1));
hi = emb_percentile_value(vals, p(2));
end

function v = emb_percentile_value(sortedVals, pct)
n = numel(sortedVals);
if n == 1
    v = sortedVals(1);
    return;
end
pos = 1 + (pct / 100) * (n - 1);
i0 = floor(pos);
i1 = ceil(pos);
if i0 == i1
    v = sortedVals(i0);
else
    w = pos - i0;
    v = (1 - w) * sortedVals(i0) + w * sortedVals(i1);
end
end

function emb_draw_colored(ax, segs, vals, lo, hi, lw, use_log)
% emb_draw_colored  Draw edge segments colored by a scalar value (Fig B).
%
% PURPOSE
%   Render each interface segment with a turbo-colormap color encoding a
%   per-edge scalar (here |l - L_pred|), for the per-edge distortion figure.
%
% INPUTS
%   ax      : target axes.
%   segs    : cell of 2x2 segments (one per value).
%   vals    : per-segment scalar values.
%   lo, hi  : color-scale limits (value -> color onto turbo(256)).
%   lw      : line width.
%   use_log : (optional, default false) if true, map value->color on a LOG scale
%             between lo and hi (lo must be > 0; values <= lo -> bottom color).
%             Pair with clim([lo hi]) + set(ax,'ColorScale','log') on the caller
%             side so the colorbar matches the per-line colors.
%
% OUTPUTS
%   none (draws into ax).
%
% ALGORITHM / EDGE CASES
%   Linear: frac = (val-lo)/(hi-lo). Log: frac = (log(val)-log(lo))/(log(hi)-
%   log(lo)), with val clamped to [lo,hi]. idx = round(frac*255)+1 clamped to
%   [1,256]. Degenerate ranges fall back so idx stays valid. Empty segs skipped.
if nargin < 7 || isempty(use_log), use_log = false; end
hold(ax,'on');
cmap = turbo(256);
if use_log
    lo = max(lo, eps); if hi <= lo, hi = lo*10; end
    denom = log(hi) - log(lo);
    frac = @(v) (log(min(max(v,lo),hi)) - log(lo)) / denom;
else
    rng = hi - lo; if rng <= 0, rng = 1; end
    frac = @(v) (v - lo) / rng;
end
for k = 1 : numel(segs)
    s = segs{k}; if isempty(s), continue; end
    if k > numel(vals) || ~isfinite(vals(k)), continue; end
    idx = round(frac(vals(k))*255) + 1;
    if idx < 1, idx = 1; elseif idx > 256, idx = 256; end
    plot(ax, s(:,1), s(:,2), '-', 'Color', cmap(idx,:), 'LineWidth', lw);
end
end

function output_paths = DCG_make_extreme_task_composites_internal(data_root, cache_dir, figures_root, dataset_to_analyze, h_bins_for_quality_analysis, DCG_CONFIG)
% DCG_make_extreme_task_composites_internal  Temporary 5x3 revision composites.
%
% This helper builds four requested figures without changing the
% normal single-dataset plotter path:
%   1. 5x3 task-family hop-distance curves, raw MAE, with Baseline curves in black.
%   2. 5x3 task-family hop-distance curves, log2 nMAE.
%   3. 5x3 task-family hop-distance curves, log2 MAE, with Baseline curves in black.
%   4. 5x3 task-family condition summaries, raw MAE.
%   5. 5x3 task-family condition summaries, log2 nMAE.
%   6. 5x3 task-family condition summaries, log2 MAE.
%   7. 5x3 median-prediction GT/prediction embedding overlays.
%   8. 5x3 median-prediction per-edge embedding-error maps.
%
% Rows are GNN models in paper order. Columns are task families: kA, shear,
% and tissue size. The curve figures overlay the three conditions in each
% family with RGB colors. The embedding figures use the most extreme condition
% per family: kA_1, Shear_1_5, and Tissue_784.
if nargin < 6 || isempty(DCG_CONFIG), DCG_CONFIG = struct(); end
if nargin < 5 || isempty(h_bins_for_quality_analysis), h_bins_for_quality_analysis = 0.4 : 0.1 : 1; end
if nargin < 4 || isempty(dataset_to_analyze), dataset_to_analyze = 'test'; end
if nargin < 3 || isempty(figures_root), figures_root = fullfile(data_root, '_figures', 'revision_2026'); end
if nargin < 2 || isempty(cache_dir), if isempty(data_root)
    error('DCG:missingDataRoot', ['Set the consolidated prediction snapshot path ', ...
        'using DCG_CONFIG.data_root, the DCG_DATA_ROOT environment variable, ', ...
        'or an untracked DCG_local_config.m file.']);
end

cache_dir = fullfile(data_root, '_analyzer_cache', 'revision_2026'); end

output_dir = extreme_cfg(DCG_CONFIG, 'extreme_output_dir', fullfile(figures_root, 'revision_extreme_task_composites'));
save_png = extreme_cfg(DCG_CONFIG, 'extreme_save_png', false);
if exist(output_dir, 'dir') ~= 7, mkdir(output_dir); end

families = extreme_task_families();
model_rows = {'GraphSAGE', 'GAT', 'GIN', 'PNA', 'PPGN'};
all_dataset_keys = {};
for f = 1 : numel(families)
    all_dataset_keys = [all_dataset_keys, families(f).datasets]; %#ok<AGROW>
end
all_dataset_keys = extreme_unique([all_dataset_keys, {'kA_1', 'Shear_1_5', 'Tissue_784'}]);

bundles = struct();
for i = 1 : numel(all_dataset_keys)
    key = all_dataset_keys{i};
    try
        bundles.(matlab.lang.makeValidName(key)) = extreme_load_dataset_bundle(key, data_root, cache_dir, ...
            dataset_to_analyze, h_bins_for_quality_analysis);
    catch ME
        warning('DCG:extremeLoadFailed', 'Could not load %s for extreme composites: %s', key, ME.message);
    end
end

output_paths = struct();
output_paths.raw_mae = extreme_plot_family_curve_grid(families, model_rows, bundles, ...
    'rawmae', 'Mean graph MAE', output_dir, save_png);
output_paths.log2_nmae = extreme_plot_family_curve_grid(families, model_rows, bundles, ...
    'log2nmae', 'log2(normalized graph MAE)', output_dir, save_png);
output_paths.log2_mae = extreme_plot_family_curve_grid(families, model_rows, bundles, ...
    'log2mae', 'log2(mean graph MAE)', output_dir, save_png);
output_paths.condition_raw_mae = extreme_plot_family_condition_grid(families, model_rows, bundles, ...
    'rawmae', 'Mean graph MAE', output_dir, save_png);
output_paths.condition_log2_nmae = extreme_plot_family_condition_grid(families, model_rows, bundles, ...
    'log2nmae', 'nMAE_{Id}', output_dir, save_png);
output_paths.condition_log2_mae = extreme_plot_family_condition_grid(families, model_rows, bundles, ...
    'log2mae', 'log2(mean graph MAE)', output_dir, save_png);
if isequal(extreme_cfg(DCG_CONFIG, 'extreme_skip_embedding', false), true)
    output_paths.embedding_overlay = '';
    output_paths.embedding_per_edge = '';
    fprintf('[DCG_make_extreme_task_composites_internal] extreme_skip_embedding=true, skipped embedding grids.\n');
else
    [output_paths.embedding_overlay, output_paths.embedding_per_edge] = extreme_plot_embedding_grids( ...
        model_rows, bundles, data_root, dataset_to_analyze, output_dir, save_png, DCG_CONFIG);
end

extreme_write_assumptions(output_dir, families, model_rows, output_paths);
fprintf('[DCG_make_extreme_task_composites_internal] wrote composites under: %s\n', output_dir);
end


function v = extreme_cfg(cfg, name, default_value)
if isstruct(cfg) && isfield(cfg, name) && ~isempty(cfg.(name))
    v = cfg.(name);
else
    v = default_value;
end
end


function keys = extreme_unique(keys_in)
keys = {};
for i = 1 : numel(keys_in)
    if isempty(keys_in{i}), continue; end
    if ~ismember(keys_in{i}, keys)
        keys{end+1} = keys_in{i}; %#ok<AGROW>
    end
end
end


function families = extreme_task_families()
rgb = {[1 0 0], [0 0.62 0], [0 0 1]};
families = repmat(struct('name','', 'title','', 'datasets',{{}}, 'labels',{{}}, 'colors',{rgb}), 1, 3);
families(1).name = 'kA';
families(1).title = 'kA';
families(1).datasets = {'v1_2_16_W', 'kA_10', 'kA_1'};
families(1).labels = {'kA=100 (v1 16)', 'kA=10', 'kA=1'};
families(2).name = 'shear';
families(2).title = 'Shear';
families(2).datasets = {'v1_2_16_W', 'Shear_1_2', 'Shear_1_5'};
families(2).labels = {'lambda=1', 'lambda=1.2', 'lambda=1.5'};
families(3).name = 'tissue_size';
families(3).title = 'Tissue size';
families(3).datasets = {'v1_2_16_W', 'Tissue_484', 'Tissue_784'};
families(3).labels = {'256 cells', '484 cells', '784 cells'};
end


function bundle = extreme_load_dataset_bundle(dataset_key, data_root, cache_dir, dataset_to_analyze, h_bins_for_quality_analysis)
meta = extreme_dataset_meta(dataset_key);
summary_file = fullfile(cache_dir, [meta.summary_dataset, ' - results_summary.mat']);
if exist(summary_file, 'file') ~= 2
    error('summary file not found: %s', summary_file);
end

loaded = load(summary_file, 'S', 'all_models', 'tasks', 'data_sets');
S = loaded.S;
all_models = loaded.all_models;
loaded_tasks = loaded.tasks;
data_sets = loaded.data_sets;
all_models{strcmp(all_models, 'no_learning')} = 'Baseline';

if ~isequal(data_sets, {'test'})
    error('summary %s was not generated from the test split only', summary_file);
end
if ~all(ismember(meta.tasks, loaded_tasks))
    error('summary %s does not contain requested tasks [%s]', summary_file, strjoin(meta.tasks, ', '));
end
tasks = meta.tasks;

if isfield(S, 'hexagonality'), S.hexagonality = S.hexagonality(1,:)'; end
if isfield(S, 'disorder'), S.disorder = S.disorder(1,:)'; end
if ~isempty(meta.size_bins_to_keep)
    S = keep_summary_size_bins(S, tasks, meta.size_bins_to_keep);
end

max_cell_dist = meta.max_cell_dist;
for t = 1 : numel(tasks)
    pe = S.prediction_errors.(tasks{t});
    for ii = 1 : numel(pe)
        if isempty(pe{ii}), continue; end
        ds_cells = pe{ii}.(dataset_to_analyze);
        max_cell_dist = max(max_cell_dist, max(cellfun(@numel, ds_cells)));
    end
end

[MAE_individuals, MAE_size_avg_lin, MAE_size_sd_lin, ~, ~, ~, ~, S] = extract_MAEs(tasks, S, all_models, ...
    dataset_to_analyze, max_cell_dist, h_bins_for_quality_analysis);
S = calculate_normalization_factors(S, dataset_to_analyze, h_bins_for_quality_analysis, max_cell_dist);

[~, MAE_size_avg_lin, MAE_size_sd_lin, ~, ~, MAE_dists_avg_lin, MAE_dists_sd_lin, ~] = extract_MAEs(tasks, S, all_models, ...
    dataset_to_analyze, max_cell_dist, h_bins_for_quality_analysis, 0);

[~, MAE_size_avg_log, MAE_size_sd_log, ~, ~, MAE_dists_avg_log, MAE_dists_sd_log, ~] = extract_MAEs(tasks, S, all_models, ...
    dataset_to_analyze, max_cell_dist, h_bins_for_quality_analysis, 1);

S_norm_size = S;
S_norm_size.prediction_errors = extreme_normalize_prediction_errors_by_size(S_norm_size.prediction_errors, ...
    S_norm_size.normalization.(dataset_to_analyze), tasks, dataset_to_analyze);
[~, MAE_size_avg_norm, MAE_size_sd_norm, ~, ~, ~, ~, ~] = extract_MAEs(tasks, S_norm_size, all_models, ...
    dataset_to_analyze, max_cell_dist, h_bins_for_quality_analysis, 1, 1);

S_norm = S;
S_norm.prediction_errors = extreme_normalize_prediction_errors_by_dist(S_norm.prediction_errors, ...
    S_norm.normalization.(dataset_to_analyze), tasks, dataset_to_analyze);
[~, ~, ~, ~, ~, MAE_dists_avg_norm, MAE_dists_sd_norm, ~] = extract_MAEs(tasks, S_norm, all_models, ...
    dataset_to_analyze, max_cell_dist, h_bins_for_quality_analysis, 1, 1);

baseline_idx = find(strcmp(all_models, 'Baseline') | strcmp(all_models, 'no_learning'), 1);
if isempty(baseline_idx), baseline_idx = numel(all_models); end
MAE_size_avg_norm = force_baseline_reference_to_zero(MAE_size_avg_norm, tasks, baseline_idx);
MAE_size_sd_norm = force_baseline_reference_to_zero(MAE_size_sd_norm, tasks, baseline_idx);
MAE_dists_avg_norm = force_baseline_reference_to_zero(MAE_dists_avg_norm, tasks, baseline_idx);
MAE_dists_sd_norm = force_baseline_reference_to_zero(MAE_dists_sd_norm, tasks, baseline_idx);

bundle = struct();
bundle.dataset_key = dataset_key;
bundle.data_root = data_root;
bundle.summary_file = summary_file;
bundle.S = S;
bundle.MAE_individuals = MAE_individuals;
bundle.all_models = all_models;
bundle.tasks = tasks;
bundle.dataset_to_analyze = dataset_to_analyze;
bundle.max_cell_dist = max_cell_dist;
bundle.rawmae.avg = MAE_dists_avg_lin;
bundle.rawmae.sd = MAE_dists_sd_lin;
bundle.rawmae.size_avg = MAE_size_avg_lin;
bundle.rawmae.size_sd = MAE_size_sd_lin;
bundle.log2mae.avg = MAE_dists_avg_log;
bundle.log2mae.sd = MAE_dists_sd_log;
bundle.log2mae.size_avg = MAE_size_avg_log;
bundle.log2mae.size_sd = MAE_size_sd_log;
bundle.log2nmae.avg = MAE_dists_avg_norm;
bundle.log2nmae.sd = MAE_dists_sd_norm;
bundle.log2nmae.size_avg = MAE_size_avg_norm;
bundle.log2nmae.size_sd = MAE_size_sd_norm;
end


function prediction_errors = extreme_normalize_prediction_errors_by_size(prediction_errors, norm_struct, tasks, dataset_to_analyze)
for t = 1 : numel(tasks)
    task = tasks{t};
    for s = 1 : size(prediction_errors.(task), 1)
        for ss = 1 : size(prediction_errors.(task), 2)
            if isempty(prediction_errors.(task){s,ss}), continue; end
            if numel(norm_struct.per_size) < ss || ~isfinite(norm_struct.per_size(ss)) || norm_struct.per_size(ss) <= 0
                for g = 1 : numel(prediction_errors.(task){s,ss}.(dataset_to_analyze))
                    for h = 1 : numel(prediction_errors.(task){s,ss}.(dataset_to_analyze){g})
                        prediction_errors.(task){s,ss}.(dataset_to_analyze){g}{h}(:) = NaN;
                    end
                end
                continue;
            end
            denom = norm_struct.per_size(ss);
            for g = 1 : numel(prediction_errors.(task){s,ss}.(dataset_to_analyze))
                for h = 1 : numel(prediction_errors.(task){s,ss}.(dataset_to_analyze){g})
                    prediction_errors.(task){s,ss}.(dataset_to_analyze){g}{h} = ...
                        prediction_errors.(task){s,ss}.(dataset_to_analyze){g}{h} ./ denom;
                end
            end
        end
    end
end
end


function meta = extreme_dataset_meta(dataset_key)
meta = struct('summary_dataset', dataset_key, 'tasks', {{'lengths_to_lengths'}}, ...
    'size_bins_to_keep', [], 'max_cell_dist', 24);
switch dataset_key
    case 'v1_2_16_W'
        meta.summary_dataset = 'v1_W';
        meta.size_bins_to_keep = 5;
        meta.max_cell_dist = 24;
    case {'kA_1', 'kA_10', 'Shear_1_2', 'Shear_1_5'}
        meta.summary_dataset = dataset_key;
        meta.max_cell_dist = 24;
    case 'Tissue_484'
        meta.summary_dataset = dataset_key;
        meta.max_cell_dist = 40;
    case 'Tissue_784'
        meta.summary_dataset = dataset_key;
        meta.max_cell_dist = 50;
    otherwise
        meta.summary_dataset = dataset_key;
end
end


function prediction_errors = extreme_normalize_prediction_errors_by_dist(prediction_errors, norm_struct, tasks, dataset_to_analyze)
for t = 1 : numel(tasks)
    task = tasks{t};
    for s = 1 : size(prediction_errors.(task), 1)
        for ss = 1 : size(prediction_errors.(task), 2)
            if isempty(prediction_errors.(task){s,ss}), continue; end
            if numel(norm_struct.per_dist) < ss || isempty(norm_struct.per_dist{ss}), continue; end
            denoms = norm_struct.per_dist{ss};
            for g = 1 : numel(prediction_errors.(task){s,ss}.(dataset_to_analyze))
                for h = 1 : numel(prediction_errors.(task){s,ss}.(dataset_to_analyze){g})
                    if h > numel(denoms) || ~isfinite(denoms(h)) || denoms(h) <= 0
                        prediction_errors.(task){s,ss}.(dataset_to_analyze){g}{h}(:) = NaN;
                    else
                        prediction_errors.(task){s,ss}.(dataset_to_analyze){g}{h} = ...
                            prediction_errors.(task){s,ss}.(dataset_to_analyze){g}{h} ./ denoms(h);
                    end
                end
            end
        end
    end
end
end


function out_path = extreme_plot_family_curve_grid(families, model_rows, bundles, metric_name, y_label, output_dir, save_png)
fig = figure('Position', [60 60 1180 1540], 'Color', 'w', 'NumberTitle', 'off', ...
    'Name', ['Extreme task families - ', metric_name]);
tl = tiledlayout(fig, numel(model_rows), numel(families), 'TileSpacing', 'compact', 'Padding', 'compact');
axes_list = gobjects(numel(model_rows), numel(families));
global_y = [inf, -inf];
draw_baseline_curves = strcmp(metric_name, 'rawmae') || strcmp(metric_name, 'log2mae');

for mrow = 1 : numel(model_rows)
    for f = 1 : numel(families)
        ax = nexttile(tl, (mrow - 1) * numel(families) + f);
        axes_list(mrow, f) = ax;
        hold(ax, 'on');
        family = families(f);
        max_x = 0;
        for d = 1 : numel(family.datasets)
            key = family.datasets{d};
            bname = matlab.lang.makeValidName(key);
            if ~isfield(bundles, bname)
                continue;
            end
            [x, y, e] = extreme_curve_for_model(bundles.(bname), metric_name, model_rows{mrow});
            if isempty(x)
                continue;
            end
            extreme_draw_curve(ax, x, y, e, family.colors{d}, family.labels{d});
            max_x = max(max_x, max(x));
            finite_band = [y(:) - e(:); y(:) + e(:)];
            finite_band = finite_band(isfinite(finite_band));
            if ~isempty(finite_band)
                global_y(1) = min(global_y(1), min(finite_band));
                global_y(2) = max(global_y(2), max(finite_band));
            end
        end
        if draw_baseline_curves
            for d = 1 : numel(family.datasets)
                key = family.datasets{d};
                bname = matlab.lang.makeValidName(key);
                if ~isfield(bundles, bname)
                    continue;
                end
                [xB, yB, eB] = extreme_curve_for_model(bundles.(bname), metric_name, 'Baseline');
                if isempty(xB)
                    continue;
                end
                extreme_draw_baseline_curve(ax, xB, yB, eB);
                max_x = max(max_x, max(xB));
                finite_band = [yB(:) - eB(:); yB(:) + eB(:)];
                finite_band = finite_band(isfinite(finite_band));
                if ~isempty(finite_band)
                    global_y(1) = min(global_y(1), min(finite_band));
                    global_y(2) = max(global_y(2), max(finite_band));
                end
            end
        end
        if strcmp(metric_name, 'log2nmae') && max_x > 0
            plot(ax, [-0.5, max_x + 0.5], [0, 0], '-k', ...
                'LineWidth', 1.0, 'HandleVisibility', 'off');
            global_y(1) = min(global_y(1), 0);
            global_y(2) = max(global_y(2), 0);
        end
        if max_x > 0, xlim(ax, [-0.5, max_x + 0.5]); else, xlim(ax, [-0.5, 0.5]); end
        if mrow == 1
            title(ax, family.title);
        end
        if f == 1
            ylabel(ax, {model_rows{mrow}, y_label});
        else
            ylabel(ax, '');
        end
        if mrow == numel(model_rows)
            xlabel(ax, 'Hops from T1 interface');
        else
            xlabel(ax, '');
        end
        axis(ax, 'square');
    end
end

if isfinite(global_y(1)) && isfinite(global_y(2)) && global_y(1) < global_y(2)
    yl = extreme_padded_limits(global_y(1), global_y(2), false);
    for i = 1 : numel(axes_list)
        if isgraphics(axes_list(i)), ylim(axes_list(i), yl); end
    end
end

if strcmp(metric_name, 'log2nmae')
    label = 'log2 nMAE';
elseif strcmp(metric_name, 'rawmae')
    label = 'raw MAE';
else
    label = 'log2 MAE';
end
out_path = fullfile(output_dir, ['Extreme task families by model (', label, ').fig']);
extreme_save_figure(fig, out_path, save_png, 300);
end


function [x, y, e] = extreme_curve_for_model(bundle, metric_name, model_name)
x = []; y = []; e = [];
model_idx = find(strcmpi(bundle.all_models, model_name), 1);
if isempty(model_idx), return; end
task = bundle.tasks{1};
avg_cells = bundle.(metric_name).avg.(task);
sd_cells = bundle.(metric_name).sd.(task);
nonempty = find(~cellfun(@isempty, avg_cells));
if isempty(nonempty), return; end
siz = nonempty(end);
if model_idx > size(avg_cells{siz}, 2), return; end
y = avg_cells{siz}(:, model_idx);
e = sd_cells{siz}(:, model_idx);
x = (0 : numel(y) - 1)';
keep = isfinite(y);
x = x(keep);
y = y(keep);
e = e(keep);
e(~isfinite(e)) = 0;
end


function out_path = extreme_plot_family_condition_grid(families, model_rows, bundles, metric_name, y_label, output_dir, save_png)
fig = figure('Position', [80 80 1180 1540], 'Color', 'w', 'NumberTitle', 'off', ...
    'Name', ['Extreme task-family condition summary - ', metric_name]);
tl = tiledlayout(fig, numel(model_rows), numel(families), 'TileSpacing', 'compact', 'Padding', 'compact');
axes_list = gobjects(numel(model_rows), numel(families));
global_y = [inf, -inf];
draw_baseline = strcmp(metric_name, 'rawmae') || strcmp(metric_name, 'log2mae');

for mrow = 1 : numel(model_rows)
    for f = 1 : numel(families)
        ax = nexttile(tl, (mrow - 1) * numel(families) + f);
        axes_list(mrow, f) = ax;
        hold(ax, 'on');
        family = families(f);
        x = 1 : numel(family.datasets);
        y = nan(size(x));
        e = nan(size(x));
        yB = nan(size(x));
        eB = nan(size(x));

        for d = 1 : numel(family.datasets)
            key = family.datasets{d};
            bname = matlab.lang.makeValidName(key);
            if ~isfield(bundles, bname)
                continue;
            end
            [y(d), e(d)] = extreme_condition_value_for_model(bundles.(bname), metric_name, model_rows{mrow});
            if draw_baseline
                [yB(d), eB(d)] = extreme_condition_value_for_model(bundles.(bname), metric_name, 'Baseline');
            end
        end

        if draw_baseline
            okB = isfinite(yB);
            if any(okB)
                errorbar(ax, x(okB), yB(okB), eB(okB), '--k', ...
                    'LineWidth', 1.0, 'CapSize', 4, 'DisplayName', 'Baseline');
                finite_band = [yB(okB) - eB(okB), yB(okB) + eB(okB)];
                global_y(1) = min(global_y(1), min(finite_band));
                global_y(2) = max(global_y(2), max(finite_band));
            end
        elseif strcmp(metric_name, 'log2nmae')
            plot(ax, [0.5, numel(x) + 0.5], [0, 0], '-k', ...
                'LineWidth', 1.0, 'HandleVisibility', 'off');
            global_y(1) = min(global_y(1), 0);
            global_y(2) = max(global_y(2), 0);
        end

        ok = isfinite(y);
        if any(ok)
            plot(ax, x(ok), y(ok), '-', 'Color', [0.35 0.35 0.35], ...
                'LineWidth', 0.8, 'HandleVisibility', 'off');
            for d = find(ok)
                errorbar(ax, x(d), y(d), e(d), 'o', ...
                    'Color', family.colors{d}, 'MarkerFaceColor', family.colors{d}, ...
                    'MarkerEdgeColor', family.colors{d}, 'LineWidth', 1.2, ...
                    'MarkerSize', 5.5, 'CapSize', 5, 'DisplayName', family.labels{d});
            end
            finite_band = [y(ok) - e(ok), y(ok) + e(ok)];
            global_y(1) = min(global_y(1), min(finite_band));
            global_y(2) = max(global_y(2), max(finite_band));
        end

        xlim(ax, [0.5, numel(x) + 0.5]);
        set(ax, 'XTick', x);
        if mrow == numel(model_rows)
            set(ax, 'XTickLabel', family.labels, 'XTickLabelRotation', 35, 'FontSize', 7);
        else
            set(ax, 'XTickLabel', {});
        end
        if mrow == 1
            title(ax, family.title);
        end
        if f == 1
            ylabel(ax, {model_rows{mrow}, y_label});
        else
            ylabel(ax, '');
        end
        grid(ax, 'on');
        axis(ax, 'square');
    end
end

if isfinite(global_y(1)) && isfinite(global_y(2)) && global_y(1) < global_y(2)
    yl = extreme_padded_limits(global_y(1), global_y(2), strcmp(metric_name, 'rawmae'));
    for i = 1 : numel(axes_list)
        if isgraphics(axes_list(i)), ylim(axes_list(i), yl); end
    end
end

if strcmp(metric_name, 'log2nmae')
    label = 'log2 nMAE';
elseif strcmp(metric_name, 'rawmae')
    label = 'raw MAE';
else
    label = 'log2 MAE';
end
out_path = fullfile(output_dir, ['Extreme task families condition summary (', label, ').fig']);
extreme_save_figure(fig, out_path, save_png, 300);
end


function [y, e] = extreme_condition_value_for_model(bundle, metric_name, model_name)
y = NaN; e = NaN;
model_idx = find(strcmpi(bundle.all_models, model_name), 1);
if isempty(model_idx), return; end
task = bundle.tasks{1};
if ~isfield(bundle.(metric_name), 'size_avg') || ~isfield(bundle.(metric_name), 'size_sd')
    return;
end
avg_mat = bundle.(metric_name).size_avg.(task);
sd_mat = bundle.(metric_name).size_sd.(task);
if isempty(avg_mat) || model_idx > size(avg_mat, 2), return; end
% Select the populated size row for THIS model, not "any finite" row. The
% log2(nMAE) pass forces the Baseline column to zero, including rows whose
% model columns are otherwise empty/NaN; an any-finite selector would then
% accidentally choose a blank v1 size row and drop the red default condition.
rows = find(isfinite(avg_mat(:, model_idx)));
if isempty(rows), return; end
siz = rows(end);
y = avg_mat(siz, model_idx);
if ~isempty(sd_mat) && siz <= size(sd_mat, 1) && model_idx <= size(sd_mat, 2)
    e = sd_mat(siz, model_idx);
end
if ~isfinite(e), e = 0; end
end


function extreme_draw_curve(ax, x, y, e, color, display_name)
x = x(:); y = y(:); e = e(:);
if isempty(x), return; end
h = shadedErrorBar(x, y, e, 'lineprops', {'-', 'Color', color}, 'transparent', true);
if isstruct(h) && isfield(h, 'mainLine') && isgraphics(h.mainLine)
    set(h.mainLine, 'DisplayName', display_name);
end
end


function extreme_draw_baseline_curve(ax, x, y, e)
x = x(:); y = y(:); e = e(:);
if isempty(x), return; end
h = shadedErrorBar(x, y, e, 'lineprops', {'-', 'Color', [0 0 0]}, 'transparent', true);
if isstruct(h)
    if isfield(h, 'mainLine') && isgraphics(h.mainLine)
        set(h.mainLine, 'LineWidth', 1.0, 'HandleVisibility', 'off');
    end
    if isfield(h, 'patch') && isgraphics(h.patch)
        set(h.patch, 'FaceAlpha', 0.06, 'HandleVisibility', 'off');
    end
    if isfield(h, 'edge') && all(isgraphics(h.edge))
        set(h.edge, 'HandleVisibility', 'off');
    end
end
end


function lims = extreme_padded_limits(ymin, ymax, clamp_zero)
if ~isfinite(ymin) || ~isfinite(ymax)
    lims = [0, 1];
    return;
end
if ymin == ymax
    pad = max(0.1, abs(ymin) * 0.05);
else
    pad = 0.05 * (ymax - ymin);
end
lims = [ymin - pad, ymax + pad];
if clamp_zero
    lims(1) = min(0, lims(1));
end
end


function [overlay_path, per_edge_path] = extreme_plot_embedding_grids(model_rows, bundles, data_root, dataset_to_analyze, output_dir, save_png, DCG_CONFIG)
extreme_datasets = {'kA_1', 'Shear_1_5', 'Tissue_784'};
extreme_titles = {'kA = 1', 'Shear 1.5', 'Tissue 784'};
[sel_grid, geom_grid] = extreme_collect_embedding_geometry(model_rows, extreme_datasets, bundles, ...
    data_root, dataset_to_analyze, DCG_CONFIG);

figA = figure('Position', [40 40 1180 1540], 'Color', 'w', 'NumberTitle', 'off', ...
    'Name', 'Extreme median embedding overlays');
tlA = tiledlayout(figA, numel(model_rows), numel(extreme_datasets), 'TileSpacing', 'compact', 'Padding', 'compact');
for r = 1 : numel(model_rows)
    for c = 1 : numel(extreme_datasets)
        ax = nexttile(tlA, (r - 1) * numel(extreme_datasets) + c);
        gg = geom_grid(r,c);
        ss = sel_grid(r,c);
        if gg.ok
            emb_draw_overlay(ax, gg.GTu, gg.GTseg, gg.PRu, gg.PRseg, ...
                'k', [0 0.2 0.85], [0 0.6 0], [0.85 0 0], 0.55, 2.3, gg.xl, gg.yl);
            title(ax, {sprintf('%s | %s', model_rows{r}, extreme_titles{c}), ...
                sprintf('50%% MAE=%s seed=%d graph=%d', emb_fmt_mae(ss.sMAE), ss.seedrow, ss.best_r)}, ...
                'FontWeight', 'normal', 'FontSize', 8, 'Interpreter', 'none');
        else
            axis(ax, 'off');
            title(ax, {sprintf('%s | %s', model_rows{r}, extreme_titles{c}), ss.reason}, ...
                'FontWeight', 'normal', 'FontSize', 8, 'Interpreter', 'none');
        end
    end
end
overlay_path = fullfile(output_dir, 'Extreme median embeddings overlay.fig');
extreme_save_figure(figA, overlay_path, save_png, 250);

figB = figure('Position', [40 40 1180 1540], 'Color', 'w', 'NumberTitle', 'off', ...
    'Name', 'Extreme median embedding per-edge errors');
tlB = tiledlayout(figB, numel(model_rows), numel(extreme_datasets), 'TileSpacing', 'compact', 'Padding', 'compact');
useLogColor = strcmpi(extreme_cfg(DCG_CONFIG, 'embed_color_scale', 'linear'), 'log');
colorPct = extreme_cfg(DCG_CONFIG, 'embed_color_percentiles', []);
for r = 1 : numel(model_rows)
    for c = 1 : numel(extreme_datasets)
        ax = nexttile(tlB, (r - 1) * numel(extreme_datasets) + c);
        gg = geom_grid(r,c);
        ss = sel_grid(r,c);
        if gg.ok
            vi = gg.valsB(:);
            vi = vi(isfinite(vi));
            [lo, hi] = emb_color_limits(vi, useLogColor, colorPct);
            emb_draw_colored(ax, gg.segsB, gg.valsB, lo, hi, 1.25, useLogColor);
            axis(ax, 'equal'); axis(ax, 'off'); xlim(ax, gg.xl); ylim(ax, gg.yl);
            colormap(ax, turbo); clim(ax, [lo hi]);
            if useLogColor, set(ax, 'ColorScale', 'log'); else, set(ax, 'ColorScale', 'linear'); end
            cb = colorbar(ax);
            emb_configure_embedding_colorbar(cb, lo, hi, useLogColor);
            cb.Label.String = '|l - L_{pred}|';
            title(ax, {sprintf('%s | %s', model_rows{r}, extreme_titles{c}), ...
                sprintf('MAE(emb,pred)=%s', emb_fmt_mae(gg.maePT))}, ...
                'FontWeight', 'normal', 'FontSize', 8, 'Interpreter', 'none');
        else
            axis(ax, 'off');
            title(ax, {sprintf('%s | %s', model_rows{r}, extreme_titles{c}), ss.reason}, ...
                'FontWeight', 'normal', 'FontSize', 8, 'Interpreter', 'none');
        end
    end
end
per_edge_path = fullfile(output_dir, 'Extreme median embeddings per-edge error.fig');
extreme_save_figure(figB, per_edge_path, save_png, 250);
end


function [sel_grid, geom_grid] = extreme_collect_embedding_geometry(model_rows, dataset_keys, bundles, data_root, dataset_to_analyze, DCG_CONFIG)
blank_sel = extreme_blank_selection('not selected');
blank_geom = extreme_blank_geometry();
sel_grid = repmat(blank_sel, numel(model_rows), numel(dataset_keys));
geom_grid = repmat(blank_geom, numel(model_rows), numel(dataset_keys));

engine = extreme_cfg(DCG_CONFIG, 'embed_engine', '');
workdir = extreme_cfg(DCG_CONFIG, 'embed_workdir', fullfile(tempdir, 'dcg_springs_embed'));
stdRoot = extreme_cfg(DCG_CONFIG, 'embed_vt2d_std', '');
revRoot = extreme_cfg(DCG_CONFIG, 'embed_vt2d_rev', '');
predRoot = extreme_cfg(DCG_CONFIG, 'embed_pred_root', data_root);
indsRoot = extreme_cfg(DCG_CONFIG, 'embed_inds_root', fullfile(data_root, 'inds'));
recompute = extreme_cfg(DCG_CONFIG, 'embed_recompute', false);
shearVt2d = extreme_cfg(DCG_CONFIG, 'embed_shear_affine_vt2d', true);
emb_consolidated = DCG_consolidated_paths('is_consolidated', predRoot);

if exist(engine, 'file') ~= 2
    for i = 1 : numel(sel_grid)
        sel_grid(i).reason = sprintf('engine not found: %s', engine);
    end
    return;
end

variants = {'gt',5; 'pred',6};
runRD = {}; runVT = {}; runPT = {}; runSI = {};
for r = 1 : numel(model_rows)
    for c = 1 : numel(dataset_keys)
        key = dataset_keys{c};
        bname = matlab.lang.makeValidName(key);
        if ~isfield(bundles, bname)
            sel_grid(r,c).reason = 'summary not loaded';
            continue;
        end
        bundle = bundles.(bname);
        sel = extreme_select_embedding_example(bundle, model_rows{r}, 50, predRoot, indsRoot, ...
            emb_consolidated, dataset_to_analyze, shearVt2d);
        if ~sel.ok
            sel_grid(r,c) = sel;
            continue;
        end
        vt2d = emb_resolve_vt2d(sel.sim_id, stdRoot, revRoot, key);
        if isempty(vt2d)
            sel.ok = false;
            sel.reason = 'missing vt2d';
            sel_grid(r,c) = sel;
            continue;
        end
        if shearVt2d
            [vt2d, shear_msg] = emb_prepare_shear_vt2d(vt2d, workdir, key);
            if ~isempty(shear_msg)
                fprintf('[extreme embed shear vt2d] %s: %s\n', sel.cfgkey, shear_msg);
            end
        end
        sel.vt2d = vt2d;
        [fm, fp] = emb_flip(sel.wpred_file, sel.sim_id);
        if isempty(fm), [fm, fp] = emb_flip(sel.pred_file, sel.sim_id); end
        sel.flipMap = fm;
        sel.flipPairs = fp;
        for v = 1 : size(variants, 1)
            tag = variants{v,1};
            srccol = variants{v,2};
            rd = fullfile(workdir, 'emb', sel.cfgkey, tag);
            sel.rundir.(tag) = rd;
            marker = fullfile(rd, 'output', ['out_', sel.sim_id]);
            if recompute || exist(marker, 'file') ~= 2
                if exist(fullfile(rd, 'output'), 'dir') ~= 7, mkdir(fullfile(rd, 'output')); end
                tmp = fullfile(workdir, sprintf('%s_%s_%s', tag, sel.cfgkey, sel.sim_id));
                try
                    emb_write_pred(tmp, sel.sim_id, sel.pred_file, srccol, fm);
                    runRD{end+1} = rd; %#ok<AGROW>
                    runVT{end+1} = vt2d; %#ok<AGROW>
                    runPT{end+1} = tmp; %#ok<AGROW>
                    runSI{end+1} = sel.sim_id; %#ok<AGROW>
                catch ME
                    warning('DCG:extremeEmbedWrite', 'Could not write embedding input for %s/%s: %s', sel.cfgkey, tag, ME.message);
                end
            end
        end
        sel_grid(r,c) = sel;
    end
end

if ~isempty(runRD)
    fprintf('[extreme embed] %d relaxations queued for missing median examples.\n', numel(runRD));
    eng = engine;
    parfor q = 1 : numel(runRD)
        emb_run_engine(eng, runVT{q}, runPT{q}, runSI{q}, runRD{q});
    end
end

for r = 1 : numel(model_rows)
    for c = 1 : numel(dataset_keys)
        sel = sel_grid(r,c);
        if ~sel.ok, continue; end
        geom_grid(r,c) = extreme_load_embedding_geometry(sel);
        if ~geom_grid(r,c).ok
            sel_grid(r,c).reason = 'geometry failed';
        end
    end
end
end


function sel = extreme_select_embedding_example(bundle, model_name, pct, predRoot, indsRoot, emb_consolidated, dataset_to_analyze, shearVt2d)
sel = extreme_blank_selection('not selected');
sel.pct = pct;
sel.model = model_name;
task = bundle.tasks{1};
model_idx = find(strcmpi(bundle.all_models, model_name), 1);
if isempty(model_idx)
    sel.reason = 'model not available';
    return;
end

scoresArr = []; repsArr = []; sizesArr = []; graphsArr = [];
for r = 1 : size(bundle.MAE_individuals.(task), 1)
    for siz = 1 : size(bundle.MAE_individuals.(task), 2)
        cm = bundle.MAE_individuals.(task){r,siz};
        if isempty(cm) || model_idx > size(cm, 2), continue; end
        vg = find(isfinite(cm(:, model_idx)));
        scoresArr = [scoresArr; cm(vg, model_idx)]; %#ok<AGROW>
        repsArr = [repsArr; repmat(r, numel(vg), 1)]; %#ok<AGROW>
        sizesArr = [sizesArr; repmat(siz, numel(vg), 1)]; %#ok<AGROW>
        graphsArr = [graphsArr; vg(:)]; %#ok<AGROW>
    end
end
if isempty(scoresArr)
    sel.reason = 'no finite graph MAEs';
    return;
end

[scoresSorted, order] = sort(scoresArr, 'ascend');
rank_i = max(1, min(numel(scoresSorted), round(pct * numel(scoresSorted) / 100)));
rec_i = order(rank_i);
r = repsArr(rec_i);
siz = sizesArr(rec_i);
g = graphsArr(rec_i);
wstr = 'W';
prefix = emb_dataset_prefix(bundle.dataset_key, siz, wstr);
wprefix = emb_dataset_prefix(bundle.dataset_key, siz, 'W');

try
    if emb_consolidated
        inds_dir = DCG_consolidated_paths('inds_dir', predRoot, prefix);
        if isempty(inds_dir), error('no split folder for prefix %s', prefix); end
        inds_filename = fullfile(inds_dir, [dataset_to_analyze, '.inds']);
    else
        inds_filename = fullfile(indsRoot, prefix, [dataset_to_analyze, '.inds']);
    end
    cf = fopen(inds_filename, 'rt');
    if cf < 0, error('inds file not found: %s', inds_filename); end
    inds = fread(cf, inf, '*char')';
    fclose(cf);
    inds = str2num(inds) + 1; %#ok<ST2NM>
    curr_graph_ind = inds(g);

    if emb_consolidated
        predf = DCG_consolidated_paths('pred_file', predRoot, prefix, model_name, r - 1);
        wpredf = DCG_consolidated_paths('pred_file', predRoot, wprefix, model_name, r - 1);
    else
        predf = fullfile(predRoot, sprintf('pred_%s__%s_s%d.txt', prefix, model_name, r - 1));
        wpredf = fullfile(predRoot, sprintf('pred_%s__%s_s%d.txt', wprefix, model_name, r - 1));
    end
    if exist(predf, 'file') ~= 2, error('pred file not found: %s', predf); end
    gn = load_dataset(predf);
    egn = gn{curr_graph_ind};
    sim_id = [egn, '.txt'];
    sMAE = emb_sMAE(bundle.S, task, dataset_to_analyze, r, siz, g, model_idx);
    prows = emb_read_block(predf, sim_id);
    fileMAE = NaN;
    if ~isempty(prows)
        c_true = cellfun(@(rr) str2double(rr{end-1}), prows);
        c_pred = cellfun(@(rr) str2double(rr{end}), prows);
        fileMAE = mean(abs(c_pred - c_true), 'omitnan');
    end
    if isfinite(sMAE) && isfinite(fileMAE) && abs(fileMAE - sMAE) > max(5e-3, 0.1 * abs(sMAE))
        error('file MAE %.4f != S MAE %.4f', fileMAE, sMAE);
    end

    sel.ok = true;
    sel.reason = '';
    sel.dataset = bundle.dataset_key;
    sel.model = model_name;
    sel.subset_size = siz;
    sel.best_r = g;
    sel.seedrow = r;
    sel.model_idx = model_idx;
    sel.sMAE = sMAE;
    sel.sim_id = sim_id;
    sel.pred_file = predf;
    sel.wpred_file = wpredf;
    sel.cfgkey = matlab.lang.makeValidName(sprintf('%s_%s_%s', wstr, model_name, egn));
    if shearVt2d && ~isempty(emb_shear_lambda(bundle.dataset_key))
        sel.cfgkey = [sel.cfgkey, '_affineShear'];
    end
catch ME
    sel.ok = false;
    sel.reason = ME.message;
end
end


function sel = extreme_blank_selection(reason)
sel = struct('ok', false, 'reason', reason, 'pct', 50, 'dataset', '', 'model', '', ...
    'subset_size', 1, 'best_r', 0, 'seedrow', 1, 'model_idx', NaN, 'sMAE', NaN, ...
    'sim_id', '', 'pred_file', '', 'wpred_file', '', 'cfgkey', '', 'vt2d', '', ...
    'flipMap', [], 'flipPairs', [], 'rundir', struct('gt', '', 'pred', ''));
end


function geom = extreme_blank_geometry()
geom = struct('ok', false, 'GTu', {{}}, 'PRu', {{}}, 'GTseg', {{}}, 'PRseg', {{}}, ...
    'segsB', {{}}, 'valsB', [], 'maePT', NaN, 'maePG', NaN, 'xl', [0 1], 'yl', [0 1]);
end


function geom = extreme_load_embedding_geometry(sel)
geom = extreme_blank_geometry();
try
    if exist(fullfile(sel.rundir.gt, 'output', ['out_', sel.sim_id]), 'file') ~= 2 || ...
            exist(fullfile(sel.rundir.pred, 'output', ['out_', sel.sim_id]), 'file') ~= 2
        return;
    end
    perio = emb_read_perio(sel.vt2d);
    GTp3 = emb_read_polys(sel.rundir.gt, 3);
    GTp1 = emb_read_polys(sel.rundir.gt, 1);
    PRp3 = emb_read_polys(sel.rundir.pred, 3);
    Nc = numel(GTp3);

    [~, cgGT] = emb_topology(GTp3, perio);
    [~, cgPR] = emb_topology(PRp3, perio);
    [adj1, ~] = emb_topology(GTp1, perio);

    fp = sel.flipPairs;
    nflip = size(fp, 1);
    newPairs = zeros(nflip, 2);
    for f = 1 : nflip
        A = fp(f, 1);
        B = fp(f, 2);
        cd = find(adj1(A,:) & adj1(B,:));
        if numel(cd) ~= 2, cd = [cd, nan(1,2)]; cd = cd(1:2); end %#ok<AGROW>
        newPairs(f,:) = cd;
    end
    if nflip == 0, seedCell = 1; else, seedCell = fp(1,1); end

    GTu = emb_unwrap_cells(GTp3, cgGT, emb_adjacency_from(cgGT, Nc), perio, seedCell);
    PRu = emb_unwrap_cells(PRp3, cgPR, emb_adjacency_from(cgPR, Nc), perio, seedCell);
    [PRu, ~] = emb_snap_frame(PRu, GTu, perio);
    [PRu, ~] = emb_align_to(PRu, GTu);

    Mp = emb_read_out(sel.rundir.pred, sel.sim_id);
    geom.GTu = GTu;
    geom.PRu = PRu;
    geom.GTseg = emb_edge_segs(GTu, cgGT, newPairs);
    geom.PRseg = emb_edge_segs(PRu, cgPR, newPairs);
    [geom.segsB, geom.valsB, ~] = emb_edge_segs_error_de_novo(PRu, cgPR, Mp(:,1:2), Mp(:,4));
    [~, valsPG, ~] = emb_edge_segs_error_de_novo(PRu, cgPR, Mp(:,1:2), Mp(:,3));
    geom.maePT = emb_mean_finite(geom.valsB);
    geom.maePG = emb_mean_finite(valsPG);

    allpts = [vertcat(GTu{:}); vertcat(PRu{:})];
    pad = 0.03 * max(max(allpts, [], 1) - min(allpts, [], 1));
    if ~isfinite(pad) || pad <= 0, pad = 1; end
    geom.xl = [min(allpts(:,1)) - pad, max(allpts(:,1)) + pad];
    geom.yl = [min(allpts(:,2)) - pad, max(allpts(:,2)) + pad];
    geom.ok = true;
catch ME
    warning('DCG:extremeEmbedGeometry', 'Could not load geometry for %s: %s', sel.cfgkey, ME.message);
end
end


function extreme_save_figure(fig, fig_path, save_png, resolution)
if nargin < 4 || isempty(resolution), resolution = 300; end
set(fig, 'Visible', 'on');
savefig(fig, fig_path);
if save_png
    png_path = regexprep(fig_path, '\.fig$', '.png');
    exportgraphics(fig, png_path, 'Resolution', resolution);
end
fprintf('[DCG extreme composites] saved %s\n', fig_path);
end


function extreme_write_assumptions(output_dir, families, model_rows, output_paths)
fid = fopen(fullfile(output_dir, 'extreme_task_composites_assumptions.txt'), 'w');
if fid < 0, return; end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'Extreme task-family composite figures\n');
fprintf(fid, 'Rows: %s\n', strjoin(model_rows, ', '));
fprintf(fid, 'Columns: kA, shear, tissue size.\n');
fprintf(fid, 'Curve colors within each column are RGB in the order listed below.\n\n');
fprintf(fid, ['Hop-distance figures plot mean graph error versus hop from the T1 edge. ' ...
    'Condition-summary figures collapse each dataset to one whole-graph value per model/condition, ' ...
    'with error bars showing SD across training seeds. Raw MAE and log2(MAE) panels include the ' ...
    'identity Baseline as a dashed black line; log2(nMAE) panels use the black y=0 baseline.\n\n']);
for f = 1 : numel(families)
    fprintf(fid, '%s:\n', families(f).title);
    for d = 1 : numel(families(f).datasets)
        fprintf(fid, '  %d. %s -> %s\n', d, families(f).datasets{d}, families(f).labels{d});
    end
end
fprintf(fid, '\nEmbedding panels use the 50th-percentile graph for each model within each extreme dataset:\n');
fprintf(fid, '  kA -> kA_1\n  shear -> Shear_1_5\n  tissue size -> Tissue_784\n');
fprintf(fid, 'PPGN/Tissue_784 is blank because no PPGN 784-cell run exists.\n\n');
fprintf(fid, 'Output paths:\n');
names = fieldnames(output_paths);
for i = 1 : numel(names)
    fprintf(fid, '  %s: %s\n', names{i}, output_paths.(names{i}));
end
end
