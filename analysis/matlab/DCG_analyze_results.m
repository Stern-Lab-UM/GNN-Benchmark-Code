%==========================================================================
% DCG_analyze_results
%
% Parse saved GNN prediction files, attach train/validation/test split indices,
% compute per-graph and per-hop prediction-error summaries, and save the
% result-summary structure used by the plotting scripts.
%
% This script is the post-prediction analyzer. It does not run Bayesian
% optimization, train neural networks, or generate prediction files. Those
% steps are handled by the Python/PyTorch model code under models/.
%
% DATA LAYOUT
%   The preferred input is a consolidated prediction snapshot:
%
%     <data_root>/
%       <task>_<model>_<W|UW>_<size>_s<seed>.pred.txt
%       splits/<key>/train.inds
%       splits/<key>/val.inds
%       splits/<key>/test.inds
%
%   Legacy flat prediction filenames are also supported through
%   DCG_consolidated_paths and read_dataset_inds.
%
% PREDICTION-FILE COLUMNS
%   W  (lengths_to_lengths), MPNN: [u v pre flag post pred]
%   UW (none_to_lengths),    MPNN: [u v post pred]
%   PPGN files may include interleaved node-feature rows, which are removed by
%   load_dataset when consider_nodes is enabled.
%
% MAIN OUTPUTS
%   output_filename             (analyses data.mat): parsed raw model data I
%   results_summary_filename    (results_summary.mat): summary S plus model,
%                                task, and split metadata
%
% HOW TO RUN
%   Set dataset directly, or pass a DCG_CONFIG struct from a driver script.
%   Set data_root through DCG_CONFIG.data_root, the DCG_DATA_ROOT environment
%   variable, or an untracked DCG_local_config.m file.
%
%   After this script finishes, use DCG_plot_results or DCG_plot_everything to
%   generate figures from the saved summaries.
%==========================================================================
% 1) DATASET SELECTION  --  set `dataset` below, then Run.
%==========================================================================
% The benchmark predictions live in one flat folder, filename-encoded as
% pred_<prefix>__<model>_s<seed>.txt. One `dataset` = one analysis run; this
% if/elseif sets every per-dataset parameter. Dataset-independent settings
% are in section 2.
%
% Supported values:
%   'v1_W'   'v1_UW'           -- v1 standard benchmark (weighted / unweighted)
%   'hex'                      -- v1 uniform-hexagonality experiment
%   'Shear_1_2'  'Shear_1_5'   -- revision: sheared tissues
%   'kA_1'       'kA_10'       -- revision: alternative K_A
%   'Flip_two'                 -- revision: two concurrent T1 transitions
%   'Tissue_484' 'Tissue_784'  -- revision: large tissues (no PPGN)
% `dataset` may be pre-set by a driver script; default it only if unset.
% The struct calling convention (DCG_run_revision_analyses.m) passes
% the dataset via DCG_CONFIG.dataset; honor it BEFORE the default and the
% dataset selection below, otherwise the analyzer silently runs as 'v1_W'.
if exist('DCG_CONFIG', 'var') && isstruct(DCG_CONFIG) ...
        && isfield(DCG_CONFIG, 'dataset') && ~isempty(DCG_CONFIG.dataset)
    dataset = DCG_CONFIG.dataset;
end
if ~exist('dataset', 'var') || isempty(dataset)
    dataset = 'v1_W';
end

if strcmp(dataset, 'v1_W')
    mp_prefixes   = {'v1_1_32_W','v1_2_16_W','v1_2_8_W','v1_2_4_W','v1_2_2_W','v1_2_1_W'};
    tasks         = {'lengths_to_lengths'};
    has_ppgn      = true;
    n_cells       = 256;
    curr_analysis = 'Standard dataset experiments';

elseif strcmp(dataset, 'v1_UW')
    mp_prefixes   = {'v1_1_32_UW','v1_2_16_UW','v1_2_8_UW','v1_2_4_UW','v1_2_2_UW','v1_2_1_UW'};
    % Load BOTH tasks: section 9 (W/UW row alignment), section 10 (no_learning
    % baseline), and section 14 (root edge + vertex-line hop graph) all reference the
    % W matrix, so a UW run cannot be UW-only. extract_MP_results swaps the
    % _UW prefix to _W when loading the lengths_to_lengths files.
    tasks         = {'lengths_to_lengths', 'none_to_lengths'};
    has_ppgn      = true;
    n_cells       = 256;
    curr_analysis = 'Standard dataset experiments';

elseif strcmp(dataset, 'hex')
    mp_prefixes   = {'hex_2_8_W'};
    tasks         = {'lengths_to_lengths'};
    has_ppgn      = true;
    n_cells       = 256;
    curr_analysis = 'Uniform hexagonality experiments';

elseif strcmp(dataset, 'Shear_1_2')
    mp_prefixes   = {'rev_Shear_1_2'};
    tasks         = {'lengths_to_lengths'};
    has_ppgn      = true;
    n_cells       = 256;
    curr_analysis = 'Standard dataset experiments';

elseif strcmp(dataset, 'Shear_1_5')
    mp_prefixes   = {'rev_Shear_1_5'};
    tasks         = {'lengths_to_lengths'};
    has_ppgn      = true;
    n_cells       = 256;
    curr_analysis = 'Standard dataset experiments';

elseif strcmp(dataset, 'kA_1')
    mp_prefixes   = {'rev_kA_1'};
    tasks         = {'lengths_to_lengths'};
    has_ppgn      = true;
    n_cells       = 256;
    curr_analysis = 'Standard dataset experiments';

elseif strcmp(dataset, 'kA_10')
    mp_prefixes   = {'rev_kA_10'};
    tasks         = {'lengths_to_lengths'};
    has_ppgn      = true;
    n_cells       = 256;
    curr_analysis = 'Standard dataset experiments';

elseif strcmp(dataset, 'Flip_two')
    mp_prefixes   = {'rev_Flip_two'};
    tasks         = {'lengths_to_lengths'};
    has_ppgn      = true;
    n_cells       = 256;
    curr_analysis = 'Standard dataset experiments';

elseif strcmp(dataset, 'Tissue_484')
    mp_prefixes   = {'rev_Tissue_484'};
    tasks         = {'lengths_to_lengths'};
    has_ppgn      = false;   % large tissue -- PPGN never run here (VRAM)
    n_cells       = 484;
    curr_analysis = 'Standard dataset experiments';

elseif strcmp(dataset, 'Tissue_784')
    mp_prefixes   = {'rev_Tissue_784'};
    tasks         = {'lengths_to_lengths'};
    has_ppgn      = false;   % large tissue -- PPGN never run here (VRAM)
    n_cells       = 784;
    curr_analysis = 'Standard dataset experiments';

else
    error('DCG:unknownDataset', ['Unknown dataset "%s". Valid: v1_W, v1_UW, ', ...
        'hex, Shear_1_2, Shear_1_5, kA_1, kA_10, Flip_two, Tissue_484, Tissue_784.'], ...
        dataset);
end

%==========================================================================
% 2) DATASET-INDEPENDENT CONFIGURATION
%==========================================================================
% Prediction snapshot folder; per-dataset split indices are read
% by the read_dataset_inds() helper at the end of this file.
path_cfg = DCG_publication_config();
data_root = path_cfg.data_root;
inds_root = '';

% The new prediction files are 0-indexed: pred_..._s0 .. pred_..._s4.
seeds = 0:4;

% The four message-passing architectures. PPGN is loaded separately, and
% only when has_ppgn is true (set per-dataset in section 1).
MP_models = {'GraphSAGE', 'GAT', 'GIN', 'PNA'};

% Which split the main loop analyzes (only data_sets{1} is used).
data_sets = {'test'};

% 1 = reuse the cached I struct instead of re-parsing from disk.
load_precomputed_data = 0;

% PPGN files carry interleaved node-feature rows.
add_node_features = 1;

% Gate for the drop_flag1 baseline fix (1 = apply; always 1 for this data).
is_new_interface_extra.PPGN = 1;
is_new_interface_extra.MP   = 1;

% Optional caller override. DCG_run_all_datasets.m can pre-set `dataset`;
% callers may also pre-set DCG_CONFIG with any of these fields:
% data_root, inds_root, seeds, MP_models, data_sets, load_precomputed_data,
% add_node_features, is_new_interface_extra, cache_dir, output_filename,
% results_summary_filename.
cache_dir = '';
output_filename = '';
results_summary_filename = '';
analysis_algorithm_version = '2026-06-06_triple_vertex_hops_v1';
if exist('DCG_CONFIG', 'var') && isstruct(DCG_CONFIG)
    if isfield(DCG_CONFIG, 'data_root'), data_root = DCG_CONFIG.data_root; end
    if isfield(DCG_CONFIG, 'inds_root'), inds_root = DCG_CONFIG.inds_root; end
    if isfield(DCG_CONFIG, 'seeds'), seeds = DCG_CONFIG.seeds; end
    if isfield(DCG_CONFIG, 'MP_models'), MP_models = DCG_CONFIG.MP_models; end
    if isfield(DCG_CONFIG, 'data_sets'), data_sets = DCG_CONFIG.data_sets; end
    if isfield(DCG_CONFIG, 'load_precomputed_data'), load_precomputed_data = DCG_CONFIG.load_precomputed_data; end
    if isfield(DCG_CONFIG, 'add_node_features'), add_node_features = DCG_CONFIG.add_node_features; end
    if isfield(DCG_CONFIG, 'cache_dir'), cache_dir = DCG_CONFIG.cache_dir; end
    if isfield(DCG_CONFIG, 'output_filename'), output_filename = DCG_CONFIG.output_filename; end
    if isfield(DCG_CONFIG, 'results_summary_filename'), results_summary_filename = DCG_CONFIG.results_summary_filename; end
    if isfield(DCG_CONFIG, 'is_new_interface_extra')
        is_new_interface_extra = DCG_CONFIG.is_new_interface_extra;
    end
end

if isempty(data_root)
    error('DCG:missingDataRoot', ['Set the consolidated prediction snapshot path ', ...
        'using DCG_CONFIG.data_root, DCG_DATA_ROOT, or DCG_local_config.m.']);
end

if ~isequal(data_sets, {'test'})
    error('DCG:testOnlyRequired', ...
        'Manuscript analyses must be generated from the test split only. data_sets must be {''test''}.');
end

if isempty(inds_root)
    inds_root = fullfile(data_root, 'inds');
end
if isempty(cache_dir)
    cache_dir = fullfile(data_root, '_analyzer_cache');
end
if isempty(output_filename)
    output_filename = fullfile(cache_dir, [dataset, ' - analyses data.mat']);
end
if isempty(results_summary_filename)
    results_summary_filename = fullfile(cache_dir, [dataset, ' - results_summary.mat']);
end

% Consolidated-snapshot layout: predictions are <task>_<model>_<W|UW>_<size>_sN
% .pred.txt at the data_root, and splits live under splits\<key>\ (deduped,
% mapped by _applies_to.txt). Detect once; extract_*_results and
% read_dataset_inds switch path resolution on this flag. Legacy flat layout
% (pred_<prefix>__<model>_s<seed>.txt + inds\<prefix>\) is used otherwise.
is_consolidated = DCG_consolidated_paths('is_consolidated', data_root);
if is_consolidated
    inds_root = fullfile(data_root, 'splits');
end

% PPGN availability: enable PPGN only if this selected dataset has at least
% one matching PPGN file. Partial seed/cohort coverage is still tolerated;
% extract_PPGN_results logs missing files and section 6 NaN-fills them.
ppgn_expected_prefixes = {};
ppgn_available_prefixes = {};
if has_ppgn
    for t = 1 : length(tasks)
        for i = 1 : numel(mp_prefixes)
            prefix = mp_prefixes{i};
            if strcmp(tasks{t}, 'lengths_to_lengths') && endsWith(prefix, '_UW')
                prefix_for_file = [prefix(1:end-2), 'W'];
            elseif strcmp(tasks{t}, 'none_to_lengths') && endsWith(prefix, '_W')
                prefix_for_file = [prefix(1:end-1), 'UW'];
            else
                prefix_for_file = prefix;
            end
            ppgn_expected_prefixes{end+1} = prefix_for_file; %#ok<SAGROW>
        end
    end
    ppgn_expected_prefixes = unique(ppgn_expected_prefixes, 'stable');
    for i = 1 : numel(ppgn_expected_prefixes)
        if is_consolidated
            glob_path = DCG_consolidated_paths('pred_glob', ...
                data_root, ppgn_expected_prefixes{i}, 'PPGN');
        else
            glob_path = fullfile(data_root, ...
                sprintf('pred_%s__PPGN_s*.txt', ppgn_expected_prefixes{i}));
        end
        if ~isempty(dir(glob_path))
            ppgn_available_prefixes{end+1} = ppgn_expected_prefixes{i}; %#ok<SAGROW>
        end
    end
end
ppgn_available = ~isempty(ppgn_available_prefixes);
has_ppgn = has_ppgn && ppgn_available;

% Per-dataset output paths, kept beside the data so each run is self-contained.
if ~isfolder(cache_dir), mkdir(cache_dir); end


fprintf('[DCG_analyze_results] dataset=%s | tasks=%s | has_ppgn=%d | n_cells=%d\n', ...
    dataset, strjoin(tasks, ','), has_ppgn, n_cells);
if ~isempty(ppgn_expected_prefixes)
    fprintf('[DCG_analyze_results] PPGN prefixes matched: %d/%d\n', ...
        numel(ppgn_available_prefixes), numel(ppgn_expected_prefixes));
end

%==========================================================================
% 3) (dataset selection is the if/elseif in section 1 above)
%==========================================================================

%==========================================================================
% 4) PARSE PREDICTION FILES INTO `I`  (or load from cache)
%==========================================================================
% Either reuse a previously-saved `I` (cache path = output_filename) or
% re-parse every PPGN + MP predictions file from disk. Re-parsing all 5
% seeds x 6 sizes x 5 models is slow (~30 min on NFS); the cache is large
% (~3 GB) so make sure you trust it before reusing.
if load_precomputed_data && isfile(output_filename)
    load(output_filename);
else
    % Fresh slate per dataset: the driver runs all datasets in one MATLAB
    % session via run(), so `I` persists in the workspace. Extraction below
    % only overwrites real-model fields, so without this reset a prior
    % dataset's no_learning (added in section 10) and any has_ppgn=0 stale
    % PPGN leak in and crash section 7's split filter.
    I = struct();
    % PPGN: gated by has_ppgn (true once PPGN files appear -- see
    % ppgn_available, section 2). extract_PPGN_results matches MP's flat-
    % layout shape and tolerates partial seed coverage per prefix.
    if has_ppgn
        I.PPGN = extract_PPGN_results(data_root, inds_root, mp_prefixes, tasks, ...
            is_new_interface_extra.PPGN, seeds);
    end
    % One MP architecture at a time; all four return the same struct shape.
    for m = 1 : length(MP_models)
        I.(MP_models{m}) = extract_MP_results(data_root, inds_root, mp_prefixes, ...
            tasks, MP_models{m}, is_new_interface_extra.MP, seeds);
    end
    % Cache for subsequent runs.
    save(output_filename, 'I');
end

% Fieldnames at this point: {'PPGN','GraphSAGE','GAT','GIN','PNA'}.
all_models = fieldnames(I);

%==========================================================================
% 5) (REMOVED 2026-05-23) hexagonality cross-reference graft
%==========================================================================
% Previously: for dataset='hex', this block loaded v1_W's analyses-data
% cache and slotted hex into row 4 of v1_W's 6-cohort frame, so hex's
% results_summary.mat would carry v1+hex side by side for a combined plot.
% Removed because the only analysis we actually want for hex is "bin its
% own test graphs by per-graph hexagonality, MAE per bin, averaged over
% seeds" -- entirely self-contained in hex's own data. The v1+hex
% comparison, if needed, is now done plotter-side by loading both
% summaries separately. Hex's analyzer run is now ~6x faster (1 cohort x
% 5 seeds instead of 6 x 5) and has no order dependency on v1_W.

% Keep an untouched snapshot. The subsequent post-processing modifies `I` in
% place; reassigning `I = original_I` is mostly a marker -- handy in interactive
% sessions if you want to rewind to this point.
original_I = I;

I = original_I;

%==========================================================================
% 6) NORMALIZE EVERY MODEL TO A COMPLETE (n_subsets x n_seeds) GRID
%==========================================================================
% extract_*_results leave a slot empty wherever a (prefix, seed) prediction
% file is missing. Downstream code requires every model to share one
% (n_sub x n_seed) grid with matching per-graph edge counts:
%   - section 7's split filter does inds{i}.(split) then vals{i,s}(idcs);
%   - section 14 stacks vals{i,s}{j}(:,end) across ALL models via cell2mat,
%     so an absent model's graph j must still be a matrix of the right size.
% So for each subset i we pick a reference cohort shape (this model's first
% non-empty seed, else any model's), NaN-fill it, and drop it into every
% empty (model, i, s) vals slot; empty .inds slots get the subset's split
% struct from the first model/seed that has it (the split is model-agnostic).
% NaN slots flag "no data": no_learning / ground-truth (built from PNA) carry
% the NaN through, and section 14's isnan guard skips those (i,s); aggregates
% use 'omitnan'. This generalizes the historical PPGN-only fill (which cloned
% I.PNA at the same (i,s) and broke when PNA itself was partial) so partial MP
% coverage -- and a partial PNA template -- parse without the per-run
% MP_models/seeds workaround.
model_names = fieldnames(I);
for t = 1 : length(tasks)
    fill_grid  = size(I.(model_names{1}).(tasks{t}).vals);
    fill_nsub  = fill_grid(1);
    fill_nseed = fill_grid(2);
    for i = 1 : fill_nsub
        % Cross-model fallbacks for subset i (cohort shape + split struct);
        % both are graph properties, identical across models and seeds.
        xref_vals = {};
        xref_inds = [];
        for m = 1 : numel(model_names)
            Vm = I.(model_names{m}).(tasks{t}).vals;
            Nm = I.(model_names{m}).(tasks{t}).inds;
            for s = 1 : fill_nseed
                if isempty(xref_vals) && ~isempty(Vm{i,s})
                    xref_vals = Vm{i,s};
                end
                if isempty(xref_inds) && ~isempty(Nm{i,s}) && isstruct(Nm{i,s})
                    xref_inds = Nm{i,s};
                end
            end
        end
        if isempty(xref_vals)
            continue;   % no model has subset i at all -> leave it untouched
        end
        for m = 1 : numel(model_names)
            % Prefer this model's own cohort shape (keeps its column layout);
            % fall back to the cross-model shape only for a fully-absent model.
            sref = xref_vals;
            for s = 1 : fill_nseed
                if ~isempty(I.(model_names{m}).(tasks{t}).vals{i,s})
                    sref = I.(model_names{m}).(tasks{t}).vals{i,s};
                    break;
                end
            end
            nan_tmpl = cellfun(@(x) nan(size(x)), sref, 'UniformOutput', false);
            for s = 1 : fill_nseed
                if isempty(I.(model_names{m}).(tasks{t}).vals{i,s})
                    I.(model_names{m}).(tasks{t}).vals{i,s} = nan_tmpl;
                end
                if isempty(I.(model_names{m}).(tasks{t}).inds{i,s})
                    I.(model_names{m}).(tasks{t}).inds{i,s} = xref_inds;
                end
            end
        end
    end
end
clear model_names fill_grid fill_nsub fill_nseed xref_vals xref_inds Vm Nm sref nan_tmpl;

%==========================================================================
% 7) RESTRICT EVERY MODEL TO ONE SPLIT (train, val, or test)
%==========================================================================
% taking from each architecture only the test set values:
% After this block, I.<model>.<task>.vals{i,s} is the subset of graphs that
% belong to data_sets{1} -- typically 'test' for paper figures. The .inds
% struct holds the index lists per split; we look up the right one and
% pull those rows out of vals.
for m = 1 : length(all_models)
    for t = 1 : length(tasks)
        for i = 1 : size(I.(all_models{m}).(tasks{t}).vals,1)
            curr_idcs = I.(all_models{m}).(tasks{t}).inds{i}.(data_sets{1});
            for s = 1 : length(seeds)
                try
                    I.(all_models{m}).(tasks{t}).vals{i,s} = I.(all_models{m}).(tasks{t}).vals{i,s}(curr_idcs);
                catch ME_idx
                    % Defensive: report context if curr_idcs goes past the
                    % end of vals (which has happened when the splits and
                    % the parsed predictions came from different runs).
                    error('DCG:filterIndex', ...
                        'Index filter failed: model=%s task=%s subset_i=%d seed=%d  cell_length=%d  curr_idcs[min,max]=[%d,%d]  msg=%s', ...
                        all_models{m}, tasks{t}, i, s, numel(I.(all_models{m}).(tasks{t}).vals{i,s}), ...
                        min(curr_idcs), max(curr_idcs), ME_idx.message);
                end
            end
        end
    end
end

%--------------------------------------------------------------------------
% (Disabled) historical block that tried to align every model's subset list
% to PPGN's. Left commented out as documentation of a prior approach.
%--------------------------------------------------------------------------
%{
% making sure that all models have the same data structure:
tmp_I = I;
all_models = fieldnames(I);
for m = 2 : length(all_models)

    for t = 1 : length(tasks)
        I.(all_models{m}).(tasks{t}) = I.PPGN.(tasks{t});

        for s = 1 : length(PPGN.(tasks{t}){s}.subsets)
            curr_s = find(strcmp(tmp_I.(all_models{m}).(tasks{t}).subsets, PPGN.(tasks{t}){s}.subsets{s}));

            if isempty(curr_s)
                I.(all_models{m}).(tasks{t}).vals{s} = cellfun(@(x) nan(size(x)), I.(all_models{m}).(tasks{t}).vals{s}, 'UniformOutput', false);
            elseif isempty(tmp_I.(all_models{m}).(tasks{t}).vals{curr_s})
                I.(all_models{m}).(tasks{t}).vals{s} = cellfun(@(x) nan(size(x)), I.(all_models{m}).(tasks{t}).vals{s}, 'UniformOutput', false);
            else
                I.(all_models{m}).(tasks{t}).vals{s} = tmp_I.(all_models{m}).(tasks{t}).vals{curr_s};
            end
        end
    end
end
clear tmp_I;
%}

%==========================================================================
% 8) DROP THE "ZEROS" COLUMN IN UW PPGN
%==========================================================================
% removing the zeros column in none_to_lengths in PPGN:
% PPGN's UW files have 5 columns [u v flag post pred] whereas MP UW has 4
% [u v post pred]. The "extra" column 3 (flag) is always zero in UW, so we
% strip it here so every model's UW matrix has the same width.
if has_ppgn && ismember('none_to_lengths', tasks)
    for i = 1 : numel(I.PPGN.none_to_lengths.vals)
        for j = 1 : length(I.PPGN.none_to_lengths.vals{i})
            if size(I.PPGN.none_to_lengths.vals{i}{j},2) == 5
                I.PPGN.none_to_lengths.vals{i}{j}(:,3) = [];
            end
        end
    end
end

%==========================================================================
% 9) ALIGN W AND UW ROW ORDER PER GRAPH
%==========================================================================
% making sure that the graphs are sorted similarly (WATCH OUT! I DID IT THROUGH SORTING THE GROUND TRUTH!!)
%
% Each graph's W matrix and UW matrix list the same set of edges, but the
% rows may not be in the same order. We map each (u,v) pair to a linear
% index via sub2ind, then use setxor to find the rows that exist in one
% but not the other. The eliminated edge sat in W but not in UW (before
% drop_flag1), so we used to "move" that UW row into the W slot. After
% drop_flag1 (2026-05-14) W and UW (u,v) sets should match exactly -- meaning
% setxor returns empty `b` and `c` and this block is mostly a no-op. The
% `keyboard` on the count check fires if more than one row differs (which
% would indicate a deeper alignment problem).
if ismember('none_to_lengths', tasks)
    for m = 1 : length(all_models)
        m
        for i = 1 : size(I.(all_models{m}).(tasks{t}).vals,1)
            for j = 1 : size(I.(all_models{m}).(tasks{t}).vals,2)
                for k = 1 : length(I.(all_models{m}).(tasks{t}).vals{i,j})

                    % Linearize the W (u,v) pairs.
                    pairs = I.(all_models{m}).lengths_to_lengths.vals{i,j}{k}(:,1:2);
                    if isnan(pairs(1))
                        continue;
                    end
                    nums1 = sub2ind([n_cells,n_cells], pairs(:,1), pairs(:,2));

                    % Linearize the UW (u,v) pairs.
                    pairs = I.(all_models{m}).none_to_lengths.vals{i,j}{k}(:,1:2);
                    if isnan(pairs(1))
                        continue;
                    end
                    nums2 = sub2ind([n_cells,n_cells], pairs(:,1), pairs(:,2));

                    % b = indices in W not in UW; c = indices in UW not in W.
                    [a,b,c] = setxor(nums1, nums2);

                    % Post-drop_flag1, W and UW have identical (u,v) sets, so
                    % b and c are empty. The reassignment below would then
                    % collapse vals to 0 rows (because 1:[]-1 and []:end both
                    % evaluate to empty). Skip when there's nothing to move.
                    if ~isempty(c)
                        % Pull out the UW row(s) that need to move, delete
                        % them from their current position, then re-insert
                        % them at W's position (so UW row k matches W row k).
                        moved = I.(all_models{m}).none_to_lengths.vals{i,j}{k}(c,:);
                        I.(all_models{m}).none_to_lengths.vals{i,j}{k}(c,:) = [];
                        I.(all_models{m}).none_to_lengths.vals{i,j}{k} = [I.(all_models{m}).none_to_lengths.vals{i,j}{k}(1:b-1,:); moved; I.(all_models{m}).none_to_lengths.vals{i,j}{k}(b:end,:)];

                        % Sanity check: after the move, at most one row should
                        % still differ (the eliminated edge row in pre-drop_flag1
                        % data; zero rows post-drop_flag1).
                        if nnz(any(I.(all_models{m}).none_to_lengths.vals{i,j}{k}(:,1:2) ~= I.(all_models{m}).lengths_to_lengths.vals{i,j}{k}(:,1:2),2)) ~= 1
                            keyboard;
                        end
                    end

                    % (alternative ordering kept for reference)
                    % [~, order] = sortrows(I.(all_models{m}).(tasks{t}).vals{i,j}{k}(:,end-1));
                    % I.(all_models{m}).(tasks{t}).vals{i,j}{k} = I.(all_models{m}).(tasks{t}).vals{i,j}{k}(order,:);
                end
            end
        end
    end
end

%==========================================================================
% 10) ADD THE "no_learning" PSEUDO-MODEL (the identity-baseline)
%==========================================================================
% adding the pseudo-model "no-learning":
%
% no_learning is a synthetic model whose "prediction" for each edge is simply
% the pre-T1 edge length (column 3 of the W matrix). It represents what you
% get if you assume nothing changes across the T1 transition. Comparing
% trained models to this baseline gives the normalized-MAE (nMAE) ratio.
% We copy a reference model's struct shape (cheapest non-PPGN model to
% template from), then overwrite the last column with the pre-T1 lengths.
%
% ref_model GENERALIZATION (2026-05-31): historically this was hardcoded to
% I.PNA. The edges (vertex-line hop graph), the no_learning identity-baseline, and the
% per-edge ground-truth (post-T1 true length) were ALL pulled from I.PNA.
% Those columns (edge list cols 1:2 and the true label col end-1) are
% IDENTICAL across every model for a given graph -- they are the physical
% label, not a model output (proven: 12240/12240 columns byte-identical).
% So any present model is a valid reference. But when PNA pred files are
% absent (e.g. Shear/Flip_two/Tissue_484 while PNA is still being generated
% on another node), I.PNA is a NaN/empty template -> ground-truth + edges
% never fill -> the summary S was silently written as a ~59 KB stub with NO
% error. Pick the reference by priority: PNA when it has real data (so the
% already-good summaries stay byte-for-byte identical), else the first
% present non-PPGN model, else PPGN. Error loudly if nothing has real data.
ref_priority = {'PNA', 'GraphSAGE', 'GAT', 'GIN', 'PPGN'};
ref_model = '';
for rp = 1 : numel(ref_priority)
    cand = ref_priority{rp};
    if ~isfield(I, cand) || ~isfield(I.(cand), tasks{1}), continue; end
    v = I.(cand).(tasks{1}).vals;
    if iscell(v) && ~isempty(v) && iscell(v{1,1}) && ~isempty(v{1,1}) ...
            && isnumeric(v{1,1}{1}) && ~isempty(v{1,1}{1}) ...
            && ~any(isnan(v{1,1}{1}(:,1)))
        ref_model = cand;
        break;
    end
end
if isempty(ref_model)
    error('DCG:noReferenceModel', ['No present model has real data to ', ...
        'serve as the ground-truth/edges reference (all empty templates). ', ...
        'Check that pred files exist for at least one model.']);
end
fprintf('[DCG] ground-truth/no_learning/edges reference model: %s\n', ref_model);

n_initial_models = length(fieldnames(I));
I.no_learning = I.(ref_model);
for t = 1 : length(tasks)
    % Iterate over the no_learning shape itself (cloned from PNA above).
    % Original code used I.PPGN here, which breaks when has_ppgn=0.
    for i = 1 : numel(I.no_learning.(tasks{t}).vals)
        for j = 1 : numel(I.no_learning.(tasks{t}).vals{i})
            % PNA.lengths_to_lengths.vals{i}{j}(:,3) is pre-T1 length for every edge.
            I.no_learning.(tasks{t}).vals{i}{j}(:,end) = I.(ref_model).lengths_to_lengths.vals{i}{j}(:,3);
        end
    end
end

%==========================================================================
% 11) FIND SUBSETS COMMON TO ALL MODELS
%==========================================================================
% isolating the sets we have in all models:
% Each model's `.subsets` is a list of training-set names (cohort folders).
% We intersect them so downstream analyses only iterate over cohorts every
% model trained on. With the v1 paper data this intersection is the full
% set -- but the variable is computed regardless.
all_models = fieldnames(I);
is_initialized = 1;
available_subsets = [];
for m = 1 : length(all_models)
    for t = 1 : length(tasks)
        if is_initialized
            available_subsets = I.(all_models{m}).(tasks{t}).subsets;
            is_initialized = 0;
        else
            available_subsets = intersect(available_subsets, I.(all_models{m}).(tasks{t}).subsets);
        end
    end
end

%--------------------------------------------------------------------------
% (Disabled) historical block: hard-filter each model down to the intersection
% and sort it by (subset_siz, subset_idx). Left commented out as a hint of
% what one would do if subsets were missing.
%--------------------------------------------------------------------------
%{
% filtering all the entries we have in all regimes, and ordering them to have the same order:
for m = 1 : length(all_models)
    for t = 1 : length(tasks)
        to_keep = ismember(I.(all_models{m}).(tasks{t}).subsets, available_subsets);
        I.(all_models{m}).(tasks{t}) = structfun(@(x) x(to_keep,:), I.(all_models{m}).(tasks{t}), 'UniformOutput', false);

        [~, order] = sortrows([I.(all_models{m}).(tasks{t}).subset_siz, I.(all_models{m}).(tasks{t}).subset_idx]);
        I.(all_models{m}).(tasks{t}) = structfun(@(x) x(order,:), I.(all_models{m}).(tasks{t}), 'UniformOutput', false);
    end
end
%}

%--------------------------------------------------------------------------
% (Disabled) historical sanity-check block: verify that every model has the
% same (u,v) ordering per graph. Useful while debugging but slow.
%--------------------------------------------------------------------------
% checking that all graphs and all datasets in all tasks and all models are sorted the same way (will help later with saving time):
%{
for t = 1 : length(tasks)
    for i = 1 : length(I.(all_models{m}).(tasks{t}).subset_siz)
        for j = 1 : length(I.(all_models{m}).(tasks{t}).vals{i})
            for m = 2 : length(all_models)
                if t == 1
                    try
                        if ~isequal(I.(all_models{m}).(tasks{t}).vals{i}{j}(:,1:end-1), I.(all_models{1}).(tasks{1}).vals{i}{j}(:,1:end-1))
                            disp([t,i,j,m]);
                        end
                    catch
                        keyboard;
                    end
                else
                    try
                        if ~isequal(I.(all_models{m}).(tasks{t}).vals{i}{j}(:,1:3), I.(all_models{1}).(tasks{1}).vals{i}{j}(:,[1,2,5]))
                            disp([t,i,j,m]);
                        end
                    catch
                        keyboard;
                    end
                end
            end
        end
    end
end
%}

%==========================================================================
% 12) NORMALIZE SUBSET NAMING (one-time legacy cleanup)
%==========================================================================
% patchy corrections...
% Early naming used `_set_2_` for the 2_*_cells cohorts; we rewrite to
% `_set_1_` for uniformity. Also force subset_idx to 1 so every cohort is
% treated as a single "rep index" (the v1 data only has one rep per size).
for t = 1 : length(tasks)
    for m = 1 : length(all_models)
        I.(all_models{m}).(tasks{t}).subsets = regexprep(I.(all_models{m}).(tasks{t}).subsets, 'set_2', 'set_1');
        I.(all_models{m}).(tasks{t}).subset_idx(:) = 1;
    end
end

%==========================================================================
% 13) PREALLOCATE THE SUMMARY STRUCTURE `S`
%==========================================================================
% res_siz = (n_reps, n_size_bins, n_seeds). n_size_bins = log2(max_cohort)+1
% so cohorts at sizes 1,2,4,8,16,32 land in bins 1..6 respectively.
res_siz = [max(I.(all_models{end}).(tasks{1}).subset_idx), log2(max(I.(all_models{end}).(tasks{1}).subset_siz))+1, length(seeds)];
S.distances = cell(res_siz);
S.ground_truth.lengths_to_lengths = cell(res_siz);
S.ground_truth.none_to_lengths = cell(res_siz);
S.prediction_errors.lengths_to_lengths = cell(res_siz);
S.prediction_errors.none_to_lengths = cell(res_siz);
S.predictions.lengths_to_lengths = cell(res_siz);
S.predictions.none_to_lengths = cell(res_siz);
S.cell_hexagonality = cell(res_siz);
S.edge_hexagonality = cell(res_siz);
S.edges = cell(res_siz);
S.hexagonality = cell(res_siz);
S.disorder = cell(res_siz);
S.edge_hexagonality_in_dist = cell(res_siz);

% (unused scratch matrix; kept to avoid breaking workspace dumps).
a = nan(0,6);

%==========================================================================
% 14) MAIN AGGREGATION LOOP
%==========================================================================
% Iterate over (cohort i, seed s, graph ss). For each graph:
%   - build the row-preserving vertex-line graph of its underlying tissue
%     graph (vertices = interfaces/edges; one hop = interfaces share a cell);
%   - compute hop distance from the T1 root edge to every other edge;
%   - assemble per-model predictions / ground truths;
%   - bin predictions and errors by hop distance;
%   - compute per-cell hexagonality and per-edge "neighborhood hexagonality";
%   - aggregate per-distance edge hexagonality.
for i = 1 : length(I.(all_models{end}).(tasks{1}).subset_siz)

    % Map cohort size 1,2,4,8,16,32 into size-bin index 1..6. Use the
    % explicit summary reference model instead of a leftover script variable
    % from an earlier `for m = ...` loop.
    summary_ref_model = all_models{end};
    curr_col = log2(I.(summary_ref_model).(tasks{1}).subset_siz(i))+1;
    curr_idx = I.(summary_ref_model).(tasks{1}).subset_idx(i);

    for s = 1 : length(seeds)

        disp([num2str(i), '/', num2str(length(I.(all_models{end}).(tasks{1}).subset_siz)), ', ', num2str(s), '/', num2str(length(data_sets))]);

        % Indices of graphs in the chosen split (e.g. test) for this cohort.
        % Preallocate per-graph cells inside S for this (rep, size, seed) slot.
        curr_dataset = I.(all_models{end}).(tasks{1}).inds{i}.(data_sets{1});
        S.distances{curr_idx, curr_col, s}.(data_sets{1}) = cell(length(curr_dataset),1);
        S.edges{curr_idx, curr_col, s}.(data_sets{1}) = cell(length(curr_dataset),1);
        S.ground_truth.lengths_to_lengths{curr_idx, curr_col, s}.(data_sets{1}) = cell(length(curr_dataset),1);
        S.ground_truth.none_to_lengths{curr_idx, curr_col, s}.(data_sets{1}) = cell(length(curr_dataset),1);
        S.prediction_errors.lengths_to_lengths{curr_idx, curr_col, s}.(data_sets{1}) = cell(length(curr_dataset),1);
        S.prediction_errors.none_to_lengths{curr_idx, curr_col, s}.(data_sets{1}) = cell(length(curr_dataset),1);
        S.predictions.lengths_to_lengths{curr_idx, curr_col, s}.(data_sets{1}) = cell(length(curr_dataset),1);
        S.predictions.none_to_lengths{curr_idx, curr_col, s}.(data_sets{1}) = cell(length(curr_dataset),1);
        S.hexagonality{curr_idx, curr_col, s}.(data_sets{1}) = nan(length(curr_dataset),1);
        S.cell_hexagonality{curr_idx, curr_col, s}.(data_sets{1}) = cell(length(curr_dataset),1);
        S.edge_hexagonality{curr_idx, curr_col, s}.(data_sets{1}) = cell(length(curr_dataset),1);
        S.disorder{curr_idx, curr_col, s}.(data_sets{1}) = nan(length(curr_dataset),1);

        for ss = 1 : length(curr_dataset)

            % j has historically been either curr_dataset(ss) (the original
            % file index) or ss (post-filter position). We use ss because
            % I.<model>.vals was already filtered to test-only above.
            % j = curr_dataset(ss);
            j = ss;

            % Skip NaN-flagged placeholder rows (only happens for PPGN
            % cohorts where no real PPGN data was loaded for that seed).
            try
                if isnan(I.(all_models{end}).lengths_to_lengths.vals{i,s}{j}(1))
                    continue;
                end
            catch
                keyboard;
            end

            % Per-graph W matrix and its row-preserving vertex-line hop graph.
            curr_G = I.(all_models{end}).lengths_to_lengths.vals{i,s}{j};
            line_G = dcg_line_graph_preserve_rows(curr_G(:,1:2));
            % drop_flag1 removed the flag=1 row, so find the newly-formed edge
            % (pre = 0) instead. It sits at the same geometric T1 location.
            root_edge_in_line_G = find(curr_G(:,3) == 0);
            if strcmp(dataset, 'Flip_two')
                expected_root_count = 2;
            else
                expected_root_count = 1;
            end
            if numel(root_edge_in_line_G) ~= expected_root_count
                error('DCG:unexpectedT1RootCount', ...
                    '%s expected %d T1 root edge(s), found %d in subset %d seed %d graph %d.', ...
                    dataset, expected_root_count, numel(root_edge_in_line_G), i, s, j);
            end
            dcg_assert_root_rows_match_line_graph(line_G, curr_G(:,1:2), root_edge_in_line_G, ...
                sprintf('%s subset %d seed %d graph %d', dataset, i, s, j));

            % Record the edge list and per-edge historical vertex-line hop distance
            % from the root (or nearest root for multi-T1 datasets).
            S.edges{curr_idx, curr_col, s}.(data_sets{1}){ss} = I.(all_models{end}).lengths_to_lengths.vals{i,s}{j}(:,1:2);
            % FIX 2026-05-22 (Flip_two): graphs with multiple concurrent T1s have
            % multiple pre==0 entries -> root_edge_in_line_G is a vector ->
            % distances() returns a k x n matrix. Collapse to hop-to-nearest-T1
            % so downstream sort/hist/mat2cell all see a single 1-D column.
            dist_mat = distances(line_G, root_edge_in_line_G);
            if size(dist_mat, 1) > 1
                dist_mat = min(dist_mat, [], 1);
            end
            S.distances{curr_idx, curr_col, s}.(data_sets{1}){ss} = dist_mat';

            % (disabled) consistency check: distances computed from UW should
            % equal those from W. Kept for reference.
            %%
            % curr_G1 = I.(all_models{end}).none_to_lengths.vals{i,s}{j};
            % line_G1 = line_graph_from_vertex_model(curr_G1(:,1:2));
            % root_edge_in_line_G1 = find(curr_G1(:,4) == 1);
            % S1.edges{curr_idx, curr_col, s}.(data_sets{1}){ss} = I.(all_models{end}).none_to_lengths.vals{i,s}{j}(:,1:2);
            % S1.distances{curr_idx, curr_col, s}.(data_sets{1}){ss} = distances(line_G1, root_edge_in_line_G1)';
            %
            % if ~isequal(S.distances{curr_idx, curr_col, s}.(data_sets{1}){ss}, S1.distances{curr_idx, curr_col, s}.(data_sets{1}){ss})
            %     % keyboard;
            % end
            %%

            % Per-edge ground-truth post-T1 length, pulled from PNA's matrix
            % (any non-PPGN model would work since the GT column is identical
            % across models -- PNA is chosen by convention).
            S.ground_truth.lengths_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){ss} = I.(ref_model).lengths_to_lengths.vals{i,s}{j}(:,end-1);

            if ismember('none_to_lengths', tasks)
                S.ground_truth.none_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){ss} = I.(ref_model).none_to_lengths.vals{i,s}{j}(:,end-1);
            end

            % Stack each model's last (prediction) and second-to-last (GT)
            % column horizontally -- one column per model -- into per-edge
            % matrices. curr_vals_* is |pred - gt|.
            curr_predictions_lengths_to_lengths = cell2mat(cellfun(@(curr_model) I.(curr_model).lengths_to_lengths.vals{i,s}{j}(:,end), all_models, 'UniformOutput', false)');
            curr_ground_truth_lengths_to_lengths = cell2mat(cellfun(@(curr_model) I.(curr_model).lengths_to_lengths.vals{i,s}{j}(:,end-1), all_models, 'UniformOutput', false)');
            curr_vals_lengths_to_lengths = abs(curr_predictions_lengths_to_lengths - curr_ground_truth_lengths_to_lengths);

            % UW (none_to_lengths) per-edge extraction and consistency check are
            % only meaningful when UW is in the task list. The hex experiment is
            % W-only by default (tasks = {'lengths_to_lengths'}), so this block
            % is gated to prevent crashes there.
            if ismember('none_to_lengths', tasks)
                curr_predictions_none_to_lengths = cell2mat(cellfun(@(curr_model) I.(curr_model).none_to_lengths.vals{i,s}{j}(:,end), all_models, 'UniformOutput', false)');
                curr_ground_truth_none_to_lengths = cell2mat(cellfun(@(curr_model) I.(curr_model).none_to_lengths.vals{i,s}{j}(:,end-1), all_models, 'UniformOutput', false)');
                curr_vals_none_to_lengths = abs(curr_predictions_none_to_lengths - curr_ground_truth_none_to_lengths);

                % Sanity: all models must agree on the GT length per edge
                % (since GT is a property of the graph, not the model).
                try
                    to_compare1 = ~isnan(curr_ground_truth_none_to_lengths(:,1));
                    to_compare2 = ~isnan(curr_ground_truth_lengths_to_lengths(:,1));
                    if ~isequal(S.ground_truth.none_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){j}(to_compare1), curr_ground_truth_none_to_lengths(to_compare1,1)) || ~isequal(S.ground_truth.lengths_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){j}(to_compare2), curr_ground_truth_lengths_to_lengths(to_compare2,1))
                        keyboard;
                    end
                catch
                    keyboard;
                end
            end

            % Sort edges by hop distance; `h` is the count of edges in each
            % integer hop bucket (0, 1, 2, ...). h' is later passed to
            % mat2cell to bucket the predictions per hop.
            [sorted_distances, distances_order] = sort(S.distances{curr_idx, curr_col, s}.(data_sets{1}){ss});
            h = hist(sorted_distances, 0:sorted_distances(end));

            % Apply the same sort permutation everywhere so all per-edge
            % arrays line up.
            S.edges{curr_idx, curr_col, s}.(data_sets{1}){ss} = S.edges{curr_idx, curr_col, s}.(data_sets{1}){ss}(distances_order,:);
            S.distances{curr_idx, curr_col, s}.(data_sets{1}){ss} = S.distances{curr_idx, curr_col, s}.(data_sets{1}){ss}(distances_order);

            S.ground_truth.lengths_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){ss} = S.ground_truth.lengths_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){ss}(distances_order);
            if ismember('none_to_lengths', tasks)
                S.ground_truth.none_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){ss} = S.ground_truth.none_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){ss}(distances_order);
            end

            curr_vals_lengths_to_lengths = curr_vals_lengths_to_lengths(distances_order,:);
            curr_predictions_lengths_to_lengths = curr_predictions_lengths_to_lengths(distances_order,:);
            if ismember('none_to_lengths', tasks)
                curr_vals_none_to_lengths = curr_vals_none_to_lengths(distances_order,:);
                curr_predictions_none_to_lengths = curr_predictions_none_to_lengths(distances_order,:);
            end

            % Bucket the now-sorted predictions and errors by hop distance.
            % After this, each S.predictions.<task>{...}{ss} is a cell of
            % length(unique hops) -- each entry is (n_edges_at_that_hop x n_models).
            S.predictions.lengths_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){ss} = mat2cell(curr_predictions_lengths_to_lengths, h', size(curr_predictions_lengths_to_lengths,2));
            S.prediction_errors.lengths_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){ss} = mat2cell(curr_vals_lengths_to_lengths, h', size(curr_vals_lengths_to_lengths,2));
            if ismember('none_to_lengths', tasks)
                S.predictions.none_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){ss} = mat2cell(curr_predictions_none_to_lengths, h', size(curr_predictions_none_to_lengths,2));
                S.prediction_errors.none_to_lengths{curr_idx, curr_col, s}.(data_sets{1}){ss} = mat2cell(curr_vals_none_to_lengths, h', size(curr_vals_none_to_lengths,2));
            end

            % Per-cell hexagonality = how many edges touch each cell (degree
            % of the cell in the dual graph). For a perfect hex tissue every
            % cell has degree 6.
            S.cell_hexagonality{curr_idx, curr_col, s}.(data_sets{1}){ss,1} = hist(reshape(I.(all_models{end}).lengths_to_lengths.vals{i,s}{j}(:,1:2), [], 1), 1:n_cells)';
            % Whole-graph "hexagonality" = fraction of cells with exactly 6 neighbors.
            S.hexagonality{curr_idx, curr_col, s}.(data_sets{1})(ss) = mean(S.cell_hexagonality{curr_idx, curr_col, s}.(data_sets{1}){ss} == 6);
            % Disorder = scale parameter of the noise used to generate this graph.
            S.disorder{curr_idx, curr_col, s}.(data_sets{1})(ss) = I.(all_models{end}).(tasks{1}).graph_id{i}.disorder(curr_dataset(ss));

            % --------- Edge hexagonality -------------------------------
            % For each edge, find the 4 cells in its "local neighborhood"
            % (the two cells the edge separates plus their shared neighbors),
            % look up their cell hexagonality (degree), and average |deg - 6|.
            % This gives a per-edge "how regular is your immediate region"
            % score. The 4-cells expectation comes from the topology of a
            % hex T1 transition.
            %
            % Vectorized 2026-05-17: build cell-adjacency (with self-loops on
            % the diagonal) once, then B = A(c1,:) & A(c2,:) is a sparse
            % n_edges x n_cells indicator of each edge's 4-cell neighborhood.
            % edge_hex = B * |cell_hex - 6| / 4 in one mat-vec product
            % (replaces ~768 per-edge mean() calls per graph).
            curr_edges = S.edges{curr_idx, curr_col, s}.(data_sets{1}){ss};
            curr_cell_hex_minus_6 = abs(S.cell_hexagonality{curr_idx, curr_col, s}.(data_sets{1}){ss} - 6);
            n_cells_here = numel(curr_cell_hex_minus_6);

            A_self = sparse([curr_edges(:,1); curr_edges(:,2); (1:n_cells_here)'], ...
                [curr_edges(:,2); curr_edges(:,1); (1:n_cells_here)'], ...
                1, n_cells_here, n_cells_here);
            A_self = A_self > 0;  % cell adjacency with self-loops, logical
            B = A_self(curr_edges(:,1), :) & A_self(curr_edges(:,2), :);

            % Sanity: every edge's 4-cell neighborhood should have exactly 4 cells.
            % Revision tissues (e.g. soft-kA) can legitimately violate this at a
            % few edges; warn instead of halting so a batch run completes.
            n_bad_nbhd = nnz(sum(B, 2) ~= 4);
            if n_bad_nbhd > 0
                warning('DCG:edgeHexNeighborhood', ...
                    'edge-hex neighborhood ~= 4 cells: dataset=%s i=%d s=%d ss=%d (%d edges)', ...
                    dataset, i, s, ss, n_bad_nbhd);
            end

            S.edge_hexagonality{curr_idx, curr_col, s}.(data_sets{1}){ss,1} = (B * curr_cell_hex_minus_6) / 4;

            % Average edge-hexagonality per integer hop distance. Allocate to
            % the graph's observed final hop instead of assuming the old
            % 256-cell/24-hop range; large revision tissues can go deeper.
            curr_edge_hexagonality = S.edge_hexagonality{curr_idx, curr_col, s}.(data_sets{1}){ss,1};
            curr_edge_distances = S.distances{curr_idx, curr_col, s}.(data_sets{1}){ss,1};
            edge_hexagonality_in_dist = nan(curr_edge_distances(end)+1,1);

            for d = 1 : curr_edge_distances(end)+1
                edge_hexagonality_in_dist(d) = mean(curr_edge_hexagonality(curr_edge_distances == d-1));
            end

            S.edge_hexagonality_in_dist{curr_idx, curr_col, s}.(data_sets{1}){ss} = edge_hexagonality_in_dist;

        end

    end

end

%==========================================================================
% 15) SQUEEZE OUT THE LEADING SINGLETON DIM AND TRANSPOSE
%==========================================================================
% Convert from (1 x n_sizes x n_seeds) cell to (n_seeds x n_sizes) cell for
% every S subfield. The plotter expects the latter shape.
S.distances = squeeze(S.distances)';
S.ground_truth.lengths_to_lengths = squeeze(S.ground_truth.lengths_to_lengths)';
S.ground_truth.none_to_lengths = squeeze(S.ground_truth.none_to_lengths)';
S.cell_hexagonality = squeeze(S.cell_hexagonality)';
S.edge_hexagonality = squeeze(S.edge_hexagonality)';
S.edges = squeeze(S.edges)';
S.hexagonality = squeeze(S.hexagonality)';
S.disorder = squeeze(S.disorder)';
S.edge_hexagonality_in_dist = squeeze(S.edge_hexagonality_in_dist)';

S.prediction_errors.lengths_to_lengths = squeeze(S.prediction_errors.lengths_to_lengths)';
S.prediction_errors.none_to_lengths = squeeze(S.prediction_errors.none_to_lengths)';
S.predictions.lengths_to_lengths = squeeze(S.predictions.lengths_to_lengths)';
S.predictions.none_to_lengths = squeeze(S.predictions.none_to_lengths)';

%==========================================================================
% 16) PERSIST RESULTS
%==========================================================================
% Save S plus the model / task / split metadata so the plotter can run
% standalone without re-running this script.
% CACHE-STALENESS GUARD (2026-06-01/06): stamp the summary with both a
% fingerprint of the source prediction set and the analysis algorithm version.
% The source fingerprint catches changed pred files; the algorithm version
% catches logic changes such as the hop-distance definition.
source_manifest = dcg_source_manifest(data_root); %#ok<NASGU>
save(results_summary_filename, 'S', 'all_models', 'tasks', 'data_sets', ...
    'source_manifest', 'analysis_algorithm_version');


%==========================================================================
% HELPER: extract_MP_results   (rewritten 2026-05-22 for the flat layout)
%==========================================================================
% MP = extract_MP_results(data_root, inds_root, mp_prefixes, tasks, model, ...
%                         is_new_interface_extra, seeds)
%
% Read every prediction file for one MP architecture across the cohorts of
% the selected dataset. The flat layout has one file per (cohort,model,seed):
%     <data_root>\pred_<prefix>__<model>_s<seed>.txt
% and one split per cohort:
%     <inds_root>\<prefix>\{train,val,test}.inds
% `mp_prefixes` is the cohort-prefix list set by the section-1 if/elseif
% (e.g. {'v1_1_32_W',...,'v1_2_1_W'} or {'rev_kA_1'}). The returned struct
% mirrors the per-task layout used by `I.<model>` above.
function MP = extract_MP_results(data_root, inds_root, mp_prefixes, tasks, model, is_new_interface_extra, seeds)

% Parallelized over (cohort, seed) file loads; falls back to serial if no
% pool / no Parallel Computing Toolbox. 6 workers (per user request).
if isempty(gcp('nocreate'))
    try, parpool('local', 6); catch, end
end

% Resolve prediction filenames against the consolidated snapshot layout when
% data_root is one; otherwise the legacy flat pred_<prefix>__<model>_s<seed>.
consolidated = DCG_consolidated_paths('is_consolidated', data_root);

for t = 1 : length(tasks)

    % Cohorts = the prefix list passed in (mp_prefixes). Parse cohort idx/siz
    % from each prefix: v1/hex prefixes encode it (<fam>_<idx>_<siz>_<W|UW>);
    % the revision datasets are single-cohort (training_set_1_16_cells).
    n_sub  = numel(mp_prefixes);
    n_seed = numel(seeds);
    subset_idx = zeros(n_sub, 1);
    subset_siz = zeros(n_sub, 1);
    for i = 1 : n_sub
        tok = regexp(mp_prefixes{i}, '^(?:v1|hex)_(\d+)_(\d+)_(?:W|UW)$', 'tokens', 'once');
        if ~isempty(tok)
            subset_idx(i) = str2double(tok{1});
            subset_siz(i) = str2double(tok{2});
        else
            subset_idx(i) = 1;     % revision: single training_set_1_16_cells cohort
            subset_siz(i) = 16;
        end
    end

    % Sort cohorts by size (then index), as the legacy loader did.
    [~, order] = sortrows([subset_siz, subset_idx]);
    MP.(tasks{t}).subsets    = mp_prefixes(order);
    MP.(tasks{t}).subset_idx = subset_idx(order);
    MP.(tasks{t}).subset_siz = subset_siz(order);

    MP.(tasks{t}).graph_names = cell(n_sub, n_seed);
    MP.(tasks{t}).file_header = cell(n_sub, n_seed);
    MP.(tasks{t}).graph_id = cell(n_sub, n_seed);
    MP.(tasks{t}).vals = cell(n_sub, n_seed);
    MP.(tasks{t}).inds = cell(n_sub, n_seed);

    % Flatten (i, s) into a single index range for parfor. Each iteration is
    % independent: writes only to its own slot of par_* sliced cells, then we
    % assemble back into MP.<task>.* after the parfor.
    N        = n_sub * n_seed;
    par_gn   = cell(N, 1);
    par_fh   = cell(N, 1);
    par_gi   = cell(N, 1);
    par_vv   = cell(N, 1);
    par_in   = cell(N, 1);
    par_ok   = false(N, 1);

    subsets_t = MP.(tasks{t}).subsets;
    task_t    = tasks{t};

    parfor k = 1 : N
        [i_par, s_par] = ind2sub([n_sub, n_seed], k);
        prefix = subsets_t{i_par};

        % Task-specific prefix: a v1_UW run carries _UW prefixes but needs
        % the _W files for the lengths_to_lengths task (and vice versa).
        % Prefixes without a W/UW suffix (hex, revision) pass through.
        if strcmp(task_t, 'lengths_to_lengths') && endsWith(prefix, '_UW')
            prefix_for_file = [prefix(1:end-2), 'W'];
        elseif strcmp(task_t, 'none_to_lengths') && endsWith(prefix, '_W')
            prefix_for_file = [prefix(1:end-1), 'UW'];
        else
            prefix_for_file = prefix;
        end

        if consolidated
            curr_filename = DCG_consolidated_paths('pred_file', ...
                data_root, prefix_for_file, model, seeds(s_par));
        else
            curr_filename = fullfile(data_root, ...
                sprintf('pred_%s__%s_s%d.txt', prefix_for_file, model, seeds(s_par)));
        end

        if ~isfile(curr_filename)
            fprintf('*** File not found: "%s"\n', curr_filename);
            continue;
        end

        % Most MP prediction files hold only edge rows, but some GAT exports
        % embed a per-graph node-feature block ("Format nodes:") just like
        % PPGN. Left in place, load_dataset reshapes those 37-col node rows
        % into the 6-col edge layout and corrupts vv_k: ~2x the rows and
        % garbage (u,v) pairs > n_cells. That surfaces downstream as a
        % sub2ind overflow in the none_to_lengths alignment (section 11) and
        % a cell2mat size mismatch in model stacking (section 14). Detect the
        % block in the file preamble and pass consider_nodes=1 so load_dataset
        % truncates each graph at its first node row -- exactly as
        % extract_PPGN_results does. Node-less files are unaffected.
        fid_probe = fopen(curr_filename, 'rt');
        header_probe = fread(fid_probe, 8192, '*char')';
        fclose(fid_probe);
        consider_nodes_k = contains(header_probe, 'Format nodes:');

        [gn_k, ~, fh_k, gi_k, vv_k] = load_dataset(curr_filename, consider_nodes_k);

        for j = 1 : length(vv_k)
            to_remove = vv_k{j}(:,1) > vv_k{j}(:,2);
            vv_k{j}(to_remove,:) = [];
            % DEDUP-FIX 2026-05-19: handle exact (u,v) duplicates that come from
            % a re-evaluation being concatenated into the file by mistake (hit
            % on Tissue_484 GraphSAGE seed 2 graph 816: 71 affected edges, each
            % with two near-identical prediction rows). Keep the first
            % occurrence per (u,v) -- predictions are equivalent to within
            % float noise so the choice is immaterial.
            [~, unique_idx] = unique(vv_k{j}(:,1:2), 'rows', 'stable');
            if numel(unique_idx) < size(vv_k{j}, 1)
                vv_k{j} = vv_k{j}(unique_idx, :);
            end
            [~, order_j] = sortrows(vv_k{j}(:,1:2));
            vv_k{j} = vv_k{j}(order_j,:);

            % BASELINE-FIX 2026-05-14 (drop_flag1). See the long comment block
            % below; semantics unchanged in the parfor version.
            if strcmp(task_t, 'lengths_to_lengths') && is_new_interface_extra
                removed_line = vv_k{j}(:,4) == 1;
                vv_k{j}(removed_line,:) = [];
            end
        end

        par_gn{k} = gn_k;
        par_fh{k} = fh_k;
        par_gi{k} = gi_k;
        par_vv{k} = vv_k;
        % split indices for this cohort, read from inds\<prefix>\
        par_in{k} = read_dataset_inds(inds_root, prefix_for_file, data_root);
        par_ok(k) = true;
    end

    % BASELINE-FIX 2026-05-14 (drop_flag1; 768 edges in W per graph).
    % Previously: the eliminated edge (flag=1, pre>0, post=0) and the
    % newly-formed edge (pre=0, post>0) were MERGED into a single row by
    % copying the new edge's post/pred onto the eliminated row's pre. The
    % merged row carried correct |new_pred-new_post| as MAE but
    % |removed_pre-new_post| (~0.49 vs ~0.12 expected) as the identity-
    % baseline contribution -- inflating apparent improvement-over-baseline
    % by ~0.07 nMAE aggregated, ~0.5 at hop 0.
    % Now: drop the eliminated edge row entirely. We KNOW a priori that the
    % eliminated edge has post-T1 length 0, so neither model error nor
    % identity baseline is evaluated on it. The newly-formed edge (pre=0,
    % post=new_post) is retained and counts as the T1 hop-0 row. Result:
    % exactly 768 W edges per graph (vs 769 in raw .txt; UW is already 768
    % since the file format omits the eliminated row). `is_new_interface_extra`
    % still gates this so legacy data without the separate eliminated-edge
    % row pass through unchanged.

    % Assemble parfor outputs into MP struct.
    for k = 1 : N
        if ~par_ok(k), continue; end
        [i_un, s_un] = ind2sub([n_sub, n_seed], k);
        MP.(tasks{t}).graph_names{i_un, s_un} = par_gn{k};
        MP.(tasks{t}).file_header{i_un, s_un} = par_fh{k};
        MP.(tasks{t}).graph_id{i_un, s_un}    = par_gi{k};
        MP.(tasks{t}).vals{i_un, s_un}        = par_vv{k};
        % Original code wrote inds{i_un} (linear -> column 1). Preserve that
        % so the main script's `inds{i}` access still finds the per-cohort
        % train/val/test struct. Storing on every (i,s) is also harmless.
        MP.(tasks{t}).inds{i_un, s_un} = par_in{k};
    end

    % if we loaded no files, we continue to the next iteration:
    if all(cellfun(@isempty, MP.(tasks{t}).vals(:)))
        continue;
    end

    % to_remove = cellfun(@isempty, MP.(tasks{t}).vals);
    % MP.(tasks{t}) = structfun(@(x) x(~to_remove,:), MP.(tasks{t}), 'UniformOutput', false);

    % verifying that all graphs have been loaded:
    n_graphs = regexp(MP.(tasks{t}).file_header, 'Total graphs\: (\d+)', 'tokens');
    n_graphs = [n_graphs{:}]';
    n_graphs = [n_graphs{:}]';
    n_graphs = cellfun(@str2num, n_graphs);
    if ~isequal(n_graphs, cellfun(@length, MP.(tasks{t}).vals(:)))
        keyboard; % mismatch in # of graphs
    end


end
end


%==========================================================================
% HELPER: extract_PPGN_results   (rewritten 2026-05-23 for the flat layout)
%==========================================================================
% PPGN = extract_PPGN_results(data_root, inds_root, mp_prefixes, tasks, ...
%                             is_new_interface_extra, seeds)
%
% Same flat-layout discovery as extract_MP_results, just with the model
% hardcoded to "PPGN" in the filename pattern:
%     <data_root>\pred_<prefix>__PPGN_s<seed>.txt
% Returns the SAME struct shape as extract_MP_results -- the historical
% per-seed-folder layer is gone (was {s}.vals{i,s}); now it's just
% PPGN.(task).vals{i,s}, indexed the same way as every other model.
%
% Partial coverage tolerance (2026-05-23): not every (prefix, seed) has a
% prediction file yet (e.g. v1_1_32_W has none, v1_1_32_UW has 2/5,
% v1_2_16_W has 4/5). Missing files are logged and the slot is left empty;
% section 6 NaN-fills them from PNA's template so downstream code is happy.
function PPGN = extract_PPGN_results(data_root, inds_root, mp_prefixes, tasks, is_new_interface_extra, seeds)

% Reuse the existing pool if MP started one already.
if isempty(gcp('nocreate'))
    try, parpool('local', 6); catch, end
end

% Consolidated snapshot vs legacy flat pred filenames (see extract_MP_results).
consolidated = DCG_consolidated_paths('is_consolidated', data_root);

for t = 1 : length(tasks)

    % Same prefix -> cohort idx/siz parsing as extract_MP_results.
    n_sub  = numel(mp_prefixes);
    n_seed = numel(seeds);
    subset_idx = zeros(n_sub, 1);
    subset_siz = zeros(n_sub, 1);
    for i = 1 : n_sub
        tok = regexp(mp_prefixes{i}, '^(?:v1|hex)_(\d+)_(\d+)_(?:W|UW)$', 'tokens', 'once');
        if ~isempty(tok)
            subset_idx(i) = str2double(tok{1});
            subset_siz(i) = str2double(tok{2});
        else
            subset_idx(i) = 1;     % revision: single training_set_1_16_cells cohort
            subset_siz(i) = 16;
        end
    end

    [~, order] = sortrows([subset_siz, subset_idx]);
    PPGN.(tasks{t}).subsets    = mp_prefixes(order);
    PPGN.(tasks{t}).subset_idx = subset_idx(order);
    PPGN.(tasks{t}).subset_siz = subset_siz(order);

    PPGN.(tasks{t}).graph_names = cell(n_sub, n_seed);
    PPGN.(tasks{t}).file_header = cell(n_sub, n_seed);
    PPGN.(tasks{t}).graph_id    = cell(n_sub, n_seed);
    PPGN.(tasks{t}).vals        = cell(n_sub, n_seed);
    PPGN.(tasks{t}).inds        = cell(n_sub, n_seed);

    N        = n_sub * n_seed;
    par_gn   = cell(N, 1);
    par_fh   = cell(N, 1);
    par_gi   = cell(N, 1);
    par_vv   = cell(N, 1);
    par_in   = cell(N, 1);
    par_ok   = false(N, 1);

    subsets_t = PPGN.(tasks{t}).subsets;
    task_t    = tasks{t};

    parfor k = 1 : N
        [i_par, s_par] = ind2sub([n_sub, n_seed], k);
        prefix = subsets_t{i_par};

        % Same task-specific prefix-swap as extract_MP_results: a v1_UW run
        % carries _UW prefixes but needs the _W files for the
        % lengths_to_lengths task (and vice versa).
        if strcmp(task_t, 'lengths_to_lengths') && endsWith(prefix, '_UW')
            prefix_for_file = [prefix(1:end-2), 'W'];
        elseif strcmp(task_t, 'none_to_lengths') && endsWith(prefix, '_W')
            prefix_for_file = [prefix(1:end-1), 'UW'];
        else
            prefix_for_file = prefix;
        end

        if consolidated
            curr_filename = DCG_consolidated_paths('pred_file', ...
                data_root, prefix_for_file, 'PPGN', seeds(s_par));
        else
            curr_filename = fullfile(data_root, ...
                sprintf('pred_%s__PPGN_s%d.txt', prefix_for_file, seeds(s_par)));
        end

        if ~isfile(curr_filename)
            fprintf('*** PPGN file not found (skipped): "%s"\n', curr_filename);
            continue;
        end

        % Pass consider_nodes=1 (parity with the historical PPGN call).
        [gn_k, ~, fh_k, gi_k, vv_k] = load_dataset(curr_filename, 1);

        % Sanity: the legacy PPGN extractor caught the un-normalized branch
        % via this >100 trip; preserve it. Skip if vv_k is empty.
        if ~isempty(vv_k) && ~isempty(vv_k{1}) && vv_k{1}(1,end) > 100
            error('extract_PPGN_results:suspiciousValue', ...
                'predicted length > 100 in %s', curr_filename);
        end

        for j = 1 : length(vv_k)
            % Same row hygiene as extract_MP_results: keep u<v direction,
            % dedup exact (u,v), stable-sort by (u,v).
            to_remove = vv_k{j}(:,1) > vv_k{j}(:,2);
            vv_k{j}(to_remove,:) = [];
            [~, unique_idx] = unique(vv_k{j}(:,1:2), 'rows', 'stable');
            if numel(unique_idx) < size(vv_k{j}, 1)
                vv_k{j} = vv_k{j}(unique_idx, :);
            end
            [~, order_j] = sortrows(vv_k{j}(:,1:2));
            vv_k{j} = vv_k{j}(order_j,:);

            % BASELINE-FIX 2026-05-14 (drop_flag1; 768 W edges per graph).
            % Identical to extract_MP_results: drop the eliminated-edge row
            % (flag=1). UW is already 768 in the file format; section 8
            % later strips PPGN's extra UW col-3 (always zero).
            if strcmp(task_t, 'lengths_to_lengths') && is_new_interface_extra
                removed_line = vv_k{j}(:,4) == 1;
                vv_k{j}(removed_line,:) = [];
            end
        end

        par_gn{k} = gn_k;
        par_fh{k} = fh_k;
        par_gi{k} = gi_k;
        par_vv{k} = vv_k;
        par_in{k} = read_dataset_inds(inds_root, prefix_for_file, data_root);
        par_ok(k) = true;
    end

    for k = 1 : N
        if ~par_ok(k), continue; end
        [i_un, s_un] = ind2sub([n_sub, n_seed], k);
        PPGN.(tasks{t}).graph_names{i_un, s_un} = par_gn{k};
        PPGN.(tasks{t}).file_header{i_un, s_un} = par_fh{k};
        PPGN.(tasks{t}).graph_id{i_un, s_un}    = par_gi{k};
        PPGN.(tasks{t}).vals{i_un, s_un}        = par_vv{k};
        PPGN.(tasks{t}).inds{i_un, s_un}        = par_in{k};
    end

    % Skip the rest of the per-task verification if nothing loaded.
    if all(cellfun(@isempty, PPGN.(tasks{t}).vals(:)))
        continue;
    end

    % Verify "Total graphs: N" header matches parsed count -- restricted to
    % slots that actually loaded (partial coverage is OK).
    loaded_mask = par_ok;
    if any(loaded_mask)
        fh_loaded = PPGN.(tasks{t}).file_header(loaded_mask);
        vv_loaded = PPGN.(tasks{t}).vals(loaded_mask);
        tok = regexp(fh_loaded, 'Total graphs\: (\d+)', 'tokens');
        tok = [tok{:}]'; tok = [tok{:}]';
        n_graphs_hdr = cellfun(@str2num, tok);
        n_graphs_got = cellfun(@length, vv_loaded);
        if ~isequal(n_graphs_hdr(:), n_graphs_got(:))
            warning('DCG:ppgnGraphCount', ...
                'PPGN: header "Total graphs" disagrees with parsed count in %d loaded slot(s) for task %s', ...
                sum(n_graphs_hdr(:) ~= n_graphs_got(:)), task_t);
        end
    end

end
end


%==========================================================================
% HELPER: read_dataset_inds   (added 2026-05-22 -- flat data layout)
%==========================================================================
% inds = read_dataset_inds(inds_root, prefix)
%
% Fetch one dataset's train/val/test split from the new flat data layout:
%     <inds_root>\<prefix>\train.inds   val.inds   test.inds
% <prefix> is the dataset tag carried by every prediction filename
% (pred_<prefix>__<model>_s<seed>.txt) -- e.g. 'v1_2_8_W', 'hex_2_8_W',
% 'rev_kA_1'. The split is a property of the dataset, so it is shared by
% all 4 models x 5 seeds of that prefix.
%
% Returns a struct with .train / .val / .test, each a column of 1-based
% graph indices. The .inds files store 0-based indices; the +1 converts to
% MATLAB indexing -- the same convention the legacy per-cohort loader used.
function inds = read_dataset_inds(inds_root, prefix, data_root)
if nargin < 3, data_root = ''; end
if ~isempty(data_root) && DCG_consolidated_paths('is_consolidated', data_root)
    % Consolidated snapshot: split is in splits\<key>\, chosen by _applies_to.
    inds_dir = DCG_consolidated_paths('inds_dir', data_root, prefix);
    if isempty(inds_dir)
        error('read_dataset_inds:noSplit', ...
              'no split folder applies to prefix "%s" under %s', ...
              prefix, fullfile(data_root, 'splits'));
    end
else
    inds_dir = fullfile(inds_root, prefix);
end
split_names = {'train', 'val', 'test'};
inds = struct();
for k = 1 : numel(split_names)
    fname = fullfile(inds_dir, [split_names{k}, '.inds']);
    fid = fopen(fname, 'rt');
    if fid < 0
        error('read_dataset_inds:missingFile', ...
              'indices file not found: %s', fname);
    end
    inds.(split_names{k}) = fscanf(fid, '%u', Inf) + 1;
    fclose(fid);
end
end


function LineG = dcg_line_graph_preserve_rows(C, varargin)
%DCG_LINE_GRAPH_PRESERVE_ROWS
% Build a row-preserving historical vertex-line interface graph.
%
% Distance analyses identify T1 roots by original prediction-matrix row, so
% the line graph must preserve that row order and row count.
%
% One hop means two interfaces meet at an inferred epithelial vertex. With no
% explicit vertex incidence, the historical helper infers triple junctions
% combinatorially: rows (i,j), (i,k), and (j,k) are mutually adjacent when all
% three cell-cell interfaces exist. This reproduces the original 22-ish hop
% definition while avoiding any future root-row shift if an input is not
% already sorted/unique.

C = sort(C, 2);
m = size(C, 1);
if m == 0
    LineG = graph(sparse(0, 0));
    LineG.Nodes.Interface = zeros(0, 2);
    return;
end

cells = unique(C(:));
[~, Cmap] = ismember(C, cells);
n = numel(cells);

pairRows = containers.Map('KeyType', 'char', 'ValueType', 'any');
nbrsByCell = cell(n, 1);
for e = 1 : m
    a = Cmap(e, 1);
    b = Cmap(e, 2);
    key = pair_key(a, b);
    if isKey(pairRows, key)
        pairRows(key) = [pairRows(key), e];
    else
        pairRows(key) = e;
    end
    nbrsByCell{a}(end+1) = b; %#ok<AGROW>
    nbrsByCell{b}(end+1) = a; %#ok<AGROW>
end

L = sparse(m, m);

if isempty(varargin)
    vertexCells = [];
else
    vertexCells = varargin{1};
end

if isempty(vertexCells)
    for i = 1 : n
        nbrs = unique(nbrsByCell{i});
        for aa = 1 : numel(nbrs)-1
            j = nbrs(aa);
            for bb = aa+1 : numel(nbrs)
                k = nbrs(bb);
                e1 = rows_for_pair(i, j);
                e2 = rows_for_pair(i, k);
                e3 = rows_for_pair(j, k);
                if isempty(e1) || isempty(e2) || isempty(e3)
                    continue;
                end
                connect_sets(e1, e2);
                connect_sets(e1, e3);
                connect_sets(e2, e3);
            end
        end
    end
else
    if ~iscell(vertexCells)
        vmat = vertexCells;
        vertexCells = cell(size(vmat, 1), 1);
        for v = 1 : size(vmat, 1)
            row = vmat(v, :);
            row(~isfinite(row) | row == 0) = [];
            vertexCells{v} = row;
        end
    end

    for v = 1 : numel(vertexCells)
        cellsHere = vertexCells{v};
        [tf, cellsHereIdx] = ismember(cellsHere, cells);
        cellsHereIdx = cellsHereIdx(tf);
        if numel(cellsHereIdx) < 2
            continue;
        end
        pairs = nchoosek(cellsHereIdx, 2);
        eIdx = [];
        for pp = 1 : size(pairs, 1)
            eIdx = [eIdx, rows_for_pair(pairs(pp,1), pairs(pp,2))]; %#ok<AGROW>
        end
        eIdx = unique(eIdx);
        if numel(eIdx) >= 2
            L(eIdx, eIdx) = 1;
        end
    end
end

L(1:m+1:end) = 0;
LineG = graph(spones(L));
LineG.Nodes.Interface = C;

    function key = pair_key(a, b)
        if a > b
            tmp = a;
            a = b;
            b = tmp;
        end
        key = sprintf('%d_%d', a, b);
    end

    function rows = rows_for_pair(a, b)
        key = pair_key(a, b);
        if isKey(pairRows, key)
            rows = pairRows(key);
        else
            rows = [];
        end
    end

    function connect_sets(rowsA, rowsB)
        L(rowsA, rowsB) = 1;
        L(rowsB, rowsA) = 1;
    end
end


function dcg_assert_root_rows_match_line_graph(line_G, original_edges, root_rows, context)
%DCG_ASSERT_ROOT_ROWS_MATCH_LINE_GRAPH
% Fail if line-graph node rows do not match original prediction rows.

if any(root_rows < 1) || any(root_rows > numnodes(line_G))
    error('DCG:rootOutOfLineGraphRange', ...
        'Root row out of line-graph range in %s.', context);
end

expected = sort(original_edges(root_rows, :), 2);
actual = line_G.Nodes.Interface(root_rows, :);
if ~isequal(expected, actual)
    error('DCG:rootLineGraphRowMismatch', ...
        ['Line-graph root rows do not match prediction rows in %s. ', ...
         'Distance analysis would start from the wrong T1 interface.'], context);
end
end
