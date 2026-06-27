function outputs = DCG_plot_Flip_two_interaction(analyses_filename, figures_output_dir, save_png, single_t1_analyses_filename)
% DCG_plot_Flip_two_interaction  Implement dcg plot flip two interaction for this MATLAB workflow.
% Inputs: analyses_filename, figures_output_dir, save_png, single_t1_analyses_filename
% Outputs: outputs
%DCG_PLOT_FLIP_TWO_INTERACTION
% Two-source diagnostics for the Flip_two revision dataset.
%
% The standard plotter collapses multiple T1 roots to nearest-root distance.
% This add-on re-loads the raw analyses cache and reconstructs distances to
% each of the two newly formed interfaces separately.
%
% ============================ FILE OVERVIEW ============================
% CONTEXT
%   Part of the DCG / GNN-benchmark analysis pipeline (npj revision). The
%   "Flip_two" dataset is a controlled experiment in which exactly TWO T1
%   transitions (cell-neighbor swaps) are introduced concurrently into each
%   tissue graph. The scientific question is how the two flips INTERACT: does
%   a model's prediction error near one flip depend on proximity to the
%   second flip? The standard DCG plotter only knows the distance to the
%   nearest T1 root, so it cannot separate "near one flip" from "near both
%   flips". This file re-derives, per edge, the distance to EACH of the two
%   flip interfaces independently and produces interaction-aware tables and
%   figures.
%
% DATA MODEL / CONVENTIONS (shared by every function below)
%   * The analyses cache is a struct I with one field per model
%     (I.PNA, I.GAT, I.GIN, I.GraphSAGE) plus task/subset/seed nesting:
%         I.<model>.<task>.vals{subset_i, seed}{graph_idx}  -> per-graph matrix
%         I.<model>.<task>.inds{subset_i, seed}.<split>      -> graph indices
%   * A per-graph matrix ("curr_G") has one ROW per graph EDGE. Columns:
%         col 1, col 2 : the two vertex-model endpoints of the edge (used to
%                        build the vertex-line graph; edges become nodes).
%         col 3        : the BASELINE prediction = preferred/initial edge
%                        length BEFORE any T1 (also called in_preferred_length).
%                        A value of exactly 0 here MARKS a T1 root interface.
%         col end-1    : the GROUND-TRUTH target length (GT).
%         col end      : the per-model PREDICTED length (only meaningful in the
%                        per-model matrices I.<model>...; for the baseline the
%                        prediction is col 3).
%   * PNA is treated as the CANONICAL source of graph topology and ground
%     truth: graph indexing, edge order, root detection, vertex-line distances
%     and GT are all read from I.PNA. The other models supply only their own
%     predicted column, aligned row-for-row to the PNA graph.
%
% KEY ANALYSIS DECISIONS (fixed in the main function; see inline notes)
%   * near_radius = 3 hops  : an edge is "near" a flip if its vertex-line hop
%                             distance to that flip's root is <= 3.
%   * task        = 'lengths_to_lengths' : the only regression task analyzed.
%   * split       = 'test' ONLY          : summaries use held-out graphs only.
%   * PPGN        : INCLUDED (2026-06-01). The analyzer normalizes PPGN to the
%                   MP flat layout, so I.PPGN is row-aligned to I.PNA -- verified
%                   4080/4080 Flip_two graphs identical on cols 1-5 (endpoints,
%                   pre-length, flip, GT). Earlier builds excluded it for a
%                   now-stale mis-alignment concern.
%   * Models analyzed: GraphSAGE, GAT, GIN, PNA, plus a Baseline derived from
%                   col 3 (the pre-flip preferred length).
%
% PER-EDGE ERROR DEFINITIONS (computed in extract_flip_two_records)
%   * model error    = abs(model_predicted_length - GT)   (per trained model)
%   * baseline error = abs(col3 - GT) = abs(preferred_length - GT)
%
% ZONES (mutually exclusive edge classes, per graph; near_radius = 3)
%   * zone 1 "near exactly one flip" : (d1<=r) XOR (d2<=r)
%   * zone 2 "near both flips"       : (d1<=r) AND (d2<=r)
%   * zone 3 "far from both flips"   : (d1>r)  AND (d2>r)
%   (d1, d2 are the vertex-line hop distances to root #1 and root #2.)
%
% INTER-FLIP DISTANCE GROUPS (per graph; tertiles, see
% inter_flip_distance_group_edges): close / middle / far, cut at the 1/3 and
% 2/3 quantiles of the per-graph inter-flip hop distance.
%
% AGGREGATION LADDER (see summarize_wide_records)
%   edge errors -> mean within graph/bin -> mean within seed -> mean across
%   seed means; reported SD is the standard deviation ACROSS seed means.
%
% OUTPUTS
%   write_outputs_tables : CSV record/summary tables + a .mat of pair records.
%   write_assumptions    : a human-readable text log of every decision above.
%   Figures (returned in the OUTPUTS struct): graph MAE vs inter-flip
%   distance, MAE vs nearest-T1 distance (raw / log2 / log2-nMAE), interaction
%   zone bars, two-distance heatmaps, close/middle/far nearest-distance curves,
%   and two reviewer-facing single-T1 vs two-T1 comparisons:
%     1. graph-level overall MAE/nMAE degradation;
%     2. edge-level nearest-T1 distance profiles.
% ======================================================================
%
% H1: Entry point. Build interaction-aware Flip_two tables/figures from cache.
%
% PURPOSE
%   Top-level driver. Loads the raw analyses cache, reconstructs two-flip
%   distance structure per edge, aggregates errors across the seed ladder,
%   writes record/summary tables plus an assumptions log, and renders the
%   full set of diagnostic figures.
%
% INPUTS
%   analyses_filename  : (char, optional) path to the '... - analyses data.mat'
%                        cache containing variable I. Defaults to the Flip_two
%                        cache under the consolidated snapshot when missing/empty.
%   figures_output_dir : (char, optional) directory for CSV/MAT/TXT/FIG/PNG
%                        outputs. Created if it does not exist. Defaults to the
%                        configured Flip_two dataset folder under _figures, so
%                        these diagnostics sit beside the standard Flip_two
%                        figures.
%   save_png           : (logical, optional) if true, also export each figure
%                        as a 300-dpi PNG alongside the .fig. Default false.
%   single_t1_analyses_filename :
%                        (char, optional) path to the weighted single-T1 v1
%                        cache. Defaults to 'v1_W - analyses data.mat' in the
%                        same analyzer-cache folder as analyses_filename.
%
% OUTPUTS
%   outputs : struct of figure file paths, one field per generated figure
%             (graph_vs_inter_flip, nearest_raw, nearest_log, nearest_nmae,
%             zones, heatmap_raw, heatmap_nmae, close_far,
%             single_vs_two_overall, single_vs_two_nearest_raw,
%             single_vs_two_nearest_nmae).
%
% ALGORITHM
%   1. Resolve default arguments and ensure the output directory exists.
%   2. Fix the analysis decisions (near_radius=3, task, split='test',
%      trained-model list, palette).
%   3. load() variable I from the cache.
%   4. extract_flip_two_records -> Graph/Nearest/Pair/Zone long tables.
%   5. summarize_wide_records -> per-group seed-ladder summaries for each view.
%   6. Derive inter-flip distance tertile groups and attach to Graph/Nearest;
%      summarize the nearest view per distance group.
%   7. Attach human-readable zone and distance-group labels.
%   8. write_outputs_tables + write_assumptions persist everything.
%   9. Render every figure, collecting their paths into OUTPUTS.
%
% DECISIONS / EDGE-CASES
%   * Defaults are resolved from DCG_publication_config; pass explicit paths to override.
%   * near_radius, task and split are the load-bearing analysis choices; see
%     the FILE OVERVIEW for rationale. (PPGN is now included, 2026-06-01.)

if nargin < 1 || isempty(analyses_filename)
    path_cfg = DCG_publication_config();
    if isempty(path_cfg.data_root)
        error('DCG:missingDataRoot', ['Pass analyses_filename explicitly or set ', ...
            'DCG_DATA_ROOT / DCG_local_config.m.']);
    end
    analyses_filename = fullfile(path_cfg.data_root, '_analyzer_cache', ...
        'revision_2026', 'Flip_two - analyses data.mat');
end
if nargin < 2 || isempty(figures_output_dir)
    path_cfg = DCG_publication_config();
    if ~isempty(path_cfg.data_root)
        figures_output_dir = fullfile(path_cfg.data_root, '_figures', ...
            'revision_2026', 'Flip_two');
    else
        figures_output_dir = fullfile(fileparts(analyses_filename), '_figures', 'Flip_two');
    end
end
if nargin < 3 || isempty(save_png)
    save_png = false;
end
if nargin < 4 || isempty(single_t1_analyses_filename)
    single_t1_analyses_filename = default_single_t1_analyses_filename(analyses_filename);
end

if ~isfolder(figures_output_dir)
    mkdir(figures_output_dir);
end

near_radius = 3;
task = 'lengths_to_lengths';
split_name = 'test';
trained_models = {'PPGN','GraphSAGE','GAT','GIN','PNA'};
model_names = [trained_models, {'Baseline'}];
colors = paper_model_colors(model_names);

fprintf('[DCG_plot_Flip_two_interaction] loading %s\n', analyses_filename);
L = load(analyses_filename, 'I');
I = L.I;

[GraphT, NearestT, PairT, ZoneT, skipped] = extract_flip_two_records(I, task, split_name, trained_models, near_radius);

fprintf('[DCG_plot_Flip_two_interaction] loading single-T1 reference %s\n', single_t1_analyses_filename);
LS = load(single_t1_analyses_filename, 'I');
SingleI = LS.I;
[SingleGraphT, SingleNearestT, SingleZoneT, skipped_single, single_subset] = extract_single_t1_records(SingleI, task, split_name, trained_models, near_radius);

GraphSummary = summarize_wide_records(GraphT, {'inter_flip_dist'}, model_names);
NearestSummary = summarize_wide_records(NearestT, {'d_near'}, model_names);
PairSummary = summarize_wide_records(PairT, {'d_near','d_far'}, model_names);
ZoneSummary = summarize_wide_records(ZoneT, {'zone_id'}, model_names);
ComparisonGraphT = combine_single_two_records(SingleGraphT, GraphT, model_names, 'graph');
ComparisonNearestT = combine_single_two_records(SingleNearestT, NearestT, model_names, 'nearest');
ComparisonZoneT = combine_single_two_zone_records(SingleZoneT, ZoneT, model_names);
OverallComparisonSummary = summarize_wide_records(ComparisonGraphT, {'condition_id'}, model_names);
NearestComparisonSummary = summarize_wide_records(ComparisonNearestT, {'condition_id','d_near'}, model_names);
ZoneComparisonSummary = summarize_wide_records(ComparisonZoneT, {'comparison_zone_id'}, model_names);
PairVsSingleSummary = add_single_reference_to_pair_summary(PairSummary, NearestComparisonSummary, model_names);

distance_group_edges = inter_flip_distance_group_edges(GraphT.inter_flip_dist);
GraphT.distance_group = inter_flip_distance_group(GraphT.inter_flip_dist, distance_group_edges);
NearestT.distance_group = inter_flip_distance_group(NearestT.inter_flip_dist, distance_group_edges);
NearestGroupSummary = summarize_wide_records(NearestT, {'distance_group','d_near'}, model_names);

ZoneSummary.zone = zone_label_from_id(ZoneSummary.zone_id, near_radius);
NearestGroupSummary.distance_group_label = distance_group_label_from_id(NearestGroupSummary.distance_group);
OverallComparisonSummary.condition = condition_label_from_id(OverallComparisonSummary.condition_id);
NearestComparisonSummary.condition = condition_label_from_id(NearestComparisonSummary.condition_id);
ZoneComparisonSummary.zone = comparison_zone_label_from_id(ZoneComparisonSummary.comparison_zone_id, near_radius);

write_outputs_tables(figures_output_dir, GraphT, NearestT, PairT, ZoneT, ...
    GraphSummary, NearestSummary, PairSummary, ZoneSummary, NearestGroupSummary);
write_single_vs_two_outputs(figures_output_dir, SingleGraphT, SingleNearestT, SingleZoneT, ...
    ComparisonGraphT, ComparisonNearestT, ComparisonZoneT, ...
    OverallComparisonSummary, NearestComparisonSummary, ZoneComparisonSummary, PairVsSingleSummary);
write_assumptions(figures_output_dir, analyses_filename, single_t1_analyses_filename, ...
    split_name, near_radius, skipped, skipped_single, distance_group_edges, single_subset);

outputs = struct();
outputs.single_vs_two_bars_2x2 = plot_single_vs_two_bar_summary_2x2(OverallComparisonSummary, ZoneComparisonSummary, model_names, colors, near_radius, figures_output_dir, save_png);
outputs.single_vs_two_bars_2x2_mae_log_scale = plot_single_vs_two_bar_summary_2x2(OverallComparisonSummary, ZoneComparisonSummary, model_names, colors, near_radius, figures_output_dir, save_png, true);
outputs.nearest_single_two_2x3 = plot_nearest_distance_single_two_2x3(NearestSummary, NearestComparisonSummary, model_names, colors, figures_output_dir, save_png);
outputs.graph_vs_inter_flip = plot_graph_mae_vs_inter_flip(GraphSummary, model_names, colors, figures_output_dir, save_png, OverallComparisonSummary);
outputs.heatmap_raw = plot_pair_heatmaps(PairSummary, trained_models, 'raw_mean', ...
    'Mean graph-edge MAE', 'Flip_two MAE heatmap by two-T1 distances (raw MAE)', figures_output_dir, save_png);
outputs.heatmap_nmae = plot_pair_heatmaps(PairSummary, trained_models, 'log2_nmae_mean', ...
    'log2(nMAE to graph-pair baseline)', 'Flip_two MAE heatmap by two-T1 distances (log2 nMAE)', figures_output_dir, save_png);
outputs.heatmap_raw_vs_single = plot_pair_heatmaps(PairVsSingleSummary, trained_models, 'raw_delta_vs_single', ...
    'Flip_two raw MAE minus single-T1 raw MAE at same d_{near}', 'Flip_two MAE heatmap by two-T1 distances (raw MAE minus single-T1)', figures_output_dir, save_png, true);
outputs.heatmap_nmae_vs_single = plot_pair_heatmaps(PairVsSingleSummary, trained_models, 'log2_nmae_delta_vs_single', ...
    'Flip_two log2(nMAE) minus single-T1 log2(nMAE) at same d_{near}', 'Flip_two MAE heatmap by two-T1 distances (log2 nMAE minus single-T1)', figures_output_dir, save_png, true);

fprintf('[DCG_plot_Flip_two_interaction] complete. Output: %s\n', figures_output_dir);

end


function [GraphT, NearestT, PairT, ZoneT, skipped] = extract_flip_two_records(I, task, split_name, trained_models, near_radius)
% extract_flip_two_records  Extract flip two records records from analysis structures.
% Inputs: I, task, split_name, trained_models, near_radius
% Outputs: GraphT, NearestT, PairT, ZoneT, skipped
%EXTRACT_FLIP_TWO_RECORDS
% H1: Reconstruct per-edge two-flip distances/errors into four long tables.
%
% PURPOSE
%   The analytical heart of the file. Walks every TEST graph (per seed) in the
%   PNA cache, finds the two T1 root interfaces, computes each edge's
%   vertex-line hop distance to BOTH roots, forms per-edge model/baseline errors,
%   and bins those errors four complementary ways (whole-graph, nearest-hop,
%   distance-pair, interaction-zone) into tidy tables for downstream summary.
%
% INPUTS
%   I              : analyses cache struct (see FILE OVERVIEW for layout).
%   task           : (char) task field name, e.g. 'lengths_to_lengths'.
%   split_name     : (char) split field name, here 'test'.
%   trained_models : cellstr of model names to read predictions for
%                    (e.g. {'GraphSAGE','GAT','GIN','PNA'}); 'Baseline' is
%                    appended internally as an extra synthetic model.
%   near_radius    : (scalar hops) threshold defining the three zone masks.
%
% OUTPUTS
%   GraphT  : one row per graph. Vars: seed, graph_idx, inter_flip_dist, plus
%             one error column per model = mean per-edge error over the graph.
%   NearestT: one row per (graph, nearest-hop value d_near). Error columns =
%             mean per-edge error over edges whose min-distance equals d_near.
%   PairT   : one row per (graph, unique (d_near,d_far) pair). Error columns =
%             mean per-edge error over edges sharing that exact distance pair.
%   ZoneT   : up to three rows per graph (one per occupied zone). Error columns
%             = mean per-edge error over edges in that zone.
%   skipped : struct counting discarded graphs:
%               .not_two_roots = graphs without exactly two col-3==0 roots (or
%                                whose distance matrix is not 2xN).
%               .bad_graph     = empty/NaN graphs, or graphs with no edge that
%                                is finite-distance to both roots with finite
%                                errors across all models.
%
% ALGORITHM
%   1. Pre-size output arrays from total test-graph count and an example graph
%      (NearestMeta allowed up to ~80 hop-bins/graph; Pair/Zone sized
%      generously; arrays grow on demand via grow_record_arrays).
%   2. For each seed, for each test graph index:
%      a. Skip empty graphs or graphs whose first row contains NaN.
%      b. Find root edges = rows with col 3 == 0; require EXACTLY two.
%      c. Build the row-preserving vertex-line graph from edge endpoints
%         (cols 1:2); edges become graph nodes. distances() from both roots
%         yields a 2xN matrix D.
%      d. d1 = D(1,:)', d2 = D(2,:)'; d_near=min(d1,d2); d_far=max(d1,d2).
%         inter_flip_dist = D(1, root_edges(2)) = hops between the two roots.
%      e. GT = col end-1 of the PNA graph. For each trained model, error =
%         abs(pred col end - GT) when its matrix aligns row-for-row; Baseline
%         error = abs(col3 - GT). Misaligned/empty model matrices stay NaN.
%      f. valid_edges = finite d1 & d2 & finite errors for ALL models.
%      g. Emit one Graph row (mean over valid edges); one Nearest row per
%         distinct d_near; one Pair row per distinct (d_near,d_far); one Zone
%         row per non-empty zone mask (xor / both / neither, vs near_radius).
%   3. Convert the filled prefixes of the meta/err arrays to tables and print
%      a count summary.
%
% MATH / DECISIONS / EDGE-CASES
%   * Root marker: col 3 == 0 encodes a freshly created T1 interface. Graphs
%     not having exactly two such rows are out of scope and skipped.
%   * Distances are LINE-GRAPH hop counts (edge-to-edge adjacency), matching
%     DCG_analyze_results; +Inf means unreachable and is filtered.
%   * inter_flip_dist is symmetric, so reading row 1 at the column of root 2
%     suffices.
%   * Per-model alignment is guarded by size(vals_m,1)==size(curr_G,1); any
%     model that fails the check contributes NaN, which then fails valid_edges
%     for the affected rows (and can skip the whole graph as bad_graph).
%   * mean(...,'omitnan') is used so a single missing model does not void a bin
%     that is otherwise populated, while the all-models-finite valid_edges gate
%     keeps the common case fully aligned.

n_models = numel(trained_models) + 1;
model_names = [trained_models, {'Baseline'}];
subset_i = 1;
n_seeds = size(I.PNA.(task).vals, 2);
n_graphs_total = 0;
for s = 1 : n_seeds
    n_graphs_total = n_graphs_total + numel(I.PNA.(task).inds{subset_i,s}.(split_name));
end

example_idx = I.PNA.(task).inds{subset_i,1}.(split_name)(1);
example_graph = I.PNA.(task).vals{subset_i,1}{example_idx};
n_edges_example = size(example_graph, 1);

GraphMeta = nan(n_graphs_total, 3);
GraphErr = nan(n_graphs_total, n_models);
NearestMeta = nan(n_graphs_total * 80, 4);
NearestErr = nan(n_graphs_total * 80, n_models);
PairMeta = nan(n_graphs_total * n_edges_example, 5);
PairErr = nan(n_graphs_total * n_edges_example, n_models);
ZoneMeta = nan(n_graphs_total * 3, 4);
ZoneErr = nan(n_graphs_total * 3, n_models);

g_row = 0;
n_row = 0;
p_row = 0;
z_row = 0;
skipped = struct('not_two_roots', 0, 'bad_graph', 0);

for s = 1 : n_seeds
    test_indices = I.PNA.(task).inds{subset_i,s}.(split_name)(:);
    for ii = 1 : numel(test_indices)
        graph_idx = test_indices(ii);
        curr_G = I.PNA.(task).vals{subset_i,s}{graph_idx};
        if isempty(curr_G) || any(isnan(curr_G(1,:)))
            skipped.bad_graph = skipped.bad_graph + 1;
            continue;
        end

        root_edges = find(curr_G(:,3) == 0);
        if numel(root_edges) ~= 2
            skipped.not_two_roots = skipped.not_two_roots + 1;
            continue;
        end

        line_G = flip_two_line_graph_preserve_rows(curr_G(:,1:2));
        assert_root_rows_match_line_graph(line_G, curr_G(:,1:2), root_edges, ...
            sprintf('Flip_two seed %d graph %d', s, graph_idx));
        D = distances(line_G, root_edges);
        if size(D, 1) ~= 2
            skipped.not_two_roots = skipped.not_two_roots + 1;
            continue;
        end

        d1 = D(1,:)';
        d2 = D(2,:)';
        d_near = min(d1, d2);
        d_far = max(d1, d2);
        inter_flip_dist = D(1, root_edges(2));

        gt = I.PNA.(task).vals{subset_i,s}{graph_idx}(:,end-1);
        errs = nan(size(curr_G,1), n_models);
        for m = 1 : numel(trained_models)
            vals_m = I.(trained_models{m}).(task).vals{subset_i,s}{graph_idx};
            if ~isempty(vals_m) && size(vals_m,1) == size(curr_G,1)
                errs(:,m) = abs(vals_m(:,end) - gt);
            end
        end
        errs(:,end) = abs(curr_G(:,3) - gt);

        valid_edges = isfinite(d1) & isfinite(d2) & all(isfinite(errs), 2);
        if ~any(valid_edges)
            skipped.bad_graph = skipped.bad_graph + 1;
            continue;
        end

        g_row = g_row + 1;
        GraphMeta(g_row,:) = [s, graph_idx, inter_flip_dist];
        GraphErr(g_row,:) = mean(errs(valid_edges,:), 1, 'omitnan');

        dn_vals = unique(d_near(valid_edges));
        for h = reshape(dn_vals, 1, [])
            take = valid_edges & d_near == h;
            n_row = n_row + 1;
            if n_row > size(NearestMeta, 1)
                [NearestMeta, NearestErr] = grow_record_arrays(NearestMeta, NearestErr, n_models);
            end
            NearestMeta(n_row,:) = [s, graph_idx, inter_flip_dist, h];
            NearestErr(n_row,:) = mean(errs(take,:), 1, 'omitnan');
        end

        pair_vals = unique([d_near(valid_edges), d_far(valid_edges)], 'rows');
        for pp = 1 : size(pair_vals, 1)
            take = valid_edges & d_near == pair_vals(pp,1) & d_far == pair_vals(pp,2);
            p_row = p_row + 1;
            if p_row > size(PairMeta, 1)
                [PairMeta, PairErr] = grow_record_arrays(PairMeta, PairErr, n_models);
            end
            PairMeta(p_row,:) = [s, graph_idx, inter_flip_dist, pair_vals(pp,1), pair_vals(pp,2)];
            PairErr(p_row,:) = mean(errs(take,:), 1, 'omitnan');
        end

        zone_masks = {
            valid_edges & xor(d1 <= near_radius, d2 <= near_radius)
            valid_edges & d1 <= near_radius & d2 <= near_radius
            valid_edges & d1 > near_radius & d2 > near_radius
            };
        for z = 1 : numel(zone_masks)
            if ~any(zone_masks{z})
                continue;
            end
            z_row = z_row + 1;
            ZoneMeta(z_row,:) = [s, graph_idx, inter_flip_dist, z];
            ZoneErr(z_row,:) = mean(errs(zone_masks{z},:), 1, 'omitnan');
        end
    end
end

GraphT = records_to_table(GraphMeta(1:g_row,:), GraphErr(1:g_row,:), ...
    {'seed','graph_idx','inter_flip_dist'}, model_names);
NearestT = records_to_table(NearestMeta(1:n_row,:), NearestErr(1:n_row,:), ...
    {'seed','graph_idx','inter_flip_dist','d_near'}, model_names);
PairT = records_to_table(PairMeta(1:p_row,:), PairErr(1:p_row,:), ...
    {'seed','graph_idx','inter_flip_dist','d_near','d_far'}, model_names);
ZoneT = records_to_table(ZoneMeta(1:z_row,:), ZoneErr(1:z_row,:), ...
    {'seed','graph_idx','inter_flip_dist','zone_id'}, model_names);

fprintf('[DCG_plot_Flip_two_interaction] extracted %d graphs, %d nearest-hop records, %d distance-pair records, %d zone records.\n', ...
    height(GraphT), height(NearestT), height(PairT), height(ZoneT));

end


function [GraphT, NearestT, ZoneT, skipped, subset_info] = extract_single_t1_records(I, task, split_name, trained_models, near_radius)
% extract_single_t1_records  Extract single t1 records records from analysis structures.
% Inputs: I, task, split_name, trained_models, near_radius
% Outputs: GraphT, NearestT, ZoneT, skipped, subset_info
%EXTRACT_SINGLE_T1_RECORDS
% H1: Build graph and nearest-distance records for the weighted single-T1 v1 task.
%
% PURPOSE
%   Produce the single-T1 reference tables needed for the reviewer-facing
%   comparison against Flip_two. The computation deliberately mirrors the
%   Flip_two extraction rules: PNA supplies topology/GT/root detection, trained
%   models supply only their prediction column, the baseline is col 3
%   (pre-flip preferred length), errors are absolute length errors, and
%   aggregation is left to summarize_wide_records.
%
% INPUTS
%   I              : weighted single-T1 analyses cache struct. For v1_W this
%                    contains multiple training-cohort sizes in the first
%                    dimension of I.<model>.<task>.vals / inds.
%   task           : task field name; this script uses 'lengths_to_lengths'.
%   split_name     : split field name; this script uses held-out 'test' only.
%   trained_models : cellstr of trained model fields to read from I. Baseline
%                    is appended internally from the PNA matrix.
%
% OUTPUTS
%   GraphT   : one row per valid single-T1 test graph. Vars: seed, graph_idx,
%              plus one absolute-error column per trained model and Baseline.
%   NearestT : one row per (graph, hop distance from the single T1 root). Vars:
%              seed, graph_idx, d_near, plus the same model error columns.
%   ZoneT    : up to two rows per graph: zone_id=1 for d<=near_radius and
%              zone_id=2 for d>near_radius, using the same threshold as the
%              two-T1 near/far zone analysis.
%   skipped  : struct counting graphs discarded because they did not have
%              exactly one root (.not_one_root) or because the graph/model
%              matrices were malformed or had no fully valid edge (.bad_graph).
%   subset_info : struct documenting the selected v1 cohort bin. For the
%              single-vs-two comparison this must be the 16-cohort weighted v1
%              bin (v1_2_16_W), matching the revision-scale training regime.
%
% ALGORITHM
%   1. Select the 16-cohort weighted v1 bin by metadata
%      (subset_siz == 16 and subset_idx == 2) and walk its seed/test indices.
%   2. Require exactly one root edge, defined by col 3 == 0.
%   3. Build the row-preserving vertex-line hop graph from endpoint columns 1:2
%      and compute each edge's hop distance to that root. This distance is
%      named d_near so the resulting table can be summarized and overlaid with
%      Flip_two's min(d1,d2) curves.
%   4. Compute per-edge abs(pred-GT) for every model and abs(col3-GT) for the
%      Baseline. Keep only edges with finite distance and finite errors for all
%      models.
%   5. Emit a whole-graph mean row plus one nearest-distance-bin row per
%      occupied hop value plus near/far rows using near_radius.
%
% DECISIONS / EDGE-CASES
%   * This is an unpaired distributional comparison. graph_idx is retained for
%     traceability, but the code does not assume that v1_W and Flip_two contain
%     the same underlying tissues.
%   * A previous version hard-coded subset_i = 1, which selected the 1-cohort
%     v1 bin (4 test graphs per seed). That made the overall single-vs-two
%     histogram insensitive to rebuilding caches and compared against the wrong
%     baseline distribution. The selector below deliberately errors if the
%     16-cohort bin is not present.
%   * The single-T1 root must be exactly one col-3-zero row. If a future cache
%     encodes the root differently, the skip counter will make that failure
%     obvious instead of silently producing a misleading comparison.

n_models = numel(trained_models) + 1;
model_names = [trained_models, {'Baseline'}];
subset_info = select_single_t1_reference_subset(I, task, split_name);
subset_i = subset_info.subset_i;
n_seeds = size(I.PNA.(task).vals, 2);
n_graphs_total = 0;
for s = 1 : n_seeds
    n_graphs_total = n_graphs_total + numel(I.PNA.(task).inds{subset_i,s}.(split_name));
end

GraphMeta = nan(n_graphs_total, 2);
GraphErr = nan(n_graphs_total, n_models);
NearestMeta = nan(max(n_graphs_total * 80, 1), 3);
NearestErr = nan(max(n_graphs_total * 80, 1), n_models);
ZoneMeta = nan(max(n_graphs_total * 2, 1), 3);
ZoneErr = nan(max(n_graphs_total * 2, 1), n_models);

g_row = 0;
n_row = 0;
z_row = 0;
skipped = struct('not_one_root', 0, 'bad_graph', 0);

for s = 1 : n_seeds
    test_indices = I.PNA.(task).inds{subset_i,s}.(split_name)(:);
    for ii = 1 : numel(test_indices)
        graph_idx = test_indices(ii);
        curr_G = I.PNA.(task).vals{subset_i,s}{graph_idx};
        if isempty(curr_G) || any(isnan(curr_G(1,:)))
            skipped.bad_graph = skipped.bad_graph + 1;
            continue;
        end

        root_edges = find(curr_G(:,3) == 0);
        if numel(root_edges) ~= 1
            skipped.not_one_root = skipped.not_one_root + 1;
            continue;
        end

        line_G = flip_two_line_graph_preserve_rows(curr_G(:,1:2));
        assert_root_rows_match_line_graph(line_G, curr_G(:,1:2), root_edges, ...
            sprintf('single-T1 seed %d graph %d', s, graph_idx));
        d_near = distances(line_G, root_edges)';

        gt = curr_G(:,end-1);
        errs = nan(size(curr_G,1), n_models);
        for m = 1 : numel(trained_models)
            if isfield(I, trained_models{m}) && isfield(I.(trained_models{m}), task)
                vals_m = I.(trained_models{m}).(task).vals{subset_i,s}{graph_idx};
                if ~isempty(vals_m) && size(vals_m,1) == size(curr_G,1)
                    errs(:,m) = abs(vals_m(:,end) - gt);
                end
            end
        end
        errs(:,end) = abs(curr_G(:,3) - gt);

        valid_edges = isfinite(d_near) & all(isfinite(errs), 2);
        if ~any(valid_edges)
            skipped.bad_graph = skipped.bad_graph + 1;
            continue;
        end

        g_row = g_row + 1;
        GraphMeta(g_row,:) = [s, graph_idx];
        GraphErr(g_row,:) = mean(errs(valid_edges,:), 1, 'omitnan');

        dn_vals = unique(d_near(valid_edges));
        for h = reshape(dn_vals, 1, [])
            take = valid_edges & d_near == h;
            n_row = n_row + 1;
            if n_row > size(NearestMeta, 1)
                [NearestMeta, NearestErr] = grow_record_arrays(NearestMeta, NearestErr, n_models);
            end
            NearestMeta(n_row,:) = [s, graph_idx, h];
            NearestErr(n_row,:) = mean(errs(take,:), 1, 'omitnan');
        end

        zone_masks = {
            valid_edges & d_near <= near_radius
            valid_edges & d_near > near_radius
            };
        for z = 1 : numel(zone_masks)
            if ~any(zone_masks{z})
                continue;
            end
            z_row = z_row + 1;
            ZoneMeta(z_row,:) = [s, graph_idx, z];
            ZoneErr(z_row,:) = mean(errs(zone_masks{z},:), 1, 'omitnan');
        end
    end
end

GraphT = records_to_table(GraphMeta(1:g_row,:), GraphErr(1:g_row,:), ...
    {'seed','graph_idx'}, model_names);
NearestT = records_to_table(NearestMeta(1:n_row,:), NearestErr(1:n_row,:), ...
    {'seed','graph_idx','d_near'}, model_names);
ZoneT = records_to_table(ZoneMeta(1:z_row,:), ZoneErr(1:z_row,:), ...
    {'seed','graph_idx','zone_id'}, model_names);

fprintf(['[DCG_plot_Flip_two_interaction] extracted single-T1 reference ', ...
    'subset_i=%d (subset_siz=%g, subset_idx=%g): %d graphs, %d nearest-hop records, %d zone records.\n'], ...
    subset_info.subset_i, subset_info.subset_siz, subset_info.subset_idx, height(GraphT), height(NearestT), height(ZoneT));

end


function subset_info = select_single_t1_reference_subset(I, task, split_name)
% select_single_t1_reference_subset  Select single t1 reference subset for the current analysis.
% Inputs: I, task, split_name
% Outputs: subset_info
%SELECT_SINGLE_T1_REFERENCE_SUBSET
% H1: Locate the weighted v1 16-cohort bin used as the single-T1 reference.
%
% PURPOSE
%   The v1_W analyses cache contains multiple training-cohort sizes. The
%   Flip_two benchmark was trained at the revision scale, which should be
%   compared against the v1_2_16_W reference bin, not the tiny 1-cohort v1 bin.
%   This helper selects that bin by analyzer metadata and fails loudly if it is
%   unavailable or empty.
%
% OUTPUT
%   subset_info.subset_i   : row index into I.<model>.<task>.vals/inds.
%   subset_info.subset_siz : cohort-size metadata, expected to be 16.
%   subset_info.subset_idx : cohort-repetition metadata, expected to be 2 for
%                            v1_2_16_W.
%   subset_info.n_test_graphs_by_seed : number of held-out graphs per seed.

target_subset_siz = 16;
target_subset_idx = 2;

if ~isfield(I, 'PNA') || ~isfield(I.PNA, task)
    error('DCG:missingSingleReference', ...
        'Single-T1 reference cache does not contain I.PNA.%s.', task);
end

T = I.PNA.(task);
if ~isfield(T, 'subset_siz') || ~isfield(T, 'subset_idx')
    error('DCG:missingSingleSubsetMetadata', ...
        'Single-T1 reference cache lacks subset_siz/subset_idx metadata; cannot select v1_2_16_W safely.');
end

subset_siz = T.subset_siz(:);
subset_idx = T.subset_idx(:);
candidates = find(subset_siz == target_subset_siz & subset_idx == target_subset_idx);
if isempty(candidates)
    error('DCG:missingSingle16CohortSubset', ...
        'Single-T1 reference cache has no subset_siz=%d, subset_idx=%d bin (expected v1_2_16_W). Available bins: %s', ...
        target_subset_siz, target_subset_idx, format_available_subset_bins(subset_siz, subset_idx));
end
if numel(candidates) > 1
    error('DCG:ambiguousSingle16CohortSubset', ...
        'Single-T1 reference cache has multiple subset_siz=%d, subset_idx=%d bins: %s', ...
        target_subset_siz, target_subset_idx, mat2str(candidates(:)'));
end

subset_i = candidates(1);
n_seeds = size(T.vals, 2);
n_test = nan(1, n_seeds);
for s = 1 : n_seeds
    if size(T.inds, 1) < subset_i || size(T.inds, 2) < s || isempty(T.inds{subset_i,s}) || ...
            ~isfield(T.inds{subset_i,s}, split_name)
        error('DCG:missingSingle16CohortSplit', ...
            'Selected single-T1 subset_i=%d is missing split "%s" for seed %d.', subset_i, split_name, s);
    end
    n_test(s) = numel(T.inds{subset_i,s}.(split_name));
end

if any(n_test == 0)
    error('DCG:emptySingle16CohortSplit', ...
        'Selected single-T1 subset_i=%d has an empty "%s" split for at least one seed: %s', ...
        subset_i, split_name, mat2str(n_test));
end

subset_info = struct();
subset_info.subset_i = subset_i;
subset_info.subset_siz = subset_siz(subset_i);
subset_info.subset_idx = subset_idx(subset_i);
subset_info.n_test_graphs_by_seed = n_test;

end


function txt = format_available_subset_bins(subset_siz, subset_idx)
% format_available_subset_bins  Implement format available subset bins for this MATLAB workflow.
% Inputs: subset_siz, subset_idx
% Outputs: txt
%FORMAT_AVAILABLE_SUBSET_BINS
% H1: Compact diagnostic string for available v1 cohort bins.

parts = cell(numel(subset_siz), 1);
for k = 1 : numel(subset_siz)
    parts{k} = sprintf('#%d:siz=%g,idx=%g', k, subset_siz(k), subset_idx(k));
end
txt = strjoin(parts, '; ');

end


function LineG = flip_two_line_graph_preserve_rows(C, varargin)
% flip_two_line_graph_preserve_rows  Build a line graph whose rows preserve source edge ordering.
% Inputs: C, varargin
% Outputs: LineG
%FLIP_TWO_LINE_GRAPH_PRESERVE_ROWS
% H1: Build a row-preserving historical vertex-line interface graph.
%
% PURPOSE
%   T1 roots are identified by rows in the original prediction matrix. This
%   helper preserves one line-graph node per original row, in the same order as
%   C, so distances(line_G,rootRows) really starts from the T1 interfaces.
%
% HOP DEFINITION
%   One hop means two interfaces meet at an inferred epithelial vertex. With no
%   explicit vertex incidence, the historical helper infers triple junctions
%   combinatorially: rows (i,j), (i,k), and (j,k) are mutually adjacent when all
%   three cell-cell interfaces exist. This reproduces the original 22-ish
%   distance scale while preserving prediction-row root indices.

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
% pair_key  Implement pair key for analysis/matlab/DCG_plot_Flip_two_interaction.m.
% Inputs: a, b
% Outputs: key
        if a > b
            tmp = a;
            a = b;
            b = tmp;
        end
        key = sprintf('%d_%d', a, b);
    end

    function rows = rows_for_pair(a, b)
% rows_for_pair  Implement rows for pair for analysis/matlab/DCG_plot_Flip_two_interaction.m.
% Inputs: a, b
% Outputs: rows
        key = pair_key(a, b);
        if isKey(pairRows, key)
            rows = pairRows(key);
        else
            rows = [];
        end
    end

    function connect_sets(rowsA, rowsB)
% connect_sets  Implement connect sets for analysis/matlab/DCG_plot_Flip_two_interaction.m.
% Inputs: rowsA, rowsB
% Outputs: none; performs side effects or updates the caller workflow.
        L(rowsA, rowsB) = 1;
        L(rowsB, rowsA) = 1;
    end
end


function assert_root_rows_match_line_graph(line_G, original_edges, root_rows, context)
% assert_root_rows_match_line_graph  Build a line graph whose rows preserve source edge ordering.
% Inputs: line_G, original_edges, root_rows, context
% Outputs: none; performs side effects or updates the caller workflow.
%ASSERT_ROOT_ROWS_MATCH_LINE_GRAPH
% H1: Fail loudly if line-graph node rows no longer match prediction rows.

if any(root_rows < 1) || any(root_rows > numnodes(line_G))
    error('DCG:FlipTwoRootOutOfRange', ...
        'Root row out of line-graph range in %s.', context);
end

expected = sort(original_edges(root_rows, :), 2);
actual = line_G.Nodes.Interface(root_rows, :);
if ~isequal(expected, actual)
    error('DCG:FlipTwoRootRowMismatch', ...
        ['Line-graph root rows do not match prediction rows in %s. ', ...
         'Distance analysis would start from the wrong T1 interface.'], context);
end
end


function T = combine_single_two_records(SingleT, FlipT, model_names, record_kind)
% combine_single_two_records  Implement combine single two records for this MATLAB workflow.
% Inputs: SingleT, FlipT, model_names, record_kind
% Outputs: T
%COMBINE_SINGLE_TWO_RECORDS
% H1: Align single-T1 and two-T1 records into one condition-coded table.
%
% PURPOSE
%   summarize_wide_records expects one table with common columns. The raw
%   single-T1 and Flip_two tables differ slightly (Flip_two carries
%   inter_flip_dist; single-T1 does not), so this helper drops condition-specific
%   metadata and adds a numeric condition_id:
%       1 = weighted single T1 (v1_W reference)
%       2 = two simultaneous T1s (Flip_two)
%
% INPUTS
%   SingleT, FlipT : graph-level or nearest-distance record tables.
%   model_names    : model columns to preserve.
%   record_kind    : 'graph' keeps seed/graph_idx; 'nearest' also keeps d_near.
%
% OUTPUTS
%   T : concatenated table ready for summarize_wide_records. No paired matching
%       of graph_idx is implied; condition_id only encodes the benchmark.

model_vars = matlab.lang.makeValidName(model_names);
switch record_kind
    case 'graph'
        keep_vars = [{'seed','graph_idx'}, model_vars];
        order_vars = [{'condition_id','seed','graph_idx'}, model_vars];
    case 'nearest'
        keep_vars = [{'seed','graph_idx','d_near'}, model_vars];
        order_vars = [{'condition_id','seed','graph_idx','d_near'}, model_vars];
    otherwise
        error('Unknown record_kind: %s', record_kind);
end

SingleOut = SingleT(:, keep_vars);
FlipOut = FlipT(:, keep_vars);
SingleOut.condition_id = ones(height(SingleOut), 1);
FlipOut.condition_id = 2 * ones(height(FlipOut), 1);
T = [SingleOut(:, order_vars); FlipOut(:, order_vars)];

end


function T = combine_single_two_zone_records(SingleZoneT, FlipZoneT, model_names)
% combine_single_two_zone_records  Implement combine single two zone records for this MATLAB workflow.
% Inputs: SingleZoneT, FlipZoneT, model_names
% Outputs: T
%COMBINE_SINGLE_TWO_ZONE_RECORDS
% H1: Build one zone table containing matched single-T1 and two-T1 zone classes.
%
% PURPOSE
%   The two-T1 zone plot has a "far from both flips" class, but the isolated
%   single-T1 reference needs its matched "far from one flip" class, using the
%   same near_radius threshold. This helper maps the native zone ids into a
%   single comparison axis:
%       1 = single: near one T1
%       2 = single: far from one T1
%       3 = two: near exactly one T1
%       4 = two: near both T1s
%       5 = two: far from both T1s

model_vars = matlab.lang.makeValidName(model_names);
keep_single = [{'seed','graph_idx','zone_id'}, model_vars];
keep_flip = [{'seed','graph_idx','zone_id'}, model_vars];

SingleOut = SingleZoneT(:, keep_single);
FlipOut = FlipZoneT(:, keep_flip);
SingleOut.condition_id = ones(height(SingleOut), 1);
FlipOut.condition_id = 2 * ones(height(FlipOut), 1);
SingleOut.comparison_zone_id = SingleOut.zone_id;
FlipOut.comparison_zone_id = FlipOut.zone_id + 2;

order_vars = [{'condition_id','comparison_zone_id','seed','graph_idx','zone_id'}, model_vars];
T = [SingleOut(:, order_vars); FlipOut(:, order_vars)];

end


function PairVsSingleSummary = add_single_reference_to_pair_summary(PairSummary, NearestComparisonSummary, model_names)
% add_single_reference_to_pair_summary  Implement add single reference to pair summary for this MATLAB workflow.
% Inputs: PairSummary, NearestComparisonSummary, model_names
% Outputs: PairVsSingleSummary
%ADD_SINGLE_REFERENCE_TO_PAIR_SUMMARY
% H1: Add heatmap-ready deltas from the same-model single-T1 d_near profile.
%
% PURPOSE
%   Pair heatmaps are intrinsically two-T1 because they need (d_near,d_far).
%   The meaningful single-T1 comparison is therefore: for each heatmap cell,
%   subtract the isolated single-T1 value at the same nearest-T1 distance and
%   same model. This asks whether the second T1 changes the error beyond what
%   nearest-distance alone predicts.

PairVsSingleSummary = PairSummary;
PairVsSingleSummary.raw_delta_vs_single = nan(height(PairSummary), 1);
PairVsSingleSummary.log2_delta_vs_single = nan(height(PairSummary), 1);
PairVsSingleSummary.log2_nmae_delta_vs_single = nan(height(PairSummary), 1);

SingleNearestSummary = NearestComparisonSummary(NearestComparisonSummary.condition_id == 1, :);
for r = 1 : height(PairVsSingleSummary)
    m = PairVsSingleSummary.model(r);
    d = PairVsSingleSummary.d_near(r);
    ref = SingleNearestSummary(SingleNearestSummary.model == m & SingleNearestSummary.d_near == d, :);
    if isempty(ref)
        continue;
    end
    PairVsSingleSummary.raw_delta_vs_single(r) = PairVsSingleSummary.raw_mean(r) - ref.raw_mean(1);
    PairVsSingleSummary.log2_delta_vs_single(r) = PairVsSingleSummary.log2_mean(r) - ref.log2_mean(1);
    PairVsSingleSummary.log2_nmae_delta_vs_single(r) = PairVsSingleSummary.log2_nmae_mean(r) - ref.log2_nmae_mean(1);
end

% Keep the usual trained-model order predictable even though this helper does
% not otherwise need model_names; passing it catches accidental caller drift.
missing = setdiff(string(model_names), unique(PairVsSingleSummary.model));
if ~isempty(missing)
    warning('DCG:missingPairVsSingleModel', ...
        'Pair-vs-single summary has no rows for model(s): %s', strjoin(missing, ', '));
end

end


function [meta_out, err_out] = grow_record_arrays(meta_in, err_in, n_models)
% grow_record_arrays  Implement grow record arrays for this MATLAB workflow.
% Inputs: meta_in, err_in, n_models
% Outputs: meta_out, err_out
%GROW_RECORD_ARRAYS
% H1: Double the row capacity of a meta/err preallocation pair with NaN pad.
%
% PURPOSE
%   Amortized-growth helper for the Nearest and Pair record buffers in
%   extract_flip_two_records, used when the pre-sized arrays fill up.
%
% INPUTS
%   meta_in  : current metadata matrix (rows = records, cols = key fields).
%   err_in   : current error matrix (rows = records, cols = models).
%   n_models : number of model columns to allocate for the err extension.
%
% OUTPUTS
%   meta_out : meta_in with an equal-size NaN block appended (rows doubled).
%   err_out  : err_in with size(err_in,1) new NaN rows appended (n_models cols).
%
% ALGORITHM
%   Vertically concatenate the input with a NaN block of matching width.
%
% DECISIONS / EDGE-CASES
%   * Doubling keeps the number of reallocations logarithmic in record count.
%   * New rows are NaN so unused tail rows are trivially identified/trimmed.

meta_out = [meta_in; nan(size(meta_in))];
err_out = [err_in; nan(size(err_in,1), n_models)];

end


function T = records_to_table(meta, err, meta_names, model_names)
% records_to_table  Implement records to table for this MATLAB workflow.
% Inputs: meta, err, meta_names, model_names
% Outputs: T
%RECORDS_TO_TABLE
% H1: Glue a metadata matrix and an error matrix into one named table.
%
% PURPOSE
%   Assemble the trimmed meta/err numeric blocks produced in
%   extract_flip_two_records into a single table with proper column names.
%
% INPUTS
%   meta        : numeric matrix of key columns (seed, graph_idx, distances...).
%   err         : numeric matrix of per-model error columns.
%   meta_names  : cellstr of variable names for the meta columns.
%   model_names : cellstr of model names; converted to valid MATLAB
%                 identifiers for the error columns.
%
% OUTPUTS
%   T : table whose columns are [meta_names, valid(model_names)].
%
% ALGORITHM
%   Horizontally concatenate [meta, err] and wrap with array2table, sanitizing
%   model names via matlab.lang.makeValidName.
%
% DECISIONS / EDGE-CASES
%   * makeValidName ensures names like a leading-digit or space-bearing model
%     become legal table variable identifiers (must match the names used by
%     summarize_wide_records when it indexes columns).

T = array2table([meta, err], 'VariableNames', [meta_names, matlab.lang.makeValidName(model_names)]);

end


function Summary = summarize_wide_records(T, group_vars, model_names)
% summarize_wide_records  Compute summarize wide records summary values.
% Inputs: T, group_vars, model_names
% Outputs: Summary
%SUMMARIZE_WIDE_RECORDS
% H1: Seed-ladder aggregate of a long record table into per-group summaries.
%
% PURPOSE
%   Collapse one of the long tables (Graph/Nearest/Pair/Zone) into a tidy
%   per-(group x model) summary, applying the project's two-stage seed ladder
%   and three metric families: raw MAE, log2 MAE, and log2 normalized-MAE
%   (nMAE = model relative to the same-bin Baseline).
%
% INPUTS
%   T           : a long record table (output of records_to_table). Must
%                 contain a 'seed' column, a 'Baseline' column, and one column
%                 per model name.
%   group_vars  : cellstr of column names to GROUP BY (the x-axis / facet
%                 keys), e.g. {'d_near'} or {'distance_group','d_near'}.
%   model_names : cellstr of model names (includes 'Baseline'); each yields its
%                 own set of summary rows.
%
% OUTPUTS
%   Summary : table with one row per (group, model). Carries the group key
%             columns plus:
%               model           : model name (string).
%               raw_mean/raw_sd : mean and SD of the per-seed mean raw error.
%               log2_mean/log2_sd      : same for per-seed log2(mean error)
%                                        (arithmetic MAE, displayed in log2).
%               log2_nmae_mean/_sd     : same for per-seed mean of
%                                        log2(model/Baseline) (geometric; 0 for Baseline).
%               n_seeds         : number of seeds with a finite raw mean.
%               n_records       : number of underlying records in the group.
%             Returns an empty table when T is empty.
%
% ALGORITHM
%   1. findgroups over group_vars -> group index G and key table keyTbl.
%   2. For each group and each model:
%        For each seed present in the group:
%          raw_seed  = mean(model error)                          (omitnan)
%          log_seed  = log2( mean(model error) ) over error>0 & finite (arithmetic)
%          nmae_seed = mean( log2(model error / Baseline error) ) over
%                      rows where both >0 & finite (geometric; 0 for Baseline)
%      Then collapse across seeds: report mean and SD (std,0) of each per-seed
%      vector, count finite seeds and total records, and append the row.
%
% MATH / DECISIONS / EDGE-CASES
%   * Two-stage ladder: average within seed first, then across seeds, so seeds
%     are weighted equally regardless of how many records each contributes; the
%     reported SD is the BETWEEN-SEED dispersion of those per-seed means.
%   * log2 metrics drop non-positive / non-finite values before taking the log.
%   * Baseline's nMAE is fixed to 0 (it is its own reference), so it plots as a
%     flat zero line on every log2-nMAE axis.
%   * LOG-PLACEMENT RULE (2026-06-01): MAE is ARITHMETIC, nMAE is GEOMETRIC, to
%     match DCG_plot_results. So log_seed = log2(mean(error)) (log
%     AFTER the within-seed mean) and nmae_seed = mean(log2(model/Baseline))
%     (log PER record, then averaged -- a geometric-mean ratio, as the paper
%     defines nMAE). The Baseline column is pinned to 0. (An interim 2026-06-01
%     edit had briefly made nmae_seed the log of a ratio of mean MAEs
%     (arithmetic); that was reverted to keep nMAE geometric.)
%   * findgroups preserves key columns in keyTbl, so the group keys flow
%     straight into each Summary row via row = keyTbl(gi,:).

if isempty(T) || height(T) == 0
    Summary = table();
    return;
end

[G, keyTbl] = findgroups(T(:, group_vars));
Summary = table();
model_vars = matlab.lang.makeValidName(model_names);

for gi = 1 : height(keyTbl)
    group_mask = G == gi;
    seeds = unique(T.seed(group_mask));
    for m = 1 : numel(model_names)
        raw_seed = nan(numel(seeds), 1);
        log_seed = nan(numel(seeds), 1);
        nmae_seed = nan(numel(seeds), 1);
        for si = 1 : numel(seeds)
            take = group_mask & T.seed == seeds(si);
            vals = T.(model_vars{m})(take);
            base = T.Baseline(take);
            vals = vals(:);
            base = base(:);
            raw_seed(si) = mean(vals, 'omitnan');
            % ARITHMETIC MAE in log2 form: log2 AFTER the within-seed mean.
            log_seed(si) = log2(mean(vals(vals > 0 & isfinite(vals)), 'omitnan'));
            if strcmp(model_names{m}, 'Baseline')
                nmae_seed(si) = 0;
            else
                % GEOMETRIC nMAE: log2 PER record, then averaged.
                ratio_ok = vals > 0 & base > 0 & isfinite(vals) & isfinite(base);
                nmae_seed(si) = mean(log2(vals(ratio_ok) ./ base(ratio_ok)), 'omitnan');
            end
        end

        row = keyTbl(gi,:);
        row.model = string(model_names{m});
        row.raw_mean = mean(raw_seed, 'omitnan');
        row.raw_sd = std(raw_seed, 0, 'omitnan');
        row.log2_mean = mean(log_seed, 'omitnan');
        row.log2_sd = std(log_seed, 0, 'omitnan');
        row.log2_nmae_mean = mean(nmae_seed, 'omitnan');
        row.log2_nmae_sd = std(nmae_seed, 0, 'omitnan');
        row.n_seeds = nnz(isfinite(raw_seed));
        row.n_records = nnz(group_mask);
        Summary = [Summary; row]; %#ok<AGROW>
    end
end

end


function edges = inter_flip_distance_group_edges(inter_flip_dist)
% inter_flip_distance_group_edges  Implement inter flip distance group edges for this MATLAB workflow.
% Inputs: inter_flip_dist
% Outputs: edges
%INTER_FLIP_DISTANCE_GROUP_EDGES
% H1: Tertile bin edges for per-graph inter-flip distance (close/middle/far).
%
% PURPOSE
%   Derive the cut points that split graphs into three inter-flip-distance
%   groups of (approximately) equal count, used by the close/middle/far views.
%
% INPUTS
%   inter_flip_dist : vector of per-graph inter-flip hop distances (one entry
%                     per graph; non-finite entries are ignored).
%
% OUTPUTS
%   edges : 1x4 vector [-Inf, q1, q2, +Inf] where q1,q2 are the 1/3 and 2/3
%           quantiles. The open ends guarantee every value lands in a bin.
%
% ALGORITHM
%   Compute the [1/3, 2/3] quantiles over the finite distances and pad with
%   -Inf / +Inf.
%
% MATH / DECISIONS / EDGE-CASES
%   * Tertiles (not fixed hop thresholds) keep the three groups balanced for
%     whatever distance distribution the dataset happens to have.
%   * Non-finite (unreachable) distances are excluded from the quantile.

q = quantile(inter_flip_dist(isfinite(inter_flip_dist)), [1/3, 2/3]);
edges = [-inf, q(1), q(2), inf];

end


function group_id = inter_flip_distance_group(inter_flip_dist, edges)
% inter_flip_distance_group  Implement inter flip distance group for this MATLAB workflow.
% Inputs: inter_flip_dist, edges
% Outputs: group_id
%INTER_FLIP_DISTANCE_GROUP
% H1: Assign each graph's inter-flip distance to a tertile group id 1/2/3.
%
% PURPOSE
%   Bucketize per-graph inter-flip distances into close(1)/middle(2)/far(3)
%   using the edges from inter_flip_distance_group_edges.
%
% INPUTS
%   inter_flip_dist : vector/array of per-record inter-flip hop distances.
%   edges           : 1x4 bin-edge vector [-Inf, q1, q2, +Inf].
%
% OUTPUTS
%   group_id : same-size array of ids: 1 if d<=edges(2); 2 if edges(2)<d<=
%              edges(3); 3 if d>edges(3); NaN where the distance is NaN.
%
% ALGORITHM
%   Three masked assignments into a NaN-initialized array (half-open bins,
%   lower edge exclusive / upper edge inclusive except the first which is <=).
%
% DECISIONS / EDGE-CASES
%   * NaN distances remain NaN (no group) because none of the masks match.
%   * Boundary values fall into the lower-or-equal bin, mirroring the cut-point
%     convention so the partition is exhaustive and non-overlapping.

group_id = nan(size(inter_flip_dist));
group_id(inter_flip_dist <= edges(2)) = 1;
group_id(inter_flip_dist > edges(2) & inter_flip_dist <= edges(3)) = 2;
group_id(inter_flip_dist > edges(3)) = 3;

end


function labels = distance_group_label_from_id(group_id)
% distance_group_label_from_id  Implement distance group label from id for this MATLAB workflow.
% Inputs: group_id
% Outputs: labels
%DISTANCE_GROUP_LABEL_FROM_ID
% H1: Map inter-flip distance group ids {1,2,3} to "close"/"middle"/"far".
%
% PURPOSE
%   Human-readable labels for the tertile groups, for axis titles and table
%   columns.
%
% INPUTS
%   group_id : array of group ids (1, 2, or 3; other values -> "").
%
% OUTPUTS
%   labels : same-size string array with "close"/"middle"/"far" (or "").
%
% ALGORITHM
%   Initialize an empty string array, then assign by id mask.
%
% DECISIONS / EDGE-CASES
%   * Ids outside {1,2,3} (e.g. NaN groups) keep the default empty string.

labels = strings(size(group_id));
labels(group_id == 1) = "close";
labels(group_id == 2) = "middle";
labels(group_id == 3) = "far";

end


function labels = zone_label_from_id(zone_id, near_radius)
% zone_label_from_id  Implement zone label from id for this MATLAB workflow.
% Inputs: zone_id, near_radius
% Outputs: labels
%ZONE_LABEL_FROM_ID
% H1: Map interaction-zone ids {1,2,3} to descriptive strings (with radius).
%
% PURPOSE
%   Human-readable labels for the three interaction zones, embedding the
%   near_radius threshold for clarity in figures/tables.
%
% INPUTS
%   zone_id     : array of zone ids (1=near exactly one, 2=near both,
%                 3=far from both; other values -> "").
%   near_radius : the hop threshold to display in each label.
%
% OUTPUTS
%   labels : same-size string array with the descriptive zone text (or "").
%
% ALGORITHM
%   Initialize an empty string array, then assign per id via sprintf.
%
% DECISIONS / EDGE-CASES
%   * Labels mirror the zone definitions in extract_flip_two_records:
%     1 = exactly one of d1,d2 within radius (xor); 2 = both within; 3 = both
%     beyond. Ids outside {1,2,3} keep the default empty string.

labels = strings(size(zone_id));
labels(zone_id == 1) = sprintf("near exactly one flip (one d<=%d)", near_radius);
labels(zone_id == 2) = sprintf("near both flips (d1,d2<=%d)", near_radius);
labels(zone_id == 3) = sprintf("far from both flips (d1,d2>%d)", near_radius);

end


function labels = comparison_zone_label_from_id(comparison_zone_id, near_radius)
% comparison_zone_label_from_id  Implement comparison zone label from id for this MATLAB workflow.
% Inputs: comparison_zone_id, near_radius
% Outputs: labels
%COMPARISON_ZONE_LABEL_FROM_ID
% H1: Label the five-zone single-vs-two comparison axis.
%
% PURPOSE
%   The native single-T1 and two-T1 zone ids are not directly comparable by
%   number, so combine_single_two_zone_records maps them to a five-category
%   axis. These labels keep the single and two-T1 meanings explicit.

labels = strings(size(comparison_zone_id));
labels(comparison_zone_id == 1) = sprintf("single: near T1 (d<=%d)", near_radius);
labels(comparison_zone_id == 2) = sprintf("single: far from T1 (d>%d)", near_radius);
labels(comparison_zone_id == 3) = sprintf("two: near exactly one (one d<=%d)", near_radius);
labels(comparison_zone_id == 4) = sprintf("two: near both (d1,d2<=%d)", near_radius);
labels(comparison_zone_id == 5) = sprintf("two: far from both (d1,d2>%d)", near_radius);

end


function labels = comparison_zone_short_label_from_id(comparison_zone_id)
% comparison_zone_short_label_from_id  Implement comparison zone short label from id for this MATLAB workflow.
% Inputs: comparison_zone_id
% Outputs: labels
%COMPARISON_ZONE_SHORT_LABEL_FROM_ID
% H1: Compact tick labels for the five-zone comparison figure.

labels = strings(size(comparison_zone_id));
labels(comparison_zone_id == 1) = "single near";
labels(comparison_zone_id == 2) = "single far";
labels(comparison_zone_id == 3) = "two near one";
labels(comparison_zone_id == 4) = "two near both";
labels(comparison_zone_id == 5) = "two far both";

end


function labels = condition_label_from_id(condition_id)
% condition_label_from_id  Implement condition label from id for this MATLAB workflow.
% Inputs: condition_id
% Outputs: labels
%CONDITION_LABEL_FROM_ID
% H1: Human-readable labels for benchmark condition ids.
%
% PURPOSE
%   Convert the numeric condition_id used for grouping and plotting into stable
%   text labels. Keeping the grouping variable numeric avoids table/csv
%   ambiguity, while this label column makes exported summaries readable.
%
% INPUTS
%   condition_id : numeric vector where 1=single T1 and 2=two T1s.
%
% OUTPUTS
%   labels : string vector, same shape as condition_id.

labels = strings(size(condition_id));
labels(condition_id == 1) = "single T1";
labels(condition_id == 2) = "two T1s";

end


function single_t1_analyses_filename = default_single_t1_analyses_filename(analyses_filename)
% default_single_t1_analyses_filename  Return the default single t1 analyses filename.
% Inputs: analyses_filename
% Outputs: single_t1_analyses_filename
%DEFAULT_SINGLE_T1_ANALYSES_FILENAME
% H1: Locate the weighted v1 single-T1 cache next to the Flip_two cache.
%
% PURPOSE
%   Keep the main function backwards compatible with its original three-argument
%   API while still loading the single-T1 reference needed for recommendations
%   1 and 2. The default is the weighted v1 cache from the same consolidated
%   analyzer snapshot as the Flip_two cache.
%
% INPUTS
%   analyses_filename : path to the Flip_two analyses cache.
%
% OUTPUTS
%   single_t1_analyses_filename : resolved path to the weighted v1 cache.
%
% DECISIONS / EDGE-CASES
%   * The first candidate is '<same folder>\v1_W - analyses data.mat'.
%   * Only the same cache folder is searched automatically. Pass an explicit
%     single_t1_analyses_filename to compare against a different cache.

cache_dir = fileparts(analyses_filename);
candidates = {fullfile(cache_dir, 'v1_W - analyses data.mat')};

single_t1_analyses_filename = candidates{1};
for i = 1 : numel(candidates)
    if isfile(candidates{i})
        single_t1_analyses_filename = candidates{i};
        return;
    end
end

end


function write_outputs_tables(figures_output_dir, GraphT, NearestT, PairT, ZoneT, GraphSummary, NearestSummary, PairSummary, ZoneSummary, NearestGroupSummary)
% write_outputs_tables  Write outputs tables to disk.
% Inputs: figures_output_dir, GraphT, NearestT, PairT, ZoneT, GraphSummary, NearestSummary, PairSummary, ZoneSummary, NearestGroupSummary
% Outputs: none; performs side effects or updates the caller workflow.
%WRITE_OUTPUTS_TABLES
% H1: Persist all record and summary tables to CSV (+ pair records to .mat).
%
% PURPOSE
%   Serialize the analysis products to disk for the manuscript supplement and
%   downstream reuse.
%
% INPUTS
%   figures_output_dir : destination directory.
%   GraphT, NearestT, PairT, ZoneT          : long record tables.
%   GraphSummary, NearestSummary, PairSummary,
%   ZoneSummary, NearestGroupSummary        : seed-ladder summary tables.
%
% OUTPUTS
%   (none returned) Writes the following files into figures_output_dir:
%     CSV record tables:
%       flip_two_graph_records.csv                         (GraphT)
%       flip_two_nearest_distance_graph_bin_records.csv    (NearestT)
%       flip_two_interaction_zone_graph_records.csv        (ZoneT)
%     MAT record table (v7.3, for the potentially large pair table):
%       flip_two_distance_pair_records.mat                 (PairT)
%     CSV summary tables:
%       flip_two_graph_mae_vs_inter_flip_distance_summary.csv   (GraphSummary)
%       flip_two_mae_vs_nearest_T1_distance_summary.csv         (NearestSummary)
%       flip_two_mae_by_two_T1_distances_summary.csv            (PairSummary)
%       flip_two_interaction_zone_summary.csv                   (ZoneSummary)
%       flip_two_nearest_T1_by_close_middle_far_summary.csv     (NearestGroupSummary)
%
% ALGORITHM
%   Sequential writetable() calls, plus one save() of PairT.
%
% DECISIONS / EDGE-CASES
%   * PairT is stored as .mat (not CSV) because the distance-pair records can
%     be the largest table; '-v7.3' supports large variables.

writetable(GraphT, fullfile(figures_output_dir, 'flip_two_graph_records.csv'));
writetable(NearestT, fullfile(figures_output_dir, 'flip_two_nearest_distance_graph_bin_records.csv'));
writetable(ZoneT, fullfile(figures_output_dir, 'flip_two_interaction_zone_graph_records.csv'));

save(fullfile(figures_output_dir, 'flip_two_distance_pair_records.mat'), 'PairT', '-v7.3');

writetable(GraphSummary, fullfile(figures_output_dir, 'flip_two_graph_mae_vs_inter_flip_distance_summary.csv'));
writetable(NearestSummary, fullfile(figures_output_dir, 'flip_two_mae_vs_nearest_T1_distance_summary.csv'));
writetable(PairSummary, fullfile(figures_output_dir, 'flip_two_mae_by_two_T1_distances_summary.csv'));
writetable(ZoneSummary, fullfile(figures_output_dir, 'flip_two_interaction_zone_summary.csv'));
writetable(NearestGroupSummary, fullfile(figures_output_dir, 'flip_two_nearest_T1_by_close_middle_far_summary.csv'));

end


function write_single_vs_two_outputs(figures_output_dir, SingleGraphT, SingleNearestT, SingleZoneT, ComparisonGraphT, ComparisonNearestT, ComparisonZoneT, OverallComparisonSummary, NearestComparisonSummary, ZoneComparisonSummary, PairVsSingleSummary)
% write_single_vs_two_outputs  Write single vs two outputs to disk.
% Inputs: figures_output_dir, SingleGraphT, SingleNearestT, SingleZoneT, ComparisonGraphT, ComparisonNearestT, ComparisonZoneT, OverallComparisonSummary, NearestComparisonSummary, ZoneComparisonSummary, PairVsSingleSummary
% Outputs: none; performs side effects or updates the caller workflow.
%WRITE_SINGLE_VS_TWO_OUTPUTS
% H1: Persist the recommendation-1/2 single-T1 vs two-T1 comparison tables.
%
% PURPOSE
%   The original Flip_two script wrote only two-source tables. These additional
%   CSVs make the new reviewer-facing comparisons auditable: the single-T1
%   record tables, the condition-combined record tables used for aggregation,
%   and the final seed-ladder summaries that feed the new plots.
%
% INPUTS
%   figures_output_dir        : destination directory.
%   SingleGraphT             : one row per valid weighted v1 single-T1 graph.
%   SingleNearestT           : one row per single-T1 graph x hop-distance bin.
%   SingleZoneT              : one row per single-T1 graph x near/far zone.
%   ComparisonGraphT         : SingleGraphT and Flip_two GraphT aligned with
%                              condition_id = 1/2.
%   ComparisonNearestT       : SingleNearestT and Flip_two NearestT aligned with
%                              condition_id = 1/2.
%   ComparisonZoneT          : SingleZoneT and Flip_two ZoneT aligned onto the
%                              five-zone single-vs-two comparison axis.
%   OverallComparisonSummary : summarize_wide_records(ComparisonGraphT,
%                              {'condition_id'}, model_names).
%   NearestComparisonSummary : summarize_wide_records(ComparisonNearestT,
%                              {'condition_id','d_near'}, model_names).
%   ZoneComparisonSummary    : summarize_wide_records(ComparisonZoneT,
%                              {'comparison_zone_id'}, model_names).
%   PairVsSingleSummary      : PairSummary plus delta-vs-single columns for
%                              same-model, same-d_near heatmap comparisons.
%
% OUTPUTS
%   (none returned) Writes CSV files into figures_output_dir.

writetable(SingleGraphT, fullfile(figures_output_dir, 'single_T1_graph_records.csv'));
writetable(SingleNearestT, fullfile(figures_output_dir, 'single_T1_nearest_distance_graph_bin_records.csv'));
writetable(SingleZoneT, fullfile(figures_output_dir, 'single_T1_zone_graph_records.csv'));
writetable(ComparisonGraphT, fullfile(figures_output_dir, 'single_vs_two_T1_graph_records.csv'));
writetable(ComparisonNearestT, fullfile(figures_output_dir, 'single_vs_two_T1_nearest_distance_graph_bin_records.csv'));
writetable(ComparisonZoneT, fullfile(figures_output_dir, 'single_vs_two_T1_zone_graph_records.csv'));
writetable(OverallComparisonSummary, fullfile(figures_output_dir, 'single_vs_two_T1_overall_summary.csv'));
writetable(NearestComparisonSummary, fullfile(figures_output_dir, 'single_vs_two_T1_nearest_distance_summary.csv'));
writetable(ZoneComparisonSummary, fullfile(figures_output_dir, 'single_vs_two_T1_zone_summary.csv'));
writetable(PairVsSingleSummary, fullfile(figures_output_dir, 'flip_two_mae_by_two_T1_distances_vs_single_T1_summary.csv'));

end


function write_assumptions(figures_output_dir, analyses_filename, single_t1_analyses_filename, split_name, near_radius, skipped, skipped_single, distance_group_edges, single_subset)
% write_assumptions  Write assumptions to disk.
% Inputs: figures_output_dir, analyses_filename, single_t1_analyses_filename, split_name, near_radius, skipped, skipped_single, distance_group_edges, single_subset
% Outputs: none; performs side effects or updates the caller workflow.
%WRITE_ASSUMPTIONS
% H1: Write a human-readable text log of every analysis decision and skip count.
%
% PURPOSE
%   Produce an auditable provenance file documenting the choices baked into the
%   run (source cache, split, model set, root definition, distance method,
%   thresholds, aggregation, normalization) plus how many graphs were skipped.
%
% INPUTS
%   figures_output_dir  : destination directory.
%   analyses_filename   : path to the Flip_two source cache (logged verbatim).
%   single_t1_analyses_filename : path to the weighted single-T1 reference
%                         cache used for the new comparison plots.
%   split_name          : split used (logged as "<split> only").
%   near_radius         : zone near-threshold (hops) to record.
%   skipped             : struct with .not_two_roots and .bad_graph counts.
%   skipped_single      : struct with .not_one_root and .bad_graph counts for
%                         the single-T1 reference extraction.
%   distance_group_edges: 1x4 tertile edges; the two finite cut points are
%                         logged as the close/middle/far boundaries.
%   single_subset       : selected v1 reference subset metadata.
%
% OUTPUTS
%   (none returned) Writes flip_two_analysis_assumptions.txt listing: source
%   cache, split, model set including PPGN, the col-3==0 root definition, the
%   vertex-line distance method
%   (matching DCG_analyze_results), d1/d2/d_near/d_far and inter-flip
%   distance definitions, the near-radius zone scheme, the tertile cut points,
%   the aggregation ladder, the normalization definition, and the two skip
%   counts.
%
% ALGORITHM
%   fopen for write; on failure return silently. Register an onCleanup to
%   fclose, then fprintf each assumption line.
%
% DECISIONS / EDGE-CASES
%   * onCleanup guarantees the file handle is closed even if an fprintf errors.
%   * CAVEAT: the line describing the normalized metric as
%     'log2(model graph-bin MAE / Baseline graph-bin MAE)' reads as a ratio of
%     means, but summarize_wide_records actually computes the MEAN of per-record
%     log2 ratios (a geometric-mean ratio). See summarize_wide_records for the
%     authoritative definition; the two differ in general.

fid = fopen(fullfile(figures_output_dir, 'flip_two_analysis_assumptions.txt'), 'w');
if fid < 0
    return;
end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, 'Flip_two two-source analysis assumptions\n');
fprintf(fid, 'Flip_two source cache: %s\n', analyses_filename);
fprintf(fid, 'Weighted single-T1 reference cache: %s\n', single_t1_analyses_filename);
fprintf(fid, 'Weighted single-T1 reference subset: subset_i=%d, subset_siz=%g, subset_idx=%g, test graphs by seed=%s.\n', ...
    single_subset.subset_i, single_subset.subset_siz, single_subset.subset_idx, mat2str(single_subset.n_test_graphs_by_seed));
fprintf(fid, 'Split: %s only.\n', split_name);
fprintf(fid, 'Models: PPGN, GraphSAGE, GAT, GIN, PNA, Baseline.\n');
fprintf(fid, 'Two T1 roots: rows with in_preferred_length == 0 in the weighted matrix.\n');
fprintf(fid, 'Single-T1 root: exactly one row with in_preferred_length == 0 in the weighted v1 reference matrix.\n');
fprintf(fid, 'Distances: row-preserving historical vertex-line shortest paths from the original prediction-row interfaces; one hop means two interfaces meet at an inferred epithelial vertex.\n');
fprintf(fid, 'Historical scale: 256-cell single-T1/Flip_two runs can reach roughly 22-23 hops under this definition; this was audited on 2026-06-06 and root rows were not shifted by sorting/uniquing.\n');
fprintf(fid, 'd1,d2: distances to the two T1 roots; d_near=min(d1,d2); d_far=max(d1,d2).\n');
fprintf(fid, 'Single-vs-two nearest-distance comparison: single-T1 distance is the same vertex-line distance to its only root and is stored as d_near for overlay with Flip_two min(d1,d2).\n');
fprintf(fid, 'Single-vs-two overall comparison: unpaired distributional benchmark comparison; graph_idx is retained for traceability but no same-tissue pairing/superposition is assumed.\n');
fprintf(fid, 'Inter-flip distance: minimal vertex-line hop distance between the two T1 root interfaces.\n');
fprintf(fid, 'Near radius for zone plots: <= %d hops. Zones are near exactly one flip, near both flips, and far from both flips; root labels are not treated as meaningful.\n', near_radius);
fprintf(fid, 'Single-T1 zone comparison: near one T1 is d<=%d and far from one T1 is d>%d, using the same vertex-line distance threshold as the two-T1 zones.\n', near_radius, near_radius);
fprintf(fid, 'Continuous single-vs-two comparison: plotted as a function of d_near; for single T1 this is distance to the only T1, and for Flip_two this is min(d1,d2).\n');
fprintf(fid, 'Pair heatmap single-T1 comparison: delta-vs-single cells subtract the same-model single-T1 summary at the same d_near; d_far has no single-T1 analogue.\n');
fprintf(fid, 'Close/middle/far split: graph-level inter-flip-distance tertiles with cut points %.6g and %.6g hops.\n', distance_group_edges(2), distance_group_edges(3));
fprintf(fid, 'Aggregation: edge errors averaged within graph/bin, graph-bin records averaged within seed, then seed means averaged. SD is across seed means.\n');
fprintf(fid, 'Normalized metric: log2(model graph-bin MAE / Baseline graph-bin MAE). Baseline is therefore exactly 0 when plotted as log2 nMAE.\n');
fprintf(fid, 'Skipped graphs with non-two-root structure: %d\n', skipped.not_two_roots);
fprintf(fid, 'Skipped malformed/empty graphs: %d\n', skipped.bad_graph);
fprintf(fid, 'Skipped single-T1 graphs with non-one-root structure: %d\n', skipped_single.not_one_root);
fprintf(fid, 'Skipped malformed/empty single-T1 graphs: %d\n', skipped_single.bad_graph);

end


function output_path = plot_single_vs_two_overall(OverallComparisonSummary, model_names, colors, figures_output_dir, save_png)
% plot_single_vs_two_overall  Render the plot single vs two overall panel or plotting primitive.
% Inputs: OverallComparisonSummary, model_names, colors, figures_output_dir, save_png
% Outputs: output_path
%PLOT_SINGLE_VS_TWO_OVERALL
% H1: Recommendation 1 plot: overall graph MAE/nMAE for single-T1 vs two-T1.
%
% PURPOSE
%   Directly test whether prediction accuracy degrades when the perturbation
%   contains two simultaneous T1 events rather than one isolated T1. Each panel
%   uses the same seed-ladder summary as the rest of this file; bars are
%   condition means and error bars are between-seed SD.
%
% INPUTS
%   OverallComparisonSummary : summarize_wide_records output grouped by
%                              condition_id, with condition_id=1 for v1_W
%                              single-T1 and condition_id=2 for Flip_two.
%   model_names              : cellstr of models / legend order.
%   colors                   : Nx3 RGB matrix aligned to model_names.
%   figures_output_dir       : output directory.
%   save_png                 : logical; export PNG if true.
%
% OUTPUTS
%   output_path : full path to the saved .fig file.

fig = figure('Position', [100 100 920 400], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', 'Single-T1 vs two-T1 | Overall graph error');
tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

plot_condition_bars(nexttile, OverallComparisonSummary, model_names, colors, ...
    'raw_mean', 'raw_sd', 'Mean graph MAE', 'Raw MAE');
plot_condition_bars(nexttile, OverallComparisonSummary, model_names, colors, ...
    'log2_nmae_mean', 'log2_nmae_sd', 'log2(nMAE)', 'Baseline-normalized');

output_path = fullfile(figures_output_dir, 'Single-vs-two T1 overall graph error.fig');
dcg_savefig_visible(fig, output_path);
maybe_export_png(fig, output_path, save_png);

end


function output_path = plot_single_vs_two_bar_summary_2x2(OverallComparisonSummary, ZoneComparisonSummary, model_names, colors, near_radius, figures_output_dir, save_png, use_log_mae_axis)
% plot_single_vs_two_bar_summary_2x2  Render the plot single vs two bar summary 2x2 panel or plotting primitive.
% Inputs: OverallComparisonSummary, ZoneComparisonSummary, model_names, colors, near_radius, figures_output_dir, save_png, use_log_mae_axis
% Outputs: output_path
%PLOT_SINGLE_VS_TWO_BAR_SUMMARY_2X2
% H1: Combine the overall and near/far bar summaries into one 2x2 figure.
%
% PURPOSE
%   Replaces the two separate two-panel bar figures with one compact panel:
%   overall MAE / overall nMAE on the top row and distance-zone MAE /
%   distance-zone nMAE on the bottom row.
%   With use_log_mae_axis=true, the MAE panels still plot raw MAE values, but
%   their y-axes are logarithmic. The nMAE panels are unchanged; these values
%   are already log2 normalized by summarize_wide_records, so their labels use
%   the paper-facing term nMAE instead of log(nMAE).

if nargin < 8 || isempty(use_log_mae_axis)
    use_log_mae_axis = false;
end

mae_mean_var = 'raw_mean';
mae_sd_var = 'raw_sd';
mae_ylabel_overall = 'Mean graph MAE';
mae_ylabel_zone = 'Mean graph-edge MAE';
if use_log_mae_axis
    mae_suffix = ' (MAE log scale)';
else
    mae_suffix = '';
end

fig = figure('Position', [80 80 1080 820], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', ['Single-T1 vs two-T1 | Bar summaries', mae_suffix]);
tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

plot_condition_bars(nexttile, OverallComparisonSummary, model_names, colors, ...
    mae_mean_var, mae_sd_var, mae_ylabel_overall, 'Overall graph error', use_log_mae_axis);
plot_condition_bars(nexttile, OverallComparisonSummary, model_names, colors, ...
    'log2_nmae_mean', 'log2_nmae_sd', 'nMAE', 'Overall baseline-normalized');
plot_comparison_zone_bars(nexttile, ZoneComparisonSummary, model_names, colors, ...
    mae_mean_var, mae_sd_var, mae_ylabel_zone, sprintf('Distance zones, near <= %d hops', near_radius), use_log_mae_axis);
plot_comparison_zone_bars(nexttile, ZoneComparisonSummary, model_names, colors, ...
    'log2_nmae_mean', 'log2_nmae_sd', 'nMAE to same-zone baseline', sprintf('Distance zones normalized, near <= %d hops', near_radius));

output_path = fullfile(figures_output_dir, ['Single-T1 vs two-T1 bar summaries 2x2', mae_suffix, '.fig']);
dcg_savefig_visible(fig, output_path);
maybe_export_png(fig, output_path, save_png);

end


function plot_condition_bars(ax, Summary, model_names, colors, mean_var, sd_var, y_label, plot_title, log_y_axis)
% plot_condition_bars  Render the plot condition bars panel or plotting primitive.
% Inputs: ax, Summary, model_names, colors, mean_var, sd_var, y_label, plot_title, log_y_axis
% Outputs: none; performs side effects or updates the caller workflow.
%PLOT_CONDITION_BARS
% H1: Draw grouped condition bars colored by model.
%
% PURPOSE
%   Bar-plot primitive for plot_single_vs_two_overall. The x axis is the
%   benchmark condition and color encodes the architecture/model, matching the
%   rest of the Flip_two palette.
%
% INPUTS
%   ax               : target axes.
%   Summary          : table with condition_id, model, mean_var, sd_var.
%   model_names      : model order and legend labels.
%   colors           : model RGB colors.
%   mean_var, sd_var : metric mean and SD columns to plot.
%   y_label          : y-axis label.
%   plot_title       : axes title.
%   log_y_axis       : optional logical; when true, use a logarithmic y-axis
%                      while leaving the plotted values in their original units.

if nargin < 9 || isempty(log_y_axis)
    log_y_axis = false;
end

condition_ids = [1 2];
condition_labels = {'single T1','two T1s'};
vals = nan(numel(condition_ids), numel(model_names));
sds = nan(numel(condition_ids), numel(model_names));
for c = 1 : numel(condition_ids)
    for m = 1 : numel(model_names)
        take = Summary.condition_id == condition_ids(c) & Summary.model == string(model_names{m});
        if any(take)
            vals(c,m) = Summary.(mean_var)(find(take, 1));
            sds(c,m) = Summary.(sd_var)(find(take, 1));
        end
    end
end

hold(ax, 'on');
bar_handles = bar(ax, vals, 'grouped', 'EdgeColor', 'none');
for m = 1 : numel(model_names)
    bar_handles(m).FaceColor = colors(m,:);
    errorbar(ax, bar_handles(m).XEndPoints, vals(:,m), sds(:,m), ...
        'k.', 'LineWidth', 0.75, 'HandleVisibility', 'off');
end
hold(ax, 'off');
set(ax, 'XTick', 1:numel(condition_ids), 'XTickLabel', condition_labels, ...
    'Box', 'off', 'TickDir', 'out');
xlabel(ax, 'Benchmark condition');
ylabel(ax, y_label);
title(ax, plot_title);
legend(ax, bar_handles, model_names, 'Location', 'best');
axis_tight_with_padding(ax, strcmp(mean_var, 'raw_mean') && ~log_y_axis);
if log_y_axis
    set_log_axis_from_positive_data(ax);
end

end


function output_path = plot_single_vs_two_nearest_distance(NearestComparisonSummary, model_names, colors, mean_var, sd_var, y_label, file_stem, figures_output_dir, save_png)
% plot_single_vs_two_nearest_distance  Render the plot single vs two nearest distance panel or plotting primitive.
% Inputs: NearestComparisonSummary, model_names, colors, mean_var, sd_var, y_label, file_stem, figures_output_dir, save_png
% Outputs: output_path
%PLOT_SINGLE_VS_TWO_NEAREST_DISTANCE
% H1: Recommendation 2 plot: error vs nearest-T1 distance, single vs two.
%
% PURPOSE
%   Overlay the spatial error profile around an isolated single T1 with the
%   Flip_two profile indexed by h_min(e)=min(h1(e),h2(e)). Same color means same
%   model; solid lines are single-T1 and dashed lines are two-T1. This asks
%   whether the near-field/far-field structure seen for isolated T1s survives
%   when a second T1 is present elsewhere in the graph.
%
% INPUTS
%   NearestComparisonSummary : summarize_wide_records output grouped by
%                              condition_id and d_near.
%   model_names              : cellstr of models / legend order.
%   colors                   : Nx3 RGB matrix aligned to model_names.
%   mean_var, sd_var         : metric mean and SD columns to plot.
%   y_label                  : y-axis label.
%   file_stem                : figure title and output file stem.
%   figures_output_dir       : output directory.
%   save_png                 : logical; export PNG if true.
%
% OUTPUTS
%   output_path : full path to the saved .fig file.

fig = figure('Position', [100 100 880 460], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', file_stem);
ax = axes(fig);
hold(ax, 'on');

condition_ids = [1 2];
condition_labels = {'single T1','two T1s'};
line_styles = {'-','--'};
line_handles = gobjects(1, numel(model_names) * numel(condition_ids));
legend_labels = cell(1, numel(line_handles));
hh = 0;
for c = 1 : numel(condition_ids)
    for m = 1 : numel(model_names)
        take = NearestComparisonSummary.condition_id == condition_ids(c) & ...
            NearestComparisonSummary.model == string(model_names{m});
        x = NearestComparisonSummary.d_near(take);
        y = NearestComparisonSummary.(mean_var)(take);
        e = NearestComparisonSummary.(sd_var)(take);
        [x, order] = sort(x);
        y = y(order);
        e = e(order);
        hh = hh + 1;
        line_handles(hh) = plot_shaded_line(ax, x, y, e, colors(m,:), line_styles{c});
        legend_labels{hh} = sprintf('%s %s', model_names{m}, condition_labels{c});
    end
end

hold(ax, 'off');
set(ax, 'Box', 'off', 'TickDir', 'out');
xlabel(ax, 'Nearest T1 distance (hops)');
ylabel(ax, y_label);
title(ax, regexprep(file_stem, '_', ' '));
legend(ax, line_handles, legend_labels, 'Location', 'eastoutside');
axis(ax, 'square');
axis_tight_with_padding(ax, strcmp(mean_var, 'raw_mean'));

output_path = fullfile(figures_output_dir, [file_stem, '.fig']);
dcg_savefig_visible(fig, output_path);
maybe_export_png(fig, output_path, save_png);

end


function output_path = plot_nearest_distance_single_two_2x3(NearestSummary, NearestComparisonSummary, model_names, colors, figures_output_dir, save_png)
% plot_nearest_distance_single_two_2x3  Render the plot nearest distance single two 2x3 panel or plotting primitive.
% Inputs: NearestSummary, NearestComparisonSummary, model_names, colors, figures_output_dir, save_png
% Outputs: output_path
%PLOT_NEAREST_DISTANCE_SINGLE_TWO_2X3
% H1: Compare nearest-distance profiles as separate single-T1 and two-T1 rows.
%
% PURPOSE
%   Replaces the three separate Flip_two nearest-distance figures and the
%   overlaid single-vs-two figures. Top row is Flip_two. Bottom row is the
%   16-cohort single-T1 reference. Columns are raw MAE, log2 MAE, and log2 nMAE.

fig = figure('Position', [60 60 1320 760], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', 'Single-T1 and two-T1 nearest-distance profiles | 2x3');
tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

SingleNearestSummary = NearestComparisonSummary(NearestComparisonSummary.condition_id == 1, :);

metric = {
    'raw_mean',       'raw_sd',       'Mean graph-edge MAE',      'raw MAE'
    'log2_mean',      'log2_sd',      'log2(mean graph-edge MAE)', 'log2 MAE'
    'log2_nmae_mean', 'log2_nmae_sd', 'log2(nMAE)',                'log2 nMAE'
    };

ax_top = gobjects(1, 3);
ax_bottom = gobjects(1, 3);
for k = 1 : 3
    ax_top(k) = nexttile(k);
    plot_summary_lines(ax_top(k), NearestSummary, 'd_near', model_names, colors, ...
        metric{k,1}, metric{k,2}, 'Nearest T1 distance (hops)', metric{k,3}, ...
        sprintf('Two T1s: %s', metric{k,4}));

    ax_bottom(k) = nexttile(k + 3);
    plot_summary_lines(ax_bottom(k), SingleNearestSummary, 'd_near', model_names, colors, ...
        metric{k,1}, metric{k,2}, 'Distance from T1 (hops)', metric{k,3}, ...
        sprintf('Single T1: %s', metric{k,4}));

    linkaxes([ax_top(k), ax_bottom(k)], 'y');
end

output_path = fullfile(figures_output_dir, 'Single-T1 and two-T1 nearest-distance profiles 2x3.fig');
dcg_savefig_visible(fig, output_path);
maybe_export_png(fig, output_path, save_png);

end


function output_path = plot_single_vs_two_nearest_distance_panels(NearestComparisonSummary, model_names, colors, mean_var, sd_var, y_label, file_stem, figures_output_dir, save_png)
% plot_single_vs_two_nearest_distance_panels  Render the plot single vs two nearest distance panels panel or plotting primitive.
% Inputs: NearestComparisonSummary, model_names, colors, mean_var, sd_var, y_label, file_stem, figures_output_dir, save_png
% Outputs: output_path
%PLOT_SINGLE_VS_TWO_NEAREST_DISTANCE_PANELS
% H1: Side-by-side nearest-distance profiles for single-T1 and two-T1 data.
%
% PURPOSE
%   The overlaid single-vs-two curve is useful for direct comparison but can be
%   visually busy. This companion view keeps the same metric and hop-distance
%   definition while separating the conditions into two panels:
%       left  = isolated single T1, d_near is distance to that T1
%       right = Flip_two, d_near is min(d1,d2)

fig = figure('Position', [100 100 900 430], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', file_stem);
tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

condition_ids = [1 2];
condition_titles = {'single T1', 'two T1s'};
axes_out = gobjects(1, numel(condition_ids));
for c = 1 : numel(condition_ids)
    ax = nexttile;
    axes_out(c) = ax;
    S = NearestComparisonSummary(NearestComparisonSummary.condition_id == condition_ids(c), :);
    plot_summary_lines(ax, S, 'd_near', model_names, colors, mean_var, sd_var, ...
        'Nearest T1 distance (hops)', y_label, condition_titles{c});
end
linkaxes(axes_out, 'y');

output_path = fullfile(figures_output_dir, [file_stem, '.fig']);
dcg_savefig_visible(fig, output_path);
maybe_export_png(fig, output_path, save_png);

end


function output_path = plot_single_vs_two_zones(ZoneComparisonSummary, model_names, colors, near_radius, figures_output_dir, save_png)
% plot_single_vs_two_zones  Render the plot single vs two zones panel or plotting primitive.
% Inputs: ZoneComparisonSummary, model_names, colors, near_radius, figures_output_dir, save_png
% Outputs: output_path
%PLOT_SINGLE_VS_TWO_ZONES
% H1: Compare single-T1 near/far zones with two-T1 interaction zones.
%
% PURPOSE
%   Adds the missing matched single-T1 "far from one T1" control. The same
%   near_radius threshold defines every category, so the two far-field bars are
%   directly comparable: single d>r vs two d1>r AND d2>r.

fig = figure('Position', [100 100 1040 460], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', 'Single-T1 vs two-T1 | Distance zones');
tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

plot_comparison_zone_bars(nexttile, ZoneComparisonSummary, model_names, colors, 'raw_mean', 'raw_sd', ...
    'Mean graph-edge MAE', sprintf('Distance zones, near <= %d hops', near_radius));
plot_comparison_zone_bars(nexttile, ZoneComparisonSummary, model_names, colors, 'log2_nmae_mean', 'log2_nmae_sd', ...
    'log2(nMAE to same-zone baseline)', sprintf('Baseline-normalized zones, near <= %d hops', near_radius));

output_path = fullfile(figures_output_dir, 'Single-T1 vs two-T1 distance zones.fig');
dcg_savefig_visible(fig, output_path);
maybe_export_png(fig, output_path, save_png);

end


function output_path = plot_graph_mae_vs_inter_flip(GraphSummary, model_names, colors, figures_output_dir, save_png, OverallComparisonSummary)
% plot_graph_mae_vs_inter_flip  Render the plot graph mae vs inter flip panel or plotting primitive.
% Inputs: GraphSummary, model_names, colors, figures_output_dir, save_png, OverallComparisonSummary
% Outputs: output_path
%PLOT_GRAPH_MAE_VS_INTER_FLIP
% H1: Two-panel line plot of whole-graph MAE vs inter-flip distance.
%
% PURPOSE
%   Show how each model's per-graph mean error depends on the hop distance
%   between the two T1 flips: left panel raw MAE, right panel baseline-
%   normalized nMAE (stored internally as log2_nmae_mean).
%
% INPUTS
%   GraphSummary       : summary table from summarize_wide_records grouped by
%                        inter_flip_dist.
%   model_names        : cellstr of models (incl. Baseline) and legend order.
%   colors             : Nx3 RGB matrix aligned to model_names.
%   figures_output_dir : output directory for the .fig (and optional .png).
%   save_png           : logical; export PNG if true.
%
% OUTPUTS
%   output_path : full path to the saved .fig file.
%
% ALGORITHM
%   Create a 2x3 tiledlayout. The top row draws linear raw MAE,
%   log2(mean MAE), and log2_nmae_mean/_sd against inter_flip_dist with dotted
%   single-T1 overall references. The bottom row repeats the exact same
%   Flip_two curves without those dotted references; the Baseline remains as
%   the solid black model line.
%
% DECISIONS / EDGE-CASES
%   * x variable is 'inter_flip_dist'; shaded bands are +/-SD across seed means.

fig = figure('Position', [100 100 1260 760], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', 'Flip_two | Graph MAE vs inter-flip distance');
tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile;
plot_summary_lines(ax1, GraphSummary, 'inter_flip_dist', model_names, colors, 'raw_mean', 'raw_sd', ...
    'Inter-flip distance (hops)', 'Mean graph MAE', 'Raw MAE (dotted = single-T1 overall)');
overlay_single_overall_reference(ax1, OverallComparisonSummary, model_names, colors, 'raw_mean', 'raw_sd');

ax2 = nexttile;
plot_summary_lines(ax2, GraphSummary, 'inter_flip_dist', model_names, colors, 'log2_mean', 'log2_sd', ...
    'Inter-flip distance (hops)', 'log2(mean graph MAE)', 'log2(MAE) (dotted = single-T1 overall)');
overlay_single_overall_reference(ax2, OverallComparisonSummary, model_names, colors, 'log2_mean', 'log2_sd');

ax3 = nexttile;
plot_summary_lines(ax3, GraphSummary, 'inter_flip_dist', model_names, colors, 'log2_nmae_mean', 'log2_nmae_sd', ...
    'Inter-flip distance (hops)', 'nMAE', 'Baseline-normalized (dotted = single-T1 overall)');
overlay_single_overall_reference(ax3, OverallComparisonSummary, model_names, colors, 'log2_nmae_mean', 'log2_nmae_sd');

ax4 = nexttile;
plot_summary_lines(ax4, GraphSummary, 'inter_flip_dist', model_names, colors, 'raw_mean', 'raw_sd', ...
    'Inter-flip distance (hops)', 'Mean graph MAE', 'Raw MAE');

ax5 = nexttile;
plot_summary_lines(ax5, GraphSummary, 'inter_flip_dist', model_names, colors, 'log2_mean', 'log2_sd', ...
    'Inter-flip distance (hops)', 'log2(mean graph MAE)', 'log2(MAE)');

ax6 = nexttile;
plot_summary_lines(ax6, GraphSummary, 'inter_flip_dist', model_names, colors, 'log2_nmae_mean', 'log2_nmae_sd', ...
    'Inter-flip distance (hops)', 'nMAE', 'Baseline-normalized');

linkaxes([ax1 ax4], 'xy');
linkaxes([ax2 ax5], 'xy');
linkaxes([ax3 ax6], 'xy');

output_path = fullfile(figures_output_dir, 'Flip_two graph MAE vs inter-flip distance.fig');
dcg_savefig_visible(fig, output_path);
maybe_export_png(fig, output_path, save_png);

end


function output_path = plot_nearest_distance(Summary, model_names, colors, mean_var, sd_var, y_label, file_stem, figures_output_dir, save_png)
% plot_nearest_distance  Render the plot nearest distance panel or plotting primitive.
% Inputs: Summary, model_names, colors, mean_var, sd_var, y_label, file_stem, figures_output_dir, save_png
% Outputs: output_path
%PLOT_NEAREST_DISTANCE
% H1: Single-axis line plot of MAE vs nearest-T1 distance for a chosen metric.
%
% PURPOSE
%   Generic nearest-distance curve renderer, called once per metric family
%   (raw MAE, log2 MAE, log2 nMAE) from the main function.
%
% INPUTS
%   Summary            : summarize_wide_records output grouped by d_near.
%   model_names        : cellstr of models / legend order.
%   colors             : Nx3 RGB matrix aligned to model_names.
%   mean_var, sd_var   : column names selecting which metric mean/SD to plot
%                        (e.g. 'raw_mean'/'raw_sd', 'log2_mean'/'log2_sd',
%                        'log2_nmae_mean'/'log2_nmae_sd').
%   y_label            : y-axis label string.
%   file_stem          : base name for the figure window, title, and file.
%   figures_output_dir : output directory.
%   save_png           : logical; export PNG if true.
%
% OUTPUTS
%   output_path : full path to the saved .fig file ([file_stem '.fig']).
%
% ALGORITHM
%   One axes; plot_summary_lines against 'd_near' with the chosen metric;
%   savefig and optionally PNG.
%
% DECISIONS / EDGE-CASES
%   * The plot title is file_stem with underscores replaced by spaces.

fig = figure('Position', [100 100 450 390], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', ['Flip_two | ', file_stem]);
ax = axes(fig);
plot_summary_lines(ax, Summary, 'd_near', model_names, colors, mean_var, sd_var, ...
    'Nearest T1 distance (hops)', y_label, regexprep(file_stem, '_', ' '));

output_path = fullfile(figures_output_dir, [file_stem, '.fig']);
dcg_savefig_visible(fig, output_path);
maybe_export_png(fig, output_path, save_png);

end


function output_path = plot_interaction_zones(ZoneSummary, model_names, colors, near_radius, figures_output_dir, save_png)
% plot_interaction_zones  Render the plot interaction zones panel or plotting primitive.
% Inputs: ZoneSummary, model_names, colors, near_radius, figures_output_dir, save_png
% Outputs: output_path
%PLOT_INTERACTION_ZONES
% H1: Two-panel grouped bar chart of MAE across the three interaction zones.
%
% PURPOSE
%   Compare model error in the near-one / near-both / far-from-both zones:
%   left panel raw MAE, right panel log2(nMAE) relative to the per-zone
%   baseline.
%
% INPUTS
%   ZoneSummary        : summarize_wide_records output grouped by zone_id.
%   model_names        : cellstr of models / legend order.
%   colors             : Nx3 RGB matrix aligned to model_names.
%   near_radius        : hop threshold, shown in the panel titles.
%   figures_output_dir : output directory.
%   save_png           : logical; export PNG if true.
%
% OUTPUTS
%   output_path : full path to the saved .fig file.
%
% ALGORITHM
%   1x2 tiledlayout; plot_zone_bars for raw then log2-nMAE metrics; savefig and
%   optionally PNG.
%
% DECISIONS / EDGE-CASES
%   * Titles embed near_radius so the figure is self-documenting.

fig = figure('Position', [100 100 900 430], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', 'Flip_two | Interaction zones');
tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

plot_zone_bars(nexttile, ZoneSummary, model_names, colors, 'raw_mean', 'raw_sd', ...
    'Mean graph-edge MAE', sprintf('Mean graph-edge MAE, near <= %d hops', near_radius));
plot_zone_bars(nexttile, ZoneSummary, model_names, colors, 'log2_nmae_mean', 'log2_nmae_sd', ...
    'log2(nMAE to graph-zone baseline)', sprintf('log2(nMAE), near <= %d hops', near_radius));

output_path = fullfile(figures_output_dir, 'Flip_two interaction zones.fig');
dcg_savefig_visible(fig, output_path);
maybe_export_png(fig, output_path, save_png);

end


function output_path = plot_pair_heatmaps(PairSummary, model_names, metric_var, metric_label, file_stem, figures_output_dir, save_png, center_zero)
% plot_pair_heatmaps  Render the plot pair heatmaps panel or plotting primitive.
% Inputs: PairSummary, model_names, metric_var, metric_label, file_stem, figures_output_dir, save_png, center_zero
% Outputs: output_path
%PLOT_PAIR_HEATMAPS
% H1: 2x2 grid of (d_near x d_far) MAE heatmaps, one per trained model.
%
% PURPOSE
%   Visualize how error jointly depends on BOTH flip distances: each model gets
%   a heatmap with d_far on x, d_near on y, sharing a common colour scale.
%
% INPUTS
%   PairSummary        : summarize_wide_records output grouped by (d_near,d_far).
%   model_names        : cellstr of TRAINED models (the four panels; Baseline
%                        is not tiled here).
%   metric_var         : column to map to colour (e.g. 'raw_mean' or
%                        'log2_nmae_mean').
%   metric_label       : colorbar label string.
%   file_stem          : base name for window/title/file.
%   figures_output_dir : output directory.
%   save_png           : logical; export PNG if true.
%   center_zero        : optional logical; when true, use symmetric color
%                        limits around zero for delta-vs-single heatmaps.
%
% OUTPUTS
%   output_path : full path to the saved .fig file ([file_stem '.fig']).
%
% ALGORITHM
%   1. Compute shared colour limits = [min,max] of metric across all rows.
%   2. 2x2 tiledlayout; for each model call plot_one_heatmap with those limits,
%      label axes (d_far / d_near), square aspect.
%   3. Add a single shared colorbar (tiled east) with metric_label; savefig and
%      optionally PNG.
%
% DECISIONS / EDGE-CASES
%   * If colour limits are non-finite or degenerate (min==max), they fall back
%     to [0,1] so imagesc/clim remain valid.
%   * A shared scale makes panels directly comparable across models.
%   * Heatmap colors are a signed log10 magnitude of the plotted statistic.
%     Negative values are transformed by taking their absolute magnitude,
%     logging that magnitude, and restoring the negative sign; zero stays zero.
%   * Delta-vs-single heatmaps use symmetric color limits so positive and
%     negative deviations are visually balanced.

if nargin < 8 || isempty(center_zero)
    center_zero = false;
end

plot_metric_var = [metric_var, '__signed_log10'];
PairSummary.(plot_metric_var) = signed_log10_magnitude(PairSummary.(metric_var));
metric_label = sprintf('signed log10 magnitude of %s', metric_label);

ncols = ceil(sqrt(numel(model_names)));         % dynamic grid (was hard-coded 2x2;
nrows = ceil(numel(model_names) / ncols);       % broke when PPGN made 5 models)
fig = figure('Position', [100 100 435*ncols 380*nrows], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', ['Flip_two | ', file_stem]);
tiledlayout(fig, nrows, ncols, 'TileSpacing', 'compact', 'Padding', 'compact');

all_vals = PairSummary.(plot_metric_var);
color_limits = [min(all_vals, [], 'omitnan'), max(all_vals, [], 'omitnan')];
if center_zero || (color_limits(1) < 0 && color_limits(2) > 0)
    max_abs = max(abs(all_vals), [], 'omitnan');
    color_limits = [-max_abs, max_abs];
end
if ~all(isfinite(color_limits)) || color_limits(1) == color_limits(2)
    color_limits = [0, 1];
end

for m = 1 : numel(model_names)
    ax = nexttile;
    plot_one_heatmap(ax, PairSummary, model_names{m}, plot_metric_var, color_limits);
    title(ax, model_names{m});
    xlabel(ax, 'Farther T1 distance, d_{far}');
    ylabel(ax, 'Nearest T1 distance, d_{near}');
    axis(ax, 'square');
end
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = metric_label;

output_path = fullfile(figures_output_dir, [file_stem, '.fig']);
dcg_savefig_visible(fig, output_path);
maybe_export_png(fig, output_path, save_png);

end


function y = signed_log10_magnitude(x)
% signed_log10_magnitude  Implement signed log10 magnitude for this MATLAB workflow.
% Inputs: x
% Outputs: y
%SIGNED_LOG10_MAGNITUDE
% H1: Signed log transform for heatmap color values.
%
% PURPOSE
%   Compress heatmap dynamic range without throwing away the sign of delta or
%   normalized metrics. For a nonzero value x, color is:
%       sign(x) * log10(1 + abs(x) / min_abs_nonzero)
%   where min_abs_nonzero is computed within the plotted figure. This is the
%   stable finite version of "take abs, log it, and restore the sign".

y = nan(size(x));
finite = isfinite(x);
is_zero = finite & x == 0;
nonzero = finite & x ~= 0;
y(is_zero) = 0;
if ~any(nonzero)
    return;
end

scale = min(abs(x(nonzero)), [], 'omitnan');
if ~isfinite(scale) || scale <= 0
    scale = eps;
end
y(nonzero) = sign(x(nonzero)) .* log10(1 + abs(x(nonzero)) ./ scale);

end


function output_path = plot_close_far_nearest(NearestGroupSummary, model_names, colors, figures_output_dir, save_png, NearestComparisonSummary)
% plot_close_far_nearest  Render the plot close far nearest panel or plotting primitive.
% Inputs: NearestGroupSummary, model_names, colors, figures_output_dir, save_png, NearestComparisonSummary
% Outputs: output_path
%PLOT_CLOSE_FAR_NEAREST
% H1: Three-panel nearest-distance log2(nMAE) curves, faceted by tertile group.
%
% PURPOSE
%   Show the nearest-T1-distance error curve separately for close / middle /
%   far inter-flip-distance graphs, exposing how the SECOND flip's proximity
%   modulates the nearest-flip response.
%
% INPUTS
%   NearestGroupSummary : summarize_wide_records output grouped by
%                         (distance_group, d_near).
%   model_names         : cellstr of models / legend order.
%   colors              : Nx3 RGB matrix aligned to model_names.
%   figures_output_dir  : output directory.
%   save_png            : logical; export PNG if true.
%
% OUTPUTS
%   output_path : full path to the saved .fig file.
%
% ALGORITHM
%   1x3 tiledlayout; for group_id 1..3, subset the summary to that group and
%   plot_summary_lines of log2_nmae_mean/_sd vs d_near, titled via
%   distance_group_label_from_id; savefig and optionally PNG.
%
% DECISIONS / EDGE-CASES
%   * Panels share the close/middle/far semantics defined by the tertile edges.

fig = figure('Position', [100 100 1080 390], 'DockControls', 'off', ...
    'NumberTitle', 'off', 'Name', 'Flip_two | Nearest-distance curves by inter-flip distance group');
tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for group_id = 1 : 3
    ax = nexttile;
    Sg = NearestGroupSummary(NearestGroupSummary.distance_group == group_id, :);
    plot_summary_lines(ax, Sg, 'd_near', model_names, colors, 'log2_nmae_mean', 'log2_nmae_sd', ...
        'Nearest T1 distance (hops)', 'log2(nMAE)', sprintf('%s (dotted = single T1)', char(distance_group_label_from_id(group_id))));
    overlay_single_nearest_reference(ax, NearestComparisonSummary, model_names, colors, 'log2_nmae_mean', 'log2_nmae_sd');
end

output_path = fullfile(figures_output_dir, 'Flip_two nearest-distance curves by close-middle-far inter-flip distance.fig');
dcg_savefig_visible(fig, output_path);
maybe_export_png(fig, output_path, save_png);

end


function plot_summary_lines(ax, Summary, x_var, model_names, colors, mean_var, sd_var, x_label, y_label, plot_title)
% plot_summary_lines  Render the plot summary lines panel or plotting primitive.
% Inputs: ax, Summary, x_var, model_names, colors, mean_var, sd_var, x_label, y_label, plot_title
% Outputs: none; performs side effects or updates the caller workflow.
%PLOT_SUMMARY_LINES
% H1: Draw one shaded mean+/-SD line per model on a shared axis.
%
% PURPOSE
%   Reusable line-plot primitive for all distance-curve figures: for each
%   model it sorts by the x variable and renders a shaded-error line.
%
% INPUTS
%   ax                : target axes handle.
%   Summary           : summary table containing 'model', x_var, mean_var,
%                       sd_var columns.
%   x_var             : column name for the x axis (e.g. 'd_near').
%   model_names       : cellstr of models / draw + legend order.
%   colors            : Nx3 RGB matrix aligned to model_names.
%   mean_var, sd_var  : column names for the line height and the band half-width.
%   x_label, y_label  : axis label strings.
%   plot_title        : axes title string.
%
% OUTPUTS
%   (none returned) Mutates ax: lines, labels, legend, square axis, padded
%   limits.
%
% ALGORITHM
%   For each model, mask its rows, pull (x,y,e), sort by x, choose line style
%   (dashed for Baseline, solid otherwise), and draw via plot_shaded_line;
%   collect handles for a legend; then style box/ticks, labels, title, square
%   aspect, and call axis_tight_with_padding (raw-axis flag set when plotting
%   raw_mean so the y-floor can clamp to 0).
%
% DECISIONS / EDGE-CASES
%   * Baseline is dashed to distinguish the reference from trained models.
%   * Sorting by x guarantees a monotone, non-self-crossing line and a clean
%     fill polygon.

hold(ax, 'on');
line_handles = gobjects(1, numel(model_names));
for m = 1 : numel(model_names)
    take = Summary.model == string(model_names{m});
    x = Summary.(x_var)(take);
    y = Summary.(mean_var)(take);
    e = Summary.(sd_var)(take);
    [x, order] = sort(x);
    y = y(order);
    e = e(order);
    if strcmp(model_names{m}, 'Baseline')
        line_style = '--';
    else
        line_style = '-';
    end
    line_handles(m) = plot_shaded_line(ax, x, y, e, colors(m,:), line_style);
end
hold(ax, 'off');
set(ax, 'Box', 'off', 'TickDir', 'out');
xlabel(ax, x_label);
ylabel(ax, y_label);
title(ax, plot_title);
legend(ax, line_handles, model_names, 'Location', 'best');
axis(ax, 'square');
axis_tight_with_padding(ax, strcmp(mean_var, 'raw_mean'));

end


function overlay_single_overall_reference(ax, OverallComparisonSummary, model_names, colors, mean_var, sd_var)
% overlay_single_overall_reference  Implement overlay single overall reference for this MATLAB workflow.
% Inputs: ax, OverallComparisonSummary, model_names, colors, mean_var, sd_var
% Outputs: none; performs side effects or updates the caller workflow.
%OVERLAY_SINGLE_OVERALL_REFERENCE
% H1: Add same-model dotted horizontal single-T1 references to a Flip_two axis.

if nargin < 6 || isempty(OverallComparisonSummary) || ~ismember('condition_id', OverallComparisonSummary.Properties.VariableNames)
    return;
end

xlim_curr = xlim(ax);
hold(ax, 'on');
for m = 1 : numel(model_names)
    take = OverallComparisonSummary.condition_id == 1 & OverallComparisonSummary.model == string(model_names{m});
    if ~any(take)
        continue;
    end
    y = OverallComparisonSummary.(mean_var)(find(take, 1));
    e = OverallComparisonSummary.(sd_var)(find(take, 1));
    if ~isfinite(y)
        continue;
    end
    xx = xlim_curr(:);
    yy = [y; y];
    ee = [e; e];
    h = plot_shaded_line(ax, xx, yy, ee, colors(m,:), ':');
    h.HandleVisibility = 'off';
end
hold(ax, 'off');
xlim(ax, xlim_curr);
axis_tight_with_padding(ax, strcmp(mean_var, 'raw_mean'));

end


function overlay_single_nearest_reference(ax, NearestComparisonSummary, model_names, colors, mean_var, sd_var)
% overlay_single_nearest_reference  Implement overlay single nearest reference for this MATLAB workflow.
% Inputs: ax, NearestComparisonSummary, model_names, colors, mean_var, sd_var
% Outputs: none; performs side effects or updates the caller workflow.
%OVERLAY_SINGLE_NEAREST_REFERENCE
% H1: Add same-model dotted single-T1 d_near profiles to a Flip_two distance axis.

if nargin < 6 || isempty(NearestComparisonSummary) || ~ismember('condition_id', NearestComparisonSummary.Properties.VariableNames)
    return;
end

SingleSummary = NearestComparisonSummary(NearestComparisonSummary.condition_id == 1, :);
if isempty(SingleSummary)
    return;
end

hold(ax, 'on');
for m = 1 : numel(model_names)
    take = SingleSummary.model == string(model_names{m});
    x = SingleSummary.d_near(take);
    y = SingleSummary.(mean_var)(take);
    e = SingleSummary.(sd_var)(take);
    [x, order] = sort(x);
    y = y(order);
    e = e(order);
    h = plot_shaded_line(ax, x, y, e, colors(m,:), ':');
    h.HandleVisibility = 'off';
end
hold(ax, 'off');
axis_tight_with_padding(ax, strcmp(mean_var, 'raw_mean'));

end


function plot_zone_bars(ax, ZoneSummary, model_names, colors, mean_var, sd_var, y_label, plot_title)
% plot_zone_bars  Render the plot zone bars panel or plotting primitive.
% Inputs: ax, ZoneSummary, model_names, colors, mean_var, sd_var, y_label, plot_title
% Outputs: none; performs side effects or updates the caller workflow.
%PLOT_ZONE_BARS
% H1: Grouped bar chart (zones x models) with error bars on one axis.
%
% PURPOSE
%   Reusable bar-plot primitive for the interaction-zone figure: clustered
%   bars per zone, one bar per model, with SD error bars.
%
% INPUTS
%   ax               : target axes handle.
%   ZoneSummary      : summary table with 'zone_id', 'model', mean_var, sd_var.
%   model_names      : cellstr of models / cluster + legend order.
%   colors           : Nx3 RGB matrix aligned to model_names.
%   mean_var, sd_var : column names for bar height and error-bar half-length.
%   y_label          : y-axis label string.
%   plot_title       : axes title string.
%
% OUTPUTS
%   (none returned) Mutates ax: grouped bars, error bars, categorical x ticks,
%   legend, padded limits.
%
% ALGORITHM
%   1. zone_ids = sorted unique zone ids; bar_width = 0.8 / nModels.
%   2. For each model, gather per-zone mean/SD, offset its cluster position by
%      (m - (N+1)/2)*bar_width, draw bars (model colour) and black '.' errorbars.
%   3. Set x ticks to {near one, near both, far both}, style box/ticks, labels,
%      title, legend, and axis_tight_with_padding (raw-axis flag for raw_mean).
%
% DECISIONS / EDGE-CASES
%   * The symmetric offset centers each model cluster on its zone tick.
%   * Tick labels are fixed text assuming the three canonical zones in order.

zone_ids = unique(ZoneSummary.zone_id)';
bar_width = 0.8 / numel(model_names);
hold(ax, 'on');
bar_handles = gobjects(1, numel(model_names));
for m = 1 : numel(model_names)
    means = nan(size(zone_ids));
    sds = nan(size(zone_ids));
    for z = 1 : numel(zone_ids)
        take = ZoneSummary.zone_id == zone_ids(z) & ZoneSummary.model == string(model_names{m});
        means(z) = ZoneSummary.(mean_var)(take);
        sds(z) = ZoneSummary.(sd_var)(take);
    end
    x = (1:numel(zone_ids)) + (m - (numel(model_names)+1)/2) * bar_width;
    bar_handles(m) = bar(ax, x, means, bar_width, 'FaceColor', colors(m,:), 'EdgeColor', 'none');
    errorbar(ax, x, means, sds, 'k.', 'LineWidth', 0.75);
end
hold(ax, 'off');
set(ax, 'XTick', 1:numel(zone_ids), 'XTickLabel', {'near one','near both','far both'}, ...
    'XTickLabelRotation', 25, 'Box', 'off', 'TickDir', 'out');
xlabel(ax, 'Distance zone');
ylabel(ax, y_label);
title(ax, plot_title);
legend(ax, bar_handles, model_names, 'Location', 'best');
axis_tight_with_padding(ax, strcmp(mean_var, 'raw_mean'));

end


function plot_comparison_zone_bars(ax, ZoneComparisonSummary, model_names, colors, mean_var, sd_var, y_label, plot_title, log_y_axis)
% plot_comparison_zone_bars  Render the plot comparison zone bars panel or plotting primitive.
% Inputs: ax, ZoneComparisonSummary, model_names, colors, mean_var, sd_var, y_label, plot_title, log_y_axis
% Outputs: none; performs side effects or updates the caller workflow.
%PLOT_COMPARISON_ZONE_BARS
% H1: Grouped bars for the five single-vs-two distance-zone categories.

if nargin < 9 || isempty(log_y_axis)
    log_y_axis = false;
end

zone_ids = unique(ZoneComparisonSummary.comparison_zone_id)';
bar_width = 0.8 / numel(model_names);
hold(ax, 'on');
bar_handles = gobjects(1, numel(model_names));
for m = 1 : numel(model_names)
    means = nan(size(zone_ids));
    sds = nan(size(zone_ids));
    for z = 1 : numel(zone_ids)
        take = ZoneComparisonSummary.comparison_zone_id == zone_ids(z) & ...
            ZoneComparisonSummary.model == string(model_names{m});
        if any(take)
            means(z) = ZoneComparisonSummary.(mean_var)(find(take, 1));
            sds(z) = ZoneComparisonSummary.(sd_var)(find(take, 1));
        end
    end
    x = (1:numel(zone_ids)) + (m - (numel(model_names)+1)/2) * bar_width;
    bar_handles(m) = bar(ax, x, means, bar_width, 'FaceColor', colors(m,:), 'EdgeColor', 'none');
    errorbar(ax, x, means, sds, 'k.', 'LineWidth', 0.75, 'HandleVisibility', 'off');
end
hold(ax, 'off');

tick_labels = comparison_zone_short_label_from_id(zone_ids);

set(ax, 'XTick', 1:numel(zone_ids), 'XTickLabel', cellstr(tick_labels), ...
    'XTickLabelRotation', 30, 'Box', 'off', 'TickDir', 'out');
xlabel(ax, 'Distance zone');
ylabel(ax, y_label);
title(ax, plot_title);
legend(ax, bar_handles, model_names, 'Location', 'best');
axis_tight_with_padding(ax, strcmp(mean_var, 'raw_mean') && ~log_y_axis);
if log_y_axis
    set_log_axis_from_positive_data(ax);
end

end


function set_log_axis_from_positive_data(ax)
% set_log_axis_from_positive_data  Implement set log axis from positive data for this MATLAB workflow.
% Inputs: ax
% Outputs: none; performs side effects or updates the caller workflow.
%SET_LOG_AXIS_FROM_POSITIVE_DATA
% H1: Put an axis on a positive-only logarithmic y scale.
%
% PURPOSE
%   Raw MAE panels can be displayed on a log y-axis, but MATLAB log axes cannot
%   include the zero floor used by linear raw-MAE plots. This helper gathers
%   positive finite YData from the plotted children, pads the range
%   multiplicatively, and applies YScale='log'.

objs = findall(ax, '-property', 'YData');
y = [];
for i = 1 : numel(objs)
    curr = get(objs(i), 'YData');
    if isnumeric(curr)
        y = [y; curr(:)]; %#ok<AGROW>
    end
end
y = y(isfinite(y) & y > 0);
if isempty(y)
    set(ax, 'YScale', 'log');
    return;
end

y_min = min(y);
y_max = max(y);
if y_min == y_max
    y_min = y_min / sqrt(10);
    y_max = y_max * sqrt(10);
else
    y_min = y_min / 1.2;
    y_max = y_max * 1.2;
end
set(ax, 'YScale', 'log');
ylim(ax, [y_min, y_max]);

end


function plot_one_heatmap(ax, PairSummary, model_name, metric_var, color_limits)
% plot_one_heatmap  Render the plot one heatmap panel or plotting primitive.
% Inputs: ax, PairSummary, model_name, metric_var, color_limits
% Outputs: none; performs side effects or updates the caller workflow.
%PLOT_ONE_HEATMAP
% H1: Render one model's (d_near x d_far) metric grid as an imagesc heatmap.
%
% PURPOSE
%   Single-panel helper for plot_pair_heatmaps: scatter the model's per-pair
%   metric values onto a dense integer-hop grid and display it.
%
% INPUTS
%   ax           : target axes handle.
%   PairSummary  : summary table with 'model', 'd_near', 'd_far', metric_var.
%   model_name   : model whose rows to select.
%   metric_var   : column name to visualize.
%   color_limits : 1x2 [lo hi] shared colour scale (set by the caller).
%
% OUTPUTS
%   (none returned) Mutates ax: imagesc image, normal y-direction, parula
%   colormap, clim set to color_limits.
%
% ALGORITHM
%   1. Mask the model's rows; pull d_near, d_far, vals.
%   2. Build integer hop axes x_levels=0:max(d_far), y_levels=0:max(d_near);
%      allocate an NaN matrix M and scatter each value into M(yi,xi) by exact
%      level match.
%   3. imagesc(x_levels,y_levels,M); set YDir normal; apply clim and parula.
%
% MATH / DECISIONS / EDGE-CASES
%   * Grid cells with no record stay NaN and render transparent/blank.
%   * d_near<=d_far always (by construction), so the populated region is
%     triangular; the upper triangle is naturally empty.
%   * Exact equality matching is safe because distances are integer hop counts.

take = PairSummary.model == string(model_name);
d_near = PairSummary.d_near(take);
d_far = PairSummary.d_far(take);
vals = PairSummary.(metric_var)(take);

x_levels = 0:max(d_far);
y_levels = 0:max(d_near);
M = nan(numel(y_levels), numel(x_levels));
for i = 1 : numel(vals)
    yi = find(y_levels == d_near(i), 1);
    xi = find(x_levels == d_far(i), 1);
    if ~isempty(xi) && ~isempty(yi)
        M(yi, xi) = vals(i);
    end
end

imagesc(ax, x_levels, y_levels, M);
set(ax, 'YDir', 'normal', 'Box', 'off', 'TickDir', 'out');
clim(ax, color_limits);
colormap(ax, parula);

end


function h_line = plot_shaded_line(ax, x, y, e, color, line_style)
% plot_shaded_line  Render the plot shaded line panel or plotting primitive.
% Inputs: ax, x, y, e, color, line_style
% Outputs: h_line
%PLOT_SHADED_LINE
% H1: Draw a line with a translucent +/-e error band; return the line handle.
%
% PURPOSE
%   Low-level drawing primitive used by plot_summary_lines to render a single
%   model's curve with a shaded uncertainty ribbon.
%
% INPUTS
%   ax         : target axes handle.
%   x, y, e    : equal-length vectors of x positions, y values, error
%                half-widths (band spans y-e .. y+e).
%   color      : 1x3 RGB for both the line and the fill.
%   line_style : MATLAB line spec (e.g. '-' or '--').
%
% OUTPUTS
%   h_line : handle to the plotted line (for legend entries).
%
% ALGORITHM
%   Force column vectors; keep only finite (x,y,e) triples. If any remain,
%   fill() the polygon [xv;flip(xv)] vs [yv-ev;flip(yv+ev)] at 16% alpha (no
%   edge, hidden from legend) and plot the line on top. If none are finite,
%   plot a NaN placeholder line so the legend entry still exists.
%
% DECISIONS / EDGE-CASES
%   * The band is excluded from the legend ('HandleVisibility','off') so only
%     the line shows.
%   * The NaN-placeholder branch preserves legend/colour ordering even for an
%     all-missing series.

x = x(:);
y = y(:);
e = e(:);
valid = isfinite(x) & isfinite(y) & isfinite(e);
if any(valid)
    xv = x(valid);
    yv = y(valid);
    ev = e(valid);
    fill(ax, [xv; flipud(xv)], [yv - ev; flipud(yv + ev)], color, ...
        'FaceAlpha', 0.16, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    h_line = plot(ax, xv, yv, line_style, 'Color', color, 'LineWidth', 1.4);
else
    h_line = plot(ax, NaN, NaN, line_style, 'Color', color, 'LineWidth', 1.4);
end

end


function axis_tight_with_padding(ax, raw_axis)
% axis_tight_with_padding  Implement axis tight with padding for this MATLAB workflow.
% Inputs: ax, raw_axis
% Outputs: none; performs side effects or updates the caller workflow.
%AXIS_TIGHT_WITH_PADDING
% H1: Set x/y limits to the data range plus a small margin (optional y=0 floor).
%
% PURPOSE
%   Tighten axes around all plotted data while leaving breathing room, and
%   optionally clamp the y floor to 0 for raw (non-negative) MAE plots.
%
% INPUTS
%   ax       : target axes handle.
%   raw_axis : logical; when true and the padded y_min is positive, force
%              y_min to 0 (appropriate for raw MAE which cannot be negative).
%
% OUTPUTS
%   (none returned) Mutates ax y- and x-limits.
%
% ALGORITHM
%   1. Gather YData from every child with that property; keep finite values.
%      pad = 5% of the data span (or a tiny floor when span is 0); expand
%      [y_min,y_max] by pad; if raw_axis and y_min>0 set y_min=0; apply ylim.
%   2. Gather XData similarly; if the x range is non-degenerate, set xlim to
%      [min-0.5, max+0.5] (half-hop margin for integer hop axes).
%
% MATH / DECISIONS / EDGE-CASES
%   * Degenerate y (all equal): pad = max(1e-6, 5% of |y|) to avoid a zero-height
%     axis.
%   * Empty/non-finite data: the corresponding limit is left untouched.
%   * The raw-axis 0-floor keeps raw MAE panels anchored at zero for honest
%     visual comparison.

objs = findall(ax, '-property', 'YData');
y = [];
for i = 1 : numel(objs)
    curr = get(objs(i), 'YData');
    if isnumeric(curr)
        y = [y; curr(:)]; %#ok<AGROW>
    end
end
y = y(isfinite(y));
if ~isempty(y)
    y_min = min(y);
    y_max = max(y);
    if y_min == y_max
        pad = max(1e-6, abs(y_min) * 0.05);
    else
        pad = 0.05 * (y_max - y_min);
    end
    y_min = y_min - pad;
    y_max = y_max + pad;
    if raw_axis && y_min > 0
        y_min = 0;
    end
    ylim(ax, [y_min, y_max]);
end

objs = findall(ax, '-property', 'XData');
x = [];
for i = 1 : numel(objs)
    curr = get(objs(i), 'XData');
    if isnumeric(curr)
        x = [x; curr(:)]; %#ok<AGROW>
    end
end
x = x(isfinite(x));
if ~isempty(x) && min(x) < max(x)
    xlim(ax, [min(x) - 0.5, max(x) + 0.5]);
end

end


function colors = paper_model_colors(model_names)
% paper_model_colors  Implement paper model colors for this MATLAB workflow.
% Inputs: model_names
% Outputs: colors
%PAPER_MODEL_COLORS
% H1: Return the fixed paper RGB palette for the requested model names.
%
% PURPOSE
%   Centralize the manuscript's per-model colours so every figure in this file
%   (and across the pipeline) draws each model in a consistent hue.
%
% INPUTS
%   model_names : cellstr/string array of model names to colour.
%
% OUTPUTS
%   colors : numel(model_names)x3 RGB matrix, row i = colour for model i.
%
% ALGORITHM
%   Switch on each model name and assign its fixed RGB triple; unknown names
%   (including 'Baseline') get black [0 0 0].
%
% DECISIONS / EDGE-CASES
%   * PPGN is included in the Flip_two and single-vs-two plots; its colour is
%     kept identical to the rest of the revision pipeline.
%   * Triples are MATLAB's default 'lines'-style colours assigned per model.
%   * Baseline falls through 'otherwise' to black, matching its dashed styling.

colors = zeros(numel(model_names), 3);
for i = 1 : numel(model_names)
    switch char(model_names{i})
        case 'PPGN'
            colors(i,:) = [0.0000, 0.4470, 0.7410];
        case 'GraphSAGE'
            colors(i,:) = [0.8500, 0.3250, 0.0980];
        case 'GAT'
            colors(i,:) = [0.4660, 0.6740, 0.1880];
        case 'GIN'
            colors(i,:) = [0.4940, 0.1840, 0.5560];
        case 'PNA'
            colors(i,:) = [0.9290, 0.6940, 0.1250];
        otherwise
            colors(i,:) = [0, 0, 0];
    end
end

end


function dcg_savefig_visible(fig_handle, filename)
% dcg_savefig_visible  Save MATLAB figures in the publication output format.
% Inputs: fig_handle, filename
% Outputs: none; performs side effects or updates the caller workflow.
%DCG_SAVEFIG_VISIBLE
% H1: Mark a figure visible, set a standard name, and save it as .fig.
%
% PURPOSE
%   Persist a figure so it reopens VISIBLE (MATLAB otherwise restores the
%   saved Visible state) and carries a consistent Flip_two window name.
%
% INPUTS
%   fig_handle : figure handle to save.
%   filename   : full target .fig path (its stem becomes part of the name).
%
% OUTPUTS
%   (none returned) Writes the .fig file to disk and mutates the figure's
%   Visible/Name properties.
%
% ALGORITHM
%   Set Visible 'on'; derive the file stem via fileparts; set
%   Name='Flip_two two-source analysis | <stem>' with NumberTitle off; savefig.
%
% DECISIONS / EDGE-CASES
%   * Forcing Visible 'on' ensures the saved .fig is not stuck hidden when
%     reopened later for the manuscript.

set(fig_handle, 'Visible', 'on');
[~, file_stem] = fileparts(char(filename));
set(fig_handle, 'Name', ['Flip_two two-source analysis | ', file_stem], ...
    'NumberTitle', 'off');
savefig(fig_handle, filename);

end


function maybe_export_png(fig_handle, fig_path, save_png)
% maybe_export_png  Save MATLAB figures in the publication output format.
% Inputs: fig_handle, fig_path, save_png
% Outputs: none; performs side effects or updates the caller workflow.
%MAYBE_EXPORT_PNG
% H1: Optionally export a figure as a 300-dpi PNG beside its .fig.
%
% PURPOSE
%   Conditionally emit a raster copy of a figure for quick previews/manuscript
%   drafts, controlled by the top-level save_png flag.
%
% INPUTS
%   fig_handle : figure handle to export.
%   fig_path   : the figure's .fig path; the PNG path is derived from it.
%   save_png   : logical gate; when false the function is a no-op.
%
% OUTPUTS
%   (none returned) Writes a .png next to fig_path when save_png is true.
%
% ALGORITHM
%   Return immediately if save_png is false; else replace the trailing '.fig'
%   with '.png' and exportgraphics at 300 dpi.
%
% DECISIONS / EDGE-CASES
%   * Resolution fixed at 300 dpi for print quality.
%   * The PNG path is derived purely by extension swap, so it sits alongside
%     the .fig with the same stem.

if ~save_png
    return;
end
png_path = regexprep(fig_path, '\.fig$', '.png');
exportgraphics(fig_handle, png_path, 'Resolution', 300);

end
