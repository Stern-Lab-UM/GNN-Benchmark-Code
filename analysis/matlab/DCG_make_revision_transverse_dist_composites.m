function output_paths = DCG_make_revision_transverse_dist_composites(figures_root, save_png)
%DCG_MAKE_REVISION_TRANSVERSE_DIST_COMPOSITES
% Build 3x3 comparison panels from the already-saved single-dataset
% MAE-vs-transverse-distance figures.
%
% ---------------------------------------------------------------------------
% FILE OVERVIEW (GNN-benchmark revision figure-assembly pipeline)
% ---------------------------------------------------------------------------
% This file is a self-contained MATLAB figure-assembly utility used during the
% npj revision of the DCG / GNN-benchmark study. It does NOT recompute any
% science: it only re-packages plots that were already produced and saved to
% disk as individual single-dataset .fig files. For each MAE flavor (raw,
% log2, log2-normalized) it stitches nine of those saved per-dataset plots into
% one publication-style 3x3 composite, harmonizes their axes, and re-saves the
% composite (optionally also as PNG).
%
% Conceptual layout of every 3x3 composite that is built:
%   - Column 1 is the unperturbed REFERENCE dataset ('v1', dir 'v1_2_16_W');
%     it is repeated on all three rows so each row's two perturbed variants can
%     be read against the same baseline.
%   - Columns 2-3 are PERTURBED variants of that baseline, grouped by row:
%       Row 1 = area-elasticity sweep (kA = 10, kA = 1)
%       Row 2 = applied-shear sweep   (Shear 1.2, Shear 1.5)
%       Row 3 = tissue-size sweep      (Tissue 484, Tissue 784)
%   - The same nine-cell grid is rendered three times, once per MAE flavor.
%
% Axis harmonization performed during assembly:
%   - The y-axis range is UNIFIED across all nine panels of a composite, so the
%     nine sub-plots are quantitatively comparable. The shared range is the
%     pooled finite YData extent over all nine sources, then padded by +/-5%.
%   - For the raw-MAE flavor ONLY, the unified lower y-bound is clamped to 0
%     (MAE is non-negative; a floating, slightly-negative lower bound would be
%     misleading). The log2 flavors keep their padded (possibly negative) bound.
%   - The x-axis is padded by +/-0.5 PER PANEL (not unified), because the
%     transverse-distance / hop axis is integer-valued and per-dataset.
%
% Companion side-effects written to the output directory:
%   - 'revision_transverse_dist_comparison_order.txt' records the row/column
%     dataset->panel mapping (see write_composite_assumptions) so the provenance
%     of every composite is auditable later.
%
% FUNCTIONS IN THIS FILE (1 entry point + 6 local helpers):
%   DCG_make_revision_transverse_dist_composites - main driver
%   best_data_axes        - pick the data-bearing axes inside a loaded .fig
%   copy_axis_contents    - clone one axes' graphics + style into a tile
%   dcg_savefig_visible   - make figure visible, name it, save as .fig
%   axis_data_extent      - pooled finite min/max of an object property (X/YData)
%   padded_axis_limits    - symmetric 5% padding (+ optional clamp-to-zero)
%   write_composite_assumptions - dump the row/column dataset mapping to a .txt
% ---------------------------------------------------------------------------
%
% ---------------------------------------------------------------------------
% FUNCTION: DCG_make_revision_transverse_dist_composites (main)
% ---------------------------------------------------------------------------
% H1: Assemble 3x3 MAE-vs-transverse-distance composite figures from saved per-
%     dataset .fig files and re-save them (one composite per MAE flavor).
%
% PURPOSE:
%   Top-level driver. For each of three MAE flavors it (a) discovers the nine
%   source .fig files, (b) computes a single unified y-range across those nine,
%   (c) lays out a 3x3 tiledlayout, (d) copies each source's plotted contents
%   into its tile, (e) applies the unified y-limit and per-panel x-padding plus
%   cosmetic styling (titles, axis labels, square boxes), and (f) saves the
%   finished composite to .fig (and optionally .png). It also writes a text file
%   documenting the row/column dataset mapping before the loop runs.
%
% INPUTS:
%   figures_root (optional char/string): root directory that contains one
%       sub-folder per dataset (e.g. 'v1_2_16_W', 'kA_10', ...). Each sub-folder
%       is expected to hold the three named source .fig files. If omitted or
%       empty, defaults to fullfile(cfg.data_root, '_figures', 'revision_2026')
%       using DCG_publication_config().
%   save_png (optional logical): if true, each saved composite .fig is also
%       exported to a same-named .png at 300 DPI. If omitted or empty, false.
%
% OUTPUTS:
%   output_paths (cell column vector, one entry per MAE flavor / per row of
%       figure_names): absolute path to each saved composite .fig. An entry is
%       left empty ([]) only if that flavor's composite was never saved (it is
%       always saved here, so entries are populated on normal completion).
%
% ALGORITHM:
%   1. Resolve defaults for figures_root and save_png.
%   2. Define three constant 3x3 cell arrays that govern the layout:
%        panel_dirs   - per-cell dataset sub-folder name (column 1 = 'v1_2_16_W'
%                       reference repeated; columns 2-3 = perturbed variants).
%        panel_titles - per-cell display title (TeX, e.g. 'kA = 10').
%      and one 3x(>=2) cell array figure_names whose columns are
%        {source .fig filename, short flavor tag, default y-axis label}.
%   3. Ensure output_dir = figures_root/'revision_transverse_dist_composites'
%      exists, then write the provenance .txt via write_composite_assumptions.
%   4. For each MAE flavor row f of figure_names:
%        a. Build the 3x3 source_paths from figures_root + panel_dirs +
%           filename; warn (and skip in the extent scan) for any missing file.
%        b. FIRST PASS - open each existing source invisibly, find its data axes
%           via best_data_axes, pool its finite YData extent via
%           axis_data_extent, and accumulate a global [min,max]; close it. Then
%           pad/clamp that global range via padded_axis_limits (clamp-to-zero
%           enabled only when the filename contains 'raw MAE').
%        c. Create a fixed-size figure and a 3x3 tiledlayout (compact spacing).
%        d. SECOND PASS - for each cell, open its source invisibly, find its
%           data axes, copy the plotted children + styling into the tile via
%           copy_axis_contents, set the TeX title, apply the unified y-limit
%           (only if it is finite and strictly increasing), apply per-panel
%           x-limits padded by +/-0.5 (only if the source's XData extent is
%           finite and strictly increasing), force a square box, and set axis
%           labels conditionally (y-label only on column 1, taken from the
%           source's own y-label or the flavor default; x-label only on the
%           bottom row). Missing sources / empty axes yield a blank titled tile.
%        e. Save the composite via dcg_savefig_visible; if save_png, also export
%           a 300-DPI PNG; print a progress line; record the .fig path.
%
% MATH / DECISIONS / EDGE-CASES:
%   - Reference repetition: panel_dirs column 1 is 'v1_2_16_W' on every row by
%     design, so the same baseline figure is loaded three times per composite.
%   - Two-pass design: a full first pass over all nine sources is required so
%     the y-range is truly global before any panel is drawn; the copy pass then
%     re-opens each source (sources are reopened, not cached).
%   - Unified y vs per-panel x: y-limits come from the global padded range and
%     are applied identically to every panel; x-limits are computed per panel
%     from that panel's own source XData and padded by a fixed +/-0.5.
%   - Guard conditions: y-limit is applied only when both bounds are finite and
%     global_y(1) < global_y(2); x-limit only when xl_min < xl_max; this avoids
%     degenerate/inverted axis ranges from empty or single-point data.
%   - Missing-data robustness: a missing file or an axes with no data produces a
%     blank ('axis off') but still-titled tile rather than an error, so a
%     partially-available dataset still yields a labeled composite.
%   - Titles use the TeX interpreter so tokens like 'kA = 10' render normally;
%     no subscripting is intended here.
%   - PNG path is derived by regex-replacing a trailing '.fig' with '.png'.

if nargin < 1 || isempty(figures_root)
    path_cfg = DCG_publication_config();
    if isempty(path_cfg.data_root)
        error('DCG:missingDataRoot', ['Pass figures_root explicitly or set ', ...
            'DCG_DATA_ROOT / DCG_local_config.m.']);
    end
    figures_root = fullfile(path_cfg.data_root, '_figures', 'revision_2026');
end
if nargin < 2 || isempty(save_png)
    save_png = false;
end

panel_dirs = {
    'v1_2_16_W', 'kA_10',     'kA_1'
    'v1_2_16_W', 'Shear_1_2', 'Shear_1_5'
    'v1_2_16_W', 'Tissue_484','Tissue_784'
    };
panel_titles = {
    'v1', 'kA = 10',    'kA = 1'
    'v1', 'Shear 1.2',  'Shear 1.5'
    'v1', 'Tissue 484', 'Tissue 784'
    };

figure_names = {
    'MAE vs traverse dist (raw MAE).fig',   'raw MAE',   'Mean graph MAE'
    'MAE vs traverse dist (log2 MAE).fig',  'log2 MAE',  'log2(mean graph MAE)'
    'MAE vs traverse dist (log2 nMAE).fig', 'log2 nMAE', 'log2(normalized graph MAE)'
    };

output_dir = fullfile(figures_root, 'revision_transverse_dist_composites');
if ~isfolder(output_dir)
    mkdir(output_dir);
end

write_composite_assumptions(output_dir, panel_dirs, panel_titles);

output_paths = cell(size(figure_names, 1), 1);
for f = 1 : size(figure_names, 1)
    figure_name = figure_names{f, 1};
    short_name = figure_names{f, 2};
    default_ylabel = figure_names{f, 3};
    source_paths = cell(size(panel_dirs));
    global_y = [inf, -inf];

    for r = 1 : size(panel_dirs, 1)
        for c = 1 : size(panel_dirs, 2)
            source_paths{r,c} = fullfile(figures_root, panel_dirs{r,c}, figure_name);
            if ~isfile(source_paths{r,c})
                warning('DCG:missingCompositeSource', 'Missing source figure: %s', source_paths{r,c});
                continue;
            end
            src_fig = openfig(source_paths{r,c}, 'invisible');
            src_ax = best_data_axes(src_fig);
            if ~isempty(src_ax)
                [yl_min, yl_max] = axis_data_extent(src_ax, 'YData');
                if all(isfinite([yl_min, yl_max]))
                    global_y(1) = min(global_y(1), yl_min);
                    global_y(2) = max(global_y(2), yl_max);
                end
            end
            close(src_fig);
        end
    end
    [global_y_min, global_y_max] = padded_axis_limits(global_y(1), global_y(2), contains(figure_name, 'raw MAE'));
    global_y = [global_y_min, global_y_max];

    fig = figure('Position', [80 80 1080 960], ...
        'DockControls', 'off', 'NumberTitle', 'off', ...
        'Name', ['Revision transverse distance comparison - ', short_name]);
    tl = tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    for r = 1 : size(panel_dirs, 1)
        for c = 1 : size(panel_dirs, 2)
            dst_ax = nexttile(tl, (r-1) * size(panel_dirs, 2) + c);
            src_path = source_paths{r,c};
            if ~isfile(src_path)
                axis(dst_ax, 'off');
                title(dst_ax, panel_titles{r,c});
                continue;
            end

            src_fig = openfig(src_path, 'invisible');
            src_ax = best_data_axes(src_fig);
            if isempty(src_ax)
                axis(dst_ax, 'off');
                title(dst_ax, panel_titles{r,c});
                close(src_fig);
                continue;
            end

            copy_axis_contents(src_ax, dst_ax);
            title(dst_ax, panel_titles{r,c}, 'Interpreter', 'tex');
            if isfinite(global_y(1)) && isfinite(global_y(2)) && global_y(1) < global_y(2)
                ylim(dst_ax, global_y);
            end
            [xl_min, xl_max] = axis_data_extent(src_ax, 'XData');
            if all(isfinite([xl_min, xl_max])) && xl_min < xl_max
                xlim(dst_ax, [xl_min - 0.5, xl_max + 0.5]);
            end
            axis(dst_ax, 'square');
            box(dst_ax, 'off');

            if c == 1
                source_ylabel = get(get(src_ax, 'YLabel'), 'String');
                if isempty(source_ylabel)
                    source_ylabel = default_ylabel;
                end
                ylabel(dst_ax, source_ylabel);
            else
                ylabel(dst_ax, '');
            end

            if r == size(panel_dirs, 1)
                xlabel(dst_ax, 'Hops from T1 interface');
            else
                xlabel(dst_ax, '');
            end

            close(src_fig);
        end
    end

    output_paths{f} = fullfile(output_dir, ['Revision transverse dist comparison (', short_name, ').fig']);
    dcg_savefig_visible(fig, output_paths{f});
    if save_png
        png_path = regexprep(output_paths{f}, '\.fig$', '.png');
        exportgraphics(fig, png_path, 'Resolution', 300);
    end
    fprintf('[DCG_make_revision_transverse_dist_composites] saved %s\n', output_paths{f});
end

end


function ax = best_data_axes(fig)
% ---------------------------------------------------------------------------
% FUNCTION: best_data_axes (local)
% ---------------------------------------------------------------------------
% H1: Return the single data-bearing axes of a loaded .fig, ignoring legend
%     and colorbar axes.
%
% PURPOSE:
%   A saved single-dataset figure may contain several axes-typed objects: the
%   real plot plus auxiliary ones (a legend, a colorbar) that MATLAB also
%   reports as Type 'axes'. This helper isolates the one true plot axes so the
%   caller can read its data extent and copy its contents, without being fooled
%   by the auxiliary axes.
%
% INPUTS:
%   fig (figure handle): a freshly opened (typically invisible) source figure.
%
% OUTPUTS:
%   ax (axes handle, or []): the chosen data axes; [] when the figure has no
%       axes at all, or none remain after legend/colorbar are excluded.
%
% ALGORITHM:
%   1. Collect every Type 'axes' object in the figure via findall (recursive).
%   2. If none exist, return [].
%   3. Drop any whose Tag is 'legend' or 'Colorbar' (the standard MATLAB tags
%      for those auxiliary axes).
%   4. If nothing survives the filter, return [].
%   5. Among survivors, count each one's direct children (allchild) and return
%      the axes with the MOST children.
%
% MATH / DECISIONS / EDGE-CASES:
%   - "Most children" is the heuristic for "the real plot": the data axes holds
%     the plotted lines/markers/patches, whereas auxiliary axes hold few or no
%     drawable children once legend/colorbar are already excluded.
%   - max returns the first maximizer on ties, so ties resolve deterministically
%     to whichever survivor findall enumerated first.
%   - Tag matching is case-sensitive (note 'Colorbar' is capitalized, matching
%     MATLAB's convention); only these two exact tags are filtered.
%   - findall (not findobj) is used so axes with hidden handles are still found.

ax_all = findall(fig, 'Type', 'axes');
if isempty(ax_all)
    ax = [];
    return;
end

keep = true(size(ax_all));
for i = 1 : numel(ax_all)
    tag = get(ax_all(i), 'Tag');
    if ismember(tag, {'legend', 'Colorbar'})
        keep(i) = false;
    end
end
ax_all = ax_all(keep);
if isempty(ax_all)
    ax = [];
    return;
end

n_children = arrayfun(@(a) numel(allchild(a)), ax_all);
[~, idx] = max(n_children);
ax = ax_all(idx);

end


function copy_axis_contents(src_ax, dst_ax)
% ---------------------------------------------------------------------------
% FUNCTION: copy_axis_contents (local)
% ---------------------------------------------------------------------------
% H1: Clone the plotted graphics objects and key visual styling from a source
%     axes into a destination tile axes.
%
% PURPOSE:
%   Transplant the actual plot (lines, markers, patches, etc.) from a loaded
%   single-dataset figure's axes into one tile of the 3x3 composite, and copy a
%   curated set of appearance properties so the tile resembles the original.
%
% INPUTS:
%   src_ax (axes handle): source data axes (as chosen by best_data_axes).
%   dst_ax (axes handle): destination tile axes (from nexttile) to populate.
%
% OUTPUTS:
%   none (operates by side effect: adds child objects to dst_ax and sets its
%   properties).
%
% ALGORITHM:
%   1. Grab the source axes' direct children with allchild.
%   2. copyobj them into dst_ax in flipud (reversed) order.
%   3. For each property in a fixed list (limits, scales, ticks + labels, font/
%      line/tick styling, axis colors, layer, grids), try to read it from
%      src_ax and set it on dst_ax, ignoring any error.
%
% MATH / DECISIONS / EDGE-CASES:
%   - flipud reverses the child order: allchild returns children top-of-stack
%     first, so flipping restores the original bottom-to-top draw/stacking order
%     in the copy (preserves which series visually sits on top).
%   - The per-property set() is wrapped in try/catch so an unsupported or
%     read-only property on a given axes type is silently skipped rather than
%     aborting the copy.
%   - XLim/YLim are copied here, but the main driver OVERWRITES them afterward
%     with the unified y-range and padded per-panel x-range; copying them is
%     only a sensible starting point.
%   - Only the listed cosmetic properties are transferred; anything not listed
%     (title text, axis-label strings, position, etc.) is intentionally left to
%     the caller to set.

kids = allchild(src_ax);
copyobj(flipud(kids), dst_ax);

props_to_copy = {'XLim','YLim','XScale','YScale','XTick','XTickLabel', ...
    'YTick','YTickLabel','FontSize','LineWidth','TickDir','XColor','YColor', ...
    'Layer','XGrid','YGrid'};
for p = 1 : numel(props_to_copy)
    try
        set(dst_ax, props_to_copy{p}, get(src_ax, props_to_copy{p}));
    catch
    end
end

end


function dcg_savefig_visible(fig_handle, filename)
% ---------------------------------------------------------------------------
% FUNCTION: dcg_savefig_visible (local)
% ---------------------------------------------------------------------------
% H1: Make a composite figure visible, give it a descriptive window name, and
%     save it to a .fig file.
%
% PURPOSE:
%   Persist a finished composite. The figure is forced Visible='on' before
%   saving so it reopens visibly later (composites are built on an invisible
%   working figure; saving it invisible would make it reopen hidden).
%
% INPUTS:
%   fig_handle (figure handle): the assembled composite figure to save.
%   filename (char/string): destination .fig path; its file stem (no extension)
%       is reused inside the figure's window Name.
%
% OUTPUTS:
%   none (side effects: mutates the figure's Visible/Name/NumberTitle and writes
%   the .fig file to disk).
%
% ALGORITHM:
%   1. Set Visible='on'.
%   2. Derive the file stem from filename via fileparts (after char()).
%   3. Set Name to 'Revision transverse distance comparison | <stem>' and turn
%      NumberTitle off (so the window shows only that name).
%   4. savefig the figure to filename.
%
% MATH / DECISIONS / EDGE-CASES:
%   - char(filename) normalizes a possible string scalar to a char row before
%     fileparts.
%   - No directory creation or overwrite guard here; the caller is responsible
%     for ensuring the target directory exists (it does, via mkdir earlier).

set(fig_handle, 'Visible', 'on');
[~, file_stem] = fileparts(char(filename));
set(fig_handle, 'Name', ['Revision transverse distance comparison | ', file_stem], ...
    'NumberTitle', 'off');
savefig(fig_handle, filename);

end


function [v_min, v_max] = axis_data_extent(ax, property_name)
% ---------------------------------------------------------------------------
% FUNCTION: axis_data_extent (local)
% ---------------------------------------------------------------------------
% H1: Return the pooled finite min/max of a named numeric property (e.g.
%     'XData' or 'YData') over all objects in an axes.
%
% PURPOSE:
%   Measure the true data span along one dimension by scanning every plotted
%   object that exposes the requested property and pooling their values. Used by
%   the driver to derive the global y-range and the per-panel x-range from the
%   actual data rather than from possibly-stale stored axis limits.
%
% INPUTS:
%   ax (axes handle): axes whose descendant objects are scanned.
%   property_name (char): the property to harvest, conventionally 'XData' or
%       'YData' (any numeric, value-bearing property name works).
%
% OUTPUTS:
%   v_min, v_max (double scalars): smallest / largest finite pooled value. Both
%       are NaN when no finite value of that property exists on any object.
%
% ALGORITHM:
%   1. Find all descendant objects that HAVE the property ('-property' filter).
%   2. Concatenate each object's property value (column-wise) into one vector,
%      skipping any value that is not numeric.
%   3. Drop non-finite entries (NaN/Inf).
%   4. If nothing remains, return NaN/NaN; else return min/max of the pool.
%
% MATH / DECISIONS / EDGE-CASES:
%   - Pooling ACROSS objects means multiple series in one axes all contribute,
%     so the extent covers the whole plotted content, not just one line.
%   - curr(:) flattens matrix-valued data (e.g. multi-column YData) before
%     pooling.
%   - Non-numeric values (e.g. categorical/datetime XData) are skipped, so such
%     axes contribute nothing and may yield NaN/NaN (caller then skips limit
%     setting via its isfinite guards).
%   - Incremental growth of 'vals' is intentional and small here; the
%     %#ok<AGROW> on the original line suppresses the editor's grow-in-loop
%     warning.

objs = findall(ax, '-property', property_name);
vals = [];
for i = 1 : numel(objs)
    curr = get(objs(i), property_name);
    if isnumeric(curr)
        vals = [vals; curr(:)]; %#ok<AGROW>
    end
end
vals = vals(isfinite(vals));
if isempty(vals)
    v_min = NaN;
    v_max = NaN;
else
    v_min = min(vals);
    v_max = max(vals);
end

end


function [y_min, y_max] = padded_axis_limits(y_min, y_max, prefer_zero_for_positive_raw)
% ---------------------------------------------------------------------------
% FUNCTION: padded_axis_limits (local)
% ---------------------------------------------------------------------------
% H1: Expand a [min,max] pair by symmetric 5% padding, with an optional clamp
%     of the lower bound to zero.
%
% PURPOSE:
%   Turn a raw data extent into pleasant axis limits: add breathing room above
%   and below, handle degenerate/empty input gracefully, and (for raw-MAE plots)
%   keep the axis anchored at zero so the non-negative quantity is not shown
%   dipping below zero.
%
% INPUTS:
%   y_min, y_max (numeric scalars, possibly NaN/Inf/empty): the raw lower/upper
%       extent to be padded.
%   prefer_zero_for_positive_raw (optional logical): when true AND the padded
%       lower bound is still > 0, force the lower bound to 0. Defaults to false.
%       The driver passes true only for the 'raw MAE' flavor.
%
% OUTPUTS:
%   y_min, y_max (double scalars): padded (and possibly zero-clamped) limits,
%       guaranteed finite and with y_min < y_max for valid input.
%
% ALGORITHM:
%   1. Default prefer_zero_for_positive_raw to false if not supplied.
%   2. If either input is empty or non-finite, return the fallback [0, 1].
%   3. Compute the pad:
%        - equal endpoints  -> pad = max(1e-6, |y_min| * 0.05)
%        - otherwise        -> pad = 0.05 * (y_max - y_min)
%   4. Subtract pad from y_min and add pad to y_max.
%   5. If clamping is requested and the padded y_min is still > 0, set y_min = 0.
%
% MATH / DECISIONS / EDGE-CASES:
%   - Padding is 5% of the span, applied symmetrically (same absolute pad to
%     both ends).
%   - Degenerate equal-endpoint case: a flat extent would give zero span, so the
%     pad falls back to 5% of |value| but never below 1e-6, guaranteeing a
%     strictly positive, non-zero-width range.
%   - Empty/non-finite guard returns a safe unit interval [0,1] so downstream
%     ylim/xlim calls never receive NaN/Inf.
%   - Clamp-to-zero applies ONLY when the requested flag is set and the padded
%     lower bound is strictly positive; it never raises a negative lower bound
%     up to 0 (so log2 flavors, called with the flag false, keep negative
%     bounds). This is the raw-MAE-only zero anchoring.

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
if prefer_zero_for_positive_raw && y_min > 0
    y_min = 0;
end

end


function write_composite_assumptions(output_dir, panel_dirs, panel_titles)
% ---------------------------------------------------------------------------
% FUNCTION: write_composite_assumptions (local)
% ---------------------------------------------------------------------------
% H1: Write a plain-text file documenting the row/column dataset->panel mapping
%     used to build the composites.
%
% PURPOSE:
%   Emit a provenance/assumptions record so anyone viewing the composites later
%   can tell exactly which dataset sub-folder each panel came from and what its
%   display title was. Captures the layout decisions in a sidecar .txt.
%
% INPUTS:
%   output_dir (char/string): directory where the .txt is written.
%   panel_dirs (3x3 cell of char): per-cell dataset sub-folder names.
%   panel_titles (3x3 cell of char): per-cell display titles (parallel to
%       panel_dirs).
%
% OUTPUTS:
%   none (side effect: writes 'revision_transverse_dist_comparison_order.txt'
%   into output_dir). Returns silently without writing if the file cannot be
%   opened.
%
% ALGORITHM:
%   1. Open output_dir/'revision_transverse_dist_comparison_order.txt' for
%      writing ('w'); if fopen fails (fid < 0), return immediately.
%   2. Register an onCleanup to fclose the file no matter how the function exits.
%   3. Write a header plus an explanatory line stating each output is a 3x3
%      panel built from independently saved single-dataset .fig files.
%   4. For each row r, print 'Row r: ' then, for each column c, print
%      'title (dir)' separated by ' | ', and end the line.
%
% MATH / DECISIONS / EDGE-CASES:
%   - The onCleanup guard guarantees the file handle is closed even on an error
%     mid-write, preventing a leaked/locked file.
%   - The ' | ' separator is emitted only BETWEEN columns (not after the last),
%     via the c < ncols check.
%   - Sizes are read from panel_dirs (assumed to match panel_titles); a mismatch
%     is not defended against here.
%   - 'w' mode truncates/overwrites any existing order file so the record always
%     reflects the current run.

fid = fopen(fullfile(output_dir, 'revision_transverse_dist_comparison_order.txt'), 'w');
if fid < 0
    return;
end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, 'Revision transverse-distance comparison figure order\n');
fprintf(fid, 'Each output figure is a 3x3 panel made from the independently saved single-dataset MAE-vs-transverse-distance .fig files.\n');
fprintf(fid, 'Rows/columns:\n');
for r = 1 : size(panel_dirs, 1)
    fprintf(fid, 'Row %d: ', r);
    for c = 1 : size(panel_dirs, 2)
        fprintf(fid, '%s (%s)', panel_titles{r,c}, panel_dirs{r,c});
        if c < size(panel_dirs, 2)
            fprintf(fid, ' | ');
        end
    end
    fprintf(fid, '\n');
end

end
