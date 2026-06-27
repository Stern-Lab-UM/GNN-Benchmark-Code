function y_limits = GNNBenchmark_calibrate_saved_fig_y_ranges(figures_root, datasets, figure_filenames, save_png)
% GNNBenchmark_calibrate_saved_fig_y_ranges  Implement GNN Benchmark calibrate saved fig y ranges for this MATLAB workflow.
% Inputs: figures_root, datasets, figure_filenames, save_png
% Outputs: y_limits
%GNNBenchmark_CALIBRATE_SAVED_FIG_Y_RANGES
% Match y-limits across corresponding saved .fig files.
%
% Example:
%   GNNBenchmark_calibrate_saved_fig_y_ranges(figures_root, datasets)
%
% NOTE: the H1 example above is incomplete -- the function signature takes
% four inputs, not two. The fuller corrected calling forms are:
%   y_limits = GNNBenchmark_calibrate_saved_fig_y_ranges(figures_root, datasets)
%   y_limits = GNNBenchmark_calibrate_saved_fig_y_ranges(figures_root, datasets, figure_filenames)
%   y_limits = GNNBenchmark_calibrate_saved_fig_y_ranges(figures_root, datasets, figure_filenames, save_png)
% where FIGURE_FILENAMES and SAVE_PNG are optional (see INPUTS below) and the
% returned Y_LIMITS struct captures the unified y-axis range chosen per figure.
%
% PURPOSE:
%   Harmonize ("calibrate") the y-axis limits of a family of previously saved
%   MATLAB .fig files so that the SAME plot, generated separately for several
%   datasets, shares one identical y-axis range. This makes per-dataset panels
%   visually comparable (e.g. for a multi-panel figure in the GNN Benchmark
%   manuscript). Figures are grouped by their EXACT filename, found in the
%   per-dataset subfolders under FIGURES_ROOT; for each group a common
%   [y_min, y_max] is computed from the data, the .fig files are re-opened,
%   their data axes are forced to that range, and each .fig is OVERWRITTEN in
%   place (optionally also exported to a 300-dpi PNG alongside it).
%
% INPUTS:
%   figures_root      - (char/string) Root directory that contains one
%                       subfolder per dataset. Figure files are looked up at
%                       fullfile(figures_root, datasets{d}, fig_name).
%   datasets          - (cellstr) Names of the dataset subfolders to scan. The
%                       grouping is across these subfolders for each fixed
%                       filename, so a given figure is matched by EXACT name
%                       across every dataset folder.
%   figure_filenames  - (cellstr, optional) Exact .fig filenames to calibrate.
%                       DEFAULTS (when omitted or empty) to these 6 hardcoded
%                       names:
%                         'MAE vs traverse dist (raw MAE).fig'
%                         'MAE vs traverse dist (log2 MAE).fig'
%                         'MAE vs traverse dist (log2 nMAE).fig'
%                         'MAE vs dataset size (raw MAE).fig'
%                         'MAE vs dataset size (log2 MAE).fig'
%                         'MAE vs dataset size (log2 nMAE).fig'
%   save_png          - (logical, optional) If true, also write a 300-dpi PNG
%                       next to each rewritten .fig (same stem, .png ext).
%                       DEFAULTS to false (when omitted or empty).
%
% OUTPUTS:
%   y_limits          - (struct) One field per successfully calibrated figure.
%                       The field NAME is the figure filename stem (filename
%                       with the trailing '.fig' removed) sanitized into a
%                       valid MATLAB identifier via matlab.lang.makeValidName,
%                       and the field VALUE is the 1x2 [y_min, y_max] applied to
%                       that group. Figures that were skipped (no files found,
%                       or a non-finite / degenerate raw range) produce NO field.
%
% ALGORITHM:
%   For each requested filename FIG_NAME:
%     1. Initialize y_min = +inf, y_max = -inf, and an empty list of the
%        actually-existing file paths for this group.
%     2. Loop over DATASETS; skip any subfolder lacking FIG_NAME. For each
%        existing .fig: open it invisibly, take only its data axes (see
%        DATA_AXES, which drops legend/colorbar pseudo-axes), and expand
%        [y_min, y_max] by the finite YData extent of every plotted object
%        (see AXIS_DATA_EXTENT). Close the figure.
%     3. If no files existed, or the accumulated range is non-finite, or it is
%        degenerate (y_min == y_max), SKIP this group (no rewrite, no field).
%     4. Otherwise pad the range (see PADDED_AXIS_LIMITS). The third argument is
%        contains(FIG_NAME, 'raw MAE'), so only raw-MAE figures get their lower
%        bound clamped to 0.
%     5. Record the padded range under the sanitized stem field of Y_LIMITS.
%     6. Re-open every existing .fig invisibly, set ylim on each data axis to
%        the common range, OVERWRITE the .fig via GNNBenchmark_SAVEFIG_VISIBLE, and if
%        SAVE_PNG also export a 300-dpi PNG. Close each figure.
%     7. Print a one-line summary of the chosen range and group size.
%
% MATH / EDGE-CASES:
%   - The common range is the elementwise min/max over EVERY data-axis object's
%     finite YData across the whole group (all datasets combined), not per file.
%   - Only finite (yl_min, yl_max) pairs from a given axis contribute; a fully
%     non-finite axis is ignored.
%   - Groups with zero matching files, non-finite totals, or an exactly
%     degenerate range (y_min == y_max BEFORE padding) are skipped entirely.
%   - Padding is symmetric 5% (with degenerate/non-finite fallbacks handled in
%     PADDED_AXIS_LIMITS); the raw-MAE lower-bound-to-zero clamp is applied
%     after padding.
%   - .fig files are modified IN PLACE (destructive overwrite); keep backups if
%     the originals must be preserved.

if nargin < 3 || isempty(figure_filenames)
    figure_filenames = {
        'MAE vs traverse dist (raw MAE).fig'
        'MAE vs traverse dist (log2 MAE).fig'
        'MAE vs traverse dist (log2 nMAE).fig'
        'MAE vs dataset size (raw MAE).fig'
        'MAE vs dataset size (log2 MAE).fig'
        'MAE vs dataset size (log2 nMAE).fig'
        };
end
if nargin < 4 || isempty(save_png)
    save_png = false;
end

y_limits = struct();

for f = 1 : numel(figure_filenames)
    fig_name = figure_filenames{f};
    y_min = inf;
    y_max = -inf;
    existing_paths = {};

    for d = 1 : numel(datasets)
        fig_path = fullfile(figures_root, datasets{d}, fig_name);
        if ~isfile(fig_path)
            continue;
        end
        existing_paths{end+1,1} = fig_path; %#ok<AGROW>
        fig = openfig(fig_path, 'invisible');
        ax = data_axes(fig);
        for a = 1 : numel(ax)
            [yl_min, yl_max] = axis_data_extent(ax(a), 'YData');
            if all(isfinite([yl_min, yl_max]))
                y_min = min(y_min, yl_min);
                y_max = max(y_max, yl_max);
            end
        end
        close(fig);
    end

    if isempty(existing_paths) || ~isfinite(y_min) || ~isfinite(y_max) || y_min == y_max
        continue;
    end
    [y_min, y_max] = padded_axis_limits(y_min, y_max, contains(fig_name, 'raw MAE'));

    field_name = matlab.lang.makeValidName(regexprep(fig_name, '\.fig$', ''));
    y_limits.(field_name) = [y_min, y_max];

    for p = 1 : numel(existing_paths)
        fig_path = existing_paths{p};
        fig = openfig(fig_path, 'invisible');
        ax = data_axes(fig);
        for a = 1 : numel(ax)
            ylim(ax(a), [y_min, y_max]);
        end
        GNNBenchmark_savefig_visible(fig, fig_path);
        if save_png
            png_path = regexprep(fig_path, '\.fig$', '.png');
            try
                exportgraphics(fig, png_path, 'Resolution', 300);
            catch ME
                warning('GNNBenchmark:exportGraphicsFailed', 'Could not export %s: %s', png_path, ME.message);
            end
        end
        close(fig);
    end

    fprintf('[GNNBenchmark_calibrate_saved_fig_y_ranges] %s -> y=[%.6g %.6g] across %d figures.\n', ...
        fig_name, y_min, y_max, numel(existing_paths));
end

end


function GNNBenchmark_savefig_visible(fig_handle, filename)
% GNNBenchmark_savefig_visible  Save MATLAB figures in the publication output format.
% Inputs: fig_handle, filename
% Outputs: none; performs side effects or updates the caller workflow.
%GNNBenchmark_SAVEFIG_VISIBLE  Save a figure to a .fig with a readable window title, visible.
%
% PURPOSE:
%   Persist FIG_HANDLE back to FILENAME (a .fig path) after first making the
%   figure visible and giving it a human-readable window Name derived from its
%   on-disk location. The calling code opens figures invisibly for processing;
%   this helper flips Visible back on so the re-saved .fig opens visible later,
%   and stamps a "dataset | figure-stem" title for easy identification.
%
% INPUTS:
%   fig_handle - Handle to the figure to save.
%   filename   - (char/string) Destination .fig path. Its parent folder name is
%                used as the dataset label and its file stem as the plot label.
%
% OUTPUTS:
%   (none) Side effects only: mutates FIG_HANDLE's Visible/Name/NumberTitle and
%   writes the .fig file via savefig (overwriting FILENAME if it exists).
%
% ALGORITHM:
%   1. Set the figure Visible = 'on'.
%   2. Split FILENAME into [parent_dir, file_stem]; take the leaf of PARENT_DIR
%      as DATASET_LABEL (the dataset subfolder name).
%   3. If DATASET_LABEL is non-empty, set Name = '<dataset> | <stem>' and turn
%      NumberTitle off.
%   4. Call savefig(fig_handle, filename) to write the .fig.
%
% MATH / EDGE-CASES:
%   - If PARENT_DIR has no leaf name (DATASET_LABEL empty), the Name/NumberTitle
%     are left untouched and only the save happens.
%   - FILENAME is wrapped with char() so string scalars are accepted.
%   - savefig overwrites the destination .fig in place.

set(fig_handle, 'Visible', 'on');
[parent_dir, file_stem] = fileparts(char(filename));
[~, dataset_label] = fileparts(parent_dir);
if ~isempty(dataset_label)
    set(fig_handle, 'Name', sprintf('%s | %s', dataset_label, file_stem), ...
        'NumberTitle', 'off');
end
savefig(fig_handle, filename);

end


function [v_min, v_max] = axis_data_extent(ax, property_name)
% axis_data_extent  Implement axis data extent for this MATLAB workflow.
% Inputs: ax, property_name
% Outputs: v_min, v_max
%AXIS_DATA_EXTENT  Finite min/max of a named data property over all objects in an axis.
%
% PURPOSE:
%   Compute the data extent (min and max) of a given coordinate property (here
%   called with 'YData') across every child object of axis AX that exposes that
%   property -- i.e. the actual span of plotted data, used to derive a unified
%   axis range. Non-finite samples are dropped before the min/max.
%
% INPUTS:
%   ax            - Handle to a single axes object to scan.
%   property_name - (char) Name of the numeric data property to aggregate,
%                   e.g. 'YData'. Passed through to findall('-property', ...)
%                   and get(...).
%
% OUTPUTS:
%   v_min - Minimum finite value of PROPERTY_NAME over all matching objects, or
%           NaN if no finite values exist.
%   v_max - Maximum finite value, or NaN if none.
%
% ALGORITHM:
%   1. objs = findall(ax, '-property', property_name) -- every descendant that
%      has the property (lines, scatter, errorbars, etc.).
%   2. Concatenate get(obj, property_name) for each numeric object into a column
%      vector VALS (non-numeric values are ignored).
%   3. Keep only finite entries (drop NaN/Inf).
%   4. If VALS is empty, return [NaN, NaN]; else return [min(VALS), max(VALS)].
%
% MATH / EDGE-CASES:
%   - Returns NaN/NaN when nothing finite is found; callers gate on isfinite
%     before using the result.
%   - Only isnumeric properties contribute; other types are skipped.
%   - CAVEAT (errorbar objects): for errorbar series, YData is the CENTER value
%     ONLY. The whisker extents are stored separately in YNegativeDelta /
%     YPositiveDelta and are NOT read here, so the data extent (and therefore
%     the unified y-range computed by the caller) can be NARROWER than the
%     visible whiskers -- error bars may be clipped at the top/bottom of the
%     calibrated axes.

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
% padded_axis_limits  Implement padded axis limits for this MATLAB workflow.
% Inputs: y_min, y_max, prefer_zero_for_positive_raw
% Outputs: y_min, y_max
%PADDED_AXIS_LIMITS  Expand a [y_min, y_max] data range with symmetric 5% padding.
%
% PURPOSE:
%   Turn a raw data extent into a slightly roomier axis range so plotted data is
%   not flush against the axes box, with robust fallbacks for degenerate or
%   non-finite inputs, and an optional clamp of the lower bound to zero for
%   strictly-positive raw-MAE plots.
%
% INPUTS:
%   y_min                        - Lower data bound (scalar).
%   y_max                        - Upper data bound (scalar).
%   prefer_zero_for_positive_raw - (logical, optional) When true AND the padded
%                                  lower bound is still > 0, force y_min = 0.
%                                  The caller passes contains(fig_name,'raw MAE')
%                                  here, so ONLY raw-MAE figures anchor at zero.
%                                  DEFAULTS to false when omitted.
%
% OUTPUTS:
%   y_min - Padded (and possibly zero-clamped) lower limit.
%   y_max - Padded upper limit.
%
% ALGORITHM:
%   1. If either bound is empty or non-finite, return the safe fallback [0, 1].
%   2. Choose the pad:
%        - degenerate range (y_min == y_max): pad = max(1e-6, abs(y_min)*0.05)
%        - otherwise:                          pad = 0.05 * (y_max - y_min)
%   3. Apply symmetrically: y_min = y_min - pad, y_max = y_max + pad.
%   4. If PREFER_ZERO_FOR_POSITIVE_RAW and y_min > 0, set y_min = 0.
%
% MATH / EDGE-CASES:
%   - Non-finite / empty input -> hard-coded [0, 1] (prevents invalid ylim).
%   - Degenerate input (zero-width range) -> a small floor pad of 1e-6 guards
%     the case y_min == 0 (where abs(y_min)*0.05 would also be 0), guaranteeing a
%     strictly positive, non-empty range.
%   - The padding is symmetric (same absolute pad added to each side) and 5% of
%     the range width in the normal case.
%   - The zero-clamp only lowers the bottom (never raises it); it is applied
%     AFTER padding and only when the padded bottom is still above zero.

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


function ax = data_axes(fig)
% data_axes  Implement data axes for this MATLAB workflow.
% Inputs: fig
% Outputs: ax
%DATA_AXES  Return the real plotting axes of a figure, excluding legend/colorbar.
%
% PURPOSE:
%   Collect only the genuine data-plotting axes of FIG, filtering out the
%   pseudo-axes that MATLAB also reports as Type 'axes' but which carry no
%   plottable data of interest -- specifically legends and colorbars. Used so
%   the y-range calibration reads/sets limits on actual data axes only.
%
% INPUTS:
%   fig - Handle to the figure to inspect.
%
% OUTPUTS:
%   ax  - Column/array of axes handles that are NOT tagged 'legend' or
%         'Colorbar'. May be empty if the figure has no axes.
%
% ALGORITHM:
%   1. ax = findall(fig, 'Type', 'axes') -- all axes-typed handles.
%   2. If none, return empty immediately.
%   3. Build a logical KEEP mask, dropping any axis whose Tag is a member of
%      {'legend', 'Colorbar'}.
%   4. Return ax(keep).
%
% MATH / EDGE-CASES:
%   - Exclusion is by Tag string equality only ('legend', 'Colorbar'); other
%     decorative axes types (if any) are NOT filtered.
%   - Tag matching is case-sensitive via ismember (note the lowercase 'legend'
%     vs capitalized 'Colorbar', matching MATLAB's default tags).
%   - Returns the empty set unchanged when the figure contains no axes.

ax = findall(fig, 'Type', 'axes');
if isempty(ax)
    return;
end

keep = true(size(ax));
for i = 1 : numel(ax)
    tag = get(ax(i), 'Tag');
    if ismember(tag, {'legend', 'Colorbar'})
        keep(i) = false;
    end
end
ax = ax(keep);

end
