% =========================================================================
% FILE HEADER
% -------------------------------------------------------------------------
%   File    : GNNBenchmark_verify_Flip_two_figures.m
%   Project : GNN Benchmark analysis pipeline.
%   Type    : Structural (visual) quality-assurance utility for saved
%             MATLAB figures produced by the "Flip_two" analysis.
%
%   SCOPE / WHAT THIS FILE IS *NOT*
%     This is a STRUCTURAL / VISUAL QA tool ONLY. It inspects the on-disk
%     *.fig files for the presence and basic sanity of graphical elements
%     (axes, lines, labels, axis limits, colors, legends/colorbars, and the
%     expected PPGN text/legend entries). It does NOT recompute, re-derive, or validate
%     any scientific quantities, statistics, or model outputs, and it does
%     NOT emit any overall PASS/FAIL verdict. Each check writes its result
%     into a per-figure row of a report table; interpretation is left to
%     the human reviewer reading that table / the emitted CSV.
%
%   NOTE ON THIS DOCUMENTATION
%     The blocks added here are COMMENT-ONLY. No executable line, name,
%     or whitespace of the original code has been changed.
% =========================================================================
function report = GNNBenchmark_verify_Flip_two_figures(figures_output_dir)
% GNNBenchmark_verify_Flip_two_figures  Implement GNN Benchmark verify flip two figures for this MATLAB workflow.
% Inputs: figures_output_dir
% Outputs: report
%GNNBenchmark_VERIFY_FLIP_TWO_FIGURES Structural QA for Flip_two figures.
%
% PURPOSE
%   Perform automated STRUCTURAL / VISUAL quality-assurance on the set of
%   MATLAB figure files (*.fig) saved by the "Flip_two" two-flip analysis.
%   For every figure found in a directory, the function opens it (without
%   displaying it), examines its graphics objects, and records a row of
%   yes/no structural findings plus free-text notes. All rows are returned
%   as a table and also written to a CSV report. The routine never
%   recomputes scientific numbers and never returns an aggregate PASS/FAIL;
%   it is purely a presence/sanity audit of the rendered figures.
%
% USAGE
%   report = GNNBenchmark_verify_Flip_two_figures()
%   report = GNNBenchmark_verify_Flip_two_figures(figures_output_dir)
%
% INPUT
%   figures_output_dir  (optional) char/string path to the directory that
%       contains the *.fig files to audit, and into which the CSV report
%       is written. If omitted or empty (nargin < 1 || isempty(...)), it
%       defaults to the configured Flip_two dataset figure folder:
%           fullfile(cfg.data_root, '_figures', 'revision_2026', 'Flip_two')
%       Only files matching '*.fig' directly in this directory are scanned
%       (dir() is non-recursive here, so sub-folders are not traversed).
%
% OUTPUT
%   report  A MATLAB table with one row per *.fig file and exactly the
%       following 10 columns (name : type -> meaning):
%         1)  file                     : string  -> figure file name (no path).
%         2)  n_axes                   : double  -> number of *data* axes kept
%                                                   (legend/colorbar axes are
%                                                   excluded; see ALGORITHM).
%         3)  n_lines                  : double  -> number of line objects in
%                                                   the whole figure.
%         4)  n_images                 : double  -> number of image objects in
%                                                   the whole figure.
%         5)  has_data                 : logical -> true if the figure contains
%                                                   at least one data-bearing
%                                                   graphic (line OR patch OR
%                                                   image OR bar OR errorbar);
%                                                   i.e. (count of those) > 0.
%         6)  has_labels               : logical -> true only if EVERY kept data
%                                                   axis has a non-empty X label
%                                                   AND a non-empty Y label.
%         7)  has_legend_or_colorbar   : logical -> true if the figure has at
%                                                   least one Legend OR ColorBar.
%         8)  ppgn_expected            : logical -> true if some text/String
%                                                   object contains "ppgn"
%                                                   (case-insensitive), as
%                                                   expected for current Flip_two
%                                                   plots.
%         9)  limits_cover_data        : logical -> true if, for every kept data
%                                                   axis, all finite YData of all
%                                                   objects lie within that axis'
%                                                   y-limits (within tolerance).
%                                                   ONLY the Y axis is checked.
%        10)  notes                    : string  -> free-text annotations; here
%                                                   used only to flag line colors
%                                                   that do not match the expected
%                                                   palette (never causes a fail).
%
% ALGORITHM (per figure, in order)
%   1) Enumerate '*.fig' files in figures_output_dir via dir().
%   2) Define an expected color palette (expected_colors): a 6x3 RGB matrix
%      whose rows correspond to PPGN, GraphSAGE, GAT, GIN, PNA, and the
%      Baseline/error-bar color (black). Used only for the color check.
%   3) Pre-allocate an empty 0x10 table 'rows' with the schema above.
%   4) For each figure file:
%        a) openfig(path,'invisible') so nothing is displayed; register an
%           onCleanup that closes the figure when the loop iteration ends
%           (guarantees the handle is released even on error).
%        b) Find all 'axes', then DROP any whose Tag (lower-cased) contains
%           "legend" or "colorbar", leaving only true data axes (data_axes).
%        c) Collect handles by type across the whole figure: line, patch,
%           image, bar, errorbar, Legend, ColorBar.
%        d) has_data := (#line + #patch + #image + #bar + #errorbar) > 0.
%        e) For each kept data axis:
%             - has_labels := false if its XLabel.String OR YLabel.String is
%               empty (otherwise it stays true).
%             - Read y_lim = ylim(ax). For every object in the axis that has
%               a 'YData' property, take its finite YData and test whether
%               min(y) or max(y) falls outside [y_lim(1), y_lim(2)] by more
%               than 'tol'; if so, limits_cover_data := false.
%        f) PPGN inclusion screen: scan every object that has a 'String'
%           property; if any string contains "ppgn" (case-insensitive), set
%           ppgn_expected := true.
%        g) Color check: for each line, read its 'Color'; if it is a 1x3 RGB
%           triple, compute the Euclidean distance to each expected palette
%           row and, if the nearest distance exceeds the color tolerance,
%           append an "unexpected line color [r g b]" string to color_notes.
%           The 'notes' column becomes "" if no anomalies, else the unique
%           anomaly strings joined by "; ". This NEVER fails a figure; it is
%           advisory only.
%        h) Append one row (file name, counts, and the booleans/notes above)
%           to 'rows'.
%   5) report := rows. Write it to '<figures_output_dir>\flip_two_visual_QA_report.csv'
%      via writetable(), then disp() the table to the console.
%
% CHECKS AND TOLERANCES (summary; faithful to the code)
%   * has_data        : presence test only -> count of
%                       {line,patch,image,bar,errorbar} > 0.
%   * has_labels      : both X and Y axis labels must be non-empty on EVERY
%                       kept data axis (a single empty label flips it false).
%   * limits_cover_data:
%                       Y-AXIS ONLY. For each kept axis, with y_lim = ylim(ax),
%                       the per-axis tolerance is
%                           tol = max(1e-10, 1e-8 * max(1, max(abs(y_lim)))).
%                       A figure fails this check if any finite YData value is
%                       below y_lim(1) - tol or above y_lim(2) + tol. The X
%                       axis and X limits are NOT examined.
%   * ppgn_expected   : case-insensitive substring search for "ppgn" over all
%                       objects exposing a 'String' property; a single hit
%                       sets the flag true. Current Flip_two plots include PPGN.
%   * line color check: Euclidean RGB distance from each line's Color to the
%                       expected palette; if the minimum distance > 1e-4 the
%                       color is reported in 'notes'. ADVISORY ONLY -- it never
%                       changes any boolean and never fails the figure.
%
% SIDE EFFECTS
%   * Writes 'flip_two_visual_QA_report.csv' into figures_output_dir
%     (overwriting any existing file of that name).
%   * Prints the report table to the command window via disp().
%   * Opens each figure invisibly and closes it again (onCleanup); no
%     figure windows are left visible.
%
% RETURNS NO VERDICT
%   There is deliberately no scalar/overall PASS or FAIL output. Downstream
%   consumers must inspect the per-figure boolean columns (and 'notes') to
%   decide whether a figure is acceptable.

if nargin < 1 || isempty(figures_output_dir)
    path_cfg = GNNBenchmark_publication_config();
    if isempty(path_cfg.data_root)
        error('GNNBenchmark:missingDataRoot', ['Pass figures_output_dir explicitly or set ', ...
            'GNN_BENCHMARK_DATA_ROOT / GNNBenchmark_local_config.m.']);
    end
    path_layout = GNNBenchmark_data_package_paths(path_cfg.data_root);
    figures_output_dir = fullfile(path_layout.revision_figures_root, 'Flip_two');
end
fig_files = dir(fullfile(figures_output_dir, '*.fig'));
expected_colors = [
    0.0000, 0.4470, 0.7410  % PPGN
    0.8500, 0.3250, 0.0980  % GraphSAGE
    0.4660, 0.6740, 0.1880  % GAT
    0.4940, 0.1840, 0.5560  % GIN
    0.9290, 0.6940, 0.1250  % PNA
    0.0000, 0.0000, 0.0000  % Baseline/error bars
    ];

rows = table('Size', [0, 10], ...
    'VariableTypes', {'string','double','double','double','logical','logical','logical','logical','logical','string'}, ...
    'VariableNames', {'file','n_axes','n_lines','n_images','has_data','has_labels','has_legend_or_colorbar','ppgn_expected','limits_cover_data','notes'});

for i = 1 : numel(fig_files)
    fig_path = fullfile(fig_files(i).folder, fig_files(i).name);
    fig = openfig(fig_path, 'invisible');
    cleaner = onCleanup(@() close(fig));

    data_axes = findall(fig, 'Type', 'axes');
    keep = true(size(data_axes));
    for a = 1 : numel(data_axes)
        tag = string(get(data_axes(a), 'Tag'));
        keep(a) = ~(contains(lower(tag), "legend") || contains(lower(tag), "colorbar"));
    end
    data_axes = data_axes(keep);

    lines = findall(fig, 'Type', 'line');
    patches = findall(fig, 'Type', 'patch');
    images = findall(fig, 'Type', 'image');
    bars = findall(fig, 'Type', 'bar');
    err_bars = findall(fig, 'Type', 'errorbar');
    legends = findall(fig, 'Type', 'Legend');
    colorbars = findall(fig, 'Type', 'ColorBar');

    has_data = numel(lines) + numel(patches) + numel(images) + numel(bars) + numel(err_bars) > 0;
    has_labels = true;
    limits_cover_data = true;
    for a = 1 : numel(data_axes)
        ax = data_axes(a);
        if isempty(string(ax.XLabel.String)) || isempty(string(ax.YLabel.String))
            has_labels = false;
        end
        y_lim = ylim(ax);
        y_objs = findall(ax, '-property', 'YData');
        for yy = 1 : numel(y_objs)
            y = get(y_objs(yy), 'YData');
            if isnumeric(y)
                y = y(isfinite(y));
                if ~isempty(y)
                    tol = max(1e-10, 1e-8 * max(1, max(abs(y_lim))));
                    if min(y) < y_lim(1) - tol || max(y) > y_lim(2) + tol
                        limits_cover_data = false;
                    end
                end
            end
        end
    end

    ppgn_expected = false;
    text_objs = findall(fig, '-property', 'String');
    for t = 1 : numel(text_objs)
        txt = string(get(text_objs(t), 'String'));
        if any(contains(lower(txt(:)), "ppgn"))
            ppgn_expected = true;
        end
    end

    color_notes = strings(0);
    for l = 1 : numel(lines)
        c = get(lines(l), 'Color');
        if isnumeric(c) && numel(c) == 3
            d = sqrt(sum((expected_colors - c).^2, 2));
            if min(d) > 1e-4
                color_notes(end+1) = sprintf('unexpected line color [%0.3f %0.3f %0.3f]', c); %#ok<AGROW>
            end
        end
    end

    if isempty(color_notes)
        notes = "";
    else
        notes = strjoin(unique(color_notes), '; ');
    end

    rows = [rows; {string(fig_files(i).name), numel(data_axes), numel(lines), numel(images), ...
        has_data, has_labels, ~isempty(legends) || ~isempty(colorbars), ppgn_expected, limits_cover_data, notes}]; %#ok<AGROW>
end

report = rows;
writetable(report, fullfile(figures_output_dir, 'flip_two_visual_QA_report.csv'));
disp(report);

end
