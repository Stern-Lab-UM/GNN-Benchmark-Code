function [graph_names, graphs_from_Matej_before_prediction, file_header, graph_id, vals] = load_dataset(filename, consider_nodes)
% load_dataset  Load dataset from disk or cache.
% Inputs: filename, consider_nodes
% Outputs: graph_names, graphs_from_Matej_before_prediction, file_header, graph_id, vals
%LOAD_DATASET  Read a merged tissue-graph dataset file into structured arrays.
%
%   [GRAPH_NAMES, GRAPHS, FILE_HEADER, GRAPH_ID, VALS] = LOAD_DATASET(FILENAME)
%   parses one of the all_2D_graphs*.txt dataset files produced by the
%   GNN Benchmark pipeline. Every graph in the file is laid out as:
%
%       Simulation id: graph_<n_cells>_<repeat>_<disorder>[...] .txt
%       <edge row 1>
%       <edge row 2>
%       ...
%       [optional node rows]
%
%   preceded by a single top-level FILE_HEADER (the 'Total graphs: ...'
%   block that ends at the first 'Simulation').
%
%   Outputs
%   -------
%     GRAPH_NAMES
%       Cell array (n_graphs x 1) of basenames, e.g. 'graph_16_3_20_...'.
%
%     GRAPHS_FROM_MATEJ_BEFORE_PREDICTION
%       Cell array (n_graphs x 1) of the raw per-graph text blocks
%       (edge rows, and node rows when present). The historical name
%       is preserved because callers depend on it by position.
%
%     FILE_HEADER
%       Char array: the 'Total graphs: ...' preamble (without the
%       trailing 'Simulation' sentinel).
%
%     GRAPH_ID
%       Struct of per-graph metadata parsed from GRAPH_NAMES. Fields
%       (n_cells, repeat, disorder, shear, flipped_interface_1,
%       flipped_interface_2) are column vectors of length n_graphs.
%       For the legacy 3-token naming scheme, shear / flipped_* are
%       filled with NaN.
%
%     VALS
%       Cell array (n_graphs x 1). Each entry is a numeric matrix
%       (n_rows x n_cols) obtained by reading the corresponding graph
%       block and reshaping based on the column count detected in the
%       second edge row (the first row after the two header-style
%       newlines).
%
%   Inputs
%   ------
%     FILENAME
%       Absolute path to the dataset file.
%
%     CONSIDER_NODES  (default 0)
%       Retained for backwards compatibility with older datasets that
%       mixed edge and node rows in the same block. When set, the
%       per-graph block is truncated just before the first row that
%       looks like a node row (the 9-column numeric pattern), so
%       downstream reshapes see only edge rows. Leave at 0 for the
%       current 2D pipeline, which keeps node rows in their own block
%       after a blank line.

if nargin < 2
    consider_nodes = 0;
end

% Slurp the whole file as a single character stream -- simpler than
% line-by-line because the regexes work directly on the buffer.
fid = fopen(filename, 'rt');
t = fread(fid, inf, '*char')';
fclose(fid);

% Extract the per-graph basename from each 'Simulation id: ...' line.
% v2 (2026-05-15): allow dots inside the captured stem so revision datasets
% with float-valued tokens (e.g. graph_256_10_0_-1_1_1.2.txt for Shear 1.2)
% match too. Non-greedy quantifier and an escaped trailing `\.txt` prevent
% the dot in `.txt` from being absorbed into the stem.
graph_names = regexp(t, 'Simulation id: (graph[\d\_\-\.]+?)\.txt', 'tokens');
graph_names = [graph_names{:}]';

% Split each basename into its numeric tokens. The original 'new' scheme
% emitted 6 tokens (n_cells, repeat, disorder, shear, flipped_1, flipped_2).
% Revision datasets (Shear N/M -> 1.N) split the trailing float into two
% extra tokens, giving 7. Legacy datasets had only 3.
tok = regexp(graph_names, '([\d\-]+)', 'tokens');

if length(tok{1}) == 6

    % New naming scheme.
    tok = [tok{:}]';
    tok = reshape([tok{:}]', 6, length(tok) / 6)';
    tok = cellfun(@str2num, tok);
    graph_id.n_cells = tok(:,1);
    graph_id.repeat = tok(:,2);
    graph_id.disorder = tok(:,3);
    graph_id.shear = tok(:,4);
    graph_id.flipped_interface_1 = tok(:,5);
    graph_id.flipped_interface_2 = tok(:,6);

elseif length(tok{1}) == 7

    % v2 revision-style naming with a 7th trailing-fraction token, e.g.
    % graph_256_10_0_-1_1_1.2 -> tokens {256, 10, 0, -1, 1, 1, 2}. We fold
    % the last two tokens back into a fractional shear/parameter (1 + 2/10
    % -> 1.2) so the existing 6-field graph_id schema is preserved.
    tok = [tok{:}]';
    tok = reshape([tok{:}]', 7, length(tok) / 7)';
    tok = cellfun(@str2num, tok);
    % Combine columns 6 (whole) and 7 (fraction). 7's magnitude is the
    % shortest decimal that gets us back the original (e.g. '2' -> 0.2,
    % '15' -> 0.15). We detect that by string-length of the column.
    raw7 = arrayfun(@(v) num2str(v), tok(:,7), 'UniformOutput', false);
    digits7 = cellfun(@length, raw7);
    frac = tok(:,7) ./ (10 .^ digits7);
    last_real = sign(tok(:,6)) .* (abs(tok(:,6)) + frac);
    graph_id.n_cells = tok(:,1);
    graph_id.repeat = tok(:,2);
    graph_id.disorder = tok(:,3);
    graph_id.shear = tok(:,4);
    graph_id.flipped_interface_1 = tok(:,5);
    graph_id.flipped_interface_2 = last_real;

else

    % Legacy naming scheme: fewer tokens, no shear / flipped-interface
    % info. Pad the missing fields with NaN so downstream code can
    % always index GRAPH_ID.shear etc. without a struct-field check.
    tok_in_line = length(tok{1});
    tok = [tok{:}]';
    tok = reshape([tok{:}]', tok_in_line, length(tok) / tok_in_line)';
    tok = cellfun(@str2num, tok);
    graph_id.n_cells = tok(:,1);
    graph_id.repeat = tok(:,2);
    graph_id.disorder = tok(:,3);
    graph_id.shear = nan(size(tok,1),1);
    graph_id.flipped_interface_1 = nan(size(tok,1),1);
    graph_id.flipped_interface_2 = nan(size(tok,1),1);

end

% The top-level header is everything before the first 'Simulation'
% token -- lazy regex on '.' (with 'Total graphs' as the anchor) gives
% exactly that block.
file_header = regexp(t, '(Total graphs(?:(?!Simulation).)+?)Simulation', 'tokens');
file_header = file_header{1}{1};

% Pull every run of numeric / whitespace characters. This is
% deliberately coarse so it catches both edge rows and node rows; the
% lens-filter below drops the short matches that are actually labels
% or counts from the header.
graphs_from_Matej_before_prediction = regexp(t, '([\d\.\s\-e]+)', 'tokens');
graphs_from_Matej_before_prediction = [graphs_from_Matej_before_prediction{:}]';
lens = cellfun(@length, graphs_from_Matej_before_prediction);
graphs_from_Matej_before_prediction(lens < 100) = [];

% Legacy datasets that interleaved edge and node rows in the same
% block need the node portion stripped so reshape() can infer the
% column count from the edge rows alone. The regex anchors on the
% 9-column numeric shape of a node row and truncates there.
if consider_nodes
    node_row_pattern = '\d+ \d+ \d+ \d+ \d+ [\d\.\-\e\E]+ [\d\.\-\e\E]+ [\d\.\-\e\E]+ [\d\.\-\e\E]+ ';
    for block_idx = 1 : numel(graphs_from_Matej_before_prediction)
        node_start = regexp(graphs_from_Matej_before_prediction{block_idx}, node_row_pattern, 'once');
        if ~isempty(node_start) && node_start > 1
            graphs_from_Matej_before_prediction{block_idx} = ...
                graphs_from_Matej_before_prediction{block_idx}(1:node_start-1);
        end
    end
end

% Parse each graph block into a numeric matrix. The column count is
% detected from the second edge row (between the 2nd and 3rd newline
% -- the first newline ends the graph header, the second ends the
% first edge row). sscanf is used because it tolerates scientific
% notation in the values.
vals = cell(length(graphs_from_Matej_before_prediction),1);
for i = 1 : length(graphs_from_Matej_before_prediction)
    newlines = find(graphs_from_Matej_before_prediction{i} == newline, 3, 'first');
    if numel(newlines) < 3
        error('load_dataset:malformedGraphBlock', ...
            'Graph block %d in %s has fewer than three newline-delimited numeric rows.', ...
            i, filename);
    end
    n_vals_per_line = nnz(graphs_from_Matej_before_prediction{i}(newlines(2)+1:newlines(3)-1) == ' ')+1;
    vals{i} = sscanf(graphs_from_Matej_before_prediction{i}, '%f');
    % v2 (2026-05-15): tolerate trailing junk in the last graph block. Some
    % prediction files have 1-2 extra numbers at the end of the very last
    % graph (data-quality artifact in a small number of files). Truncate to
    % the nearest multiple of n_vals_per_line before reshape so we don't
    % crash with "Size arguments must be real integers".
    n_complete_rows = floor(length(vals{i}) / n_vals_per_line);
    n_used = n_complete_rows * n_vals_per_line;
    if n_used < length(vals{i})
        warning('load_dataset_v2:truncated', ...
            'Graph %d: dropping %d trailing numbers (not a full row of %d)', ...
            i, length(vals{i}) - n_used, n_vals_per_line);
    end
    vals{i} = reshape(vals{i}(1:n_used), n_vals_per_line, n_complete_rows)';
end
