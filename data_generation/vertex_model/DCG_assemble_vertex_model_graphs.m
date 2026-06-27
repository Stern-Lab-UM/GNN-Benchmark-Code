function report = DCG_assemble_vertex_model_graphs(raw_output_dir, rows, output_dir, dataset_key)
%DCG_ASSEMBLE_VERTEX_MODEL_GRAPHS  Convert raw vertex-model graphs to ML files.
%
%   REPORT = DCG_ASSEMBLE_VERTEX_MODEL_GRAPHS(RAW_OUTPUT_DIR, ROWS,
%   OUTPUT_DIR, DATASET_KEY) reads raw graph_*.txt files produced by the
%   vertex-model simulator and writes two model-ready files:
%
%       <dataset_key>_weighted.txt
%       <dataset_key>_unweighted.txt
%
%   The raw simulator emits one undirected row per post-relaxation interface:
%
%       cell1 cell2 was_flipped pre_T1_length post_T1_length
%
%   For rows marked was_flipped=1, the row corresponds to the removed T1
%   interface.  The manuscript datasets represent that removed interface as a
%   target length of zero, and add the newly created interface as an extra
%   undirected edge with input length zero and target equal to the raw row's
%   post_T1_length.  The new interface is inferred as the two cells that are
%   common post-T1 neighbors of the removed edge's two cells.
%
%   Weighted output columns:
%
%       cell_id_1 cell_id_2 input_preferred_length input_was_flipped output_preferred_length
%
%   Unweighted output columns:
%
%       cell_id_1 cell_id_2 output_preferred_length
%
%   Both outputs are directed/symmetric: each undirected interface is written
%   once per direction and then sorted by (cell_id_1, cell_id_2), matching the
%   archived manuscript data format and both model loaders.

if ~isfolder(raw_output_dir)
    error('raw_output_dir does not exist: %s', raw_output_dir);
end
if ~isfolder(output_dir)
    mkdir(output_dir);
end

n_graphs = height(rows);
weighted_path = fullfile(output_dir, [dataset_key, '_weighted.txt']);
unweighted_path = fullfile(output_dir, [dataset_key, '_unweighted.txt']);

fw = fopen(weighted_path, 'wt');
if fw < 0, error('Could not open %s for writing', weighted_path); end
cu = onCleanup(@() fclose(fw)); %#ok<NASGU>

fu = fopen(unweighted_path, 'wt');
if fu < 0, error('Could not open %s for writing', unweighted_path); end
cu2 = onCleanup(@() fclose(fu)); %#ok<NASGU>

report = table('Size', [n_graphs, 7], ...
    'VariableTypes', {'string','string','double','double','double','double','double'}, ...
    'VariableNames', {'simulation_id','raw_file','n_cells','raw_edges','flipped_edges','assembled_edges','directed_rows'});

for g = 1:n_graphs
    row = rows(g, :);
    [raw_name, raw_path] = resolve_raw_graph_path(raw_output_dir, row);
    raw = dlmread(raw_path); %#ok<DLMRD>
    if size(raw, 2) ~= 5
        error('Raw graph must have five columns: %s', raw_path);
    end

    n_cells = table_number(row, 'n_cells');
    [directed_weighted, directed_unweighted, n_undirected, n_flipped] = assemble_one_graph(raw, raw_path);

    if max(max(directed_weighted(:, 1:2))) > n_cells
        error('Cell id exceeds n_cells in %s', raw_path);
    end

    if g == 1
        write_header(fw, n_graphs, n_cells, n_undirected, true);
        write_header(fu, n_graphs, n_cells, n_undirected, false);
    end

    fprintf(fw, 'Simulation id: %s\n', raw_name);
    fprintf(fu, 'Simulation id: %s\n', raw_name);

    fprintf(fw, '%d %d %.8g %d %.8g\n', directed_weighted');
    fprintf(fu, '%d %d %.8g\n', directed_unweighted');

    if g < n_graphs
        fprintf(fw, '\n');
        fprintf(fu, '\n');
    end

    report.simulation_id(g) = string(raw_name);
    report.raw_file(g) = string(raw_path);
    report.n_cells(g) = n_cells;
    report.raw_edges(g) = size(raw, 1);
    report.flipped_edges(g) = n_flipped;
    report.assembled_edges(g) = n_undirected;
    report.directed_rows(g) = size(directed_weighted, 1);
end

writetable(report, fullfile(output_dir, [dataset_key, '_assembly_report.csv']));

end

function [directed_weighted, directed_unweighted, n_undirected, n_flipped] = assemble_one_graph(raw, raw_path)
flipped = raw(:, 3) ~= 0;
base = [raw(:, 1), raw(:, 2), raw(:, 4), double(flipped), raw(:, 5)];
base(flipped, 5) = 0;

new_edges = zeros(nnz(flipped), 5);
flip_rows = find(flipped);
for i = 1:numel(flip_rows)
    r = flip_rows(i);
    old_a = raw(r, 1);
    old_b = raw(r, 2);
    common = infer_new_edge_cells(raw, old_a, old_b);
    new_edges(i, :) = [common(1), common(2), 0, 0, raw(r, 5)];
end

undirected = [base; new_edges];
n_flipped = nnz(flipped);
n_undirected = size(undirected, 1);

directed_weighted = [undirected; undirected(:, [2, 1, 3, 4, 5])];
directed_weighted = sortrows(directed_weighted, [1, 2]);

directed_unweighted = directed_weighted(:, [1, 2, 5]);

if size(directed_weighted, 1) ~= 2 * n_undirected
    error('Directed row count mismatch while assembling %s', raw_path);
end
end

function common = infer_new_edge_cells(raw, old_a, old_b)
not_flipped = raw(:, 3) == 0;
edges = raw(not_flipped, 1:2);

neigh_a = [edges(edges(:, 1) == old_a, 2); edges(edges(:, 2) == old_a, 1)];
neigh_b = [edges(edges(:, 1) == old_b, 2); edges(edges(:, 2) == old_b, 1)];
common = intersect(neigh_a, neigh_b);
common = sort(common(:))';

if numel(common) ~= 2
    error(['Could not infer the new T1 edge for removed interface (%d,%d): ', ...
           'expected exactly two common post-T1 neighbors, found %d.'], ...
        old_a, old_b, numel(common));
end
end

function write_header(fid, n_graphs, n_cells, n_edges, is_weighted)
n_vertices = 2 * n_cells;
fprintf(fid, 'Total graphs: %d\n\n', n_graphs);
fprintf(fid, '#cells: %d, #vertices: %d, #edges: %d\n\n', n_cells, n_vertices, n_edges);
if is_weighted
    fprintf(fid, 'Format: cell_id_1 cell_id_2 in_preferred_length in_was_flipped out_preferred_length\n\n');
else
    fprintf(fid, 'Format: cell_id_1 cell_id_2 out_preferred_length\n\n');
end
end

function [raw_name, raw_path] = resolve_raw_graph_path(raw_output_dir, row)
current_name = expected_raw_graph_name(row);
current_path = fullfile(raw_output_dir, current_name);
if isfile(current_path)
    raw_name = current_name;
    raw_path = current_path;
    return
end

archived_name = char(string(row.simulation_id));
archived_path = fullfile(raw_output_dir, archived_name);
if isfile(archived_path)
    raw_name = archived_name;
    raw_path = archived_path;
    return
end

error('Could not find raw graph. Tried %s and %s', current_path, archived_path);
end

function name = expected_raw_graph_name(row)
n_cells = table_number(row, 'n_cells');
package_id = table_number(row, 'package_id');
sigma_index = table_number(row, 'sigma_index');
t1_edge_1 = table_number(row, 't1_edge_1');
t1_edge_2 = table_number(row, 't1_edge_2');
shear_factor = table_number(row, 'shear_factor');
name = sprintf('graph_%d_%d_%d_%d_%d_%s.txt', ...
    n_cells, package_id, sigma_index, t1_edge_1, t1_edge_2, c_g_format(shear_factor));
end

function x = table_number(row, name)
v = row.(name);
if iscell(v), v = v{1}; end
if isstring(v) || ischar(v)
    x = str2double(v);
else
    x = double(v);
end
end

function s = c_g_format(x)
s = sprintf('%.6g', x);
end
