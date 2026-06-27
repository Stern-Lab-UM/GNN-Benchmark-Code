function report = DCG_generate_vertex_model_datasets(varargin)
% DCG_generate_vertex_model_datasets  Implement dcg generate vertex model datasets for this MATLAB workflow.
% Inputs: varargin
% Outputs: report
%DCG_GENERATE_VERTEX_MODEL_DATASETS  Generate manuscript vertex-model graphs.
%
%   REPORT = DCG_GENERATE_VERTEX_MODEL_DATASETS() builds the vertex-model
%   simulator, runs one publication-scale graph per generated condition, and
%   assembles weighted/unweighted model-ready text files.  Outputs are written
%   under:
%
%       <repo>/generated_data/vertex_model/
%
%   REPORT = DCG_GENERATE_VERTEX_MODEL_DATASETS('mode','publication') runs the
%   full graph manifests used for the manuscript datasets.  This is expensive:
%   each graph requires vertex-model relaxation, and the full run should be
%   launched on a compute node.
%
%   Name-value options:
%     'mode'          'minimal' (default) or 'publication'
%     'output_root'   output folder; defaults to repo/generated_data/vertex_model
%     'workers'       number of MATLAB workers for system calls; 1 by default
%     'overwrite'     rerun simulator outputs that already exist; false
%     'assemble_only' skip simulator and only assemble existing raw graph files
%     'datasets'      optional dataset key or cell array of keys to run
%     'max_graphs_per_dataset'
%                     optional finite graph limit per dataset, used by the
%                     end-to-end mini pipeline to create a fast but trainable
%                     subset from the publication graph order
%     'split_counts'  optional [n_train n_val n_test] split for limited runs
%     'counterfactual'          also write a perturbed weighted-input variant
%                               for the fallback-copying diagnostic; false
%     'counterfactual_h_min'    perturb interfaces whose edge-hop distance from
%                               the newly formed T1 interface is >= this value
%     'counterfactual_delta'    absolute length shift added with random sign
%     'counterfactual_seed'     fixed seed controlling the random signs
%     'counterfactual_suffix'   optional dataset-key suffix for the variant
%
%   Dataset conventions:
%     standard_16 is the canonical kA=100, shear=1.0, 16^2-cell dataset.
%     kA_100, shear_1_0, and tissue_256 are documented aliases of standard_16.
%
%   Simulator command:
%     vertex_model_generator Nx kA packageID SigI shearFactor T1EdgeID_1 T1EdgeID_2

p = inputParser;
p.addParameter('mode', 'minimal', @(x) any(strcmp(x, {'minimal','publication'})));
p.addParameter('output_root', '', @(x) ischar(x) || isstring(x));
p.addParameter('workers', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('overwrite', false, @(x) islogical(x) && isscalar(x));
p.addParameter('assemble_only', false, @(x) islogical(x) && isscalar(x));
p.addParameter('datasets', {}, @(x) iscell(x) || isstring(x) || ischar(x));
p.addParameter('max_graphs_per_dataset', inf, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter('split_counts', [], @(x) isempty(x) || (isnumeric(x) && numel(x) == 3 && all(x >= 0)));
p.addParameter('counterfactual', false, @(x) islogical(x) && isscalar(x));
p.addParameter('counterfactual_h_min', 14, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('counterfactual_delta', 0.05, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.addParameter('counterfactual_seed', 20260616, @(x) isnumeric(x) && isscalar(x));
p.addParameter('counterfactual_suffix', '', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opts = p.Results;
opts.workers = max(1, floor(opts.workers));
opts.counterfactual_h_min = floor(opts.counterfactual_h_min);
opts.counterfactual_suffix = char(opts.counterfactual_suffix);

paths = local_paths();
if isempty(opts.output_root)
    opts.output_root = fullfile(paths.repo_root, 'generated_data', 'vertex_model');
else
    opts.output_root = char(opts.output_root);
end

if ~isfolder(opts.output_root)
    mkdir(opts.output_root);
end

specs = DCG_vertex_model_publication_manifest(opts.mode);
if ~isempty(opts.datasets)
    wanted = cellstr(opts.datasets);
    keep = ismember({specs.key}, wanted);
    missing = setdiff(wanted, {specs.key});
    if ~isempty(missing)
        error('Unknown dataset key(s): %s', strjoin(missing, ', '));
    end
    specs = specs(keep);
end
if isfinite(opts.max_graphs_per_dataset)
    for k = 1:numel(specs)
        n_keep = min(height(specs(k).rows), floor(opts.max_graphs_per_dataset));
        specs(k).rows = specs(k).rows(1:n_keep, :);
    end
end
exe_path = fullfile(opts.output_root, 'build', executable_name());

if ~opts.assemble_only
    build_simulator(paths.src_dir, exe_path);
end

report = struct();
report.mode = opts.mode;
report.output_root = opts.output_root;
report.executable = exe_path;
report.datasets = struct();

for s = 1:numel(specs)
    spec = specs(s);
    fprintf('\n[vertex-model] Dataset %s: %s\n', spec.key, spec.description);

    raw_root = fullfile(opts.output_root, 'raw', spec.key);
    raw_output_dir = fullfile(raw_root, 'output');
    model_ready_dir = fullfile(opts.output_root, 'model_ready', spec.key, '2D');
    ensure_dir(raw_output_dir);
    ensure_dir(model_ready_dir);

    if ~opts.assemble_only
        run_initial_phase(exe_path, raw_root, spec.rows, opts);
        run_graph_phase(exe_path, raw_root, spec.rows, opts);
    end

    assembly = DCG_assemble_vertex_model_graphs(raw_output_dir, spec.rows, model_ready_dir, spec.key);
    splits_dir = fullfile(opts.output_root, 'model_ready', spec.key, 'splits');
    if isfinite(opts.max_graphs_per_dataset)
        write_limited_split(model_ready_dir, splits_dir, height(spec.rows), opts.split_counts);
    elseif strcmp(opts.mode, 'publication')
        copy_split_manifests(spec.split_manifest_dir, splits_dir);
        copy_default_split(splits_dir, model_ready_dir);
    else
        write_minimal_split(model_ready_dir, splits_dir);
    end

    report.datasets.(matlab.lang.makeValidName(spec.key)).raw_root = raw_root;
    report.datasets.(matlab.lang.makeValidName(spec.key)).model_ready_dir = model_ready_dir;
    report.datasets.(matlab.lang.makeValidName(spec.key)).graphs = height(assembly);
    report.datasets.(matlab.lang.makeValidName(spec.key)).assembly_report = ...
        fullfile(model_ready_dir, [spec.key, '_assembly_report.csv']);

    if opts.counterfactual
        cf_key = counterfactual_dataset_key(spec.key, opts);
        cf_model_ready_dir = fullfile(opts.output_root, 'model_ready', cf_key, '2D');
        cf_splits_dir = fullfile(opts.output_root, 'model_ready', cf_key, 'splits');
        ensure_dir(cf_model_ready_dir);
        fprintf('[vertex-model] Counterfactual variant %s: h >= %d, delta = %.8g, seed = %d\n', ...
            cf_key, opts.counterfactual_h_min, opts.counterfactual_delta, opts.counterfactual_seed);
        cf_assembly = DCG_assemble_vertex_model_graphs(raw_output_dir, spec.rows, ...
            cf_model_ready_dir, cf_key, ...
            'counterfactual', true, ...
            'counterfactual_h_min', opts.counterfactual_h_min, ...
            'counterfactual_delta', opts.counterfactual_delta, ...
            'counterfactual_seed', opts.counterfactual_seed);
        if isfinite(opts.max_graphs_per_dataset)
            write_limited_split(cf_model_ready_dir, cf_splits_dir, height(spec.rows), opts.split_counts);
        elseif strcmp(opts.mode, 'publication')
            copy_split_manifests(spec.split_manifest_dir, cf_splits_dir);
            copy_default_split(cf_splits_dir, cf_model_ready_dir);
        else
            write_minimal_split(cf_model_ready_dir, cf_splits_dir);
        end
        cf_field = matlab.lang.makeValidName(cf_key);
        report.datasets.(cf_field).raw_root = raw_root;
        report.datasets.(cf_field).model_ready_dir = cf_model_ready_dir;
        report.datasets.(cf_field).graphs = height(cf_assembly);
        report.datasets.(cf_field).assembly_report = ...
            fullfile(cf_model_ready_dir, [cf_key, '_assembly_report.csv']);
        report.datasets.(cf_field).counterfactual_h_min = opts.counterfactual_h_min;
        report.datasets.(cf_field).counterfactual_delta = opts.counterfactual_delta;
        report.datasets.(cf_field).counterfactual_seed = opts.counterfactual_seed;
    end
    if ~isempty(spec.aliases)
        write_alias_readmes(opts.output_root, spec.key, spec.aliases);
    end
end

write_run_summary(report);

fprintf('\n[vertex-model] Done. Output root:\n  %s\n', opts.output_root);
end

function key = counterfactual_dataset_key(base_key, opts)
% counterfactual_dataset_key  Build a readable, collision-resistant variant key.
if ~isempty(opts.counterfactual_suffix)
    suffix = opts.counterfactual_suffix;
else
    delta_txt = regexprep(sprintf('%.8g', opts.counterfactual_delta), '[^\dA-Za-z]+', 'p');
    suffix = sprintf('counterfactual_h%d_delta%s', opts.counterfactual_h_min, delta_txt);
end
key = sprintf('%s_%s', base_key, suffix);
end
function paths = local_paths()
% local_paths  Implement local paths for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: none.
% Outputs: paths
paths.this_dir = fileparts(mfilename('fullpath'));
paths.repo_root = fileparts(fileparts(paths.this_dir));
paths.src_dir = fullfile(paths.this_dir, 'src');
end

function build_simulator(src_dir, exe_path)
% build_simulator  Implement build simulator for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: src_dir, exe_path
% Outputs: none; performs side effects or updates the caller workflow.
ensure_dir(fileparts(exe_path));
cmd = sprintf('g++ -O2 -std=c++11 -Wall -Wextra -o "%s" "%s" -lm', ...
    exe_path, fullfile(src_dir, 'main.c'));
fprintf('[vertex-model] Building simulator:\n  %s\n', cmd);
[status, out] = system(cmd);
if status ~= 0
    error('Simulator build failed:\n%s', out);
end
end

function run_initial_phase(exe_path, run_root, rows, opts)
% run_initial_phase  Implement run initial phase for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: exe_path, run_root, rows, opts
% Outputs: none; performs side effects or updates the caller workflow.
unique_rows = unique(rows(:, {'nx','kA','package_id','sigma_index'}), 'rows');
fprintf('[vertex-model] Initial tissues queued: %d\n', height(unique_rows));
commands = cell(height(unique_rows), 1);
for i = 1:height(unique_rows)
    nx = table_number(unique_rows(i,:), 'nx');
    kA = table_number(unique_rows(i,:), 'kA');
    package_id = table_number(unique_rows(i,:), 'package_id');
    sigma_index = table_number(unique_rows(i,:), 'sigma_index');
    final_name = sprintf('final_%d_%s_%d_%d.vt2d', nx * nx, c_g_format(kA), package_id, sigma_index);
    final_path = fullfile(run_root, 'output', final_name);
    if isfile(final_path) && ~opts.overwrite
        continue
    end
    args = sprintf('%d %s %d %d 1 -1 -1', nx, c_g_format(kA), package_id, sigma_index);
    commands{i} = make_run_command(run_root, exe_path, args);
end
commands = commands(~cellfun(@isempty, commands));
run_commands(commands, opts.workers, 'initial');
end

function run_graph_phase(exe_path, run_root, rows, opts)
% run_graph_phase  Implement run graph phase for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: exe_path, run_root, rows, opts
% Outputs: none; performs side effects or updates the caller workflow.
fprintf('[vertex-model] T1 graph relaxations queued: %d\n', height(rows));
commands = cell(height(rows), 1);
for i = 1:height(rows)
    raw_name = expected_raw_graph_name(rows(i,:));
    raw_path = fullfile(run_root, 'output', raw_name);
    if isfile(raw_path) && ~opts.overwrite
        continue
    end
    nx = table_number(rows(i,:), 'nx');
    kA = table_number(rows(i,:), 'kA');
    package_id = table_number(rows(i,:), 'package_id');
    sigma_index = table_number(rows(i,:), 'sigma_index');
    shear_factor = table_number(rows(i,:), 'shear_factor');
    t1_edge_1 = table_number(rows(i,:), 't1_edge_1');
    t1_edge_2 = table_number(rows(i,:), 't1_edge_2');
    args = sprintf('%d %s %d %d %s %d %d', nx, c_g_format(kA), package_id, ...
        sigma_index, c_g_format(shear_factor), t1_edge_1, t1_edge_2);
    commands{i} = make_run_command(run_root, exe_path, args);
end
commands = commands(~cellfun(@isempty, commands));
run_commands(commands, opts.workers, 'graph');
end

function run_commands(commands, workers, label)
% run_commands  Implement run commands for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: commands, workers, label
% Outputs: none; performs side effects or updates the caller workflow.
if isempty(commands)
    fprintf('[vertex-model] No %s commands needed; files already exist.\n', label);
    return
end

fprintf('[vertex-model] Running %d %s commands with %d worker(s).\n', numel(commands), label, workers);
if workers > 1
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= workers
        if ~isempty(pool), delete(pool); end
        parpool('local', workers);
    end
    statuses = zeros(numel(commands), 1);
    outputs = cell(numel(commands), 1);
    parfor i = 1:numel(commands)
        [statuses(i), outputs{i}] = system(commands{i});
    end
else
    statuses = zeros(numel(commands), 1);
    outputs = cell(numel(commands), 1);
    for i = 1:numel(commands)
        [statuses(i), outputs{i}] = system(commands{i});
    end
end

bad = find(statuses ~= 0, 1, 'first');
if ~isempty(bad)
    error('Command failed during %s phase:\n%s\nOutput:\n%s', label, commands{bad}, outputs{bad});
end
end

function cmd = make_run_command(run_root, exe_path, args)
% make_run_command  Implement make run command for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: run_root, exe_path, args
% Outputs: cmd
ensure_dir(fullfile(run_root, 'output'));
if ispc
    cmd = sprintf('cd /d "%s" && "%s" %s', run_root, exe_path, args);
else
    cmd = sprintf('cd "%s" && "%s" %s', run_root, exe_path, args);
end
end

function copy_split_manifests(src_dir, dst_dir)
% copy_split_manifests  Implement copy split manifests for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: src_dir, dst_dir
% Outputs: none; performs side effects or updates the caller workflow.
if ~isfolder(src_dir)
    return
end
ensure_dir(dst_dir);
d = dir(fullfile(src_dir, '**', '*'));
d = d(~[d.isdir]);
for i = 1:numel(d)
    src = fullfile(d(i).folder, d(i).name);
    rel = erase(src, [src_dir, filesep]);
    dst = fullfile(dst_dir, rel);
    ensure_dir(fileparts(dst));
    copyfile(src, dst);
end
end

function copy_default_split(splits_dir, model_ready_dir)
% copy_default_split  Implement copy default split for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: splits_dir, model_ready_dir
% Outputs: none; performs side effects or updates the caller workflow.
if ~isfolder(splits_dir)
    return
end
preferred = {'standard_2_16', 'training_set_1_16_cells'};
src = '';
for i = 1:numel(preferred)
    candidate = fullfile(splits_dir, preferred{i});
    if isfolder(candidate)
        src = candidate;
        break
    end
end
if isempty(src)
    d = dir(fullfile(splits_dir, '*'));
    d = d([d.isdir] & ~ismember({d.name}, {'.','..'}));
    if isempty(d)
        return
    end
    src = fullfile(d(1).folder, d(1).name);
end
for name = ["train.inds", "val.inds", "test.inds"]
    f = fullfile(src, char(name));
    if isfile(f)
        copyfile(f, fullfile(model_ready_dir, char(name)));
    end
end
end

function write_minimal_split(model_ready_dir, splits_dir)
% write_minimal_split  Write minimal split to disk.
% Inputs: model_ready_dir, splits_dir
% Outputs: none; performs side effects or updates the caller workflow.
ensure_dir(fullfile(splits_dir, 'minimal_one_graph'));
write_text_file(fullfile(model_ready_dir, 'train.inds'), '');
write_text_file(fullfile(model_ready_dir, 'val.inds'), '');
write_text_file(fullfile(model_ready_dir, 'test.inds'), sprintf('0\n'));
write_text_file(fullfile(splits_dir, 'minimal_one_graph', 'train.inds'), '');
write_text_file(fullfile(splits_dir, 'minimal_one_graph', 'val.inds'), '');
write_text_file(fullfile(splits_dir, 'minimal_one_graph', 'test.inds'), sprintf('0\n'));
end

function write_limited_split(model_ready_dir, splits_dir, n_graphs, split_counts)
% write_limited_split  Write a train/val/test split for limited mini runs.
% Inputs:
%   model_ready_dir  Folder containing the generated model-ready graph files.
%   splits_dir       Folder where named split subdirectories are mirrored.
%   n_graphs         Number of generated graphs in the limited dataset.
%   split_counts     Optional [train val test] counts. When empty, reserve one
%                    validation and one test graph when possible.
% Outputs:
%   None. Writes train.inds, val.inds, and test.inds beside the dataset and
%   under splits/limited_<n>_graphs/.
if nargin < 4 || isempty(split_counts)
    if n_graphs >= 3
        split_counts = [n_graphs - 2, 1, 1];
    elseif n_graphs == 2
        split_counts = [1, 0, 1];
    else
        split_counts = [0, 0, 1];
    end
else
    split_counts = floor(split_counts(:).');
end
if sum(split_counts) > n_graphs
    error('write_limited_split:tooManyGraphs', ...
        'split_counts sum to %d but only %d graph(s) were generated.', ...
        sum(split_counts), n_graphs);
end
idx = 0:(n_graphs - 1);
train_idx = idx(1:split_counts(1));
val_idx = idx(split_counts(1) + (1:split_counts(2)));
test_idx = idx(sum(split_counts(1:2)) + (1:split_counts(3)));
split_name = sprintf('limited_%d_graphs', n_graphs);
split_dir = fullfile(splits_dir, split_name);
ensure_dir(split_dir);
write_index_file(fullfile(model_ready_dir, 'train.inds'), train_idx);
write_index_file(fullfile(model_ready_dir, 'val.inds'), val_idx);
write_index_file(fullfile(model_ready_dir, 'test.inds'), test_idx);
write_index_file(fullfile(split_dir, 'train.inds'), train_idx);
write_index_file(fullfile(split_dir, 'val.inds'), val_idx);
write_index_file(fullfile(split_dir, 'test.inds'), test_idx);
end

function write_index_file(path, idx)
% write_index_file  Write zero-based graph indices, one per line.
if isempty(idx)
    write_text_file(path, '');
    return
end
write_text_file(path, sprintf('%d\n', idx));
end
function write_text_file(path, txt)
% write_text_file  Write text file to disk.
% Inputs: path, txt
% Outputs: none; performs side effects or updates the caller workflow.
fid = fopen(path, 'wt');
if fid < 0, error('Could not open %s for writing', path); end
fprintf(fid, '%s', txt);
fclose(fid);
end

function write_alias_readmes(output_root, canonical_key, aliases)
% write_alias_readmes  Write alias readmes to disk.
% Inputs: output_root, canonical_key, aliases
% Outputs: none; performs side effects or updates the caller workflow.
for i = 1:numel(aliases)
    alias_dir = fullfile(output_root, 'model_ready', aliases{i});
    ensure_dir(alias_dir);
    fid = fopen(fullfile(alias_dir, 'README.txt'), 'wt');
    fprintf(fid, '%s is an alias of %s.\n', aliases{i}, canonical_key);
    fprintf(fid, 'Use ../%s for the generated graph files and split manifests.\n', canonical_key);
    fclose(fid);
end
end

function write_run_summary(report)
% write_run_summary  Write run summary to disk.
% Inputs: report
% Outputs: none; performs side effects or updates the caller workflow.
summary_path = fullfile(report.output_root, 'generation_summary.txt');
fid = fopen(summary_path, 'wt');
fprintf(fid, 'mode: %s\n', report.mode);
fprintf(fid, 'output_root: %s\n', report.output_root);
fprintf(fid, 'executable: %s\n\n', report.executable);
keys = fieldnames(report.datasets);
for i = 1:numel(keys)
    ds = report.datasets.(keys{i});
    fprintf(fid, '[%s]\n', keys{i});
    fprintf(fid, 'graphs: %d\n', ds.graphs);
    fprintf(fid, 'raw_root: %s\n', ds.raw_root);
    fprintf(fid, 'model_ready_dir: %s\n', ds.model_ready_dir);
    fprintf(fid, 'assembly_report: %s\n', ds.assembly_report);
    if isfield(ds, 'counterfactual_h_min')
        fprintf(fid, 'counterfactual_h_min: %d\n', ds.counterfactual_h_min);
        fprintf(fid, 'counterfactual_delta: %.17g\n', ds.counterfactual_delta);
        fprintf(fid, 'counterfactual_seed: %d\n', ds.counterfactual_seed);
    end
    fprintf(fid, '\n');
end
fclose(fid);
end

function name = expected_raw_graph_name(row)
% expected_raw_graph_name  Implement expected raw graph name for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: row
% Outputs: name
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
% table_number  Implement table number for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: row, name
% Outputs: x
v = row.(name);
if iscell(v), v = v{1}; end
if isstring(v) || ischar(v)
    x = str2double(v);
else
    x = double(v);
end
end

function s = c_g_format(x)
% c_g_format  Implement c g format for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: x
% Outputs: s
s = sprintf('%.6g', x);
end

function ensure_dir(d)
% ensure_dir  Implement ensure dir for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: d
% Outputs: none; performs side effects or updates the caller workflow.
if ~isfolder(d)
    mkdir(d);
end
end

function name = executable_name()
% executable_name  Implement executable name for data_generation/vertex_model/DCG_generate_vertex_model_datasets.m.
% Inputs: none.
% Outputs: name
if ispc
    name = 'vertex_model_generator.exe';
else
    name = 'vertex_model_generator';
end
end
