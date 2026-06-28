function paths = GNNBenchmark_data_package_paths(root)
%GNNBENCHMARK_DATA_PACKAGE_PATHS  Resolve public-package and consolidated roots.
%
%   PATHS = GNNBENCHMARK_DATA_PACKAGE_PATHS(ROOT) accepts either:
%
%     1. a public data-package root:
%          ROOT/predictions/consolidated/*.pred.txt
%          ROOT/predictions/consolidated/splits/...
%
%     2. the consolidated prediction folder itself:
%          ROOT/*.pred.txt
%          ROOT/splits/...
%
%   The returned struct normalizes these inputs into the durable folders used
%   by analysis scripts. It deliberately does not describe transient live-run
%   folders such as bo_runs/, best_hps/, generated_data/, or staged_inputs/.

root = char(root);
if isempty(root)
    error('GNNBenchmark:dataRootMissing', 'Root path is empty.');
end
if ~isfolder(root)
    error('GNNBenchmark:dataRootMissing', 'Root path is not a folder: %s', root);
end

paths = struct();
paths.input_root = root;
paths.package_root = '';
paths.is_public_package = false;

candidate = fullfile(root, 'predictions', 'consolidated');
if GNNBenchmark_consolidated_paths('is_consolidated', candidate)
    paths.package_root = root;
    paths.data_root = candidate;
    paths.is_public_package = true;
elseif GNNBenchmark_consolidated_paths('is_consolidated', root)
    paths.data_root = root;
    [parent_dir, leaf] = fileparts(root);
    [grandparent_dir, parent_leaf] = fileparts(parent_dir);
    if strcmpi(leaf, 'consolidated') && strcmpi(parent_leaf, 'predictions') && isfolder(grandparent_dir)
        paths.package_root = grandparent_dir;
        paths.is_public_package = isfolder(fullfile(grandparent_dir, 'predictions', 'consolidated'));
    end
else
    error('GNNBenchmark:notDataPackage', ...
        ['Could not resolve %s as either a public package root or a ', ...
        'consolidated prediction folder.'], root);
end

if paths.is_public_package
    pkg = paths.package_root;
    paths.embedding_root = existing_or_empty(fullfile(pkg, 'embeddings', 'per_graph'));
    paths.manuscript_analyses_root = existing_or_empty(fullfile(pkg, 'manuscript_analyses'));
    paths.final_models_consolidated = existing_or_empty(fullfile(pkg, 'final_models', 'consolidated'));
    paths.figures_root = fullfile(pkg, 'figures');
    paths.main_figures_root = paths.figures_root;
    paths.revision_figures_root = paths.figures_root;
    paths.legacy_revision_figures_root = first_existing_dir({
        fullfile(pkg, 'figures', 'revision_2026')
        fullfile(pkg, 'figures', 'revision_codex_2026')
        });
    paths.analysis_cache_root = fullfile(pkg, 'analysis_tables', 'analyzer_cache');
    paths.revision_cache_root = first_existing_dir({
        fullfile(paths.analysis_cache_root, 'revision_2026')
        fullfile(paths.analysis_cache_root, 'revision_codex_2026')
        });
    if isempty(paths.revision_cache_root)
        paths.revision_cache_root = fullfile(paths.analysis_cache_root, 'revision_2026');
    end
else
    paths.embedding_root = existing_or_empty(fullfile(paths.data_root, 'embeddings', 'per_graph'));
    paths.manuscript_analyses_root = '';
    paths.final_models_consolidated = existing_or_empty(fullfile(paths.data_root, 'final_models', 'consolidated'));
    paths.figures_root = fullfile(paths.data_root, '_figures');
    paths.main_figures_root = paths.figures_root;
    paths.revision_figures_root = paths.figures_root;
    paths.legacy_revision_figures_root = fullfile(paths.figures_root, 'revision_2026');
    paths.analysis_cache_root = fullfile(paths.data_root, '_analyzer_cache');
    paths.revision_cache_root = fullfile(paths.analysis_cache_root, 'revision_2026');
end
end

function out = existing_or_empty(path_in)
if isfolder(path_in)
    out = path_in;
else
    out = '';
end
end

function out = first_existing_dir(candidates)
out = '';
for i = 1:numel(candidates)
    if isfolder(candidates{i})
        out = candidates{i};
        return;
    end
end
end
