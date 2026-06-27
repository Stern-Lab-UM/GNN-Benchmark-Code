function out = GNNBenchmark_consolidated_paths(action, data_root, varargin)
% GNNBenchmark_consolidated_paths  Implement GNN Benchmark consolidated paths for this MATLAB workflow.
% Inputs: action, data_root, varargin
% Outputs: out
%GNNBenchmark_CONSOLIDATED_PATHS  Path bridge for the consolidated snapshot.
%
%   The 2026 revision predictions/models/splits were gathered into a single
%   flat snapshot folder (e.g. gnn_benchmark_consolidated_20260530):
%
%     <root>\<task>_<model>_<W|UW>_<size>_s<seed>.pred.txt   (+ .model.pth)
%     <root>\splits\<key>\{train,val,test}.inds
%     <root>\splits\<key>\_applies_to.txt
%
%   The analyzer/plotter speak the older flat nomenclature:
%
%     pred_<prefix>__<model>_s<seed>.txt        inds\<prefix>\<split>.inds
%
%   where <prefix> is one of  v1_<idx>_<siz>_<W|UW>,  hex_<idx>_<siz>_<W|UW>,
%   or  rev_<name>  (rev_kA_1, rev_Shear_1_2, rev_Flip_two, rev_Tissue_784...).
%   Pred-file CONTENT is identical between layouts; only the filename and the
%   split location changed. The split folders are md5-deduplicated, so the
%   <prefix> -> folder lookup is driven by each folder's _applies_to.txt
%   (which lists every flat prefix that folder serves).
%
%   Actions:
%     ('is_consolidated', root)                 -> logical
%     ('pred_file', root, prefix, model, seed)  -> full path to the .pred.txt
%     ('pred_glob', root, prefix, model)        -> seed glob pattern (for dir())
%     ('inds_dir',  root, prefix)               -> splits\<key> folder ('' none)

switch action
    case 'is_consolidated'
        out = isfolder(fullfile(data_root, 'splits')) && ...
              ~isempty(dir(fullfile(data_root, '*.pred.txt')));

    case 'pred_file'
        [prefix, model, seed] = varargin{:};
        [task, wuw, sz] = i_prefix_parts(prefix);
        out = fullfile(data_root, ...
            sprintf('%s_%s_%s_%s_s%d.pred.txt', task, model, wuw, sz, seed));

    case 'pred_glob'
        [prefix, model] = varargin{:};
        [task, wuw, sz] = i_prefix_parts(prefix);
        out = fullfile(data_root, ...
            sprintf('%s_%s_%s_%s_s*.pred.txt', task, model, wuw, sz));

    case 'inds_dir'
        out = i_inds_dir(data_root, varargin{1});

    otherwise
        error('GNNBenchmark_consolidated_paths:badAction', 'unknown action "%s"', action);
end
end


function [task, wuw, sz] = i_prefix_parts(prefix)
% i_prefix_parts  Implement i prefix parts for this MATLAB workflow.
% Inputs: prefix
% Outputs: task, wuw, sz
%I_PREFIX_PARTS  Decode a flat analyzer prefix into snapshot path tokens.
%
%   PURPOSE  Translate a flat prefix into the (task, weighting, size) tokens
%            that build a consolidated pred filename
%            <task>_<model>_<wuw>_<sz>_s<seed>.pred.txt.
%   INPUT    prefix  one of three schemes:
%              rev_<name>             revision sets (weighted only), e.g.
%                                       rev_Flip_two -> ('Flip-two','W','na')
%                                       rev_kA_1     -> ('kA','W','1')
%                                       rev_Shear_1_2-> ('Shear','W','1_2')
%                                       rev_Tissue_784->('Tissue','W','784')
%              hex_<idx>_<siz>_<W|UW> -> ('Hexagonality', wuw, '<idx>_<siz>')
%              v1_<idx>_<siz>_<W|UW>  -> ('standard-flip', wuw, '<idx>_<siz>')
%   OUTPUT   task  dataset/task token; wuw  'W' or 'UW'; sz  size token.
%   ERRORS   throws GNNBenchmark_consolidated_paths:prefix if none of the schemes match.
%   ALGORITHM  Regexp dispatch on the prefix family (rev_ / hex_ / v1_).
% Translate a flat prefix to the consolidated (task, weighting, size) tokens.

% Revision sets: rev_<name>, weighted only.
tok = regexp(prefix, '^rev_(.+)$', 'tokens', 'once');
if ~isempty(tok)
    wuw  = 'W';
    body = tok{1};
    if strcmp(body, 'Flip_two')
        task = 'Flip-two'; sz = 'na'; return;
    end
    t2 = regexp(body, '^(kA|Shear|Tissue)_(.+)$', 'tokens', 'once');
    if ~isempty(t2)
        task = t2{1}; sz = t2{2}; return;
    end
    error('GNNBenchmark_consolidated_paths:prefix', 'unrecognized rev prefix "%s"', prefix);
end

% Hexagonality: hex_<idx>_<siz>_<W|UW>.
tok = regexp(prefix, '^hex_(\d+_\d+)_(W|UW)$', 'tokens', 'once');
if ~isempty(tok)
    task = 'Hexagonality'; sz = tok{1}; wuw = tok{2}; return;
end

% Standard-flip (v1): v1_<idx>_<siz>_<W|UW>.
tok = regexp(prefix, '^v1_(\d+_\d+)_(W|UW)$', 'tokens', 'once');
if ~isempty(tok)
    task = 'standard-flip'; sz = tok{1}; wuw = tok{2}; return;
end

error('GNNBenchmark_consolidated_paths:prefix', 'unrecognized prefix "%s"', prefix);
end


function d = i_inds_dir(data_root, prefix)
% i_inds_dir  Implement i inds dir for this MATLAB workflow.
% Inputs: data_root, prefix
% Outputs: d
%I_INDS_DIR  Locate the dedup'd split folder that serves a given prefix.
%
%   PURPOSE  Split index files live under <data_root>/splits/ and are
%            md5-deduplicated, so several flat prefixes can share one folder.
%            Each folder names the prefixes it serves in _applies_to.txt; this
%            returns the folder whose _applies_to.txt lists `prefix`.
%   INPUT    data_root  consolidated snapshot root; prefix  flat dataset prefix.
%   OUTPUT   d  full path to the matching splits subfolder, or '' if none match.
%   ALGORITHM  Scan each splits subfolder; parse its _applies_to.txt
%              'applies to datasets:' line; return the first that lists `prefix`.
% Find the splits\<key> folder whose _applies_to.txt lists `prefix`.
d = '';
splits_root = fullfile(data_root, 'splits');
entries = dir(splits_root);
for k = 1 : numel(entries)
    if ~entries(k).isdir || startsWith(entries(k).name, '.')
        continue;
    end
    applies = fullfile(splits_root, entries(k).name, '_applies_to.txt');
    if exist(applies, 'file') ~= 2
        continue;
    end
    txt  = fileread(applies);
    line = regexp(txt, 'applies to datasets:\s*([^\r\n]*)', 'tokens', 'once');
    if isempty(line)
        continue;
    end
    names = regexp(line{1}, '\S+', 'match');
    if any(strcmp(names, prefix))
        d = fullfile(splits_root, entries(k).name);
        return;
    end
end
end
