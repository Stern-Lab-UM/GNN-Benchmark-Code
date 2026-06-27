function specs = DCG_vertex_model_publication_manifest(mode)
% DCG_vertex_model_publication_manifest  Implement dcg vertex model publication manifest for this MATLAB workflow.
% Inputs: mode
% Outputs: specs
%DCG_VERTEX_MODEL_PUBLICATION_MANIFEST  Publication vertex-model datasets.
%
%   SPECS = DCG_VERTEX_MODEL_PUBLICATION_MANIFEST(MODE) returns the dataset
%   definitions used by DCG_GENERATE_VERTEX_MODEL_DATASETS.  The manifest is
%   intentionally data-light: it stores graph order, split indices, and the
%   simulation parameters needed to regenerate the raw vertex-model graphs,
%   but not the raw generated tissues themselves.
%
%   MODE may be:
%     'minimal'     one publication-scale graph per generated condition;
%     'publication' all graph commands recorded for the manuscript datasets.
%
%   The three baseline conditions kA=100, shear=1.0, and tissue size 16^2
%   are aliases of the same standard_16 dataset.

if nargin < 1 || isempty(mode)
    mode = 'minimal';
end
mode = validatestring(mode, {'minimal', 'publication'});

here = fileparts(mfilename('fullpath'));
manifest_dir = fullfile(here, 'manifests');

defs = { ...
    'standard_16', 'standard_16_graph_order.csv', 'standard 16^2-cell reference: kA=100, shear=1.0, one T1'; ...
    'kA_10',      'kA_10_graph_order.csv',       '16^2 cells, kA=10, one T1'; ...
    'kA_1',       'kA_1_graph_order.csv',        '16^2 cells, kA=1, one T1'; ...
    'shear_1_2',  'shear_1_2_graph_order.csv',   '16^2 cells, kA=100, area-preserving shear factor 1.2, one T1'; ...
    'shear_1_5',  'shear_1_5_graph_order.csv',   '16^2 cells, kA=100, area-preserving shear factor 1.5, one T1'; ...
    'tissue_484', 'tissue_484_graph_order.csv',  '22^2 cells, kA=100, one T1'; ...
    'tissue_784', 'tissue_784_graph_order.csv',  '28^2 cells, kA=100, one T1'; ...
    'flip_two',   'flip_two_graph_order.csv',    '16^2 cells, kA=100, two T1 events' ...
    };

specs = repmat(struct( ...
    'key', '', ...
    'description', '', ...
    'graph_order_file', '', ...
    'rows', table(), ...
    'split_manifest_dir', '', ...
    'aliases', {{}}), size(defs, 1), 1);

for k = 1:size(defs, 1)
    key = defs{k, 1};
    order_file = fullfile(manifest_dir, defs{k, 2});
    rows = readtable(order_file, 'TextType', 'string');

    if strcmp(mode, 'minimal')
        rows = rows(1, :);
    end

    specs(k).key = key;
    specs(k).description = defs{k, 3};
    specs(k).graph_order_file = order_file;
    specs(k).rows = rows;
    specs(k).split_manifest_dir = fullfile(manifest_dir, 'splits', key);
    specs(k).aliases = {};
end

specs(1).aliases = {'kA_100', 'shear_1_0', 'tissue_256'};

end
