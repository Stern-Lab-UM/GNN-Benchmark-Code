function cfg = GNNBenchmark_local_config_template(cfg)
% GNNBenchmark_local_config_template  Implement GNN Benchmark local config template for this MATLAB workflow.
% Inputs: cfg
% Outputs: cfg
%GNNBenchmark_LOCAL_CONFIG_TEMPLATE  Template for machine-specific GNN Benchmark paths.
%
%   Copy this file to GNNBenchmark_local_config.m, edit the paths for your machine,
%   and keep GNNBenchmark_local_config.m untracked. The public repository intentionally
%   does not contain user-specific drive letters, scratch folders, or account
%   names.

% Public data package root, or the consolidated prediction snapshot itself.
% Both of these are valid:
%   cfg.data_root = '/path/to/gnn_benchmark_public_data';
%   cfg.data_root = '/path/to/gnn_benchmark_public_data/predictions/consolidated';

% Optional spring-embedding executable and geometry inputs. These are needed
% only for the embedded-tissue example panels.
% cfg.embed_engine   = '/path/to/spring_embed';
% cfg.embed_workdir  = fullfile(tempdir, 'GNNBenchmark_springs_embed');
% cfg.embed_vt2d_std = '/path/to/standard/vt2d/files';
% cfg.embed_vt2d_rev = '/path/to/revision/vt2d/files';

end
