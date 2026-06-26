function cfg = DCG_local_config_template(cfg)
%DCG_LOCAL_CONFIG_TEMPLATE  Template for machine-specific DCG paths.
%
%   Copy this file to DCG_local_config.m, edit the paths for your machine,
%   and keep DCG_local_config.m untracked. The public repository intentionally
%   does not contain user-specific drive letters, scratch folders, or account
%   names.

% Consolidated prediction snapshot, containing *.pred.txt and splits/.
% cfg.data_root = '/path/to/gnn_benchmark_consolidated_20260530';

% Optional spring-embedding executable and geometry inputs. These are needed
% only for the embedded-tissue example panels.
% cfg.embed_engine   = '/path/to/spring_embed';
% cfg.embed_workdir  = fullfile(tempdir, 'dcg_springs_embed');
% cfg.embed_vt2d_std = '/path/to/standard/vt2d/files';
% cfg.embed_vt2d_rev = '/path/to/revision/vt2d/files';

end
