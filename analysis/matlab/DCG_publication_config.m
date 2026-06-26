function cfg = DCG_publication_config(overrides)
%DCG_PUBLICATION_CONFIG  User-local paths for the DCG analysis pipeline.
%
%   cfg = DCG_PUBLICATION_CONFIG() returns a struct containing paths used by
%   the MATLAB analysis and plotting scripts. The repository does not assume
%   a particular drive, cluster, or user account. Paths can be supplied in
%   three ways, in increasing order of priority:
%
%     1. Environment variables, for command-line or cluster runs.
%     2. An optional untracked DCG_local_config.m file on the MATLAB path.
%     3. An overrides struct passed by the caller.
%
%   Required for full analysis:
%     cfg.data_root
%       Folder containing the consolidated prediction snapshot:
%         *.pred.txt
%         splits/<key>/{train,val,test}.inds
%
%   Optional for embedding example figures:
%     cfg.embed_engine
%     cfg.embed_workdir
%     cfg.embed_vt2d_std
%     cfg.embed_vt2d_rev
%
%   Environment variables:
%     DCG_DATA_ROOT
%     DCG_EMBED_ENGINE
%     DCG_EMBED_WORKDIR
%     DCG_EMBED_VT2D_STD
%     DCG_EMBED_VT2D_REV

if nargin < 1 || isempty(overrides)
    overrides = struct();
end

cfg = struct();
cfg.data_root      = getenv_if_set('DCG_DATA_ROOT', '');
cfg.embed_engine   = getenv_if_set('DCG_EMBED_ENGINE', '');
cfg.embed_workdir  = getenv_if_set('DCG_EMBED_WORKDIR', fullfile(tempdir, 'dcg_springs_embed'));
cfg.embed_vt2d_std = getenv_if_set('DCG_EMBED_VT2D_STD', '');
cfg.embed_vt2d_rev = getenv_if_set('DCG_EMBED_VT2D_REV', '');

% Private user/machine configuration. This file is intentionally ignored by
% git; use DCG_local_config_template.m as a starting point.
if exist('DCG_local_config', 'file') == 2
    local_cfg = DCG_local_config(cfg);
    if isstruct(local_cfg)
        cfg = merge_structs(cfg, local_cfg);
    else
        error('DCG:badLocalConfig', 'DCG_local_config must return a struct.');
    end
end

cfg = merge_structs(cfg, overrides);

end


function value = getenv_if_set(name, fallback)
value = getenv(name);
if isempty(value)
    value = fallback;
end
end


function out = merge_structs(base, extra)
out = base;
if isempty(extra)
    return;
end
names = fieldnames(extra);
for i = 1:numel(names)
    out.(names{i}) = extra.(names{i});
end
end
