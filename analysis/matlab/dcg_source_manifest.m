function m = dcg_source_manifest(data_root)
% dcg_source_manifest  Implement dcg source manifest for this MATLAB workflow.
% Inputs: data_root
% Outputs: m
%DCG_SOURCE_MANIFEST  Deterministic fingerprint of a prediction-file set.
%
% PURPOSE
%   Returns a stable string listing every prediction file in data_root with its
%   byte size and modification time, sorted by name. The analyzer stamps a
%   results_summary.mat with this fingerprint; the plotter recomputes it at plot
%   time and REFUSES to plot if it differs -- so a summary that is stale relative
%   to its source folder ERRORS instead of silently producing wrong figures.
%   (Cache-staleness guard, added 2026-06-01 after a stale summary silently drove
%   a whole batch of wrong figures.)
%
% INPUT
%   data_root : folder holding the prediction files (consolidated flat layout
%               <task>_<model>_..._s*.pred.txt, or legacy pred_*.txt).
%
% OUTPUT
%   m : char row vector fingerprint; 'NO_SOURCE_FILES' if the folder has none.
%
% NOTES
%   * Fingerprints the WHOLE folder's prediction set (name+size+mtime), so ANY
%     change -- a file added, removed, replaced, or re-touched -- invalidates
%     every summary built from that folder. This is intentionally conservative:
%     it errs toward forcing a rebuild rather than risking a silent stale plot.
%   * Reads only dir() metadata (no file contents) -> cheap even for GB-scale
%     prediction files. The one gap it cannot see is an in-place overwrite that
%     preserves BOTH size and mtime; the canonical workflow's force-rebuild
%     (rebuild_summaries=true) covers that residual case.

d = dir(fullfile(data_root, '*.pred.txt'));                 % consolidated flat layout
if isempty(d)
    d = dir(fullfile(data_root, 'pred_*.txt'));             % legacy flat layout
end
if ~isempty(d)
    d = d(~[d.isdir]);
end
if isempty(d)
    m = 'NO_SOURCE_FILES';
    return;
end
names = {d.name};
[~, ord] = sort(names);
d = d(ord);
parts = cell(1, numel(d));
for i = 1:numel(d)
    % name | bytes | mtime-in-whole-seconds (round to avoid float noise)
    parts{i} = sprintf('%s|%d|%.0f', d(i).name, d(i).bytes, d(i).datenum * 86400);
end
m = strjoin(parts, ';');
end
