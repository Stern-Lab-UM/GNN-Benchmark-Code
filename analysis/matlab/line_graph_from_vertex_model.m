function LineG = line_graph_from_vertex_model(C, vertexCells)
%LINE_GRAPH_FROM_VERTEX_MODEL Build the line graph of an epithelial vertex model.
%
%   LineG = line_graph_from_vertex_model(C)
%   LineG = line_graph_from_vertex_model(C, vertexCells)
%
% INPUTS
%   C           N x 2 array. Each row [cellA cellB] is one unordered
%               cell-cell interface. Duplicates are tolerated.
%
%   vertexCells Optional vertex incidence. If supplied, each row/cell-array
%               element lists the cell IDs that meet at one vertex. If omitted,
%               the historical pipeline infers triple junctions
%               combinatorially: interfaces (i,j), (i,k), and (j,k) are linked
%               when all three cell-cell interfaces exist.
%
% OUTPUT
%   LineG       MATLAB graph object with one node per unique interface. Two
%               nodes are adjacent when their interfaces meet at a vertex.
%               LineG.Nodes.Interface stores the sorted unique interface list.
%
% NOTE
%   This is the historical "22-ish" hop-distance definition used by the
%   manuscript/revision summaries before the 2026-06-06 cell-share
%   experiment. It intentionally uses unique(sort(C,2),'rows'), matching the
%   old helper exactly. The official prediction matrices are already sorted
%   and unique, so this does not move T1 root rows in the audited data.

C = sort(C, 2);
C = unique(C, 'rows');
m = size(C, 1);

if m == 0
    LineG = graph(sparse(0, 0));
    LineG.Nodes.Interface = zeros(0, 2);
    return;
end

cells = unique(C(:));
[~, Cmap] = ismember(C, cells);
n = numel(cells);

edgeIdx = full(sparse( ...
    [Cmap(:,1); Cmap(:,2)], ...
    [Cmap(:,2); Cmap(:,1)], ...
    [(1:m)'; (1:m)'], ...
    n, n));

L = false(m, m);

if nargin < 2 || isempty(vertexCells)
    for i = 1 : n
        nbrs = find(edgeIdx(i, :));
        for a = 1 : numel(nbrs)-1
            j = nbrs(a);
            for b = a+1 : numel(nbrs)
                k = nbrs(b);
                e3 = edgeIdx(j, k);
                if e3
                    e1 = edgeIdx(i, j);
                    e2 = edgeIdx(i, k);
                    L([e1 e1 e2], [e2 e3 e3]) = 1;
                    L([e2 e3 e3], [e1 e1 e2]) = 1;
                end
            end
        end
    end
else
    if ~iscell(vertexCells)
        vmat = vertexCells;
        vertexCells = cell(size(vmat, 1), 1);
        for v = 1 : size(vmat, 1)
            row = vmat(v, :);
            row(~isfinite(row) | row == 0) = [];
            vertexCells{v} = row;
        end
    end

    for v = 1 : numel(vertexCells)
        cellsHere = vertexCells{v};
        if numel(cellsHere) < 2
            continue;
        end

        [tf, cellsHereIdx] = ismember(cellsHere, cells);
        cellsHereIdx = cellsHereIdx(tf);
        if numel(cellsHereIdx) < 2
            continue;
        end

        pairs = nchoosek(cellsHereIdx, 2);
        eIdx = edgeIdx(sub2ind([n n], pairs(:,1), pairs(:,2)));
        eIdx = eIdx(eIdx > 0);
        if numel(eIdx) >= 2
            L(eIdx, eIdx) = 1;
        end
    end
end

L(1:m+1:end) = 0;
LineG = graph(spones(L));
LineG.Nodes.Interface = C;
end
