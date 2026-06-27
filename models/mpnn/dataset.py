"""Utilities for models / mpnn / dataset.py in the GNN Benchmark codebase."""

import os
import os.path as osp
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose, Constant, LocalDegreeProfile, AddLaplacianEigenvectorPE


class CanonicalLapPESign:
    """Deterministic, canonical sign for the Laplacian-PE columns (audit I5).

    PyG's ``AddLaplacianEigenvectorPE`` randomises every eigenvector's sign on
    each call, so the PE is non-reproducible (train vs predict re-process) and
    inconsistent with PPGN. This transform runs straight after it and re-fixes
    the sign of each of the last ``k`` columns of ``data.x`` so the
    largest-magnitude entry is positive -- the same rule the MATLAB pipeline
    (gnn_benchmark_pipeline_2D_*.m) applies to PPGN's eigenvectors, so the two families'
    PE signs align.

    Role:
        CanonicalLapPESign groups state and methods for this repository component.
    """

    def __init__(self, k: int):
        """
        Initialize the CanonicalLapPESign instance and store constructor configuration.

        Args:
            k: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.k = k

    def __call__(self, data):
        """
        Apply this callable transform to the supplied graph/data object.

        Args:
            data: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        pe = data.x[:, -self.k:]
        max_idx = pe.abs().argmax(dim=0)                       # row of max |entry| per column
        signs = torch.sign(pe[max_idx, torch.arange(self.k)])
        signs[signs == 0] = 1.0
        data.x[:, -self.k:] = pe * signs
        return data


class Nano(InMemoryDataset):
    """
    Provide the nano component used by models / mpnn / dataset.py.


    Role:
        Nano groups state and methods for this repository component.
    """

    def __init__(self, root, is_weighted, use_node_feats=True, dim=None):
        """
        Initialize the Nano instance and store constructor configuration.

        Args:
            root: Caller-supplied value used by this routine.
            is_weighted: Caller-supplied value used by this routine.
            use_node_feats: Caller-supplied value used by this routine.
            dim: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        # Optional node features:
        #  - True  => 5× LocalDegreeProfile + 30× Laplacian PE  (total 35 dims)
        #  - False => 1D constant feature per node (keeps the models working)
        if use_node_feats:
            transform = Compose([
                LocalDegreeProfile(),
                AddLaplacianEigenvectorPE(k=30, attr_name=None, is_undirected=True),
                CanonicalLapPESign(k=30),  # audit I5: deterministic, PPGN-aligned PE signs
            ])
        else:
            # Provide a minimal 1D feature so GNNs still have x:
            transform = Compose([Constant(value=1)])

        # `dim` is '2D' or '3D'. If caller didn't pass it, fall back to sniffing
        # the substring out of `root` (keeps predict.py and older callers working).
        if dim is None:
            if '3D' in root:
                dim = '3D'
            elif '2D' in root:
                dim = '2D'
            else:
                raise ValueError(
                    f"Nano: could not infer dim from root path '{root}'. "
                    f"Pass dim='2D' or dim='3D' explicitly."
                )
        assert dim in ('2D', '3D'), f"dim must be '2D' or '3D', got {dim!r}"
        self.dim = dim

        self.is_weighted = is_weighted
        self.use_node_feats = use_node_feats
        super().__init__(root, pre_transform=transform)
        self.data, self.slices, self.idx_split, self.meta_data = torch.load(self.processed_paths[0], weights_only=False)
        # Compute y range from training graphs only to avoid test-set leakage.
        # When no split is present (e.g. single-file inference), fall back to
        # the whole dataset — predict.py overrides these with the values saved
        # in the training checkpoint.
        train_idx = self.idx_split.get('train', []) if isinstance(self.idx_split, dict) else []
        if len(train_idx) > 0 and 'y' in self.slices:
            y_slices = self.slices['y']
            train_y = torch.cat(
                [self.data.y[int(y_slices[g]):int(y_slices[g + 1])] for g in train_idx],
                dim=0,
            )
            self.y_max, self.y_min = train_y.max(), train_y.min()
        else:
            self.y_max, self.y_min = self.data.y.max(), self.data.y.min()
        # deg_max is used by trainer.py to scale the 5 LocalDegreeProfile
        # columns (data.x[:, :5]). Compute it over the LDP columns of the
        # training graphs only; fall back to the legacy first-graph value when
        # no split is present (predict.py overrides with the checkpoint value).
        if len(train_idx) > 0 and 'x' in self.slices and self.data.x.size(1) >= 5:
            x_slices = self.slices['x']
            train_x_ldp = torch.cat(
                [self.data.x[int(x_slices[g]):int(x_slices[g + 1]), :5] for g in train_idx],
                dim=0,
            )
            self.deg_max = train_x_ldp.max()
        else:
            self.deg_max = self.data.x[0].max()
        self.new_min, self.new_max = 0, 1

    def set_y_range(self, y_min, y_max):
        """
        Set y range state used by later calls.

        Args:
            y_min: Caller-supplied value used by this routine.
            y_max: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.y_min = torch.as_tensor(y_min, dtype=self.data.y.dtype)
        self.y_max = torch.as_tensor(y_max, dtype=self.data.y.dtype)

    def set_deg_max(self, deg_max):
        """
        Set deg max state used by later calls.

        Args:
            deg_max: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.deg_max = torch.as_tensor(deg_max, dtype=self.data.x.dtype)

    def scale_y(self, y):
        """
        Map physical target values into the configured normalized interval.

        Args:
            y: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        return (y - self.y_min) / (self.y_max - self.y_min) * (self.new_max - self.new_min) + self.new_min

    def inverse_scale_y(self, y):
        """
        Map physical target values into the configured normalized interval.

        Args:
            y: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        return (y - self.new_min) / (self.new_max - self.new_min) * (self.y_max - self.y_min) + self.y_min

    @property
    def processed_dir(self) -> str:
        """
        Convert raw graph files into processed tensors for later loading.

        Returns:
            Computed value used by the caller.
        """
        flag = 'weighted' if self.is_weighted else 'unweighted'
        nf = 'nodefeats_on' if getattr(self, 'use_node_feats', True) else 'nodefeats_off'
        if self.dim == '3D':
            return osp.join(self.root, f'processed_{nf}')
        else:
            return osp.join(self.root, f'processed_{flag}_{nf}')

    @property
    def raw_file_names(self):
        """
        Implement the raw file names step for models / mpnn / dataset.py.

        Returns:
            Computed value used by the caller.
        """
        return []

    @property
    def processed_file_names(self):
        """
        Convert raw graph files into processed tensors for later loading.

        Returns:
            Computed value used by the caller.
        """
        return [f'data.pt']

    def download(self):
        """
        Implement the download step for models / mpnn / dataset.py.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        pass

    def process(self):
        """
        Convert raw graph files into processed tensors for later loading.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        file_name = self.get_file_name()
        print(f'Processing {file_name}...')
        idx_split = self.get_idx_split()

        raw_file = open(osp.join(self.root, file_name), 'r').read().split('Simulation id: ')
        global_meta_data = self.get_global_meta_data(raw_file[0])

        data_list = []
        for i in tqdm(range(1, len(raw_file))):
            graph = raw_file[i].strip().split('\n')

            if self.dim == '2D':
                meta_data = global_meta_data
            else:
                assert self.dim == '3D'
                meta_data = {each.split(': ')[0]: eval(each.split(': ')[1]) for each in graph[2].split(', ')}
                del graph[1:4]

            data = self.get_graph_data(graph)
            assert data.edge_index.shape[1] == data.y.shape[0] == meta_data['#edges'] * 2
            if data.edge_attr.shape[0] != 0:
                assert data.edge_attr.shape[0] == data.y.shape[0]
            assert data.edge_index.max() == meta_data['#cells'] - 1
            data.num_nodes = data.edge_index.max() + 1
            data.meta_data = meta_data
            data_list.append(data)

        assert len(data_list) == global_meta_data['Total graphs']
        assert len(idx_split['train']) + len(idx_split['val']) + len(idx_split['test']) in [0, global_meta_data['Total graphs']]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices, idx_split, global_meta_data), self.processed_paths[0])

    def save_preds(self, y_pred, y_true, log_dir):
        """
        Write predictions back into the benchmark text-file format.

        Args:
            y_pred: Caller-supplied value used by this routine.
            y_true: Caller-supplied value used by this routine.
            log_dir: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        file_name = self.get_file_name()
        raw_file = open(osp.join(self.root, file_name), 'r').read().split('Simulation id: ')

        new_file = [raw_file[0].strip() + ' predicted_length']
        y_pred = (each.item() for each in y_pred)
        y_true = (each.item() for each in y_true)
        for i in tqdm(range(1, len(raw_file))):
            graph = raw_file[i].strip().split('\n')
            lines_for_this_graph = []
            for j in range(len(graph)):
                if j == 0:
                    lines_for_this_graph.append('Simulation id: ' + graph[j])
                elif self.dim == '3D' and j in range(1, 4):
                    lines_for_this_graph.append(graph[j])
                else:
                    # Mirror the edge filter in get_graph_data: newer files
                    # may append node-feature lines after the edges; those
                    # don't correspond to a prediction, so copy them through.
                    fields = graph[j].split(' ')
                    is_edge = (len(fields) == 5) if self.is_weighted else (len(fields) in (3, 4))
                    if not is_edge:
                        lines_for_this_graph.append(graph[j])
                        continue
                    y = eval(fields[-1])
                    assert abs(y - next(y_true)) < 1.0e-5
                    lines_for_this_graph.append(graph[j] + f' {next(y_pred)}')
            new_file.append('\n'.join(lines_for_this_graph))

        with open(osp.join(log_dir, file_name.split('.txt')[0] + '_pred.txt'), 'w') as f:
            f.write('\n\n'.join(new_file))

    def get_graph_data(self, graph):
        """
        Parse one raw graph block into PyTorch Geometric tensors.

        Args:
            graph: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        gid = graph[0]
        edge_index = []
        edge_attr = []
        y = []
        for edge in graph[1:]:
            edge_feat = edge.split(' ')
            n = len(edge_feat)
            # Newer files append a node-feature block after the edges; stop
            # once a line no longer looks like an edge. Node features are
            # recomputed by the PyG pre_transform, so we ignore the inline ones.
            if self.is_weighted:
                if n != 5:
                    break
                edge_attr.append([float(each) for each in edge_feat[2:-1]])
            else:
                if n not in (3, 4):
                    break
            edge_index.append([int(each) - 1 for each in edge_feat[:2]])
            y.append(float(edge_feat[-1]))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(-1, 1)
        return Data(edge_index=edge_index, edge_attr=edge_attr, y=y, gid=gid)

    def get_idx_split(self):
        """
        Load train/validation/test split indices when split files are present.

        Returns:
            Computed value used by the caller.
        """
        if not osp.isfile(osp.join(self.root, 'train.inds')):
            print('No split file found.')
            return {'train': [], 'val': [], 'test': []}

        train_idx = np.array(element_rstrip(open(osp.join(self.root, 'train.inds'), 'r').readlines()), dtype=int)
        val_idx = np.array(element_rstrip(open(osp.join(self.root, 'val.inds'), 'r').readlines()), dtype=int)
        test_idx = np.array(element_rstrip(open(osp.join(self.root, 'test.inds'), 'r').readlines()), dtype=int)
        return {'train': train_idx.tolist(), 'val': val_idx.tolist(), 'test': test_idx.tolist()}

    def get_global_meta_data(self, raw_meta_data):
        """
        Parse global metadata from a raw benchmark file header.

        Args:
            raw_meta_data: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        meta_data = []
        raw_meta_data = raw_meta_data.split('\n\n')
        meta_data.append(raw_meta_data[0].split(': '))
        if self.dim == '2D':
            meta_data.extend([each.split(': ') for each in raw_meta_data[1].split(', ')])
        return {each[0]: eval(each[1]) for each in meta_data}

    def get_file_name(self):
        """
        Select the raw dataset text file used by this dataset object.

        Returns:
            Computed value used by the caller.
        """
        all_file_names = os.listdir(self.root)
        if self.is_weighted is None:
            assert self.dim == '3D'
            for file_name in all_file_names:
                if '.txt' in file_name:
                    return file_name
        elif self.is_weighted:
            assert self.dim == '2D'
            for file_name in all_file_names:
                if '_weighted.txt' in file_name:
                    return file_name
        else:
            assert self.dim == '2D'
            for file_name in all_file_names:
                if '_unweighted.txt' in file_name:
                    return file_name

def element_rstrip(a):
    """
    Implement the element rstrip step for models / mpnn / dataset.py.

    Args:
        a: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    return [each.rstrip() for each in a]
