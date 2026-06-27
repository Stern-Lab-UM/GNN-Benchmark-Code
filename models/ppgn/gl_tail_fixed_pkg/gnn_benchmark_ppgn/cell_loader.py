"""Utilities for models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / cell_loader.py in the GNN Benchmark codebase."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import itertools

from gnn_benchmark_ppgn.file_reader import FileReader


class CellLoader():
    """
    Coordinate cell loader responsibilities for the gnn_benchmark_ppgn PPGN workflow.


    Role:
        CellLoader groups state and methods for this repository component.
    """
    def __init__(self):
        """
        Initialize the CellLoader instance and store constructor configuration.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        # user can modify before loading
        self.subset = None  # load a subset of the overall dataset
        # which features to target
        self.input_features = None  # List of str
        self.target_features = None  # List of str
        self.node_inds = (None, None)  # input and output indices
        self.batch_size = 64
        self.training_fraction = 0.8
        self.normalize = "all"
        self.non_negative = None
        # When True, z-score each input feature per-graph using that
        # graph's own mean/std rather than global training-set stats.
        # Only features also listed in self.normalize are affected;
        # target features keep their global (training-only) normalization.
        self.instance_normalization = False

        # may be loaded from previous instance
        self.norm_factors = None  # pd.Dataframe with mean and std
        self.indices = None  # dict of indices for train, val and test

        # generated while loading
        self.data = None  # dict of DataLoaders

    def apply_config(self, config):
        """
        Implement the apply config step for models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / cell_loader.py.

        Args:
            config: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        config_settings = {
            'training_info': ('subset', 'batch_size',
                              'training_fraction', 'input_features',
                              'target_features', 'normalize', 'non_negative',
                              'instance_normalization'),
        }

        for category, keys in config_settings.items():
            if category in config:
                for key in keys:
                    if key in config[category]:
                        self.__dict__[key] = config[category][key]

    def load_cells(self, file, apply_norm=True, only_test=False):
        """
        Load graphs and split assignments into memory for training/evaluation.

        Args:
            file: Caller-supplied value used by this routine.
            apply_norm: Caller-supplied value used by this routine.
            only_test: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        if only_test:
            # want all graphs
            subset = self.subset
            if self.indices is None:
                self.subset = None
                # set indices so they aren't calculated
                self.indices = {
                    'train': [],
                    'val': [],
                    'test': [],
                }
            else:
                self.subset = self.indices['train']

        graphs, targets = self.read_cells(file)

        if self.indices is None:
            self.split_indices(graphs.shape[0],
                               self.training_fraction)

        if self.norm_factors is None and apply_norm and not only_test:
            self.calculate_norm_factors(
                graphs[self.indices['train']],
                targets[self.indices['train']])

        if apply_norm and self.norm_factors is not None:
            graphs, targets = self.apply_norm_factors(graphs, targets)

        # Instance (per-graph) normalization runs for both training and
        # prediction paths, regardless of apply_norm: during training it
        # augments the global pass (which is a no-op for input features when
        # instance mode is on); during prediction the model's saved in_means
        # and in_stds are identity, so the loader must do the normalization.
        if self.instance_normalization:
            graphs = self.apply_input_instance_norm(graphs)

        if only_test:
            self.subset = subset  # recover saved value
            # set index of test
            self.indices['test'] = np.arange(graphs.shape[0])
            self.data = {'test': self.build_dataloader(graphs, targets),
                         'train': None,
                         'val': None}
        else:
            self.data = {
                name: self.build_dataloader(graphs[inds], targets[inds])
                for name, inds in self.indices.items()}

        return graphs.shape[0]

    def read_cells(self, file):
        """
        Read cell graph tensors and target arrays from the configured file.

        Args:
            file: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        reader = FileReader(self.subset,
                            self.target_features,
                            self.input_features)

        graphs, targets = reader.read_cells(file)
        self.target_features = reader.targets
        self.input_features = reader.inputs
        self.node_inds = reader.node_inds

        return graphs, targets

    def split_indices(self, n, training_fraction=0.8):
        '''
        split the indices from 0..n, returning dictionary with
        train, val and test as keys, the indices as values
        Can specify the training fraction, the remainder will be split between
        validation and testing.
        Validation and testing will contain at least one value

        Args:
            n: Caller-supplied value used by this routine.
            training_fraction: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        '''
        if n < 3:
            raise ValueError('Must have at least 3 indices to split')

        indices = np.arange(n)
        np.random.shuffle(indices)

        num_val = np.ceil((1-training_fraction) * n / 2)
        num_val = int(num_val if num_val > 0 else 1)  # at least 1
        num_test = n - 2*num_val

        self.indices = {
            'train': indices[:num_test],
            'val': indices[num_test:num_test + num_val],
            'test': indices[num_test + num_val:],
        }

    def calculate_norm_factors(self, training_graphs, training_targets):
        """
        Compute training-set normalization factors for inputs and targets.

        Args:
            training_graphs: Caller-supplied value used by this routine.
            training_targets: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        if self.target_features is None or self.input_features is None:
            raise ValueError('No feature names are present!')

        # have to check first element to handle multiple sized graphs
        if len(self.target_features) != training_targets[0].shape[0]:
            raise ValueError(f'Expected {len(self.target_features)} features, '
                             f'have {training_targets[0].shape[0]} targets')

        if len(self.input_features) != training_graphs[0].shape[0]-1:
            raise ValueError(f'Expected {len(self.input_features)} features, '
                             f'have {training_graphs[0].shape[0]-1} inputs')

        input_node, output_node = self.node_inds

        self.norm_factors = pd.DataFrame(
            columns=['mean', 'std'],
            index=self.input_features + self.target_features,
            dtype=float)

        stats = {
            'in_edge': StreamingStats(),
            'in_node': StreamingStats(),
            'out_edge': StreamingStats(),
            'out_node': StreamingStats(),
        }

        has_inputs = len(self.input_features) != 0
        has_input_nodes = input_node != len(self.input_features) + 1
        has_output_nodes = output_node != len(self.target_features)

        for graph, target in zip(training_graphs, training_targets):
            if has_inputs:
                stats['in_edge'].update(
                    graph[1:input_node, graph[0].astype(bool)])
            if has_input_nodes:
                stats['in_node'].update(
                    graph[input_node:].diagonal(axis1=1, axis2=2))

            stats['out_edge'].update(
                target[:output_node, graph[0].astype(bool)])

            if has_output_nodes:
                stats['out_node'].update(
                    target[output_node:].diagonal(axis1=1, axis2=2))

        self.norm_factors.loc[:, 'mean'] = list(itertools.chain(
            *[s.mean for s in stats.values()]))
        self.norm_factors.loc[:, 'std'] = list(itertools.chain(
            *[s.std for s in stats.values()]))

        # handle non-negative targets
        if self.non_negative is not None:
            for feature, max_val in zip(
                self.target_features,
                itertools.chain(stats['out_edge'].max, stats['out_node'].max)
            ):
                if self.non_negative == 'all' or feature in self.non_negative:
                    self.norm_factors.loc[feature, :] = [0, max_val]

        # handle normalization overrides
        if self.normalize == 'all':
            return

        if self.normalize is None:
            self.norm_factors['mean'] = 0
            self.norm_factors['std'] = 1
            return

        for feature in self.norm_factors.index:
            if feature not in self.normalize:
                self.norm_factors.loc[feature, :] = [0, 1]

        if self.instance_normalization:
            # In instance mode the global pass must be a no-op for input
            # features; the actual per-graph normalization happens in
            # apply_input_instance_norm. Target features keep their global
            # (training-only) mean/std so the model's saved out_means/out_stds
            # still correctly denormalize predictions at inference time.
            for feature in self.input_features:
                self.norm_factors.loc[feature, :] = [0, 1]

    def apply_input_instance_norm(self, graphs):
        '''
        Per-graph (instance) z-scoring of input features.

        For each graph independently, compute that graph's own mean/std for
        every input feature in self.normalize and normalize in place. Edge
        features are pooled over existing edges only (where row 0 = adjacency
        is nonzero); node features are pooled over the diagonal.
        Features not listed in self.normalize are left untouched. No-op if
        there are no input features, or if self.normalize is None.

        Args:
            graphs: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        '''
        if len(self.input_features) == 0:
            return graphs
        if self.normalize is None:
            return graphs

        normalize_all = (self.normalize == 'all')

        input_node, _ = self.node_inds
        if input_node is None:
            input_node = len(self.input_features) + 1

        n_edge_input = input_node - 1
        edge_names = self.input_features[:n_edge_input]
        node_names = self.input_features[n_edge_input:]

        def _should_norm(name):
            """
            Implement the should norm step for models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / cell_loader.py.

            Args:
                name: Caller-supplied value used by this routine.

            Returns:
                Computed value used by the caller.
            """
            return normalize_all or (name in self.normalize)

        eps = 0.01  # matches the floor used by StreamingStats
        for i in range(len(graphs)):
            adj_mask = graphs[i][0].astype(bool)
            # Edge features: rows 1 .. input_node
            for j, name in enumerate(edge_names):
                if not _should_norm(name):
                    continue
                row = 1 + j
                vals = graphs[i][row][adj_mask]
                if vals.size < 2:
                    continue
                m = vals.mean()
                s = max(vals.std(ddof=1), eps)
                graphs[i][row] = ((graphs[i][row] - m) / s) * graphs[i][0]
            # Node features: rows input_node .. end of input block
            if len(node_names) > 0:
                n = graphs[i].shape[2]
                eye = np.eye(n)
                for j, name in enumerate(node_names):
                    if not _should_norm(name):
                        continue
                    row = input_node + j
                    diag = np.diag(graphs[i][row])
                    if diag.size < 2:
                        continue
                    m = diag.mean()
                    s = max(diag.std(ddof=1), eps)
                    graphs[i][row] = ((graphs[i][row] - m) / s) * eye
        return graphs

    def apply_norm_factors(self, graphs, targets):
        """
        Apply stored normalization factors to graph tensors and targets.

        Args:
            graphs: Caller-supplied value used by this routine.
            targets: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        if self.norm_factors is None:
            raise ValueError('No norm factors are loaded!')

        if len(self.norm_factors) != \
                (targets[0].shape[0] + graphs[0].shape[0] - 1):
            raise ValueError(f'Expected {len(self.norm_factors)} features, '
                             f'have {targets.shape[1] + graphs.shape[1] - 1} '
                             'targets')

        if self.norm_factors.columns.tolist() != ['mean', 'std']:
            raise ValueError('DataFrame column names are unexpected!')

        mean = self.norm_factors.loc[self.target_features, 'mean'].to_numpy()
        std = self.norm_factors.loc[self.target_features, 'std'].to_numpy()

        input_node, output_node = self.node_inds
        if input_node is None:
            input_node = len(self.input_features) + 1

        if output_node is None:
            output_node = len(self.target_features)

        for i in range(len(graphs)):
            targets[i][:output_node] = (
                (targets[i][:output_node] - mean[:output_node, None, None])
                / std[:output_node, None, None]) * graphs[i][[0], :, :]
            # node targets
            targets[i][output_node:] = (
                (targets[i][output_node:] - mean[output_node:, None, None])
                / std[output_node:, None, None]) * np.eye(targets[i].shape[2])

        if len(self.input_features) != 0:
            mean = self.norm_factors.loc[
                self.input_features, 'mean'].to_numpy()
            std = self.norm_factors.loc[self.input_features, 'std'].to_numpy()

            for i in range(len(graphs)):
                graphs[i][1:input_node, ...] = (
                    (graphs[i][1:input_node, ...]
                     - mean[:input_node-1, None, None])
                    / std[:input_node-1, None, None]) * graphs[i][[0], :, :]
                # node inputs
                graphs[i][input_node:, ...] = (
                    (graphs[i][input_node:, ...]
                     - mean[input_node-1:, None, None])
                    / std[input_node-1:, None, None]
                ) * np.eye(graphs[i].shape[2])

        return graphs, targets

    def build_dataloader(self, graphs, targets):
        """
        Wrap graph examples in PyTorch DataLoader objects grouped by split.

        Args:
            graphs: Caller-supplied value used by this routine.
            targets: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        return DataLoader(
            CellsData(graphs, targets),
            batch_size=self.batch_size,
            collate_fn=collate_by_size
        )

    @property
    def train(self):
        """
        Implement the train step for models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / cell_loader.py.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        for batch in self.data['train']:
            for size in batch:
                yield size

    @property
    def test(self):
        """
        Evaluate the test split.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        for batch in self.data['test']:
            for size in batch:
                yield size

    @property
    def val(self):
        """
        Implement the val step for models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / cell_loader.py.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        for batch in self.data['val']:
            for size in batch:
                yield size


def collate_by_size(batch):
    """
    Batch same-sized PPGN graph tensors together.

    Args:
        batch: Caller-supplied value used by this routine.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    """
    # batch is a list of (graph, target) tuples
    # iterate over all graphs (x[0]) and store the number of nodes (shape[-1])
    sizes = set(x[0].shape[-1] for x in batch)
    for size in sizes:
        yield (
            torch.stack([torch.from_numpy(x[0]).to(torch.float) for x in batch
                         if x[0].shape[-1] == size]),
            torch.stack([torch.from_numpy(x[1]).to(torch.float) for x in batch
                         if x[1].shape[-1] == size])
        )


class CellsData(Dataset):
    """
    Provide the cells data component used by models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / cell_loader.py.


    Role:
        CellsData groups state and methods for this repository component.
    """
    def __init__(self, graphs, targets):
        """
        Initialize the CellsData instance and store constructor configuration.

        Args:
            graphs: Caller-supplied value used by this routine.
            targets: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.graphs = graphs
        self.targets = targets

    def __len__(self):
        """
        Returns the length of the dataset

        Returns:
            Computed value used by the caller.
        """
        return self.graphs.shape[0]

    def __getitem__(self, index):
        """
        Generates a single instance of data

        Args:
            index: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        return self.graphs[index], self.targets[index]


class StreamingStats():
    '''
    Keeps running tally of values to determine mean and stdev
    Assumes you want to perform operations along axis 1 of values

    Role:
        StreamingStats groups state and methods for this repository component.
    '''
    def __init__(self):
        """
        Initialize the StreamingStats instance and store constructor configuration.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.sum = 0
        self.count = 0
        self.sum_sq = 0
        self.max = None

    def update(self, values: np.array):
        """
        Implement the update step for models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / cell_loader.py.

        Args:
            values: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.count += values.shape[-1]
        self.sum += values.sum(axis=-1)
        self.sum_sq += (values**2).sum(axis=-1)
        if len(values) > 0:
            if self.max is None:
                self.max = values.max(axis=-1)
            else:
                self.max = np.maximum(
                    self.max,
                    values.max(axis=-1))

    @property
    def mean(self):
        """
        Implement the mean step for models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / cell_loader.py.

        Returns:
            Computed value used by the caller.
        """
        if self.count == 0:
            return []
        return (self.sum / self.count).tolist()

    @property
    def std(self):
        """
        Implement the std step for models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / cell_loader.py.

        Returns:
            Computed value used by the caller.
        """
        if self.count == 0:
            return []
        # set 0.01 as minimum for stdev
        return np.fmax(np.sqrt((self.sum_sq - self.sum**2 / self.count) /
                               (self.count - 1)),
                       0.01).tolist()
