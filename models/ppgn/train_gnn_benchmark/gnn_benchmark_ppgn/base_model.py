"""Utilities for models / ppgn / train_gnn_benchmark / gnn_benchmark_ppgn / base_model.py in the GNN Benchmark codebase."""

import torch
import torch.nn as nn
from torch.utils import checkpoint
from gnn_benchmark_ppgn.modules import RegularBlock
import numpy as np


class BaseModel(nn.Module):
    """
    implement the PPGN model wrapper


    Role:
        BaseModel groups state and methods for this repository component.
    """
    def __init__(self, in_features, out_features,
                 block_features, depth_of_mlp, disable_first_skip=False):
        """
        Build the model computation graph, until scores/values
        are returned at the end

        Args:
            in_features: Caller-supplied value used by this routine.
            out_features: Caller-supplied value used by this routine.
            block_features: Caller-supplied value used by this routine.
            depth_of_mlp: Caller-supplied value used by this routine.
            disable_first_skip: Ablation switch. False keeps the normal PPGN;
                True removes only the first RegularBlock skip path.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        super().__init__()

        self.num_targets = out_features
        self.disable_first_skip = bool(disable_first_skip)

        # default no scaling
        self.in_means = np.zeros(in_features)
        self.in_stds = np.ones(in_features)
        self.out_means = np.zeros(out_features)
        self.out_stds = np.ones(out_features)
        # indices of nodes in inputs/outputs, used to handle scaling/masking
        self.in_nodes = in_features
        self.out_nodes = out_features
        # target indices to constrain as non-negative
        self.non_negative = []
        # targets to mask as along edges only
        self.edge_only = list(range(out_features))
        self.relu = torch.nn.ReLU(inplace=True)

        # sequential ppgn blocks
        last_layer_features = in_features
        self.reg_blocks = nn.ModuleList()
        for block_i, next_layer_features in enumerate(block_features):
            skip_enabled = not (self.disable_first_skip and block_i == 0)
            self.reg_blocks.append(RegularBlock(
                last_layer_features, next_layer_features, depth_of_mlp,
                skip_enabled=skip_enabled))
            last_layer_features = next_layer_features

        # last layer to out_features
        self.reg_blocks.append(RegularBlock(
            last_layer_features, self.num_targets, depth_of_mlp))

    def forward(self, x):
        """
        Run the neural-network forward pass for this module.

        Args:
            x: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        # x of shape (B,X,N,N)
        adj = self.output_mask(x)
        for block in self.reg_blocks:
            if x.requires_grad:
                x = checkpoint.checkpoint(block, x, preserve_rng_state=False)
            else:
                x = block(x)

        x[~adj] = 0.  # Remove values outside of the original adjacency

        # symmetric constraint in undirected graph
        x = (x + torch.transpose(x, 2, 3))/2

        # relu on non_negative indices, inplace
        for ind in self.non_negative:
            self.relu(x[:, ind])

        return x

    def predict(self, x):
        """
        Run model inference and return or save predictions.

        Args:
            x: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        self.eval()
        with torch.no_grad():
            adj = self.output_mask(x)
            x = ((x - self.in_means[None, :, None, None]) /
                 self.in_stds[None, :, None, None] *
                 self.input_mask(x)).to(torch.float)

            for block in self.reg_blocks:
                x = block(x)

            x = (x * self.out_stds[None, :, None, None] +
                 self.out_means[None, :, None, None])

            x[~adj] = 0.  # Remove values outside of the original adjacency

            # symmetric constraint in undirected graph
            x = (x + torch.transpose(x, 2, 3))/2

            # relu on non_negative indices, inplace
            for ind in self.non_negative:
                self.relu(x[:, ind])

            return x

    def output_mask(self, x):
        """
        Implement the output mask step for models / ppgn / train_gnn_benchmark / gnn_benchmark_ppgn / base_model.py.

        Args:
            x: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        return self._generate_mask(x, self.num_targets, self.out_nodes, bool)

    def input_mask(self, x):
        """
        Implement the input mask step for models / ppgn / train_gnn_benchmark / gnn_benchmark_ppgn / base_model.py.

        Args:
            x: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        return self._generate_mask(x, x.shape[1], self.in_nodes, torch.float)

    def _generate_mask(self, x, total, nodes, dtype):
        '''
        True where value should be retained, false where value should be
        zeroed
        edge_only is applied only to output masks (dtype == bool)

        Args:
            x: Caller-supplied value used by this routine.
            total: Caller-supplied value used by this routine.
            nodes: Caller-supplied value used by this routine.
            dtype: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        '''
        # assume x[:,0] is adjacency
        adj = x[:, [0], :, :].to(dtype).repeat(1, total, 1, 1)
        if dtype is bool:
            for i in range(nodes):
                # want to keep the mask as is
                if i in self.edge_only:
                    continue
                # want to retain all values
                adj[:, i, :, :] = ~torch.eye(x.shape[-1], dtype=bool)

        # mask with adjacency until out_nodes, then mask diagonal
        adj[:, nodes:total, :, :] = torch.eye(x.shape[-1],
                                              dtype=dtype)
        return adj

    def set_norm_factors(self, norm_factors, config, device):
        """
        Set norm factors state used by later calls.

        Args:
            norm_factors: Caller-supplied value used by this routine.
            config: Caller-supplied value used by this routine.
            device: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        names = config.config['training_info']['target_features']
        self.out_means = torch.from_numpy(
            norm_factors.loc[names, 'mean'].values).to(torch.float).to(device)
        self.out_stds = torch.from_numpy(
            norm_factors.loc[names, 'std'].values).to(torch.float).to(device)

        names = config.config['training_info']['input_features']
        # don't scale the adjacency (input 1)
        self.in_means[1:] = norm_factors.loc[names, 'mean'].values
        self.in_means = torch.from_numpy(
            self.in_means).to(torch.float).to(device)
        self.in_stds[1:] = norm_factors.loc[names, 'std'].values
        self.in_stds = torch.from_numpy(
            self.in_stds).to(torch.float).to(device)

    def set_indices(self, config):
        """
        Set indices state used by later calls.

        Args:
            config: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        if 'input_node' in config.config['training_info']:
            self.in_nodes = config.config['training_info']['input_node']
        if 'target_node' in config.config['training_info']:
            self.out_nodes = config.config['training_info']['target_node']

    def set_non_negative(self, config):
        """
        Set non negative state used by later calls.

        Args:
            config: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        if 'non_negative' in config.config['training_info']:
            non_neg = config.config['training_info']['non_negative']
            targets = config.config['training_info']['target_features']

            self.non_negative = self._find_in_targets(non_neg, targets)

    def set_edge_only(self, config):
        """
        Set edge only state used by later calls.

        Args:
            config: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        if 'edge_only' in config.config['training_info']:
            edge_only = config.config['training_info']['edge_only']
            targets = config.config['training_info']['target_features']

            self.edge_only = self._find_in_targets(edge_only, targets)

    def _find_in_targets(self, args, targets):
        """
        Implement the find in targets step for models / ppgn / train_gnn_benchmark / gnn_benchmark_ppgn / base_model.py.

        Args:
            args: Caller-supplied value used by this routine.
            targets: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        if args is None:
            return []
        if args == 'all':
            return list(range(len(targets)))
        return [targets.index(n) for n in args if n in targets]

    def initialize_from(self, other):
        """
        Implement the initialize from step for models / ppgn / train_gnn_benchmark / gnn_benchmark_ppgn / base_model.py.

        Args:
            other: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        if len(self.reg_blocks) != len(other.reg_blocks):
            raise ValueError('Unable to initialize model from previous, '
                             'different number of layers')

        for mine, theirs in zip(self.reg_blocks[1:-1],
                                other.reg_blocks[1:-1]):
            if not mine.compatible_size(theirs):
                raise ValueError('Unable to initialize model from previous, '
                                 'different layer architecture')

        for mine, theirs in zip(self.reg_blocks, other.reg_blocks):
            # test compatable sizes, allows first and last layer to differ
            if mine.compatible_size(theirs):
                mine.load_state_dict(theirs.state_dict())
