import torch
import torch.nn as nn
from torch.utils import checkpoint
from dcg.modules import RegularBlock
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, in_features, out_features,
                 block_features, depth_of_mlp):
        """
        Build the model computation graph, until scores/values
        are returned at the end
        """
        super().__init__()

        self.num_targets = out_features

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
        for next_layer_features in block_features:
            self.reg_blocks.append(RegularBlock(
                last_layer_features, next_layer_features, depth_of_mlp))
            last_layer_features = next_layer_features

        # last layer to out_features
        self.reg_blocks.append(RegularBlock(
            last_layer_features, self.num_targets, depth_of_mlp))

    def forward(self, x):
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
        return self._generate_mask(x, self.num_targets, self.out_nodes, bool)

    def input_mask(self, x):
        return self._generate_mask(x, x.shape[1], self.in_nodes, torch.float)

    def _generate_mask(self, x, total, nodes, dtype):
        '''
        True where value should be retained, false where value should be
        zeroed
        edge_only is applied only to output masks (dtype == bool)
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
        if 'input_node' in config.config['training_info']:
            self.in_nodes = config.config['training_info']['input_node']
        if 'target_node' in config.config['training_info']:
            self.out_nodes = config.config['training_info']['target_node']

    def set_non_negative(self, config):
        if 'non_negative' in config.config['training_info']:
            non_neg = config.config['training_info']['non_negative']
            targets = config.config['training_info']['target_features']

            self.non_negative = self._find_in_targets(non_neg, targets)

    def set_edge_only(self, config):
        if 'edge_only' in config.config['training_info']:
            edge_only = config.config['training_info']['edge_only']
            targets = config.config['training_info']['target_features']

            self.edge_only = self._find_in_targets(edge_only, targets)

    def _find_in_targets(self, args, targets):
        if args is None:
            return []
        if args == 'all':
            return list(range(len(targets)))
        return [targets.index(n) for n in args if n in targets]

    def initialize_from(self, other):
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
