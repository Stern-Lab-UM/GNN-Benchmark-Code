"""Utilities for models / mpnn / models / link_predictor.py in the DCG benchmark codebase."""

import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP


class LinkPredictor(nn.Module):
    """
    Provide the link predictor component used by models / mpnn / models / link_predictor.py.


    Role:
        LinkPredictor groups state and methods for this repository component.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        """
        Initialize the LinkPredictor instance and store constructor configuration.

        Args:
            in_channels: Caller-supplied value used by this routine.
            hidden_channels: Caller-supplied value used by this routine.
            out_channels: Caller-supplied value used by this routine.
            num_layers: Caller-supplied value used by this routine.
            dropout: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        super(LinkPredictor, self).__init__()

        self.mlp = MLP(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                       num_layers=num_layers, dropout=dropout, norm='instance_norm', act='relu', plain_last=True)

    def forward(self, x, edge_index):
        """
        Run the neural-network forward pass for this module.

        Args:
            x: Caller-supplied value used by this routine.
            edge_index: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        x_i, x_j = x[edge_index[0]], x[edge_index[1]]
        x = x_i * x_j
        x = self.mlp(x)
        return torch.sigmoid(x)
