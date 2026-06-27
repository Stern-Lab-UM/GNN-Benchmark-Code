"""Edge-regression head: a standard concat-MLP head with linear output, no sigmoid.

Replaces the link-prediction-style LinkPredictor (sigmoid(MLP(x_i ⊙ x_j))) used
in the original benchmark. The motivation is to remove the sigmoid-floor
saturation that pins PNA's prediction at the T1 edge to y_min for all graphs.

Input per edge: concat([h_u, h_v, edge_attr]).
Output:         scalar (raw, unscaled).
"""
import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP


class EdgeRegressor(nn.Module):
    """
    Provide the edge regressor component used by models / mpnn / models / edge_regressor.py.


    Role:
        EdgeRegressor groups state and methods for this repository component.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, edge_dim=2):
        """
        Initialize the EdgeRegressor instance and store constructor configuration.

        Args:
            in_channels: Caller-supplied value used by this routine.
            hidden_channels: Caller-supplied value used by this routine.
            out_channels: Caller-supplied value used by this routine.
            num_layers: Caller-supplied value used by this routine.
            dropout: Caller-supplied value used by this routine.
            edge_dim: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        super().__init__()
        self.edge_dim = int(edge_dim) if edge_dim is not None else 0
        mlp_in = 2 * in_channels + self.edge_dim
        self.mlp = MLP(in_channels=mlp_in, hidden_channels=hidden_channels,
                       out_channels=out_channels, num_layers=num_layers,
                       dropout=dropout, norm='instance_norm', act='relu',
                       plain_last=True)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Run the neural-network forward pass for this module.

        Args:
            x: Caller-supplied value used by this routine.
            edge_index: Caller-supplied value used by this routine.
            edge_attr: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        x_i, x_j = x[edge_index[0]], x[edge_index[1]]
        if edge_attr is not None and self.edge_dim > 0:
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)
        return self.mlp(h)  # linear output, no sigmoid
