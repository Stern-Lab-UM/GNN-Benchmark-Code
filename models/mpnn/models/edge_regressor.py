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
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, edge_dim=2):
        super().__init__()
        # Ablation: keep the trainer/predictor constructor signature unchanged,
        # but do not expose raw edge attributes to the final regression head.
        self.edge_dim = 0
        mlp_in = 2 * in_channels
        self.mlp = MLP(in_channels=mlp_in, hidden_channels=hidden_channels,
                       out_channels=out_channels, num_layers=num_layers,
                       dropout=dropout, norm='instance_norm', act='relu',
                       plain_last=True)

    def forward(self, x, edge_index, edge_attr=None):
        x_i, x_j = x[edge_index[0]], x[edge_index[1]]
        h = torch.cat([x_i, x_j], dim=-1)
        return self.mlp(h)  # linear output, no sigmoid
