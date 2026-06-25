import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP


class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.mlp = MLP(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                       num_layers=num_layers, dropout=dropout, norm='instance_norm', act='relu', plain_last=True)

    def forward(self, x, edge_index):
        x_i, x_j = x[edge_index[0]], x[edge_index[1]]
        x = x_i * x_j
        x = self.mlp(x)
        return torch.sigmoid(x)
