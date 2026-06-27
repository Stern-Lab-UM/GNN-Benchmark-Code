"""Utilities for models / mpnn / models / sage.py in the GNN Benchmark codebase."""

from torch import Tensor
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.nn import SAGEConv as BaseSAGEConv
from torch_geometric.nn.models import GraphSAGE as BaseGraphSAGE
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils import scatter


class GraphSAGE(BaseGraphSAGE):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        edge_dim=None,
        **kwargs,
    ):
        """
        Initialize the GraphSAGE instance and store constructor configuration.

        Args:
            in_channels: Caller-supplied value used by this routine.
            hidden_channels: Caller-supplied value used by this routine.
            num_layers: Caller-supplied value used by this routine.
            out_channels: Caller-supplied value used by this routine.
            dropout: Caller-supplied value used by this routine.
            act: Caller-supplied value used by this routine.
            act_first: Caller-supplied value used by this routine.
            act_kwargs: Caller-supplied value used by this routine.
            norm: Caller-supplied value used by this routine.
            norm_kwargs: Caller-supplied value used by this routine.
            jk: Caller-supplied value used by this routine.
            edge_dim: Caller-supplied value used by this routine.
            **kwargs: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        super().__init__(hidden_channels, hidden_channels, num_layers, out_channels, dropout,
                         act, act_first, act_kwargs, norm, norm_kwargs, jk, **kwargs)
        self.node_encoder = MLP([in_channels, hidden_channels])
        if edge_dim:
            self.edge_encoder = MLP([edge_dim, hidden_channels])

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        """
        Implement the init conv step for models / mpnn / models / sage.py.

        Args:
            in_channels: Caller-supplied value used by this routine.
            out_channels: Caller-supplied value used by this routine.
            **kwargs: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        return SAGEConv(in_channels, out_channels, **kwargs)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ) -> Tensor:
        """

        Args:
            x: Caller-supplied value used by this routine.
            edge_index: Caller-supplied value used by this routine.
            edge_weight: Caller-supplied value used by this routine.
            edge_attr: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        xs: List[Tensor] = []
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.num_layers):
            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight,
                                  edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x


class SAGEConv(BaseSAGEConv):
    """
    Provide the sageconv component used by models / mpnn / models / sage.py.


    Role:
        SAGEConv groups state and methods for this repository component.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        """
        Initialize the SAGEConv instance and store constructor configuration.

        Args:
            in_channels: Caller-supplied value used by this routine.
            out_channels: Caller-supplied value used by this routine.
            aggr: Caller-supplied value used by this routine.
            normalize: Caller-supplied value used by this routine.
            root_weight: Caller-supplied value used by this routine.
            project: Caller-supplied value used by this routine.
            bias: Caller-supplied value used by this routine.
            **kwargs: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        super().__init__(in_channels, out_channels, aggr, normalize, root_weight, project, bias, **kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr=None,
                size: Size = None) -> Tensor:
        """

        Args:
            x: Caller-supplied value used by this routine.
            edge_index: Caller-supplied value used by this routine.
            edge_attr: Caller-supplied value used by this routine.
            size: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        row, col = edge_index
        msg = x[0][row]
        if edge_attr is not None:
            msg = (msg + edge_attr).relu()
        dim_size = x[1].size(0) if x[1] is not None else None
        out = scatter(msg, col, dim=0, dim_size=dim_size, reduce='mean')
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr) -> Tensor:
        """
        Implement the message step for models / mpnn / models / sage.py.

        Args:
            x_j: Caller-supplied value used by this routine.
            edge_attr: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        if edge_attr is not None:
            return (x_j + edge_attr).relu()
        return x_j
