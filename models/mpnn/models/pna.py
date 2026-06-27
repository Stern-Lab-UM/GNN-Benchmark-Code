"""Utilities for models / mpnn / models / pna.py in the GNN Benchmark codebase."""

from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.nn import PNAConv
from torch_geometric.nn.models import PNA as BasePNA


class PNA(BasePNA):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
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
            :class:`torch_geometric.nn.conv.PNAConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        """
        Implement the init conv step for models / mpnn / models / pna.py.

        Args:
            in_channels: Caller-supplied value used by this routine.
            out_channels: Caller-supplied value used by this routine.
            **kwargs: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        return PNAConv(in_channels, out_channels,
                       aggregators=['sum', 'mean', 'max', 'min', 'std'],
                       scalers=['identity', 'amplification', 'attenuation', 'linear', 'inverse_linear'],
                       **kwargs)
