"""Utilities for models / ppgn / train_dcg / dcg / modules.py in the DCG benchmark codebase."""

import torch
import torch.nn as nn


class RegularBlock(nn.Module):
    """
    Inputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes,
    multiply the result, and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth
    :param in_features: number of features for input tensor
    :param out_features: number of features for output tensor
    :param depth: depth of mlp
    :param x: Tensor of shape N x in_features x m x m
    :return: Tensor of shape N x out_features x m x m
    """
    def __init__(self, in_features, out_features, depth):
        """
        Initialize the RegularBlock instance and store constructor configuration.

        Args:
            in_features: Caller-supplied value used by this routine.
            out_features: Caller-supplied value used by this routine.
            depth: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.depth = depth

        self.mlp1 = MlpBlock(in_features, out_features, depth)
        self.mlp2 = MlpBlock(in_features, out_features, depth)

        self.skip = SkipConnection(in_features+out_features, out_features)

    def forward(self, inputs):
        """
        Run the neural-network forward pass for this module.

        Args:
            inputs: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        mult = torch.matmul(self.mlp1(inputs),
                            self.mlp2(inputs))

        out = self.skip(in1=inputs, in2=mult)
        return out

    def compatible_size(self, other):
        """
        Implement the compatible size step for models / ppgn / train_dcg / dcg / modules.py.

        Args:
            other: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        return (self.in_features == other.in_features and
                self.out_features == other.out_features and
                self.depth == other.depth)


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    :param in_features: number of features for input tensor
    :param out_features: number of features for output tensor
    :param depth_of_mlp: number of layers to use
    :param activation_fn: activate function to apply to each layer,
        default relu
    :param x: Tensor of shape N x in_features x m x m
    :return: Tensor of shape N x out_features x m x m
    """
    def __init__(self, in_features, out_features,
                 depth_of_mlp, activation_fn=nn.functional.relu):
        """
        Initialize the MlpBlock instance and store constructor configuration.

        Args:
            in_features: Caller-supplied value used by this routine.
            out_features: Caller-supplied value used by this routine.
            depth_of_mlp: Caller-supplied value used by this routine.
            activation_fn: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for _ in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features,
                                        kernel_size=1, padding=0, bias=True))
            in_features = out_features

    def forward(self, x):
        """
        Run the neural-network forward pass for this module.

        Args:
            x: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        for conv_layer in self.convs:
            x = self.activation(conv_layer(x))
        return x


class SkipConnection(nn.Module):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param in_features: d1+d2
    :param out_features: output num of features
    :return: Tensor of shape N x output_depth x m x m
    """
    def __init__(self, in_features, out_features):
        """
        Initialize the SkipConnection instance and store constructor configuration.

        Args:
            in_features: Caller-supplied value used by this routine.
            out_features: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size=1, padding=0, bias=True)

    def forward(self, in1, in2):
        """
        Run the neural-network forward pass for this module.

        Args:
            in1: Caller-supplied value used by this routine.
            in2: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out
