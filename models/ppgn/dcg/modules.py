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
    def __init__(self, in_features, out_features, depth, use_skip=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.depth = depth
        self.use_skip = bool(use_skip)

        self.mlp1 = MlpBlock(in_features, out_features, depth)
        self.mlp2 = MlpBlock(in_features, out_features, depth)

        self.skip = (SkipConnection(in_features+out_features, out_features)
                     if self.use_skip else None)

    def forward(self, inputs):
        mult = torch.matmul(self.mlp1(inputs),
                            self.mlp2(inputs))

        if self.use_skip:
            out = self.skip(in1=inputs, in2=mult)
        else:
            out = mult
        return out

    def compatible_size(self, other):
        return (self.in_features == other.in_features and
                self.out_features == other.out_features and
                self.depth == other.depth and
                self.use_skip == other.use_skip)


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
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for _ in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features,
                                        kernel_size=1, padding=0, bias=True))
            in_features = out_features

    def forward(self, x):
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
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size=1, padding=0, bias=True)

    def forward(self, in1, in2):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out
