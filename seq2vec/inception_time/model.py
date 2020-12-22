import os

import torch
from torch import nn
import torch.nn.functional as F

from typing import cast, Union, List


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class InceptionModel(nn.Module):
    """A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    num_blocks:
        The number of inception blocks to use. One inception block consists
        of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
        connector
    in_channels:
        The number of input channels (i.e. input.shape[-1])
    out_channels:
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    bottleneck_channels:
        The number of channels to use for the bottleneck. Can be list or int. If 0, no
        bottleneck is applied
    kernel_sizes:
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    output_dim:
        The number of output features
    """

    def __init__(self, num_blocks: int, in_channels: int, out_channels: Union[List[int], int],
                 bottleneck_channels: Union[List[int], int], kernel_sizes: Union[List[int], int],
                 use_residuals: Union[List[bool], bool, str] = 'default', additional_input_dim: int = 1,
                 output_dim: int = 1
                 ) -> None:
        super().__init__()

        # for easier saving and loading
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = bottleneck_channels
        self.kernel_sizes = kernel_sizes
        self.use_residuals = use_residuals
        self.additional_input_dim = additional_input_dim
        self.output_dim = output_dim

        self.input_limits = None
        self.additional_input_limits = None
        self.output_limits = None

        channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels,
                                                                          num_blocks))
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels,
                                                                     num_blocks))
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(List[bool], self._expand_to_blocks(
            cast(Union[bool, List[bool]], use_residuals), num_blocks)
                             )

        # Inception-based feature extractor
        self.blocks = nn.Sequential(*[
            InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                           residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                           kernel_size=kernel_sizes[i]) for i in range(num_blocks)
        ])

        # dense net to fold in the additional input
        self.linear = nn.Sequential(
            nn.Linear(in_features=channels[-1] + self.additional_input_dim, out_features=64),
            nn.SELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.SELU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, self.output_dim)
        )

    @staticmethod
    def _expand_to_blocks(value: Union[int, bool, List[int], List[bool]],
                          num_blocks: int) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: torch.Tensor, additional_inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = x.permute(0, 2, 1)  # need (batch_size, features, seq)
        x = self.blocks(x).mean(dim=-1)  # the mean is the global average pooling
        x = torch.cat((x, additional_inputs), dim=-1)
        return self.linear(x)
    
    def save(self, archive_dir, filename=None):
        print("Saving model to {}".format(archive_dir))
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        filename = filename if filename is not None else "model.pt"
        torch.save({
            "state_dict": self.state_dict(),
            "num_blocks": self.num_blocks,
            "in_channels": self.in_channels,
            "out_channels" : self.out_channels,
            "bottleneck_channels" : self.bottleneck_channels,
            "kernel_sizes" : self.kernel_sizes,
            "use_residuals" : self.use_residuals,
            "additional_input_dim" : self.additional_input_dim,
            "output_dim": self.output_dim,
            "input_limits": self.input_limits,
            "additional_input_limits": self.additional_input_limits,
            "output_limits": self.output_limits
        }, os.path.join(archive_dir, filename))

    @staticmethod
    def load(archive_file, device):
        print("Loading model from {}".format(archive_file))
        checkpoint = torch.load(archive_file, map_location=device)
        model = InceptionModel(checkpoint["num_blocks"], checkpoint["in_channels"], checkpoint["out_channels"],
                               checkpoint["bottleneck_channels"], checkpoint["kernel_sizes"],
                               checkpoint["use_residuals"], checkpoint["additional_input_dim"],
                               checkpoint["output_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        input_limits = checkpoint["input_limits"].to(device)
        additional_input_limits = checkpoint["additional_input_limits"].to(device)
        output_limits = checkpoint["output_limits"].to(device)
        model.input_limits = input_limits
        model.additional_input_limits = additional_input_limits
        model.output_limits = output_limits
        model = model.to(device)
        return model


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 residual: bool, stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41) -> None:
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels,
                                                kernel_size=1, bias=False)
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
                              kernel_size=kernel_size_s[i], stride=stride, bias=False)
            for i in range(len(kernel_size_s))
        ])

        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)

        if self.use_residual:
            x = x + self.residual(org_x)
        return x
