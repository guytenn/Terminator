import numpy as np
from typing import Dict, List
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, same_padding, \
    SlimConv2d, SlimFC
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


class CNN(nn.Module):
    def __init__(self,
                 obs_shape: tuple,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 final_activation="relu"):
        nn.Module.__init__(self)

        self.model_config = model_config
        self.final_activation = final_activation

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, \
            "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch")

        self._logits = None

        layers = []
        w, h, in_channels = obs_shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation))
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]

        layers.append(
            SlimConv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                None,  # padding=valid
                activation_fn=activation))

        in_size = [
            np.ceil((in_size[0] - kernel[0]) / stride),
            np.ceil((in_size[1] - kernel[1]) / stride)
        ]
        padding, _ = same_padding(in_size, [1, 1], [1, 1])

        layers.append(nn.Flatten())
        in_size = out_channels
        # Add (optional) post-fc-stack after last Conv2D layer.
        for i, out_size in enumerate(post_fcnet_hiddens +
                                     [num_outputs]):
            layers.append(
                SlimFC(
                    in_size=in_size,
                    out_size=out_size,
                    activation_fn=post_fcnet_activation
                    if i < len(post_fcnet_hiddens) - 1 else None,
                    initializer=normc_initializer(1.0)))
            in_size = out_size
        # Last layer is logits layer.
        self._logits = layers.pop()

        self._convs = nn.Sequential(*layers)

        # Holds the current "base" output (before logits layer).
        self._features = None

    def forward(self, x):
        self._features = x.float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        conv_out = self._logits(conv_out)

        if len(conv_out.shape) == 4:
            if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                raise ValueError(
                    "Given `conv_filters` ({}) do not result in a [B, {} "
                    "(`num_outputs`), 1, 1] shape (but in {})! Please "
                    "adjust your Conv2D stack such that the last 2 dims "
                    "are both 1.".format(self.model_config["conv_filters"],
                                         self.num_outputs,
                                         list(conv_out.shape)))
            logits = conv_out.squeeze(3)
            logits = logits.squeeze(2)
        else:
            logits = conv_out

        if self.final_activation == "relu":
            return torch.relu(logits)
        else:
            return logits
