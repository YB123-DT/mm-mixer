"""PyTorch DenseFace-compatible DenseNet-BC-100 backbone.

The channel schedule matches the public DenseFace FER+ implementation:
growth rate 12, three 16-layer dense blocks, compression 0.5, and a
342-dimensional vector immediately before the eight-class classifier.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        bottleneck_channels = 4 * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels, growth_rate, 3, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        return torch.cat((x, out), dim=1)


class _DenseBlock(nn.Sequential):
    def __init__(self, in_channels: int, growth_rate: int, layers: int) -> None:
        modules = []
        for layer_index in range(layers):
            modules.append(_DenseLayer(in_channels + layer_index * growth_rate, growth_rate))
        super().__init__(*modules)


class _Transition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.relu(self.norm(x), inplace=True))
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class DenseFace(nn.Module):
    """DenseNet-BC-100 with a 342-D pre-classifier representation."""

    feature_dim = 342

    def __init__(
        self,
        num_classes: int = 8,
        growth_rate: int = 12,
        block_layers: tuple[int, int, int] = (16, 16, 16),
        compression: float = 0.5,
    ) -> None:
        super().__init__()
        if growth_rate != 12 or block_layers != (16, 16, 16) or compression != 0.5:
            raise ValueError(
                "DenseFace compatibility requires growth_rate=12, "
                "block_layers=(16, 16, 16), and compression=0.5"
            )

        channels = 2 * growth_rate
        self.stem = nn.Conv2d(1, channels, 3, stride=2, padding=1, bias=False)
        self.block1 = _DenseBlock(channels, growth_rate, block_layers[0])
        channels += block_layers[0] * growth_rate  # 216
        next_channels = int(channels * compression)
        self.transition1 = _Transition(channels, next_channels)
        channels = next_channels  # 108

        self.block2 = _DenseBlock(channels, growth_rate, block_layers[1])
        channels += block_layers[1] * growth_rate  # 300
        next_channels = int(channels * compression)
        self.transition2 = _Transition(channels, next_channels)
        channels = next_channels  # 150

        self.block3 = _DenseBlock(channels, growth_rate, block_layers[2])
        channels += block_layers[2] * growth_rate  # 342
        if channels != self.feature_dim:
            raise AssertionError(f"unexpected DenseFace feature width: {channels}")
        self.final_norm = nn.BatchNorm2d(channels)
        self.classifier = nn.Linear(channels, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """Return the final 342-channel spatial map before global pooling."""
        x = self.stem(images)
        x = self.transition1(self.block1(x))
        x = self.transition2(self.block2(x))
        return F.relu(self.final_norm(self.block3(x)), inplace=True)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.forward_feature_map(images)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(images))


def checkpoint_model(checkpoint: dict, device: torch.device | str = "cpu") -> DenseFace:
    """Restore a model from the checkpoint emitted by ``train.py``."""
    config = checkpoint.get("model", {})
    model = DenseFace(num_classes=int(config.get("num_classes", 8)))
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device)
