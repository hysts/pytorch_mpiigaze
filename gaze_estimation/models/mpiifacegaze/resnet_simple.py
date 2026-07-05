from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig
from torch import nn

from .backbones import create_backbone


class Model(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.feature_extractor = create_backbone(config)
        n_channels = cast("int", self.feature_extractor.n_features)

        self.conv = nn.Conv2d(n_channels, 1, kernel_size=1, stride=1, padding=0)
        # This model assumes the input image size is 224x224.
        self.fc = nn.Linear(n_channels * 14**2, 2)

        self._register_hook()
        self._initialize_weight()

    def _initialize_weight(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _register_hook(self) -> None:
        n_channels = cast("int", self.feature_extractor.n_features)

        def hook(
            _module: nn.Module,
            grad_in: tuple[torch.Tensor, ...] | torch.Tensor,
            _grad_out: tuple[torch.Tensor, ...] | torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            return tuple(grad / n_channels for grad in grad_in)

        self.conv.register_backward_hook(hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        y = F.relu(self.conv(x))
        x = x * y
        x = x.view(x.size(0), -1)
        return self.fc(x)
