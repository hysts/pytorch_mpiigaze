import torch
import torch.nn as nn
import torch.nn.functional as F
import yacs.config


def initialize_weights(module: torch.nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class Model(nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(3600, 500)
        self.fc2 = nn.Linear(502, 2)

        self._initialize_weight()

    def _initialize_weight(self) -> None:
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True)
        x = torch.cat([x, y], dim=1)
        x = self.fc2(x)
        return x
