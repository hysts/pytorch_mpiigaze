import torch
import torch.nn as nn
import torch.nn.functional as F
import yacs.config


def initialize_weights(module: torch.nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.zeros_(module.bias)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(x)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class Model(nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super().__init__()

        depth = 8
        base_channels = 16
        input_shape = (1, 1, 36, 60)

        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth

        n_channels = [base_channels, base_channels * 2, base_channels * 4]

        self.conv = nn.Conv2d(input_shape[1],
                              n_channels[0],
                              kernel_size=(3, 3),
                              stride=1,
                              padding=1,
                              bias=False)

        self.stage1 = self._make_stage(n_channels[0],
                                       n_channels[0],
                                       n_blocks_per_stage,
                                       BasicBlock,
                                       stride=1)
        self.stage2 = self._make_stage(n_channels[0],
                                       n_channels[1],
                                       n_blocks_per_stage,
                                       BasicBlock,
                                       stride=2)
        self.stage3 = self._make_stage(n_channels[1],
                                       n_channels[2],
                                       n_blocks_per_stage,
                                       BasicBlock,
                                       stride=2)
        self.bn = nn.BatchNorm2d(n_channels[2])

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).size(0)

        self.fc = nn.Linear(self.feature_size + 2, 2)

        self.apply(initialize_weights)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, n_blocks: int,
                    block: torch.nn.Module, stride: int) -> torch.nn.Module:
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name, block(in_channels, out_channels,
                                      stride=stride))
            else:
                stage.add_module(block_name,
                                 block(out_channels, out_channels, stride=1))
        return stage

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, y], dim=1)
        x = self.fc(x)
        return x
