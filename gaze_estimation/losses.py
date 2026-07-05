from omegaconf import DictConfig
from torch import nn

from .types import LossType


def create_loss(config: DictConfig) -> nn.Module:
    loss_name = config.train.loss
    if loss_name == LossType.L1.name:
        return nn.L1Loss(reduction="mean")
    if loss_name == LossType.L2.name:
        return nn.MSELoss(reduction="mean")
    if loss_name == LossType.SmoothL1.name:
        return nn.SmoothL1Loss(reduction="mean")
    raise ValueError
