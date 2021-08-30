import torch.nn as nn
import yacs.config

from .types import LossType


def create_loss(config: yacs.config.CfgNode) -> nn.Module:
    loss_name = config.train.loss
    if loss_name == LossType.L1.name:
        return nn.L1Loss(reduction='mean')
    elif loss_name == LossType.L2.name:
        return nn.MSELoss(reduction='mean')
    elif loss_name == LossType.SmoothL1.name:
        return nn.SmoothL1Loss(reduction='mean')
    else:
        raise ValueError
