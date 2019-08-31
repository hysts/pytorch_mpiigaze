import importlib

import torch.nn as nn
import yacs.config


def create_backbone(config: yacs.config.CfgNode) -> nn.Module:
    backbone_name = config.model.backbone.name
    module = importlib.import_module(
        f'gaze_estimation.models.mpiifacegaze.backbones.{backbone_name}')
    return module.Model(config)
