import importlib

from omegaconf import DictConfig
from torch import nn


def create_backbone(config: DictConfig) -> nn.Module:
    backbone_name = config.model.backbone.name
    module = importlib.import_module(f"gaze_estimation.models.mpiifacegaze.backbones.{backbone_name}")
    return module.Model(config)
