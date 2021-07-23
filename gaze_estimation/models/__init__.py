import importlib

import torch
import yacs.config


def create_model(config: yacs.config.CfgNode) -> torch.nn.Module:
    dataset_name = config.mode.lower()
    module = importlib.import_module(
        f'gaze_estimation.models.{dataset_name}.{config.model.name}')
    model = module.Model(config)
    device = torch.device(config.device)
    model.to(device)
    return model
