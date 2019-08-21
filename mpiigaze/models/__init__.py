import importlib
import torch
import yacs.config


def create_model(config: yacs.config.CfgNode) -> torch.nn.Module:
    module = importlib.import_module(f'mpiigaze.models.{config.model.name}')
    model = module.Model()
    device = torch.device(config.train.device)
    model.to(device)
    return model
