import importlib
import torch


def create_model(config):
    module = importlib.import_module(f'mpiigaze.models.{config.model.name}')
    model = module.Model()
    device = torch.device(config.train.device)
    model.to(device)
    return model
