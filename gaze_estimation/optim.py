import torch
from omegaconf import DictConfig


def get_param_list(config: DictConfig, model: torch.nn.Module) -> list[dict[str, object]]:
    if config.train.no_weight_decay_on_bn:
        param_list = []
        for name, params in model.named_parameters():
            if "conv.weight" in name:
                param_list.append(
                    {
                        "params": params,
                        "weight_decay": config.train.weight_decay,
                    }
                )
            else:
                param_list.append(
                    {
                        "params": params,
                        "weight_decay": 0,
                    }
                )
    else:
        param_list = [
            {
                "params": list(model.parameters()),
                "weight_decay": config.train.weight_decay,
            }
        ]
    return param_list


def create_optimizer(config: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    params = get_param_list(config, model)

    if config.train.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=config.train.base_lr, momentum=config.train.momentum, nesterov=config.train.nesterov
        )
    if config.train.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.train.base_lr, betas=tuple(config.optim.adam.betas))
    if config.train.optimizer == "amsgrad":
        return torch.optim.Adam(params, lr=config.train.base_lr, betas=tuple(config.optim.adam.betas), amsgrad=True)
    raise ValueError
