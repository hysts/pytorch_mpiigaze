import torch
from omegaconf import DictConfig


def create_scheduler(config: DictConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    if config.scheduler.type == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(config.scheduler.milestones), gamma=config.scheduler.lr_decay
        )
    if config.scheduler.type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.scheduler.epochs, eta_min=config.scheduler.lr_min_factor
        )
    raise ValueError
