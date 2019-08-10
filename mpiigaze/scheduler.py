import torch


def create_scheduler(config, optimizer):
    if config.scheduler.type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.scheduler.milestones,
            gamma=config.scheduler.lr_decay)
    elif config.scheduler.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.epochs,
            eta_min=config.scheduler.lr_min_factor)
    else:
        raise ValueError()
    return scheduler
