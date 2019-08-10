#!/usr/bin/env python

import importlib
import pathlib
import time

import torch
import torch.nn as nn

from mpiigaze.dataloader import create_dataloader
from mpiigaze.utils import (load_config, compute_angle_error, AverageMeter)
from mpiigaze.logger import create_logger


def test(model, criterion, test_loader, config, logger):
    model.eval()

    device = torch.device(config.test.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    with torch.no_grad():
        for step, (images, poses, gazes) in enumerate(test_loader):
            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)

            outputs = model(images, poses)
            loss = criterion(outputs, gazes)

            angle_error = compute_angle_error(outputs, gazes).mean()

            num = images.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(angle_error.item(), num)

    logger.info(f'loss {loss_meter.avg:.4f} '
                f'angle error {angle_error_meter.avg:.2f}')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')


def main():
    config = load_config()

    outdir = pathlib.Path(config.test.outdir)
    if outdir.exists():
        raise RuntimeError(
            f'Output directory `{outdir.as_posix()}` already exists')
    outdir.mkdir(exist_ok=True, parents=True)

    logger = create_logger(name=__name__, outdir=outdir, filename='log.txt')
    logger.info(config)

    test_loader = create_dataloader(config, is_train=False)

    ckpt = torch.load(config.test.checkpoint)

    device = torch.device(config.test.device)
    module = importlib.import_module(f'mpiigaze.models.{config.model.name}')
    model = module.Model()
    model.load_state_dict(ckpt['model'])
    model.to(device)

    criterion = nn.MSELoss(reduction='mean')

    test(model, criterion, test_loader, config, logger)


if __name__ == '__main__':
    main()
