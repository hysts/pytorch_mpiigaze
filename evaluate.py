#!/usr/bin/env python

import pathlib
import torch
import torch.nn as nn

from mpiigaze.dataloader import create_dataloader
from mpiigaze.models import create_model
from mpiigaze.utils import (load_config, compute_angle_error, AverageMeter)


def test(model, criterion, test_loader, config):
    model.eval()

    device = torch.device(config.test.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()

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

    return angle_error_meter.avg


def main():
    config = load_config()

    outdir = pathlib.Path(config.test.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    test_loader = create_dataloader(config, is_train=False)

    model = create_model(config)
    ckpt = torch.load(config.test.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    criterion = nn.MSELoss(reduction='mean')

    angle_error = test(model, criterion, test_loader, config)

    outpath = outdir / 'results.txt'
    with open(outpath, 'w') as fout:
        fout.write(f'{angle_error}')


if __name__ == '__main__':
    main()
