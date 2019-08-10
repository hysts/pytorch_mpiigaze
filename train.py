#!/usr/bin/env python

import argparse
import importlib
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from tensorboardX import SummaryWriter

from mpiigaze.config import get_default_config
from mpiigaze.dataloader import create_dataloader
from mpiigaze.utils import set_seeds, AverageMeter
from mpiigaze.logger import create_logger

global_step = 0


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.train.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config.freeze()
    return config


def save_config(config, outdir):
    with open(outdir / 'config.yaml', 'w') as fout:
        fout.write(str(config))


def convert_to_unit_vector(angles):
    x = -torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
    y = -torch.sin(angles[:, 0])
    z = -torch.cos(angles[:, 1]) * torch.cos(angles[:, 1])
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def compute_angle_error(preds, labels):
    pred_x, pred_y, pred_z = convert_to_unit_vector(preds)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi


def train(epoch, model, optimizer, scheduler, criterion, train_loader, config,
          writer, logger):
    global global_step

    logger.info(f'Train {epoch}')

    model.train()

    device = torch.device(config.train.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    for step, (images, poses, gazes) in enumerate(train_loader):
        global_step += 1

        if (config.train.use_tensorboard and config.tensorboard.train_images
                and step == 0):
            image = torchvision.utils.make_grid(images,
                                                normalize=True,
                                                scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        images = images.to(device)
        poses = poses.to(device)
        gazes = gazes.to(device)

        optimizer.zero_grad()

        outputs = model(images, poses)
        loss = criterion(outputs, gazes)
        loss.backward()

        optimizer.step()

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        if config.train.use_tensorboard:
            writer.add_scalar('Train/RunningLoss', loss_meter.val, global_step)

        if step % 100 == 0:
            logger.info(f'Epoch {epoch} Step {step}/{len(train_loader)} '
                        f'lr {scheduler.get_lr()[0]:.6f} '
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'angle error {angle_error_meter.val:.2f} '
                        f'({angle_error_meter.avg:.2f})')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')

    if config.train.use_tensorboard:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/lr', scheduler.get_lr()[0], epoch)
        writer.add_scalar('Train/AngleError', angle_error_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)


def validate(epoch, model, criterion, val_loader, config, writer, logger):
    logger.info(f'Val {epoch}')

    model.eval()

    device = torch.device(config.train.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    with torch.no_grad():
        for step, (images, poses, gazes) in enumerate(val_loader):
            if (config.train.use_tensorboard and config.tensorboard.val_images
                    and epoch == 0 and step == 0):
                image = torchvision.utils.make_grid(images,
                                                    normalize=True,
                                                    scale_each=True)
                writer.add_image('Val/Image', image, epoch)

            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)

            outputs = model(images, poses)
            loss = criterion(outputs, gazes)

            angle_error = compute_angle_error(outputs, gazes).mean()

            num = images.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(angle_error.item(), num)

    logger.info(f'Epoch {epoch} Loss {loss_meter.avg:.4f} '
                f'AngleError {angle_error_meter.avg:.2f}')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')

    if config.train.use_tensorboard:
        if epoch > 0:
            writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
            writer.add_scalar('Val/AngleError', angle_error_meter.avg, epoch)
        writer.add_scalar('Val/Time', elapsed, epoch)

    if config.tensorboard.model_params:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    return angle_error_meter.avg


def main():
    config = load_config()

    torch.backends.cudnn.benchmark = config.cudnn.benchmark
    torch.backends.cudnn.deterministic = config.cudnn.deterministic

    set_seeds(config.train.seed)

    outdir = pathlib.Path(config.train.outdir)
    if not config.train.resume and outdir.exists():
        raise RuntimeError(
            f'Output directory `{outdir.as_posix()}` already exists')
    outdir.mkdir(exist_ok=True, parents=True)
    if not config.train.resume:
        save_config(config, outdir)

    logger = create_logger(name=__name__, outdir=outdir, filename='log.txt')
    logger.info(config)

    start_epoch = config.train.start_epoch
    if config.train.use_tensorboard:
        if start_epoch > 0:
            writer = SummaryWriter(outdir.as_posix(),
                                   purge_step=start_epoch + 1)
        else:
            writer = SummaryWriter(outdir.as_posix())
    else:
        writer = None

    train_loader, val_loader = create_dataloader(config, is_train=True)

    device = torch.device(config.train.device)
    module = importlib.import_module(f'mpiigaze.models.{config.model.name}')
    model = module.Model()
    model.to(device)

    criterion = nn.MSELoss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.train.base_lr,
                                momentum=config.train.momentum,
                                weight_decay=config.train.weight_decay,
                                nesterov=config.train.nesterov)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.scheduler.milestones,
        gamma=config.scheduler.lr_decay)

    # run validation before start training
    validate(0, model, criterion, val_loader, config, writer, logger)

    for epoch in range(start_epoch, config.scheduler.epochs):
        epoch += 1
        train(epoch, model, optimizer, scheduler, criterion, train_loader,
              config, writer, logger)
        angle_error = validate(epoch, model, criterion, val_loader, config,
                               writer, logger)
        scheduler.step()

        state = {
            'config': config,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'angle_error': angle_error,
        }
        model_path = outdir / 'model_state.pth'
        torch.save(state, model_path)


if __name__ == '__main__':
    main()
