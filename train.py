#!/usr/bin/env python

import pathlib
import time

import torch
import torch.nn as nn
import torchvision.utils
from tensorboardX import SummaryWriter

from mpiigaze.checkpoint import CheckPointer
from mpiigaze.dataloader import create_dataloader
from mpiigaze.logger import create_logger
from mpiigaze.models import create_model
from mpiigaze.optim import create_optimizer
from mpiigaze.scheduler import create_scheduler
from mpiigaze.utils import (set_seeds, load_config, save_config,
                            compute_angle_error, AverageMeter)

global_step = 0


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

        if step % config.train.log_period == 0:
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

    model = create_model(config)
    criterion = nn.MSELoss(reduction='mean')

    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)

    checkpointer = CheckPointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                checkpoint_dir=outdir,
                                logger=logger)

    if config.train.val_first:
        validate(0, model, criterion, val_loader, config, writer, logger)

    for epoch in range(start_epoch, config.scheduler.epochs):
        epoch += 1
        train(epoch, model, optimizer, scheduler, criterion, train_loader,
              config, writer, logger)
        scheduler.step()

        if epoch % config.train.val_period == 0:
            validate(epoch, model, criterion, val_loader, config, writer,
                     logger)

        if epoch % config.train.checkpoint_period == 0:
            ckpt_config = {'epoch': epoch, 'config': config}
            checkpointer.save(f'model_{epoch:05d}', **ckpt_config)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
