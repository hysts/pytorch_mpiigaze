#!/usr/bin/env python

import logging
import time

import torch
import torchvision.utils
from fvcore.common.checkpoint import Checkpointer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from gaze_estimation import (
    GazeEstimationMethod,
    create_dataloader,
    create_logger,
    create_loss,
    create_model,
    create_optimizer,
    create_scheduler,
    create_tensorboard_writer,
)
from gaze_estimation.tensorboard import TensorboardWriter
from gaze_estimation.utils import (
    AverageMeter,
    compute_angle_error,
    create_train_output_dir,
    load_config,
    save_config,
    set_seeds,
    setup_cudnn,
)


def train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_function: torch.nn.Module,
    train_loader: DataLoader,
    config: DictConfig,
    tensorboard_writer: TensorboardWriter,
    logger: logging.Logger,
) -> None:
    logger.info("Train %d", epoch)

    model.train()

    device = torch.device(config.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    for step, batch in enumerate(train_loader):
        if config.tensorboard.train_images and step == 0:
            image = torchvision.utils.make_grid(batch[0], normalize=True, scale_each=True)
            tensorboard_writer.add_image("Train/Image", image, epoch)

        images, poses, gazes = (tensor.to(device) for tensor in batch)

        optimizer.zero_grad()

        if config.mode == GazeEstimationMethod.MPIIGaze.name:
            outputs = model(images, poses)
        elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            outputs = model(images)
        else:
            raise ValueError
        loss = loss_function(outputs, gazes)
        loss.backward()

        optimizer.step()

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        if step % config.train.log_period == 0:
            logger.info(
                "Epoch %d Step %d/%d lr %.6f loss %.4f (%.4f) angle error %.2f (%.2f)",
                epoch,
                step,
                len(train_loader),
                scheduler.get_last_lr()[0],
                loss_meter.val,
                loss_meter.avg,
                angle_error_meter.val,
                angle_error_meter.avg,
            )

    elapsed = time.time() - start
    logger.info("Elapsed %.2f", elapsed)

    tensorboard_writer.add_scalar("Train/Loss", loss_meter.avg, epoch)
    tensorboard_writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], epoch)
    tensorboard_writer.add_scalar("Train/AngleError", angle_error_meter.avg, epoch)
    tensorboard_writer.add_scalar("Train/Time", elapsed, epoch)


def validate(
    epoch: int,
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    val_loader: DataLoader,
    config: DictConfig,
    tensorboard_writer: TensorboardWriter,
    logger: logging.Logger,
) -> None:
    logger.info("Val %d", epoch)

    model.eval()

    device = torch.device(config.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if config.tensorboard.val_images and epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(batch[0], normalize=True, scale_each=True)
                tensorboard_writer.add_image("Val/Image", image, epoch)

            images, poses, gazes = (tensor.to(device) for tensor in batch)

            if config.mode == GazeEstimationMethod.MPIIGaze.name:
                outputs = model(images, poses)
            elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
                outputs = model(images)
            else:
                raise ValueError
            loss = loss_function(outputs, gazes)

            angle_error = compute_angle_error(outputs, gazes).mean()

            num = images.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(angle_error.item(), num)

    logger.info("Epoch %d loss %.4f angle error %.2f", epoch, loss_meter.avg, angle_error_meter.avg)

    elapsed = time.time() - start
    logger.info("Elapsed %.2f", elapsed)

    if epoch > 0:
        tensorboard_writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
        tensorboard_writer.add_scalar("Val/AngleError", angle_error_meter.avg, epoch)
    tensorboard_writer.add_scalar("Val/Time", elapsed, epoch)

    if config.tensorboard.model_params:
        for name, param in model.named_parameters():
            tensorboard_writer.add_histogram(name, param, epoch)


def main() -> None:
    config = load_config()

    set_seeds(config.train.seed)
    setup_cudnn(config)

    output_dir = create_train_output_dir(config)
    save_config(config, output_dir)
    logger = create_logger(name=__name__, output_dir=output_dir, filename="log.txt")
    logger.info(OmegaConf.to_yaml(config))

    train_loader, val_loader = create_dataloader(config, is_train=True)
    model = create_model(config)
    loss_function = create_loss(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    checkpointer = Checkpointer(
        model, optimizer=optimizer, scheduler=scheduler, save_dir=output_dir.as_posix(), save_to_disk=True
    )
    tensorboard_writer = create_tensorboard_writer(config, output_dir)

    if config.train.val_first:
        validate(0, model, loss_function, val_loader, config, tensorboard_writer, logger)

    for epoch in range(1, config.scheduler.epochs + 1):
        train(epoch, model, optimizer, scheduler, loss_function, train_loader, config, tensorboard_writer, logger)
        scheduler.step()

        if epoch % config.train.val_period == 0:
            validate(epoch, model, loss_function, val_loader, config, tensorboard_writer, logger)

        if epoch % config.train.checkpoint_period == 0 or epoch == config.scheduler.epochs:
            checkpoint_config = {"epoch": epoch, "config": OmegaConf.to_container(config, resolve=True)}
            checkpointer.save(f"checkpoint_{epoch:04d}", **checkpoint_config)

    tensorboard_writer.close()


if __name__ == "__main__":
    main()
