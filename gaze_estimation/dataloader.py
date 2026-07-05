from typing import cast

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .datasets import create_dataset


def create_dataloader(config: DictConfig, is_train: bool) -> tuple[DataLoader, DataLoader] | DataLoader:
    if is_train:
        train_dataset, val_dataset = cast("tuple[Dataset, Dataset]", create_dataset(config, is_train))
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.train_dataloader.num_workers,
            pin_memory=config.train.train_dataloader.pin_memory,
            drop_last=config.train.train_dataloader.drop_last,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.val_dataloader.num_workers,
            pin_memory=config.train.val_dataloader.pin_memory,
            drop_last=False,
        )
        return train_loader, val_loader
    test_dataset = cast("Dataset", create_dataset(config, is_train))
    return DataLoader(
        test_dataset,
        batch_size=config.test.batch_size,
        num_workers=config.test.dataloader.num_workers,
        shuffle=False,
        pin_memory=config.test.dataloader.pin_memory,
        drop_last=False,
    )
