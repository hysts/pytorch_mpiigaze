import pathlib
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable

from gaze_estimation.transforms import create_transform
from gaze_estimation.types import GazeEstimationMethod

from . import mpiifacegaze, mpiigaze

NUM_PERSONS = 15
NUM_SAMPLES_PER_PERSON = 3000


def _validate_config(config: DictConfig, dataset_dir: pathlib.Path) -> None:
    if not dataset_dir.exists():
        error_msg = f"Dataset file {dataset_dir} does not exist."
        raise FileNotFoundError(error_msg)
    if config.train.test_id not in range(-1, NUM_PERSONS):
        error_msg = f"train.test_id must be in [-1, {NUM_PERSONS - 1}], but got {config.train.test_id}."
        raise ValueError(error_msg)
    if config.test.test_id not in range(NUM_PERSONS):
        error_msg = f"test.test_id must be in [0, {NUM_PERSONS - 1}], but got {config.test.test_id}."
        raise ValueError(error_msg)


def create_dataset(config: DictConfig, is_train: bool = True) -> tuple[Dataset, Dataset] | Dataset:
    dataset_class: Callable[[str, pathlib.Path, Callable], Dataset]
    if config.mode == GazeEstimationMethod.MPIIGaze.name:
        dataset_class = mpiigaze.OnePersonDataset
    elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        dataset_class = mpiifacegaze.OnePersonDataset
    else:
        raise ValueError

    dataset_dir = pathlib.Path(config.dataset.dataset_dir)
    _validate_config(config, dataset_dir)
    person_ids = [f"p{index:02}" for index in range(NUM_PERSONS)]

    transform = create_transform(config)

    if is_train:
        train_dataset: torch.utils.data.ConcatDataset
        if config.train.test_id == -1:
            train_dataset = torch.utils.data.ConcatDataset(
                [dataset_class(person_id, dataset_dir, transform) for person_id in person_ids]
            )
            expected_len = NUM_PERSONS * NUM_SAMPLES_PER_PERSON
        else:
            test_person_id = person_ids[config.train.test_id]
            train_dataset = torch.utils.data.ConcatDataset(
                [
                    dataset_class(person_id, dataset_dir, transform)
                    for person_id in person_ids
                    if person_id != test_person_id
                ]
            )
            expected_len = (NUM_PERSONS - 1) * NUM_SAMPLES_PER_PERSON
        if len(train_dataset) != expected_len:
            error_msg = f"Expected {expected_len} training samples, but got {len(train_dataset)}."
            raise ValueError(error_msg)

        val_ratio = config.train.val_ratio
        if not 0 <= val_ratio < 1:
            error_msg = f"train.val_ratio must be in [0, 1), but got {val_ratio}."
            raise ValueError(error_msg)
        val_num = int(len(train_dataset) * val_ratio)
        train_num = len(train_dataset) - val_num
        lengths = [train_num, val_num]
        train_subset, val_subset = torch.utils.data.dataset.random_split(train_dataset, lengths)
        return train_subset, val_subset
    test_person_id = person_ids[config.test.test_id]
    test_dataset = dataset_class(test_person_id, dataset_dir, transform)
    if len(test_dataset) != NUM_SAMPLES_PER_PERSON:
        error_msg = f"Expected {NUM_SAMPLES_PER_PERSON} test samples, but got {len(test_dataset)}."
        raise ValueError(error_msg)
    return test_dataset
