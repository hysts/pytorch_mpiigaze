import pathlib
from typing import List, Union

import torch
import yacs.config
from torch.utils.data import Dataset

from ..transforms import create_transform
from ..types import GazeEstimationMethod


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool = True) -> Union[List[Dataset], Dataset]:
    if config.mode == GazeEstimationMethod.MPIIGaze.name:
        from .mpiigaze import OnePersonDataset
    elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        from .mpiifacegaze import OnePersonDataset
    else:
        raise ValueError

    dataset_dir = pathlib.Path(config.dataset.dataset_dir)

    assert dataset_dir.exists()
    assert config.train.test_id in range(-1, 15)
    assert config.test.test_id in range(15)
    person_ids = [f'p{index:02}' for index in range(15)]

    transform = create_transform(config)

    if is_train:
        if config.train.test_id == -1:
            train_dataset = torch.utils.data.ConcatDataset([
                OnePersonDataset(person_id, dataset_dir, transform)
                for person_id in person_ids
            ])
            assert len(train_dataset) == 45000
        else:
            test_person_id = person_ids[config.train.test_id]
            train_dataset = torch.utils.data.ConcatDataset([
                OnePersonDataset(person_id, dataset_dir, transform)
                for person_id in person_ids if person_id != test_person_id
            ])
            assert len(train_dataset) == 42000

        val_ratio = config.train.val_ratio
        assert val_ratio < 1
        val_num = int(len(train_dataset) * val_ratio)
        train_num = len(train_dataset) - val_num
        lengths = [train_num, val_num]
        return torch.utils.data.dataset.random_split(train_dataset, lengths)
    else:
        test_person_id = person_ids[config.test.test_id]
        test_dataset = OnePersonDataset(test_person_id, dataset_dir, transform)
        assert len(test_dataset) == 3000
        return test_dataset
