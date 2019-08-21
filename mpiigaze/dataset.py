from typing import Tuple

import pathlib

import numpy as np
import torch
import yacs.config


class MPIIGazeOnePersonDataset(torch.utils.data.Dataset):
    def __init__(self, person_id_str: str, dataset_dir: pathlib.Path):
        path = dataset_dir / f'{person_id_str}.npz'
        with np.load(path) as fin:
            images = fin['image'].astype(np.float32) / 255
            poses = fin['pose']
            gazes = fin['gaze']

        self.images = torch.unsqueeze(torch.from_numpy(images), dim=1)
        self.poses = torch.from_numpy(poses)
        self.gazes = torch.from_numpy(gazes)

    def __getitem__(self, index: int
                    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self.images[index], self.poses[index], self.gazes[index]

    def __len__(self) -> int:
        return len(self.images)

    def __repr__(self) -> str:
        return self.__class__.__name__


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool = True) -> torch.utils.data.Dataset:
    dataset_dir = pathlib.Path(config.dataset.dataset_dir)

    assert dataset_dir.exists()
    assert config.train.test_id in range(15)
    person_ids = [f'p{index:02}' for index in range(15)]
    test_person_id = person_ids[config.train.test_id]

    if is_train:
        train_dataset = torch.utils.data.ConcatDataset([
            MPIIGazeOnePersonDataset(person_id, dataset_dir)
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
        test_dataset = MPIIGazeOnePersonDataset(test_person_id, dataset_dir)
        assert len(test_dataset) == 3000
        return test_dataset
