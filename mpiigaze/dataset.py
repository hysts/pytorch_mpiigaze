import pathlib
import numpy as np
import torch


class MPIIGazeOneSubjectDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id, dataset_dir):
        path = dataset_dir / f'{subject_id}.npz'
        with np.load(path) as fin:
            images = fin['image'].astype(np.float32) / 255
            poses = fin['pose']
            gazes = fin['gaze']

        self.images = torch.unsqueeze(torch.from_numpy(images), dim=1)
        self.poses = torch.from_numpy(poses)
        self.gazes = torch.from_numpy(gazes)

    def __getitem__(self, index):
        return self.images[index], self.poses[index], self.gazes[index]

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return self.__class__.__name__


def create_dataset(config, is_train=True):
    dataset_dir = pathlib.Path(config.dataset.dataset_dir)

    assert dataset_dir.exists()
    assert config.train.test_id in range(15)
    subject_ids = [f'p{index:02}' for index in range(15)]
    test_subject_id = subject_ids[config.train.test_id]

    if is_train:
        train_dataset = torch.utils.data.ConcatDataset([
            MPIIGazeOneSubjectDataset(subject_id, dataset_dir)
            for subject_id in subject_ids if subject_id != test_subject_id
        ])
        assert len(train_dataset) == 42000

        val_ratio = config.train.val_ratio
        assert val_ratio < 1
        val_num = int(len(train_dataset) * val_ratio)
        train_num = len(train_dataset) - val_num
        lengths = [train_num, val_num]
        return torch.utils.data.dataset.random_split(train_dataset, lengths)
    else:
        test_dataset = MPIIGazeOneSubjectDataset(test_subject_id, dataset_dir)
        assert len(test_dataset) == 3000
        return test_dataset
