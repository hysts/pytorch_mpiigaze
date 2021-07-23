import pathlib
from typing import Callable, Tuple

import h5py
import torch
from torch.utils.data import Dataset


class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable):
        self.transform = transform

        # In case of the MPIIGaze dataset, each image is so small that
        # reading image will become a bottleneck even with HDF5.
        # So, first load them all into memory.
        with h5py.File(dataset_path, 'r') as f:
            images = f.get(f'{person_id_str}/image')[()]
            poses = f.get(f'{person_id_str}/pose')[()]
            gazes = f.get(f'{person_id_str}/gaze')[()]
        assert len(images) == 3000
        assert len(poses) == 3000
        assert len(gazes) == 3000
        self.images = images
        self.poses = poses
        self.gazes = gazes

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        pose = torch.from_numpy(self.poses[index])
        gaze = torch.from_numpy(self.gazes[index])
        return image, pose, gaze

    def __len__(self) -> int:
        return len(self.images)
