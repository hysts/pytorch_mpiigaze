import pathlib
from typing import Callable, Tuple

import h5py
import torch
from torch.utils.data import Dataset


class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable):
        self.person_id_str = person_id_str
        self.dataset_path = dataset_path
        self.transform = transform

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.dataset_path, 'r') as f:
            image = f.get(f'{self.person_id_str}/image/{index:04}')[()]
            pose = f.get(f'{self.person_id_str}/pose/{index:04}')[()]
            gaze = f.get(f'{self.person_id_str}/gaze/{index:04}')[()]
        image = self.transform(image)
        pose = torch.from_numpy(pose)
        gaze = torch.from_numpy(gaze)
        return image, pose, gaze

    def __len__(self) -> int:
        return 3000
