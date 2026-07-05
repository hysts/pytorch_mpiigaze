import pathlib
from collections.abc import Callable

import h5py
import torch
from torch.utils.data import Dataset

NUM_SAMPLES_PER_PERSON = 3000


class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path, transform: Callable) -> None:
        self.transform = transform

        # In case of the MPIIGaze dataset, each image is so small that
        # reading image will become a bottleneck even with HDF5.
        # So, first load them all into memory.
        with h5py.File(dataset_path, "r") as f:
            images = f.get(f"{person_id_str}/image")[()]
            poses = f.get(f"{person_id_str}/pose")[()]
            gazes = f.get(f"{person_id_str}/gaze")[()]
        if not (len(images) == len(poses) == len(gazes) == NUM_SAMPLES_PER_PERSON):
            error_msg = (
                f"Expected {NUM_SAMPLES_PER_PERSON} samples for {person_id_str}, "
                f"but got {len(images)} images, {len(poses)} poses, {len(gazes)} gazes."
            )
            raise ValueError(error_msg)
        self.images = images
        self.poses = poses
        self.gazes = gazes

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        pose = torch.from_numpy(self.poses[index])
        gaze = torch.from_numpy(self.gazes[index])
        return image, pose, gaze

    def __len__(self) -> int:
        return len(self.images)
