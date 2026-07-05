from collections.abc import Callable

import cv2
import numpy as np
import torch
import torchvision
from omegaconf import DictConfig

from .types import GazeEstimationMethod

# Size of the face images stored in the MPIIFaceGaze h5 file. Images are
# only resized when a different input size is requested.
DATASET_FACE_SIZE = 448


def create_transform(config: DictConfig) -> Callable:
    if config.mode == GazeEstimationMethod.MPIIGaze.name:
        return _create_mpiigaze_transform()
    if config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        return _create_mpiifacegaze_transform(config)
    raise ValueError


def _create_mpiigaze_transform() -> Callable:
    scale = torchvision.transforms.Lambda(lambda x: x.astype(np.float32) / 255)
    return torchvision.transforms.Compose(
        [
            scale,
            torch.from_numpy,
            torchvision.transforms.Lambda(lambda x: x[None, :, :]),
        ]
    )


def _create_mpiifacegaze_transform(config: DictConfig) -> Callable:
    scale = torchvision.transforms.Lambda(lambda x: x.astype(np.float32) / 255)
    identity = torchvision.transforms.Lambda(lambda x: x)
    size = config.transform.mpiifacegaze_face_size
    resize = (
        torchvision.transforms.Lambda(lambda x: cv2.resize(x, (size, size))) if size != DATASET_FACE_SIZE else identity
    )
    if config.transform.mpiifacegaze_gray:
        to_gray = torchvision.transforms.Lambda(
            lambda x: cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)
        )
    else:
        to_gray = identity

    return torchvision.transforms.Compose(
        [
            resize,
            to_gray,
            torchvision.transforms.Lambda(lambda x: x.transpose(2, 0, 1)),
            scale,
            torch.from_numpy,
            torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
        ]
    )
