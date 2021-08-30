import argparse
import pathlib
import random
from typing import Tuple

import numpy as np
import torch
import yacs.config

from .config import get_default_config


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_cudnn(config) -> None:
    torch.backends.cudnn.benchmark = config.cudnn.benchmark
    torch.backends.cudnn.deterministic = config.cudnn.deterministic


def load_config() -> yacs.config.CfgNode:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.train_dataloader.pin_memory = False
        config.train.val_dataloader.pin_memory = False
        config.test.dataloader.pin_memory = False
    config.freeze()
    return config


def save_config(config: yacs.config.CfgNode, output_dir: pathlib.Path) -> None:
    with open(output_dir / 'config.yaml', 'w') as f:
        f.write(str(config))


def create_train_output_dir(config: yacs.config.CfgNode) -> pathlib.Path:
    output_root_dir = pathlib.Path(config.train.output_dir)
    if config.train.test_id != -1:
        output_dir = output_root_dir / f'{config.train.test_id:02}'
    else:
        output_dir = output_root_dir / 'all'
    if output_dir.exists():
        raise RuntimeError(
            f'Output directory `{output_dir.as_posix()}` already exists.')
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def convert_to_unit_vector(
        angles: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pitches = angles[:, 0]
    yaws = angles[:, 1]
    x = -torch.cos(pitches) * torch.sin(yaws)
    y = -torch.sin(pitches)
    z = -torch.cos(pitches) * torch.cos(yaws)
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def compute_angle_error(predictions: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
    pred_x, pred_y, pred_z = convert_to_unit_vector(predictions)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count
