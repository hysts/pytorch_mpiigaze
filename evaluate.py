#!/usr/bin/env python

import pathlib
from typing import cast

import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from gaze_estimation import GazeEstimationMethod, create_dataloader, create_model
from gaze_estimation.utils import compute_angle_error, load_config, save_config


def test(
    model: torch.nn.Module, test_loader: DataLoader, config: DictConfig
) -> tuple[torch.Tensor, torch.Tensor, float]:
    model.eval()
    device = torch.device(config.device)

    predictions = []
    gts = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            images, poses, gazes = (tensor.to(device) for tensor in batch)

            if config.mode == GazeEstimationMethod.MPIIGaze.name:
                outputs = model(images, poses)
            elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
                outputs = model(images)
            else:
                raise ValueError
            predictions.append(outputs.cpu())
            gts.append(gazes.cpu())

    prediction_tensor = torch.cat(predictions)
    gt_tensor = torch.cat(gts)
    angle_error = float(compute_angle_error(prediction_tensor, gt_tensor).mean())
    return prediction_tensor, gt_tensor, angle_error


def main() -> None:
    config = load_config()

    output_rootdir = pathlib.Path(config.test.output_dir)
    checkpoint_name = pathlib.Path(config.test.checkpoint).stem
    output_dir = output_rootdir / checkpoint_name
    output_dir.mkdir(exist_ok=True, parents=True)
    save_config(config, output_dir)

    test_loader = cast("DataLoader", create_dataloader(config, is_train=False))

    model = create_model(config)
    checkpoint = torch.load(config.test.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    predictions, gts, angle_error = test(model, test_loader, config)

    print(f"The mean angle error (deg): {angle_error:.2f}")  # noqa: T201

    output_path = output_dir / "predictions.npy"
    np.save(output_path, predictions.numpy())
    output_path = output_dir / "gts.npy"
    np.save(output_path, gts.numpy())
    output_path = output_dir / "error.txt"
    with output_path.open("w") as f:
        f.write(f"{angle_error}")


if __name__ == "__main__":
    main()
