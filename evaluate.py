#!/usr/bin/env python

import pathlib

import numpy as np
import torch
import tqdm

from gaze_estimation import (GazeEstimationMethod, create_dataloader,
                             create_model)
from gaze_estimation.utils import compute_angle_error, load_config, save_config


def test(model, test_loader, config):
    model.eval()
    device = torch.device(config.device)

    predictions = []
    gts = []
    with torch.no_grad():
        for images, poses, gazes in tqdm.tqdm(test_loader):
            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)

            if config.mode == GazeEstimationMethod.MPIIGaze.name:
                outputs = model(images, poses)
            elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
                outputs = model(images)
            else:
                raise ValueError
            predictions.append(outputs.cpu())
            gts.append(gazes.cpu())

    predictions = torch.cat(predictions)
    gts = torch.cat(gts)
    angle_error = float(compute_angle_error(predictions, gts).mean())
    return predictions, gts, angle_error


def main():
    config = load_config()

    output_rootdir = pathlib.Path(config.test.output_dir)
    checkpoint_name = pathlib.Path(config.test.checkpoint).stem
    output_dir = output_rootdir / checkpoint_name
    output_dir.mkdir(exist_ok=True, parents=True)
    save_config(config, output_dir)

    test_loader = create_dataloader(config, is_train=False)

    model = create_model(config)
    checkpoint = torch.load(config.test.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    predictions, gts, angle_error = test(model, test_loader, config)

    print(f'The mean angle error (deg): {angle_error:.2f}')

    output_path = output_dir / 'predictions.npy'
    np.save(output_path, predictions.numpy())
    output_path = output_dir / 'gts.npy'
    np.save(output_path, gts.numpy())
    output_path = output_dir / 'error.txt'
    with open(output_path, 'w') as f:
        f.write(f'{angle_error}')


if __name__ == '__main__':
    main()
