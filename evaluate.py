#!/usr/bin/env python

import pathlib
import numpy as np
import torch
import tqdm

from mpiigaze.dataloader import create_dataloader
from mpiigaze.models import create_model
from mpiigaze.utils import (load_config, compute_angle_error)


def test(model, test_loader, config):
    model.eval()
    device = torch.device(config.test.device)

    preds = []
    gts = []
    with torch.no_grad():
        for images, poses, gazes in tqdm.tqdm(test_loader):
            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)

            outputs = model(images, poses)
            preds.append(outputs.cpu())
            gts.append(gazes.cpu())

    preds = torch.cat(preds)
    gts = torch.cat(gts)
    angle_error = float(compute_angle_error(preds, gts).mean())
    return preds, gts, angle_error


def main():
    config = load_config()

    outdir = pathlib.Path(config.test.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    test_loader = create_dataloader(config, is_train=False)

    model = create_model(config)
    ckpt = torch.load(config.test.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    preds, gts, angle_error = test(model, test_loader, config)

    outpath = outdir / 'preds.npy'
    np.save(outpath, preds.numpy())
    outpath = outdir / 'gts.npy'
    np.save(outpath, gts.numpy())
    outpath = outdir / 'error.txt'
    with open(outpath, 'w') as fout:
        fout.write(f'{angle_error}')


if __name__ == '__main__':
    main()
