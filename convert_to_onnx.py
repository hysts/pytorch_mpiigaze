#!/usr/bin/env python

import argparse

import torch

from gaze_estimation import create_model, get_default_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--weight', type=str)
    parser.add_argument('--output-path', '-o', type=str, required=True)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)

    device = torch.device(config.device)

    model = create_model(config)
    if args.weight is not None:
        checkpoint = torch.load(args.weight, map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.eval()

    if config.mode == 'MPIIGaze':
        x = torch.zeros((1, 1, 36, 60), dtype=torch.float32, device=device)
        y = torch.zeros((1, 2), dtype=torch.float32, device=device)
        data = (x, y)
    elif config.mode == 'MPIIFaceGaze':
        x = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)
        data = (x, )
    else:
        raise ValueError

    torch.onnx.export(model, data, args.output_path)


if __name__ == '__main__':
    main()
