#!/usr/bin/env python

import argparse
import pathlib

import h5py
import numpy as np
import tqdm


def add_mat_data_to_hdf5(person_id: str, dataset_dir: pathlib.Path,
                         output_path: pathlib.Path) -> None:
    with h5py.File(dataset_dir / f'{person_id}.mat', 'r') as f_input:
        images = f_input.get('Data/data')[()]
        labels = f_input.get('Data/label')[()][:, :4]
    assert len(images) == len(labels) == 3000

    images = images.transpose(0, 2, 3, 1).astype(np.uint8)
    poses = labels[:, 2:]
    gazes = labels[:, :2]

    with h5py.File(output_path, 'a') as f_output:
        for index, (image, gaze,
                    pose) in tqdm.tqdm(enumerate(zip(images, gazes, poses)),
                                       leave=False):
            f_output.create_dataset(f'{person_id}/image/{index:04}',
                                    data=image)
            f_output.create_dataset(f'{person_id}/pose/{index:04}', data=pose)
            f_output.create_dataset(f'{person_id}/gaze/{index:04}', data=gaze)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'MPIIFaceGaze.h5'
    if output_path.exists():
        raise ValueError(f'{output_path} already exists.')

    dataset_dir = pathlib.Path(args.dataset)
    for person_id in tqdm.tqdm(range(15)):
        person_id = f'p{person_id:02}'
        add_mat_data_to_hdf5(person_id, dataset_dir, output_path)


if __name__ == '__main__':
    main()
