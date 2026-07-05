#!/usr/bin/env python

import argparse
import pathlib

import h5py
import numpy as np
import tqdm

NUM_PERSONS = 15
NUM_SAMPLES_PER_PERSON = 3000


def add_mat_data_to_hdf5(person_id: str, dataset_dir: pathlib.Path, output_path: pathlib.Path) -> None:
    with h5py.File(dataset_dir / f"{person_id}.mat", "r") as f_input:
        images = f_input.get("Data/data")[()]
        labels = f_input.get("Data/label")[()][:, :4]
    if not (len(images) == len(labels) == NUM_SAMPLES_PER_PERSON):
        error_msg = (
            f"Expected {NUM_SAMPLES_PER_PERSON} samples for {person_id}, "
            f"but got {len(images)} images and {len(labels)} labels."
        )
        raise ValueError(error_msg)

    images = images.transpose(0, 2, 3, 1).astype(np.uint8)
    poses = labels[:, 2:]
    gazes = labels[:, :2]

    with h5py.File(output_path, "a") as f_output:
        for index, (image, gaze, pose) in tqdm.tqdm(enumerate(zip(images, gazes, poses, strict=True)), leave=False):
            f_output.create_dataset(f"{person_id}/image/{index:04}", data=image)
            f_output.create_dataset(f"{person_id}/pose/{index:04}", data=pose)
            f_output.create_dataset(f"{person_id}/gaze/{index:04}", data=gaze)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "MPIIFaceGaze.h5"
    if output_path.exists():
        error_msg = f"{output_path} already exists."
        raise ValueError(error_msg)

    dataset_dir = pathlib.Path(args.dataset)
    for person_index in tqdm.tqdm(range(NUM_PERSONS)):
        person_id = f"p{person_index:02}"
        add_mat_data_to_hdf5(person_id, dataset_dir, output_path)


if __name__ == "__main__":
    main()
