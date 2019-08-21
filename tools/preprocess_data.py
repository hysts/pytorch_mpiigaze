#!/usr/bin/env python

from typing import Tuple

import argparse
import pathlib

import cv2
import numpy as np
import pandas as pd
import scipy.io
import tqdm


def convert_pose(vector: np.ndarray) -> np.ndarray:
    rot = cv2.Rodrigues(np.array(vector).astype(np.float32))[0]
    vec = rot[:, 2]
    pitch = np.arcsin(vec[1])
    yaw = np.arctan2(vec[0], vec[2])
    return np.array([pitch, yaw])


def convert_gaze(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw])


def get_eval_info(person_id: str, eval_dir: pathlib.Path) -> pd.DataFrame:
    eval_path = eval_dir / f'{person_id}.txt'
    df = pd.read_csv(eval_path,
                     delimiter=' ',
                     header=None,
                     names=['path', 'side'])
    df['day'] = df.path.apply(lambda path: path.split('/')[0])
    df['filename'] = df.path.apply(lambda path: path.split('/')[1])
    df = df.drop(['path'], axis=1)
    return df


def get_person_data(person_id: str, data_dir: pathlib.Path,
                    eval_dir: pathlib.Path
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_images = dict()
    left_poses = dict()
    left_gazes = dict()
    right_images = dict()
    right_poses = dict()
    right_gazes = dict()
    filenames = dict()
    dirpath = data_dir / person_id
    for path in sorted(dirpath.glob('*')):
        mat_data = scipy.io.loadmat(path.as_posix(),
                                    struct_as_record=False,
                                    squeeze_me=True)
        data = mat_data['data']

        day = path.stem
        left_images[day] = data.left.image
        left_poses[day] = data.left.pose
        left_gazes[day] = data.left.gaze

        right_images[day] = data.right.image
        right_poses[day] = data.right.pose
        right_gazes[day] = data.right.gaze

        filenames[day] = mat_data['filenames']

        if not isinstance(filenames[day], np.ndarray):
            left_images[day] = np.array([left_images[day]])
            left_poses[day] = np.array([left_poses[day]])
            left_gazes[day] = np.array([left_gazes[day]])
            right_images[day] = np.array([right_images[day]])
            right_poses[day] = np.array([right_poses[day]])
            right_gazes[day] = np.array([right_gazes[day]])
            filenames[day] = np.array([filenames[day]])

    images = []
    poses = []
    gazes = []
    df = get_eval_info(person_id, eval_dir)
    for _, row in df.iterrows():
        day = row.day
        index = np.where(filenames[day] == row.filename)[0][0]
        if row.side == 'left':
            image = left_images[day][index]
            pose = convert_pose(left_poses[day][index])
            gaze = convert_gaze(left_gazes[day][index])
        else:
            image = right_images[day][index][:, ::-1]
            pose = convert_pose(right_poses[day][index]) * np.array([1, -1])
            gaze = convert_gaze(right_gazes[day][index]) * np.array([1, -1])
        images.append(image)
        poses.append(pose)
        gazes.append(gaze)

    images = np.array(images).astype(np.uint8)
    poses = np.array(poses).astype(np.float32)
    gazes = np.array(gazes).astype(np.float32)

    return images, poses, gazes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset_dir = pathlib.Path(args.dataset)

    for person_id in tqdm.tqdm(range(15)):
        person_id = f'p{person_id:02}'
        data_dir = dataset_dir / 'Data' / 'Normalized'
        eval_dir = dataset_dir / 'Evaluation Subset' / 'sample list for eye image'
        images, poses, gazes = get_person_data(person_id, data_dir, eval_dir)

        output_path = output_dir / person_id
        np.savez(output_path, image=images, pose=poses, gaze=gazes)


if __name__ == '__main__':
    main()
