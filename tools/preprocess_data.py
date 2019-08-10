#!/usr/bin/env python

import argparse
import pathlib
import numpy as np
import pandas as pd
import scipy.io
import cv2


def convert_pose(vect):
    M, _ = cv2.Rodrigues(np.array(vect).astype(np.float32))
    vec = M[:, 2]
    yaw = np.arctan2(vec[0], vec[2])
    pitch = np.arcsin(vec[1])
    return np.array([yaw, pitch])


def convert_gaze(vect):
    x, y, z = vect
    yaw = np.arctan2(-x, -z)
    pitch = np.arcsin(-y)
    return np.array([yaw, pitch])


def get_eval_info(subject_id, evaldir):
    path = evaldir / f'{subject_id}.txt'
    df = pd.read_csv(path, delimiter=' ', header=None, names=['path', 'side'])
    df['day'] = df.path.apply(lambda path: path.split('/')[0])
    df['filename'] = df.path.apply(lambda path: path.split('/')[1])
    df = df.drop(['path'], axis=1)
    return df


def get_subject_data(subject_id, datadir, evaldir):
    left_images = {}
    left_poses = {}
    left_gazes = {}
    right_images = {}
    right_poses = {}
    right_gazes = {}
    filenames = {}
    dirpath = datadir / subject_id
    for path in sorted(dirpath.glob('*')):
        matdata = scipy.io.loadmat(path.as_posix(),
                                   struct_as_record=False,
                                   squeeze_me=True)
        data = matdata['data']

        day = path.stem
        left_images[day] = data.left.image
        left_poses[day] = data.left.pose
        left_gazes[day] = data.left.gaze

        right_images[day] = data.right.image
        right_poses[day] = data.right.pose
        right_gazes[day] = data.right.gaze

        filenames[day] = matdata['filenames']

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
    df = get_eval_info(subject_id, evaldir)
    for _, row in df.iterrows():
        day = row.day
        index = np.where(filenames[day] == row.filename)[0][0]
        if row.side == 'left':
            image = left_images[day][index]
            pose = convert_pose(left_poses[day][index])
            gaze = convert_gaze(left_gazes[day][index])
        else:
            image = right_images[day][index][:, ::-1]
            pose = convert_pose(right_poses[day][index]) * np.array([-1, 1])
            gaze = convert_gaze(right_gazes[day][index]) * np.array([-1, 1])
        images.append(image)
        poses.append(pose)
        gazes.append(gaze)

    images = np.array(images).astype(np.float32) / 255
    poses = np.array(poses).astype(np.float32)
    gazes = np.array(gazes).astype(np.float32)

    return images, poses, gazes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    dataset_dir = pathlib.Path(args.dataset)

    for subject_id in range(15):
        subject_id = f'p{subject_id:02}'
        datadir = dataset_dir / 'Data' / 'Normalized'
        evaldir = dataset_dir / 'Evaluation Subset' / 'sample list for eye image'
        images, poses, gazes = get_subject_data(subject_id, datadir, evaldir)

        outpath = outdir / subject_id
        np.savez(outpath, image=images, pose=poses, gaze=gazes)


if __name__ == '__main__':
    main()
