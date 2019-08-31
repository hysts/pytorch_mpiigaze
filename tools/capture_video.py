#!/usr/bin/env python

import argparse
import datetime
import pathlib

import cv2

QUIT_KEYS = {27, ord('q')}


def create_timestamp():
    dt = datetime.datetime.now()
    return dt.strftime('%Y%m%d_%H%M%S')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default='videos')
    parser.add_argument('--cap-size', type=int, nargs=2, default=(640, 480))
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    width, height = args.cap_size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f'{create_timestamp()}.mp4'
    writer = cv2.VideoWriter(output_path.as_posix(),
                             cv2.VideoWriter_fourcc(*'H264'), 30,
                             (width, height))

    while True:
        key = cv2.waitKey(1) & 0xff
        if key in QUIT_KEYS:
            break

        ok, frame = cap.read()
        if not ok:
            break

        writer.write(frame)
        cv2.imshow('frame', frame[:, ::-1])

    cap.release()
    writer.release()


if __name__ == '__main__':
    main()
