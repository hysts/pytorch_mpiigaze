#!/usr/bin/env python
"""
Download pretrained model
```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```
"""

import logging

import cv2
import numpy as np

from common import Visualizer
from mpiigaze.gaze_estimator import GazeEstimator
from mpiigaze.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    gaze_estimator = GazeEstimator(config)
    visualizer = Visualizer(gaze_estimator.camera)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, gaze_estimator.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, gaze_estimator.camera.height)

    QUIT_KEYS = {27, ord('q')}
    show_bbox = True
    show_head_pose = True
    show_landmarks = True
    show_normalized_image = False
    show_template_model = False
    while True:
        key = cv2.waitKey(1) & 0xff
        if key in QUIT_KEYS:
            break
        elif key == ord('b'):
            show_bbox = not show_bbox
        elif key == ord('l'):
            show_landmarks = not show_landmarks
        elif key == ord('h'):
            show_head_pose = not show_head_pose
        elif key == ord('n'):
            show_normalized_image = not show_normalized_image
        elif key == ord('t'):
            show_template_model = not show_template_model

        ok, frame = cap.read()
        if not ok:
            break

        visualizer.set_image(frame.copy())
        faces = gaze_estimator.detect_faces(frame)
        for face in faces:
            gaze_estimator.estimate_gaze(frame, face)

            for key in ['reye', 'leye']:
                eye = getattr(face, key)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(f'[{key}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')

            if show_bbox:
                visualizer.draw_bbox(face.bbox)
            if show_head_pose:
                # Draw the axes of the model coordinate system
                visualizer.draw_model_axes(face, lw=2)

                euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
                pitch, yaw, roll = face.change_coordinate_system(euler_angles)
                logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                            f'roll: {roll:.2f}, distance: {face.distance:.2f}')
            if show_landmarks:
                visualizer.draw_points(face.landmarks,
                                       color=(0, 255, 255),
                                       size=1)
            if show_template_model:
                # Draw the transformed template model
                visualizer.draw_3d_points(face.model3d,
                                          color=(255, 0, 525),
                                          size=1)
            if show_normalized_image:
                reye = face.reye.normalized_image
                leye = face.leye.normalized_image
                normalized = np.hstack([reye, leye])
                cv2.imshow('normalized', normalized[:, ::-1])

            # Draw the gaze vectors
            length = config.demo.gaze_visualization_length
            for key in gaze_estimator.EYE_KEYS:
                eye = getattr(face, key)
                visualizer.draw_3d_line(eye.center,
                                        eye.center + length * eye.gaze_vector)

        cv2.imshow('frame', visualizer.image[:, ::-1])

    cap.release()


if __name__ == '__main__':
    main()
