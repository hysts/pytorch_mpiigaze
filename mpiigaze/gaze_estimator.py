from typing import List

import logging

import numpy as np
import torch
import yacs.config

from common import Camera, Face, MODEL3D
from head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
from mpiigaze.models import create_model

logger = logging.getLogger(__name__)


class GazeEstimator:
    EYE_KEYS = ['reye', 'leye']

    def __init__(self, config: yacs.config.CfgNode):
        self.config = config

        self.camera = Camera(config.demo.camera_params)
        self.normalized_camera = Camera(config.demo.normalized_camera_params)

        self._landmark_estimator = LandmarkEstimator(config)
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self.normalized_camera,
            self.config.demo.normalized_camera_distance)
        self.mpiigaze_model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self.config)
        ckpt = torch.load(self.config.demo.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        model.to(torch.device(self.config.demo.device))
        model.eval()
        return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        MODEL3D.estimate_head_pose(face, self.camera)
        MODEL3D.compute_3d_pose(face)
        MODEL3D.compute_face_eye_centers(face)
        for key in self.EYE_KEYS:
            eye = getattr(face, key)
            self._head_pose_normalizer.normalize(image, eye)
        self._run_mpiigaze_model(face)

    def _run_mpiigaze_model(self, face: Face) -> None:
        images = []
        head_poses = []
        for key in self.EYE_KEYS:
            eye = getattr(face, key)
            image = eye.normalized_image
            normalized_head_pose = eye.normalized_head_rot2d
            if eye.name == 'reye':
                image = image[:, ::-1]
                normalized_head_pose *= np.array([1, -1])
            images.append(image[None, :, :])
            head_poses.append(normalized_head_pose)
        images = np.array(images).astype(np.float32) / 255
        head_poses = np.array(head_poses).astype(np.float32)

        images = torch.from_numpy(images)
        head_poses = torch.from_numpy(head_poses)

        device = torch.device(self.config.demo.device)
        with torch.no_grad():
            images = images.to(device)
            head_poses = head_poses.to(device)
            predictions = self.mpiigaze_model(images, head_poses)
            predictions = predictions.cpu().numpy()

        for i, key in enumerate(self.EYE_KEYS):
            eye = getattr(face, key)
            eye.normalized_gaze_angles = predictions[i]
            if key == 'reye':
                eye.normalized_gaze_angles *= np.array([1, -1])
            eye.angle_to_vector()
            eye.denormalize_gaze_vector()
