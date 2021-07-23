import dataclasses
from typing import Optional

import cv2
import numpy as np
import yaml


@dataclasses.dataclass()
class Camera:
    width: int = dataclasses.field(init=False)
    height: int = dataclasses.field(init=False)
    camera_matrix: np.ndarray = dataclasses.field(init=False)
    dist_coefficients: np.ndarray = dataclasses.field(init=False)

    camera_params_path: dataclasses.InitVar[str] = None

    def __post_init__(self, camera_params_path):
        with open(camera_params_path) as f:
            data = yaml.safe_load(f)
        self.width = data['image_width']
        self.height = data['image_height']
        self.camera_matrix = np.array(data['camera_matrix']['data']).reshape(
            3, 3)
        self.dist_coefficients = np.array(
            data['distortion_coefficients']['data']).reshape(-1, 1)

    def project_points(self,
                       points3d: np.ndarray,
                       rvec: Optional[np.ndarray] = None,
                       tvec: Optional[np.ndarray] = None) -> np.ndarray:
        assert points3d.shape[1] == 3
        if rvec is None:
            rvec = np.zeros(3, dtype=np.float)
        if tvec is None:
            tvec = np.zeros(3, dtype=np.float)
        points2d, _ = cv2.projectPoints(points3d, rvec, tvec,
                                        self.camera_matrix,
                                        self.dist_coefficients)
        return points2d.reshape(-1, 2)
