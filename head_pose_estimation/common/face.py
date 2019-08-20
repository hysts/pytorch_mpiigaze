from typing import Optional

import numpy as np

from head_pose_estimation.common import Eye, FaceParts


class Face(FaceParts):
    def __init__(self, bbox: np.ndarray, landmarks: np.ndarray):
        super().__init__('face')
        self.bbox = bbox
        self.landmarks = landmarks

        self.reye: Eye = Eye('reye')
        self.leye: Eye = Eye('leye')

        self.head_position: Optional[np.ndarray] = None
        self.model3d: Optional[np.ndarray] = None

    @staticmethod
    def change_coordinate_system(euler_angles: np.ndarray) -> np.ndarray:
        return euler_angles * np.array([-1, 1, -1])
