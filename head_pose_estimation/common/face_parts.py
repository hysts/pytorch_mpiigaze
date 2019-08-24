from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


class FaceParts:
    def __init__(self, name: str):
        self.name = name
        self.center: Optional[np.ndarray] = None
        self.head_pose_rot: Optional[Rotation] = None
        self.normalizing_rot: Optional[Rotation] = None
        self.normalized_head_rot2d: Optional[np.ndarray] = None
        self.normalized_image: Optional[np.ndarray] = None

    @property
    def distance(self):
        return np.linalg.norm(self.center)
