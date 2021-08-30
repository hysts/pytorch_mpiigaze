from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Camera
from .face import Face
from .face_model import MODEL3D

AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


class Visualizer:
    def __init__(self, camera: Camera):
        self._camera = camera

        self.image: Optional[np.ndarray] = None

    def set_image(self, image: np.ndarray) -> None:
        self.image = image

    def draw_bbox(self,
                  bbox: np.ndarray,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  lw: int = 1) -> None:
        assert self.image is not None
        assert bbox.shape == (2, 2)
        bbox = np.round(bbox).astype(np.int).tolist()
        cv2.rectangle(self.image, tuple(bbox[0]), tuple(bbox[1]), color, lw)

    @staticmethod
    def _convert_pt(point: np.ndarray) -> Tuple[int, int]:
        return tuple(np.round(point).astype(np.int).tolist())

    def draw_points(self,
                    points: np.ndarray,
                    color: Tuple[int, int, int] = (0, 0, 255),
                    size: int = 3) -> None:
        assert self.image is not None
        assert points.shape[1] == 2
        for pt in points:
            pt = self._convert_pt(pt)
            cv2.circle(self.image, pt, size, color, cv2.FILLED)

    def draw_3d_points(self,
                       points3d: np.ndarray,
                       color: Tuple[int, int, int] = (255, 0, 255),
                       size=3) -> None:
        assert self.image is not None
        assert points3d.shape[1] == 3
        points2d = self._camera.project_points(points3d)
        self.draw_points(points2d, color=color, size=size)

    def draw_3d_line(self,
                     point0: np.ndarray,
                     point1: np.ndarray,
                     color: Tuple[int, int, int] = (255, 255, 0),
                     lw=2) -> None:
        assert self.image is not None
        assert point0.shape == point1.shape == (3, )
        points3d = np.vstack([point0, point1])
        points2d = self._camera.project_points(points3d)
        pt0 = self._convert_pt(points2d[0])
        pt1 = self._convert_pt(points2d[1])
        cv2.line(self.image, pt0, pt1, color, lw, cv2.LINE_AA)

    def draw_model_axes(self, face: Face, length: float, lw: int = 2) -> None:
        assert self.image is not None
        assert face is not None
        assert face.head_pose_rot is not None
        assert face.head_position is not None
        assert face.landmarks is not None
        # Get the axes of the model coordinate system
        axes3d = np.eye(3, dtype=np.float) @ Rotation.from_euler(
            'XYZ', [0, np.pi, 0]).as_matrix()
        axes3d = axes3d * length
        axes2d = self._camera.project_points(axes3d,
                                             face.head_pose_rot.as_rotvec(),
                                             face.head_position)
        center = face.landmarks[MODEL3D.NOSE_INDEX]
        center = self._convert_pt(center)
        for pt, color in zip(axes2d, AXIS_COLORS):
            pt = self._convert_pt(pt)
            cv2.line(self.image, center, pt, color, lw, cv2.LINE_AA)
