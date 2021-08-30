import dataclasses

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Camera
from .face import Face


@dataclasses.dataclass(frozen=True)
class FaceModel:
    """3D face model for Multi-PIE 68 points mark-up.

    In the camera coordinate system, the X axis points to the right from
    camera, the Y axis points down, and the Z axis points forward.

    The face model is facing the camera. Here, the Z axis is
    perpendicular to the plane passing through the three midpoints of
    the eyes and mouth, the X axis is parallel to the line passing
    through the midpoints of both eyes, and the origin is at the tip of
    the nose.

    The units of the coordinate system are meters and the distance
    between outer eye corners of the model is set to 90mm.

    The model coordinate system is defined as the camera coordinate
    system rotated 180 degrees around the Y axis.
    """
    LANDMARKS: np.ndarray = np.array([
        [-0.07141807, -0.02827123, 0.08114384],
        [-0.07067417, -0.00961522, 0.08035654],
        [-0.06844646, 0.00895837, 0.08046731],
        [-0.06474301, 0.02708319, 0.08045689],
        [-0.05778475, 0.04384917, 0.07802191],
        [-0.04673809, 0.05812865, 0.07192291],
        [-0.03293922, 0.06962711, 0.06106274],
        [-0.01744018, 0.07850638, 0.04752971],
        [0., 0.08105961, 0.0425195],
        [0.01744018, 0.07850638, 0.04752971],
        [0.03293922, 0.06962711, 0.06106274],
        [0.04673809, 0.05812865, 0.07192291],
        [0.05778475, 0.04384917, 0.07802191],
        [0.06474301, 0.02708319, 0.08045689],
        [0.06844646, 0.00895837, 0.08046731],
        [0.07067417, -0.00961522, 0.08035654],
        [0.07141807, -0.02827123, 0.08114384],
        [-0.05977758, -0.0447858, 0.04562813],
        [-0.05055506, -0.05334294, 0.03834846],
        [-0.0375633, -0.05609241, 0.03158344],
        [-0.02423648, -0.05463779, 0.02510117],
        [-0.01168798, -0.04986641, 0.02050337],
        [0.01168798, -0.04986641, 0.02050337],
        [0.02423648, -0.05463779, 0.02510117],
        [0.0375633, -0.05609241, 0.03158344],
        [0.05055506, -0.05334294, 0.03834846],
        [0.05977758, -0.0447858, 0.04562813],
        [0., -0.03515768, 0.02038099],
        [0., -0.02350421, 0.01366667],
        [0., -0.01196914, 0.00658284],
        [0., 0., 0.],
        [-0.01479319, 0.00949072, 0.01708772],
        [-0.00762319, 0.01179908, 0.01419133],
        [0., 0.01381676, 0.01205559],
        [0.00762319, 0.01179908, 0.01419133],
        [0.01479319, 0.00949072, 0.01708772],
        [-0.045, -0.032415, 0.03976718],
        [-0.0370546, -0.0371723, 0.03579593],
        [-0.0275166, -0.03714814, 0.03425518],
        [-0.01919724, -0.03101962, 0.03359268],
        [-0.02813814, -0.0294397, 0.03345652],
        [-0.03763013, -0.02948442, 0.03497732],
        [0.01919724, -0.03101962, 0.03359268],
        [0.0275166, -0.03714814, 0.03425518],
        [0.0370546, -0.0371723, 0.03579593],
        [0.045, -0.032415, 0.03976718],
        [0.03763013, -0.02948442, 0.03497732],
        [0.02813814, -0.0294397, 0.03345652],
        [-0.02847002, 0.03331642, 0.03667993],
        [-0.01796181, 0.02843251, 0.02335485],
        [-0.00742947, 0.0258057, 0.01630812],
        [0., 0.0275555, 0.01538404],
        [0.00742947, 0.0258057, 0.01630812],
        [0.01796181, 0.02843251, 0.02335485],
        [0.02847002, 0.03331642, 0.03667993],
        [0.0183606, 0.0423393, 0.02523355],
        [0.00808323, 0.04614537, 0.01820142],
        [0., 0.04688623, 0.01716318],
        [-0.00808323, 0.04614537, 0.01820142],
        [-0.0183606, 0.0423393, 0.02523355],
        [-0.02409981, 0.03367606, 0.03421466],
        [-0.00756874, 0.03192644, 0.01851247],
        [0., 0.03263345, 0.01732347],
        [0.00756874, 0.03192644, 0.01851247],
        [0.02409981, 0.03367606, 0.03421466],
        [0.00771924, 0.03711846, 0.01940396],
        [0., 0.03791103, 0.0180805],
        [-0.00771924, 0.03711846, 0.01940396],
    ],
                                     dtype=np.float)

    REYE_INDICES: np.ndarray = np.array([36, 39])
    LEYE_INDICES: np.ndarray = np.array([42, 45])
    MOUTH_INDICES: np.ndarray = np.array([48, 54])

    CHIN_INDEX: int = 8
    NOSE_INDEX: int = 30

    def estimate_head_pose(self, face: Face, camera: Camera) -> None:
        """Estimate the head pose by fitting 3D template model."""
        # If the number of the template points is small, cv2.solvePnP
        # becomes unstable, so set the default value for rvec and tvec
        # and set useExtrinsicGuess to True.
        # The default values of rvec and tvec below mean that the
        # initial estimate of the head pose is not rotated and the
        # face is in front of the camera.
        rvec = np.zeros(3, dtype=np.float)
        tvec = np.array([0, 0, 1], dtype=np.float)
        _, rvec, tvec = cv2.solvePnP(self.LANDMARKS,
                                     face.landmarks,
                                     camera.camera_matrix,
                                     camera.dist_coefficients,
                                     rvec,
                                     tvec,
                                     useExtrinsicGuess=True,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        rot = Rotation.from_rotvec(rvec)
        face.head_pose_rot = rot
        face.head_position = tvec
        face.reye.head_pose_rot = rot
        face.leye.head_pose_rot = rot

    def compute_3d_pose(self, face: Face) -> None:
        """Compute the transformed model."""
        rot = face.head_pose_rot.as_matrix()
        face.model3d = self.LANDMARKS @ rot.T + face.head_position

    def compute_face_eye_centers(self, face: Face) -> None:
        """Compute the centers of the face and eyes.

        The face center is defined as the average coordinates of the six
        points at the corners of both eyes and the mouth. The eye
        centers are defined as the average coordinates of the corners of
        each eye.
        """
        face.center = face.model3d[np.concatenate(
            [self.REYE_INDICES, self.LEYE_INDICES,
             self.MOUTH_INDICES])].mean(axis=0)
        face.reye.center = face.model3d[self.REYE_INDICES].mean(axis=0)
        face.leye.center = face.model3d[self.LEYE_INDICES].mean(axis=0)


MODEL3D = FaceModel()
