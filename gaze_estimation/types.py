import enum


class GazeEstimationMethod(enum.Enum):
    MPIIGaze = enum.auto()
    MPIIFaceGaze = enum.auto()


class LossType(enum.Enum):
    L1 = enum.auto()
    L2 = enum.auto()
    SmoothL1 = enum.auto()
