"""Structured schema for the config files.

Loaded configs are merged with this schema so that unknown keys and
type mismatches are rejected at startup instead of failing deep inside
the training pipeline.
"""

import dataclasses


@dataclasses.dataclass
class DatasetConfig:
    dataset_dir: str = "datasets/MPIIGaze.h5"


@dataclasses.dataclass
class TransformConfig:
    mpiifacegaze_face_size: int = 224
    mpiifacegaze_gray: bool = False


@dataclasses.dataclass
class ModelBackboneConfig:
    name: str = "resnet_simple"
    pretrained: str = "resnet18"
    resnet_block: str = "basic"
    resnet_layers: list[int] = dataclasses.field(default_factory=lambda: [2, 2, 2])


@dataclasses.dataclass
class ModelConfig:
    name: str = "lenet"
    backbone: ModelBackboneConfig = dataclasses.field(default_factory=ModelBackboneConfig)


@dataclasses.dataclass
class TrainDataloaderConfig:
    num_workers: int = 2
    drop_last: bool = True
    pin_memory: bool = False


@dataclasses.dataclass
class ValDataloaderConfig:
    num_workers: int = 1
    pin_memory: bool = False


@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 64
    # options: sgd, adam, amsgrad
    optimizer: str = "sgd"
    base_lr: float = 0.01
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 1e-4
    no_weight_decay_on_bn: bool = False
    # options: L1, L2, SmoothL1
    loss: str = "L2"
    seed: int = 0
    val_first: bool = True
    val_period: int = 1
    test_id: int = 0
    val_ratio: float = 0.1
    output_dir: str = "experiments/mpiigaze/exp00"
    log_period: int = 100
    checkpoint_period: int = 10
    use_tensorboard: bool = True
    train_dataloader: TrainDataloaderConfig = dataclasses.field(default_factory=TrainDataloaderConfig)
    val_dataloader: ValDataloaderConfig = dataclasses.field(default_factory=ValDataloaderConfig)


@dataclasses.dataclass
class TensorboardConfig:
    train_images: bool = False
    val_images: bool = False
    model_params: bool = False


@dataclasses.dataclass
class AdamConfig:
    betas: list[float] = dataclasses.field(default_factory=lambda: [0.9, 0.999])


@dataclasses.dataclass
class OptimConfig:
    adam: AdamConfig = dataclasses.field(default_factory=AdamConfig)


@dataclasses.dataclass
class SchedulerConfig:
    epochs: int = 40
    # options: multistep, cosine
    type: str = "multistep"
    milestones: list[int] = dataclasses.field(default_factory=lambda: [20, 30])
    lr_decay: float = 0.1
    lr_min_factor: float = 0.001


@dataclasses.dataclass
class TestDataloaderConfig:
    num_workers: int = 2
    pin_memory: bool = False


@dataclasses.dataclass
class TestConfig:
    test_id: int = 0
    checkpoint: str = ""
    output_dir: str = ""
    batch_size: int = 256
    dataloader: TestDataloaderConfig = dataclasses.field(default_factory=TestDataloaderConfig)


@dataclasses.dataclass
class CudnnConfig:
    benchmark: bool = True
    deterministic: bool = False


@dataclasses.dataclass
class AppConfig:
    # options: MPIIGaze, MPIIFaceGaze
    mode: str = "MPIIGaze"
    device: str = "cuda"
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    transform: TransformConfig = dataclasses.field(default_factory=TransformConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    tensorboard: TensorboardConfig = dataclasses.field(default_factory=TensorboardConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    test: TestConfig = dataclasses.field(default_factory=TestConfig)
    cudnn: CudnnConfig = dataclasses.field(default_factory=CudnnConfig)
