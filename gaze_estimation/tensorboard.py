import pathlib

from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter


class DummyWriter:
    """No-op stand-in for SummaryWriter used when tensorboard is disabled."""

    def add_scalar(self, *args: object, **kwargs: object) -> None:
        pass

    def add_image(self, *args: object, **kwargs: object) -> None:
        pass

    def add_histogram(self, *args: object, **kwargs: object) -> None:
        pass

    def close(self) -> None:
        pass


TensorboardWriter = SummaryWriter | DummyWriter


def create_tensorboard_writer(config: DictConfig, output_dir: pathlib.Path) -> TensorboardWriter:
    if config.train.use_tensorboard:
        return SummaryWriter(output_dir.as_posix())
    return DummyWriter()
