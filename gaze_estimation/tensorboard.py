import pathlib
from typing import Any

import yacs.config
from torch.utils.tensorboard import SummaryWriter


class DummyWriter:
    """No-op stand-in for SummaryWriter used when tensorboard is disabled."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_image(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_histogram(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


def create_tensorboard_writer(config: yacs.config.CfgNode, output_dir: pathlib.Path):
    if config.train.use_tensorboard:
        return SummaryWriter(output_dir.as_posix())
    return DummyWriter()
