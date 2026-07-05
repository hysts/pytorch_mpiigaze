from omegaconf import DictConfig, OmegaConf

from .config_schema import AppConfig


def get_default_config() -> DictConfig:
    return OmegaConf.structured(AppConfig)
