"""Hydra integration for configuration loading."""

from typing import Any

from omegaconf import DictConfig, OmegaConf

from .base import Config
from .eval import EvalConfig
from .infrastructure import BufferConfig, LmdbConfig, ParallelismConfig
from .networking import NetworkingConfig
from .paths import PathsConfig
from .profiler import ProfilerConfig
from .rollout import RolloutConfig
from .training import TrainingConfig


def _dict_to_dataclass(data: dict[str, Any], cls: type) -> Any:
    """Convert a dict to a dataclass instance, handling nested structures."""
    field_types = cls.__annotations__
    init_kwargs = {}

    for field_name, field_type in field_types.items():
        if field_name not in data:
            continue

        value = data[field_name]

        # Check if this field is a dataclass type
        if hasattr(field_type, "__annotations__"):
            # Nested dataclass
            init_kwargs[field_name] = _dict_to_dataclass(
                value if isinstance(value, dict) else {}, field_type
            )
        else:
            init_kwargs[field_name] = value

    return cls(**init_kwargs)


def from_hydra_config(hydra_cfg: DictConfig) -> Config:
    """Convert Hydra DictConfig to Config dataclass.

    Args:
        hydra_cfg: Hydra DictConfig from @hydra.main()

    Returns:
        Config instance populated from Hydra config
    """
    # Convert to dict
    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)

    # Build nested dataclasses from top-level keys
    networking = _dict_to_dataclass(cfg_dict.get("networking", {}), NetworkingConfig)
    paths = _dict_to_dataclass(cfg_dict.get("paths", {}), PathsConfig)
    buffers = _dict_to_dataclass(cfg_dict.get("buffers", {}), BufferConfig)
    lmdb = _dict_to_dataclass(cfg_dict.get("lmdb", {}), LmdbConfig)
    parallelism = _dict_to_dataclass(cfg_dict.get("parallelism", {}), ParallelismConfig)
    training = _dict_to_dataclass(cfg_dict.get("training", {}), TrainingConfig)
    rollout = _dict_to_dataclass(cfg_dict.get("rollout", {}), RolloutConfig)
    evaluation_cfg = _dict_to_dataclass(cfg_dict.get("evaluation", {}), EvalConfig)
    profiler = _dict_to_dataclass(cfg_dict.get("profiler", {}), ProfilerConfig)

    cfg = Config(
        experience_name=cfg_dict.get("experience_name"),
        networking=networking,
        paths=paths,
        buffers=buffers,
        lmdb=lmdb,
        parallelism=parallelism,
        training=training,
        rollout=rollout,
        evaluation=evaluation_cfg,
        profiler=profiler,
        log_file=cfg_dict.get("log_file"),
    )

    cfg.validate()
    return cfg
