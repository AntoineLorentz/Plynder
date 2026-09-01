"""Configuration module for plynder.

Provides a hierarchical configuration system based on dataclasses,
designed to work with Hydra for CLI overrides.

Example usage:
    from plynder.core.config import Config

    # Use defaults
    cfg = Config()

    # Override specific values
    cfg = Config(
        training=TrainingConfig(
            optimizer=OptimizerConfig(learning_rate=1e-3)
        )
    )

    # With Hydra:
    # python train.py training.optimizer.learning_rate=1e-3
"""

from .base import Config
from .eval import EvalConfig
from .infrastructure import BufferConfig, LmdbConfig, ParallelismConfig
from .networking import NetworkingConfig, PortsConfig
from .paths import PathsConfig
from .profiler import ProfilerConfig
from .rollout import RolloutConfig, SamplingConfig, VllmRolloutConfig
from .training import (
    BatchingConfig,
    CheckpointConfig,
    OptimizerConfig,
    TrainingConfig,
)

__all__ = [
    # Main config
    "Config",
    # Networking
    "NetworkingConfig",
    "PortsConfig",
    # Paths
    "PathsConfig",
    # Infrastructure
    "BufferConfig",
    "LmdbConfig",
    "ParallelismConfig",
    # Training
    "TrainingConfig",
    "OptimizerConfig",
    "BatchingConfig",
    "CheckpointConfig",
    # Rollout
    "RolloutConfig",
    "VllmRolloutConfig",
    "SamplingConfig",
    # Eval
    "EvalConfig",
    # Profiler
    "ProfilerConfig",
]
