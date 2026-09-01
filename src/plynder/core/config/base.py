"""Base configuration class that composes all sub-configurations."""

from dataclasses import dataclass, field

from .eval import EvalConfig
from .infrastructure import BufferConfig, LmdbConfig, ParallelismConfig
from .networking import NetworkingConfig
from .paths import PathsConfig
from .profiler import ProfilerConfig
from .rollout import RolloutConfig
from .training import TrainingConfig


@dataclass
class Config:
    """Root configuration class for plynder.

    Composes all sub-configurations into a single hierarchical structure.
    Designed to work with Hydra for CLI overrides.
    """

    experience_name: str = "plynder"
    """Name of the training run. Used for organizing logs and checkpoints."""

    networking: NetworkingConfig = field(default_factory=NetworkingConfig)
    """ZMQ networking and address configuration."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    """Filesystem paths configuration."""

    buffers: BufferConfig = field(default_factory=BufferConfig)
    """Buffer configuration for data transfer."""

    lmdb: LmdbConfig = field(default_factory=LmdbConfig)
    """LMDB database configuration."""

    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    """Data loading parallelism configuration."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Training-specific configuration."""

    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    """Rollout-specific configuration."""

    evaluation: EvalConfig = field(default_factory=EvalConfig)
    """Evaluation-specific configuration"""

    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    """Profiler configuration for torch.profiler traces."""

    # Logging configuration
    log_file: str | None = None
    """Path to log file. If None, auto-generated based on timestamp."""

    def validate(self) -> None:
        """Validate configuration consistency.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Buffer size must be divisible by batch size
        if self.buffers.samples_buffering_sampler % self.training.batching.total_batch_size != 0:
            raise ValueError(
                f"buffer_size ({self.buffers.samples_buffering_sampler}) must be divisible by "
                f"total_batch_size ({self.training.batching.total_batch_size})"
            )
