"""Infrastructure configuration: LMDB, buffers, parallelism."""

from dataclasses import dataclass, field


@dataclass
class BufferConfig:
    """Buffer configuration for data transfer between components.

    The samples_buffer is shared between training and rollout and must be identical.
    """

    samples_buffer: int = 64
    """Number of trajectories to pack together before pyarrow serialization.
    Must be identical in both training and rollout modules."""

    samples_buffering_sampler: int = 6144
    """Data buffer size for training batching.
    Must be divisible by total_batch_size."""

    ring_record_capacity: int = 1280
    """Ring buffer record capacity for LMDB."""


@dataclass
class LmdbConfig:
    """LMDB database configuration."""

    db_path: str = "/dev/shm/msgbuf"
    """Path to LMDB database."""

    map_size: int = field(default=10 * (1 << 30))  # 10GB
    """LMDB map size in bytes."""


@dataclass
class ParallelismConfig:
    """Data loading parallelism configuration."""

    dataloader_num_workers: int = 0
    """Number of subprocesses for data loading.
    0 means data is loaded in the main process."""
