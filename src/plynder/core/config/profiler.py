"""Profiler configuration for torch.profiler."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ProfilerConfig:
    """Configuration for torch.profiler profiling.

    Controls the profiling of training steps for Chrome trace visualization.
    Profiling is skipped unless ``enabled`` is ``True``.
    """

    enabled: bool = False
    """Whether to enable the profiler."""

    wait_steps: int = 5
    """Number of steps to wait before starting to record."""

    warmup_steps: int = 3
    """Number of warmup steps (recorded but not saved)."""

    active_steps: int = 5
    """Number of steps to actively record and save traces."""

    record_shapes: bool = False
    """Whether to record tensor shapes."""

    profile_memory: bool = True
    """Whether to profile memory usage."""

    with_stack: bool = True
    """Whether to record Python call stacks."""

    with_flops: bool = True
    """Whether to estimate FLOPs."""

    output_dir: str | None = None
    """Directory for trace output files. If None, uses the training log directory."""

    activities: list[Literal["cpu", "cuda"]] = field(default_factory=lambda: ["cpu", "cuda"])
    """Activities to profile. Defaults to all available."""
