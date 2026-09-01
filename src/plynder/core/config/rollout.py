"""Rollout configuration."""

from dataclasses import dataclass, field


@dataclass
class VllmRolloutConfig:
    """vLLM engine configuration for rollout."""

    num_engines_per_gpu: int = 4
    """Number of vLLM engines per GPU."""

    kv_cache_memory_bytes: int = field(default=5 * 1024**3)  # 5GB
    """vLLM KV cache memory in bytes."""

    max_concurrent_groups: int = 32
    """Maximum number of concurrent groups per engine."""


@dataclass
class SamplingConfig:
    """Sampling configuration."""

    group_size: int = 16
    """Number of continuations sampled per opening (group size for group-relative advantages)."""

    temperature: float = 1.0
    """Sampling temperature for rollout."""

    data_timeout: int = 900  # 15 minutes
    """Timeout in seconds before discarding a group."""

    max_tokens: int = 1000
    """Maximum number of tokens to sample per group (rollout-side vLLM generation cap).
    This is NOT the training microbatch_max_tokens; it caps per-group generation length."""

    opening_abandonment_threshold: int | None = None
    """Number of consecutive discarded groups before abandoning an opening."""

    probability_retry_discarded_opening: float = 0.01
    """Probability to retry a previously discarded opening."""


@dataclass
class RolloutConfig:
    """Main rollout configuration."""

    vllm: VllmRolloutConfig = field(default_factory=VllmRolloutConfig)
    """vLLM engine settings."""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    """Sampling settings."""
