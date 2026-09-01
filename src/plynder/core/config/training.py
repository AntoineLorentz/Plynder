"""Training configuration."""

from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    learning_rate: float = 1e-4
    """Initial learning rate (after warmup)."""

    start_learning_rate: float | None = None
    """Starting learning rate for warmup."""

    lr_decay_steps: int | None = None
    """Number of steps for learning rate decay."""

    adam_beta1: float = 0.9
    """Beta1 parameter for Adam optimizer."""

    adam_beta2: float = 0.999
    """Beta2 parameter for Adam optimizer."""

    adam_weight_decay: float = 1e-4
    """Weight decay for Adam optimizer."""

    adam_epsilon: float = 1e-8
    """Epsilon value for Adam optimizer."""

    gradient_clipping: float | None = None
    """Gradient clipping value."""


@dataclass
class BatchingConfig:
    """Batching configuration."""

    total_batch_size: int = 1024
    """Total batch size accounting for dynamic grad acc and distribution."""

    microbatch_max_tokens: int = 102_400
    """Maximum number of tokens in one micro-batch per GPU. Controls VRAM usage (larger = more OOM risk).
    Does NOT affect total batch size, number of gradient updates, or rollout throughput.
    Gradient accumulation is adjusted dynamically to reach total_batch_size."""


@dataclass
class CheckpointConfig:
    """Checkpoint and model saving configuration."""

    save_steps: int = 10_000
    """Steps between checkpoint saves."""

    archive_steps: int = 50_000
    """Steps between archive checkpoint saves."""


@dataclass
class TrainingConfig:
    """Main training configuration."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer settings."""

    batching: BatchingConfig = field(default_factory=BatchingConfig)
    """Batching settings."""

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Checkpoint settings."""

    seed: int | None = None
    """Seed for reproducible training."""

    send_model_frequency: int = 16
    """How often to send model to rollout."""

    training_time: int = 12 * 3600
    """Maximum training time in sec."""

    # CISPO hyperparameters (MiniMax-M1 Eq.4-5 + Eq.7 IS mask). Defaults are
    # the values used to train the released model.
    cispo_epsilon_low: float = 0.1
    """CISPO lower IS clip: clamp weight at 1 - eps_low."""
    cispo_epsilon_high: float = 0.2
    """CISPO upper IS clip: clamp weight at 1 + eps_high."""
    cispo_use_mask: bool = True
    """Enable the Eq.7 IS mask: zero the gradient for over-confident good
    tokens (A>0, r > 1+eps_high) and abandoned bad tokens (A<=0, r < 1-eps_low)."""
