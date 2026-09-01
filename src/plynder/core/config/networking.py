"""Networking configuration: ZMQ addresses and ports."""

from dataclasses import dataclass, field


@dataclass
class PortsConfig:
    """Port configuration for ZMQ communication.

    Communication flows:
    - Samples flow: Rollout PUSH -> Trainer PULL
    - Model update (trainer->rollout): Trainer PUB -> Rollout SUB
    - Model update (rollout->rollout): Rollout PUB -> Rollout SUB (collective RPC)
    """

    control: int = 5554
    """Port for control messages (e.g., synchronize, shutdown)."""

    samples: int = 5555
    """Trainer receives samples from rollout."""

    model: int = 5556
    """Trainer publishes model updates to rollout/evaluation."""

    stats: int = 5557
    """Trainer/Rollout publishes stats."""

    microbatch_assignment: int = 5558
    """Sampler address (trainer side)."""

    model_rollout: int = 5559
    """Rollout publishes model for collective RPC (intra-rollout communication)."""

    samples_rollout: int = 5560
    """Rollout Workers and rust backend push data to Sender."""


@dataclass
class NetworkingConfig:
    """Networking configuration for all ZMQ communication."""

    train_ip: str = "localhost"
    """Trainer IP address. Rollout connects to this address."""

    ports: PortsConfig = field(default_factory=PortsConfig)

    # ========== Topics for PUB/SUB model ==========

    rollout_topic: str = "rollout"
    """Topic used by the trainer PUB `model` to signal the model is for rollout update."""

    eval_topic: str = "eval"
    """Topic used by the trainer PUB `model` to signal the model is for evaluation."""

    # ========== Other settings ==========

    synchronize: bool = True

    # ========== Trainer addresses (bind) ==========

    @property
    def control_router(self) -> str:
        """Address for control messages (e.g., synchronize, shutdown)."""
        return f"tcp://*:{self.ports.control}"

    @property
    def samples_pull(self) -> str:
        """Address for trainer to pull samples from rollout."""
        return f"tcp://*:{self.ports.samples}"

    @property
    def model_pub(self) -> str:
        """Address for trainer to publish model updates."""
        return f"tcp://*:{self.ports.model}"

    @property
    def stats_trainer_sub(self) -> str:
        """Address for trainer to receive stats."""
        return f"tcp://*:{self.ports.stats}"

    @property
    def stats_trainer_pub(self) -> str:
        """Address for trainer to send stats."""
        return f"tcp://localhost:{self.ports.stats}"

    @property
    def microbatch_assignment_router(self) -> str:
        """Address for trainer sampler to send microbatch assignment."""
        return f"tcp://*:{self.ports.microbatch_assignment}"

    @property
    def microbatch_assignment_dealer(self) -> str:
        """Address for trainer dataloader workers to receive microbatch assignment."""
        return f"tcp://localhost:{self.ports.microbatch_assignment}"

    # ========== Rollout/Eval addresses (connect to trainer) ==========

    @property
    def control_dealer(self) -> str:
        """Address for rollout/eval to receive control messages (e.g., synchronize, shutdown)."""
        return f"tcp://{self.train_ip}:{self.ports.control}"

    @property
    def samples_push(self) -> str:
        """Address for rollout to push ready samples to trainer."""
        return f"tcp://{self.train_ip}:{self.ports.samples}"

    @property
    def model_sub(self) -> str:
        """Address for rollout/eval to subscribe to trainer model updates."""
        return f"tcp://{self.train_ip}:{self.ports.model}"

    @property
    def stats_rollout_pub(self) -> str:
        """Address for rollout to send to trainer stats."""
        return f"tcp://{self.train_ip}:{self.ports.stats}"

    # ========== Rollout internal (collective RPC) ==========

    @property
    def rollout_model_pub(self) -> str:
        """Address for rollout to publish model (collective RPC)."""
        return f"tcp://*:{self.ports.model_rollout}"

    @property
    def rollout_model_sub(self) -> str:
        """Address for rollout to subscribe to model (collective RPC)."""
        return f"tcp://localhost:{self.ports.model_rollout}"

    @property
    def rollout_samples_pull(self) -> str:
        """Address for rollout to pull (incompletes) samples from Workers and rust backend."""
        return f"tcp://*:{self.ports.samples_rollout}"

    @property
    def rollout_samples_push(self) -> str:
        """Address for rollout Workers and rust backend to push (incompletes) samples to Sender."""
        return f"tcp://localhost:{self.ports.samples_rollout}"
