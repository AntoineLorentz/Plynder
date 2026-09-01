"""Evaluation configuration."""

from dataclasses import dataclass


@dataclass
class EvalConfig:
    """Evaluation configuration for Stockfish matches."""

    temperature: float = 1e-4
    """Temperature used when sampling the model to evaluate."""

    start_ranking: float = 0
    """Starting ranking for the model."""

    num_matches: int = 512
    """Number of matches played per model evaluation."""

    num_workers: int = 8
    """Number of concurrent workers playing matches."""

    concurrent_per_worker: int = 8
    """Number of concurrent matches played by each worker."""
