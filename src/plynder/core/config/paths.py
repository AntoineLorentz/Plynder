"""Path configuration."""

from dataclasses import dataclass


@dataclass
class PathsConfig:
    """Filesystem paths configuration."""

    logs: str = "logs"
    """Directory for log files."""

    model_config: str = "model_configs/qwen3_50M_512_16"
    """Path to model config directory (for loading model architecture)."""

    openings_jsonl: str = "openings.jsonl"
    """Path to the list of openings considered for rollout and evaluation."""

    stockfish_path: str = "stockfish/stockfish-ubuntu-x86-64-avx2"
    """Path to the Stockfish binary."""

    rankings_json: str = "rankings.json"
    """Path to rankings JSON file mapping Stockfish temperatures to rankings."""

    tokenizer_path: str = "models/plynder_tokenizer"
    """Path to tokenizer."""

    checkpoints: str = "models/plynder"
    """Directory for model checkpoints."""

    mlflow_tracking_uri: str = "sqlite:///logs/mlflow.db"
    """MLflow tracking URI (SQL backend). Three slashes = relative path; use
    four slashes (sqlite:////abs/path) for an absolute path."""
