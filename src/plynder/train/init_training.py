"""Initialize training environment with Accelerate."""

import logging
import os

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)

from plynder.core.config import Config

logger = get_logger(__name__)


def init_training(cfg: Config) -> Accelerator:
    """Initialize the Accelerator for distributed training.

    Args:
        cfg: Training configuration.

    Returns:
        Configured Accelerator instance.
    """

    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.paths.checkpoints, logging_dir=cfg.paths.logs
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="mlflow",
        project_config=accelerator_project_config,
        gradient_accumulation_steps=1,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)

    # Handle the repository creation
    if accelerator.is_main_process and cfg.paths.checkpoints is not None:
        os.makedirs(cfg.paths.checkpoints, exist_ok=True)

    return accelerator  # type: ignore[return-value]
