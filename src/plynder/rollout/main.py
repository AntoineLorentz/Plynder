"""Rollout entry point: starts the model receiver, sender, and vLLM workers."""

import logging
import os
import signal
import sys
import time
import warnings

import torch
import zmq

from plynder.core import setup_logging
from plynder.core.config import Config
from plynder.rollout.async_worker import Worker
from plynder.rollout.model_receiver import ModelReceiver
from plynder.rollout.sender import Sender

# vLLM emits noisy UserWarnings at engine startup; silence them here.
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


def global_main(cfg: Config) -> None:
    dp_size = torch.cuda.device_count()

    model_receiver = ModelReceiver(
        model_sub_address=cfg.networking.model_sub,
        model_sub_topic=cfg.networking.rollout_topic,
        rollout_model_pub_address=cfg.networking.rollout_model_pub,
    )
    model_receiver.start()

    sender = Sender(
        rollout_samples_pull_address=cfg.networking.rollout_samples_pull,
        samples_push_address=cfg.networking.samples_push,
        stats_rollout_pub_address=cfg.networking.stats_rollout_pub,
        data_timeout=cfg.rollout.sampling.data_timeout,
        samples_buffer=cfg.buffers.samples_buffer,
    )
    sender.start()

    workers: list[Worker] = []
    for global_engine_id in range(dp_size * cfg.rollout.vllm.num_engines_per_gpu):
        rollout_worker = Worker(
            rank=global_engine_id // cfg.rollout.vllm.num_engines_per_gpu,
            global_engine_id=global_engine_id,
            config_path=cfg.paths.model_config,
            wait_first_model=cfg.networking.synchronize,
            rollout_model_sub_address=cfg.networking.rollout_model_sub,
            rollout_samples_push_address=cfg.networking.rollout_samples_push,
            openings_jsonl_path=cfg.paths.openings_jsonl,
            rollout_config=cfg.rollout,
        )
        rollout_worker.start()
        workers.append(rollout_worker)

    def shutdown(_signum, _frame):
        logger.info("\nShutdown signal received, stopping all processes...")
        for w in workers:
            os.kill(w.pid, signal.SIGINT)

        # Join with timeout to avoid hanging
        for w in workers:
            w.join(timeout=30)

        sender.terminate()
        model_receiver.terminate()
        sender.join(timeout=5)
        model_receiver.join(timeout=5)

        logger.info("All processes stopped.")

        control_dealer.close()
        ctx.term()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    time.sleep(5)  # Give time for workers to start and connect to the model receiver
    ctx = zmq.Context()
    control_dealer = ctx.socket(zmq.DEALER)
    control_dealer.connect(cfg.networking.control_dealer)
    control_dealer.send(cfg.networking.rollout_topic.encode())
    logger.info("Rollout started, topic sent to control dealer.")

    # Waiting for shutdown
    _ = control_dealer.recv()
    shutdown(None, None)


if __name__ == "__main__":
    from datetime import datetime

    import hydra
    from omegaconf import DictConfig

    from plynder.core.config.hydra import from_hydra_config

    @hydra.main(
        version_base=None,
        config_path="../../../configs",
        config_name="config",
    )
    def rollout_app(cfg: DictConfig) -> None:
        """Main rollout entry point with Hydra."""
        config = from_hydra_config(cfg)

        # Setup logging
        if config.log_file is not None:
            config.log_file = (
                config.log_file + "_rollout_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.txt")
            )

        setup_logging(config.log_file)
        global_main(config)

    rollout_app()
