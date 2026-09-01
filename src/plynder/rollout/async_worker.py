"""Rollout worker process: runs a vLLM AsyncLLM engine and generates game trajectories."""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import warnings
from itertools import count

import numpy as np
import transformers
import vllm
import zmq
import zmq.asyncio
from transformers import AutoConfig, AutoModelForCausalLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from plynder.core.config import RolloutConfig
from plynder.rollout.async_group import Group

logger = logging.getLogger(__name__)


def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"{message} ({category.__name__}: {filename}:{lineno})")


warnings.showwarning = custom_warning_handler
warnings.filterwarnings("ignore", category=UserWarning)


class Worker(mp.Process):
    """One vLLM engine per process; spawns ``Group`` objects for group sampling."""

    def __init__(
        self,
        rank: int,
        global_engine_id: int,
        config_path: str,
        wait_first_model: bool,
        rollout_model_sub_address: str,
        rollout_samples_push_address: str,
        openings_jsonl_path: str,
        rollout_config: RolloutConfig,
    ) -> None:
        super().__init__(daemon=False)
        self.rank = rank
        self.global_engine_id = global_engine_id
        self.config_path = config_path
        self.wait_first_model = wait_first_model
        self.rollout_config = rollout_config

        self.rollout_model_sub_address = rollout_model_sub_address
        self.rollout_samples_push_address = rollout_samples_push_address
        self.openings_jsonl_path = openings_jsonl_path

    def _startup(self) -> None:
        """Initialize vLLM engine, ZMQ sockets, and load openings."""
        transformers.utils.logging.disable_progress_bar()
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.rank)
        os.environ["PLYNDER_ROLLOUT_ADDRESS"] = str(self.rollout_samples_push_address)
        os.environ["PLYNDER_GLOBAL_ENGINE_ID"] = str(self.global_engine_id)

        self.ctx = zmq.asyncio.Context()

        self.model_socket = self.ctx.socket(zmq.SUB)
        self.model_socket.connect(self.rollout_model_sub_address)
        self.model_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.samples_socket = self.ctx.socket(zmq.PUSH)
        self.samples_socket.connect(self.rollout_samples_push_address)

        vllm_version = vllm.__version__.replace(".", "_")

        config = AutoConfig.from_pretrained(self.config_path)
        os.environ["PLYNDER_VOCAB_SIZE"] = str(config.vocab_size)
        model = AutoModelForCausalLM.from_config(
            config
        )  # Random model, will be overwritten by the trainer

        self.tmp_model_dir = tempfile.mkdtemp(prefix="vllm_model_", dir="/tmp")
        model.save_pretrained(self.tmp_model_dir)

        engine_args = AsyncEngineArgs(
            model=self.tmp_model_dir,
            skip_tokenizer_init=True,
            gpu_memory_utilization=0.01,
            kv_cache_memory_bytes=self.rollout_config.vllm.kv_cache_memory_bytes,
            performance_mode="throughput",
            worker_cls=f"plynder.rollout.vllm_plugin.{vllm_version}.gpu_worker.PatchedWorker",
            scheduling_policy="priority",
            use_tqdm_on_load=False,
        )

        self.engine = AsyncLLM.from_engine_args(engine_args)

        with open(self.openings_jsonl_path) as f:
            self.openings = [json.loads(line) for line in f]

        self.openings_patience = np.zeros(len(self.openings), dtype=np.uint32)

        self.received_first_model = asyncio.Event()

    async def _sample_groups(self, queue: asyncio.Queue) -> None:
        """Consume groups from the queue and sample them concurrently."""
        while True:
            try:
                group = await queue.get()
                discarded = await group.sample()
                self.openings_patience[group.opening_id] += int(discarded)
            except Exception:
                logging.error("Exception caught in Worker._sample_groups", exc_info=True)

    async def _main_loop(self) -> None:
        """Start the weight-update task, wait for the first model, then pump groups."""
        asyncio.create_task(self.update_weight())

        if self.wait_first_model:
            await self.received_first_model.wait()

        queue = asyncio.Queue(maxsize=self.rollout_config.vllm.max_concurrent_groups)

        for _ in range(self.rollout_config.vllm.max_concurrent_groups):
            asyncio.create_task(self._sample_groups(queue))

        for group_id in count():
            new_group = Group(
                self.engine,
                self.global_engine_id,
                group_id,
                self.openings,
                self.openings_patience,
                self.samples_socket,
                self.rollout_config.sampling,
            )
            await queue.put(new_group)

    async def update_weight(self) -> None:
        """Listen for model updates from the trainer and hot-swap weights via collective RPC."""
        while True:
            new_model_info = await self.model_socket.recv_json()
            await self.engine.collective_rpc("load_model_from_shm", kwargs=new_model_info)

            self.received_first_model.set()
            if self.tmp_model_dir:
                shutil.rmtree(self.tmp_model_dir, ignore_errors=True)
                self.tmp_model_dir = None

    async def shutdown_async(self) -> None:
        """Shut down the vLLM engine and close all ZMQ sockets."""
        if hasattr(self, "engine") and self.engine is not None:
            self.engine.shutdown()

        for sock in ("model_socket", "samples_socket"):
            if hasattr(self, sock):
                getattr(self, sock).close()

        if hasattr(self, "ctx"):
            self.ctx.term()

    def run(self) -> None:
        """Process entry point: startup, run the async main loop, then clean up."""
        self._startup()
        try:
            asyncio.run(self._main_loop())
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            # Run shutdown in a new event loop to ensure cleanup
            asyncio.run(self.shutdown_async())
