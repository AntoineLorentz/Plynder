"""Patches vLLM's GPU Worker to support shared-memory model hot-swapping."""

import io
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

import torch
from transformers import AutoModelForCausalLM

# ruff: noqa: I001
from .gpu_model_runner import PatchedGPUModelRunner
import vllm.v1.worker.gpu_model_runner  # noqa: E402

vllm.v1.worker.gpu_model_runner.GPUModelRunner = PatchedGPUModelRunner

from vllm.v1.worker.gpu_worker import Worker  # noqa: E402


class PatchedWorker(Worker):
    """vLLM Worker subclass with shared-memory model hot-swapping."""

    def load_model_from_path(self, model_path: str, dtype_name: str) -> None:
        """Load model weights from a file path into the vLLM engine."""
        dtype = getattr(torch, dtype_name)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        self.model_runner.model.load_weights(weights=model.named_parameters())

    def load_model_from_shm(self, shm_name: str, size: int) -> None:
        """Load model weights from a shared-memory segment into the vLLM engine."""
        shm = SharedMemory(name=shm_name)
        # Unregister: this process is a consumer, not the owner.
        # Without this, each worker's resource tracker will try to unlink
        # the segment at exit, even though ModelReceiver already did it.
        resource_tracker.unregister(f"/{shm_name}", "shared_memory")
        try:
            buf = io.BytesIO(bytes(shm.buf[:size]))
            state_dict = torch.load(buf, map_location="cpu")
            torch.cuda.synchronize()
            self.model_runner.model.load_weights(weights=state_dict.items())
            torch.cuda.synchronize()
        finally:
            shm.close()
