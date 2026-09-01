"""vLLM 0.18.1 CPU worker patch (re-exports GPU patches for CPU mode)."""

import vllm.v1.worker.gpu_model_runner

from .gpu_model_runner import PatchedGPUModelRunner

vllm.v1.worker.gpu_model_runner.GPUModelRunner = PatchedGPUModelRunner

import vllm.v1.worker.cpu_worker  # noqa: E402

from .gpu_worker import PatchedWorker  # noqa: E402

vllm.v1.worker.cpu_worker.Worker = PatchedWorker


class PatchedCPUWorker(vllm.v1.worker.cpu_worker.CPUWorker):
    """CPU worker variant of PatchedWorker (uses the same model runner patches)."""

    pass
