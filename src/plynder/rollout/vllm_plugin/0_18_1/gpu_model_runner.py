"""Patches vLLM's GPUModelRunner to spawn Rust valid-token mask computation."""

from typing import TYPE_CHECKING

import torch

from plynder.rollout.vllm_plugin import sequence_states

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


import vllm.v1.worker.gpu_input_batch  # noqa: E402

from .gpu_input_batch import PatchedInputBatch

vllm.v1.worker.gpu_input_batch.InputBatch = PatchedInputBatch

import vllm.v1.sample.sampler  # noqa: E402

from .sampler import PatchedSampler  # noqa: E402

vllm.v1.sample.sampler.Sampler = PatchedSampler

from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402


class PatchedGPUModelRunner(GPUModelRunner):
    """GPUModelRunner subclass that spawns Rust valid-token mask computation."""

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update input batch, then spawn Rust mask computation for active sequences."""
        res = super()._update_states(scheduler_output)

        sequence_states.rs_state.spawn_valid_tokens_mask(self.input_batch.req_ids)
        sequence_states.rs_state.spawn_send_sequence_data(list(scheduler_output.finished_req_ids))

        return res

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run a dummy sampling pass (profiling) that skips Rust mask application."""
        sequence_states.dummy_sampler_run = True
        return super()._dummy_sampler_run(hidden_states)
