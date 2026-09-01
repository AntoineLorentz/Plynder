"""Patches vLLM's Sampler to apply the Rust legal-move mask before sampling."""

import torch
from vllm.v1.sample.sampler import _SAMPLING_EPS, Sampler

from plynder.rollout.vllm_plugin import sequence_states


class PatchedSampler(Sampler):
    """Sampler subclass that applies the Rust legal-move mask before sampling."""

    def sample(self, *args, **kwargs):
        """Sample tokens, then apply the sampled moves to the Rust sequence states."""
        sampled, processed_logprobs = super().sample(*args, **kwargs)

        if not sequence_states.dummy_sampler_run:
            logprobs = sequence_states.logprobs_scaled.gather(-1, sampled[:, None])
            sequence_states.rs_state.apply_sampled_batch(
                sampled.tolist(), logprobs.flatten().tolist()
            )
        else:
            sequence_states.dummy_sampler_run = False

        return sampled, processed_logprobs

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        """Apply temperature scaling and the Rust legal-move mask to logits."""
        # Use in-place division to avoid creating a new tensor.
        # Avoid division by zero if there are greedy requests.
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)

        if not sequence_states.dummy_sampler_run:
            logits.masked_fill_(
                sequence_states.rs_state.join_get_valid_tokens_mask(), float("-inf")
            )
            sequence_states.logprobs_scaled = logits.log_softmax(dim=-1, dtype=torch.float32)

        return logits.div_(temp.unsqueeze(dim=1))
