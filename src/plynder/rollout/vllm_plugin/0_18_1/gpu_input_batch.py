"""Patches vLLM's InputBatch to sync Rust sequence states on add/remove."""

from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from plynder.rollout.vllm_plugin import sequence_states


class PatchedInputBatch(InputBatch):
    """InputBatch subclass that syncs Rust sequence states on add/remove."""

    def add_request(self, request: "CachedRequestState") -> None:
        """Register the sequence with the Rust engine, then delegate to vLLM."""
        sequence_states.rs_state.add_sequence(
            request.req_id, request.prompt_token_ids + request.output_token_ids
        )
        return super().add_request(request)

    def remove_request(self, req_id: str) -> int | None:
        """Remove the sequence from the Rust engine, then delegate to vLLM."""
        sequence_states.rs_state.remove_sequence(req_id)
        return super().remove_request(req_id)
