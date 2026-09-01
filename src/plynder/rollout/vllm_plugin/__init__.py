"""Global singleton bridging the Rust legal-move engine and the vLLM plugin.

The ``sequence_states`` singleton is constructed at import time using environment
variables set by the rollout worker (``PLYNDER_VOCAB_SIZE``, ``PLYNDER_RS_DEVICE``,
etc.).  The vLLM plugin patches in ``0_18_1/`` read from this singleton to
apply legal-move masks during sampling.
"""

import os

# Import torch first: the plynder_rs extension links against libtorch_python.so
# (no RUNPATH), so the lib must be loaded into the process before the
# extension is imported.
# ruff: noqa: I001
import torch  # noqa: F401

from plynder_rs import SequenceStatesAsync, TerminalTokens

from plynder.core import SpecialTokens


class SequenceStates:
    """Wraps the Rust ``SequenceStatesAsync`` with plugin-specific state."""

    def __init__(self) -> None:
        vocab_size = int(os.environ.get("PLYNDER_VOCAB_SIZE", "1972"))

        self.rs_state = SequenceStatesAsync(
            vocab_size=vocab_size,
            device=os.environ.get("PLYNDER_RS_DEVICE", "cuda:0"),
            terminal_tokens=TerminalTokens(
                draw_id=SpecialTokens.DRAW,
                terminal_token_win_0=SpecialTokens.WHITE_WIN,
                terminal_token_win_1=SpecialTokens.BLACK_WIN,
            ),
            rollout_address=os.environ.get("PLYNDER_ROLLOUT_ADDRESS", None),
            global_engine_id=int(os.environ.get("PLYNDER_GLOBAL_ENGINE_ID", "0")),
        )

        self.dummy_sampler_run = False
        self.logprobs_scaled = None
        self.is_real_batch = False


sequence_states = SequenceStates()
