"""Sampling group: generates continuations from an opening for rollout."""

from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING, Any

import msgpack
import numpy as np

if TYPE_CHECKING:
    import zmq.asyncio
from vllm import SamplingParams
from vllm.inputs import TokenInputs
from vllm.sampling_params import RequestOutputKind

from plynder.core import SpecialTokens
from plynder.core.config.rollout import SamplingConfig

warnings.filterwarnings("ignore", category=RuntimeWarning)

#: Depth of the opening prefix (must match scripts/generate_openings.py depth_max)
OPENING_DEPTH = 5

class Group:
    """Wraps a single sampling group: picks an opening, generates ``group_size`` continuations."""

    def __init__(
        self,
        engine: Any,
        global_engine_id: int,
        group_id: int,
        openings: list[list[int]],
        openings_patience: np.ndarray,
        samples_socket: zmq.asyncio.Socket,
        sampling_config: SamplingConfig,
    ) -> None:
        self.engine = engine
        self.global_engine_id = global_engine_id
        self.group_id = group_id
        self.openings = openings
        self.openings_patience = openings_patience
        self.sampling_config = sampling_config
        self.samples_socket = samples_socket

        self.end_opening = OPENING_DEPTH

        self.token_ids = []

        self.sampling_params = SamplingParams(
            temperature=sampling_config.temperature,
            max_tokens=sampling_config.max_tokens,
            stop_token_ids=[
                SpecialTokens.DRAW,
                SpecialTokens.WHITE_WIN,
                SpecialTokens.BLACK_WIN,
            ],
            output_kind=RequestOutputKind.FINAL_ONLY,
            detokenize=False,
        )

    async def sample(self) -> bool:
        """Sample one group: pick an opening, generate continuations, send to Sender.

        Returns ``True`` if the group was discarded (all continuations had the
        same outcome), ``False`` otherwise.
        """
        request_id = f"{self.group_id}-0"

        idx = np.arange(len(self.openings), dtype=np.int32)

        living_mask = self.openings_patience < self.sampling_config.opening_abandonment_threshold

        dead_opening_idx = idx[~living_mask]
        relived_opening_idx = dead_opening_idx[
            np.random.rand(len(dead_opening_idx))
            < self.sampling_config.probability_retry_discarded_opening
        ]
        living_opening_idx = np.concat([idx[living_mask], relived_opening_idx])

        self.opening_id = int(np.random.choice(living_opening_idx))

        prompt = TokenInputs(prompt_token_ids=self.openings[self.opening_id])
        collector = await self.engine.add_request(
            params=self.sampling_params,
            request_id=request_id,
            prompt=prompt,
            priority=self.group_id,
        )

        output = await collector.get()

        token_ids = output.prompt_token_ids + list(output.outputs[0].token_ids)
        self.token_ids.append(np.array(token_ids, dtype=np.int32))

        if len(token_ids) - 2 <= self.end_opening:
            # Bad game, we don't send data to socket, a timeout will clean rust data
            discarded_group = True
            return discarded_group

        # Sample the split point (start of the continuation shared by the group)
        self.start_index = random.randint(
            self.end_opening, len(token_ids) - 2
        )  # -2 because of EOS token

        # Adding all groups requests
        question_prompt = TokenInputs(prompt_token_ids=token_ids[: self.start_index])
        collectors = [
            await self.engine.add_request(
                request_id=f"{self.group_id}-{i}",
                prompt=question_prompt,
                params=self.sampling_params,
                priority=self.group_id,
            )
            for i in range(1, self.sampling_config.group_size)
        ]

        for collector in collectors:
            output = await collector.get()
            token_ids = output.prompt_token_ids + list(output.outputs[0].token_ids)
            self.token_ids.append(np.array(token_ids, dtype=np.int32))

        await self._send_data()

        discarded_group = all(tok[-1] == self.token_ids[0][-1] for tok in self.token_ids)
        return discarded_group

    async def _send_data(self) -> None:
        """Send the group's token IDs to the Sender via msgpack-over-ZMQ."""
        info = {
            "type": "group",
            "global_engine_id": self.global_engine_id,
            "group_id": self.group_id,
            "start_index": self.start_index,
            "end_opening": self.end_opening,
        }

        await self.samples_socket.send_multipart([msgpack.dumps(info), *self.token_ids])
