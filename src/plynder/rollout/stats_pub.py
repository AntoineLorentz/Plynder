"""Statistics publisher: accumulates rollout metrics and publishes them via ZMQ."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import zmq

from plynder.core import SpecialTokens

if TYPE_CHECKING:
    from plynder.rollout.sender import OpeningGroupData


class StatsPub:
    """Accumulates win rates, throughput, and latency stats; publishes periodically via ZMQ."""

    def __init__(
        self,
        ctx: zmq.Context,
        stats_rollout_pub_address: str,
        stats_sampling_frequency: int = 4,
        publish_interval: int = 20,
    ) -> None:
        self.stats_sampling_frequency = stats_sampling_frequency
        self.publish_interval = publish_interval

        self.socket = ctx.socket(zmq.PUB)
        self.socket.connect(stats_rollout_pub_address)

        self.num_samples = 0

        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.tokens_white_win = 0
        self.tokens_black_win = 0
        self.tokens_draw = 0
        self.tokens = 0
        self.kept = 0
        self.group_latencies = []
        self.last_publish = time.time()

        self.timeout = 0

    def update(self, group_data: OpeningGroupData) -> None:
        """Record stats from a completed group; publish if the interval has elapsed."""
        self.num_samples += 1

        if self.num_samples % self.stats_sampling_frequency == 0:
            for i, tok in enumerate(group_data.token_ids):
                # token_ids[1:] are continuations, so we need to add start_index
                self.tokens += len(tok) if i == 0 else len(tok) - group_data.start_index

                if tok[-1] == SpecialTokens.WHITE_WIN:
                    self.white_wins += 1
                    self.tokens_white_win += len(tok)
                elif tok[-1] == SpecialTokens.BLACK_WIN:
                    self.black_wins += 1
                    self.tokens_black_win += len(tok)
                elif tok[-1] == SpecialTokens.DRAW:
                    self.draws += 1
                    self.tokens_draw += len(tok)

            self.kept += (
                len(group_data.token_ids)
                if any(tok[-1] != group_data.token_ids[0][-1] for tok in group_data.token_ids)
                else 0
            )
            self.group_latencies.append(group_data.last_update - group_data.creation_time)

            if time.time() - self.last_publish > self.publish_interval:
                self.publish()
                self.num_samples = 0

    def update_timeout(self, group_data: OpeningGroupData) -> None:
        """Record a timed-out group."""
        self.timeout += 1

    def publish(self) -> None:
        """Compute aggregate stats, send them via ZMQ, and reset all accumulators."""
        num_sequences = max(self.white_wins + self.black_wins + self.draws, 1)

        data = {
            "white_wins": self.white_wins / num_sequences,
            "black_wins": self.black_wins / num_sequences,
            "draws": self.draws / num_sequences,
            "average_len_white_win": self.tokens_white_win / max(self.white_wins, 1),
            "average_len_black_win": self.tokens_black_win / max(self.black_wins, 1),
            "average_len_draw": self.tokens_draw / max(self.draws, 1),
            "kept_ratio": self.kept / num_sequences,
            "throughput_inference": self.tokens
            * self.stats_sampling_frequency
            / (time.time() - self.last_publish),
            "throughput_total": (self.tokens_white_win + self.tokens_black_win + self.tokens_draw)
            * self.stats_sampling_frequency
            / (time.time() - self.last_publish),
            "throughput_sequences": num_sequences
            * self.stats_sampling_frequency
            / (time.time() - self.last_publish),
            "p50_group_latency": np.percentile(self.group_latencies, 50)
            if self.group_latencies
            else 0,
            "p90_group_latency": np.percentile(self.group_latencies, 90)
            if self.group_latencies
            else 0,
            "p99_group_latency": np.percentile(self.group_latencies, 99)
            if self.group_latencies
            else 0,
            "timeout_per_sec": self.timeout / (time.time() - self.last_publish),
        }

        self.socket.send_json(data)

        # reset data
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.tokens_white_win = 0
        self.tokens_black_win = 0
        self.tokens_draw = 0
        self.tokens = 0
        self.kept = 0
        self.last_publish = time.time()
        self.group_latencies = []
        self.timeout = 0
