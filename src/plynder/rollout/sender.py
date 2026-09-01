"""Sender process: aggregates rollout trajectories and pushes them to the trainer.

Runs as a subprocess alongside the vLLM workers.  Receives partial data from
both the vLLM ``Group`` objects (move tokens) and the Rust backend (logprobs,
allowed-token masks), assembles them into complete ``TrajectoryData`` records,
and serializes batches via PyArrow IPC before pushing to the trainer's ZMQ
PULL socket.
"""

import atexit
import multiprocessing as mp
import time
from collections import defaultdict
from dataclasses import dataclass, field

import msgpack
import numpy as np
import pyarrow as pa
import zmq

from plynder.core import SpecialTokens
from plynder.rollout.stats_pub import StatsPub


@dataclass
class TrajectoryData:
    """A single complete game trajectory ready for training."""

    start_index: int
    end_opening: int
    token_ids: np.ndarray
    token_logprobs: np.ndarray
    allowed_tokens_flat: np.ndarray
    allowed_tokens_offsets: np.ndarray
    white_wins: int
    black_wins: int
    draws: int


@dataclass
class OpeningGroupData:
    """Accumulates partial data for a group of continuations until all are complete."""

    creation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    start_index: int = -1
    end_opening: int = -1
    token_ids: list = field(default_factory=list)
    token_logprobs: dict = field(default_factory=dict)
    allowed_tokens_flat: dict = field(default_factory=dict)
    allowed_tokens_offsets: dict = field(default_factory=dict)

    def is_complete(self) -> bool:
        n = len(self.token_ids)
        set_n = set(range(n))
        return (
            n > 0
            and set(self.token_logprobs.keys()) == set_n
            and set(self.allowed_tokens_flat.keys()) == set_n
            and set(self.allowed_tokens_offsets.keys()) == set_n
            and self.start_index >= 0
            and self.end_opening >= 0
        )

    def to_trajectory_data_list(self) -> list[TrajectoryData]:
        trajectory_data_list = []

        if all(tok[-1] == self.token_ids[0][-1] for tok in self.token_ids):
            # Group discarded as all outcomes are identical
            return trajectory_data_list

        white_wins = sum(tok[-1] == SpecialTokens.WHITE_WIN for tok in self.token_ids)
        black_wins = sum(tok[-1] == SpecialTokens.BLACK_WIN for tok in self.token_ids)
        draws = sum(tok[-1] == SpecialTokens.DRAW for tok in self.token_ids)

        start_offset = self.start_index - self.end_opening

        for i in range(len(self.token_ids)):
            if i > 0:
                _token_logprobs = np.concatenate(
                    [self.token_logprobs[0][:start_offset], self.token_logprobs[i]]
                )

                offset_start = self.allowed_tokens_offsets[0][start_offset]
                _allowed_tokens_flat = np.concatenate(
                    [
                        self.allowed_tokens_flat[0][:offset_start],
                        self.allowed_tokens_flat[i],
                    ]
                )
                _allowed_tokens_offsets = np.concatenate(
                    [
                        self.allowed_tokens_offsets[0][:start_offset],
                        self.allowed_tokens_offsets[0][start_offset]
                        + self.allowed_tokens_offsets[i],
                    ]
                )
            else:
                _token_logprobs = self.token_logprobs[0]
                _allowed_tokens_flat = self.allowed_tokens_flat[0]
                _allowed_tokens_offsets = self.allowed_tokens_offsets[0]

            trajectory_data_list.append(
                TrajectoryData(
                    start_index=self.start_index,
                    end_opening=self.end_opening,
                    token_ids=self.token_ids[i],
                    token_logprobs=_token_logprobs,
                    allowed_tokens_flat=_allowed_tokens_flat,
                    allowed_tokens_offsets=_allowed_tokens_offsets,
                    white_wins=white_wins,
                    black_wins=black_wins,
                    draws=draws,
                )
            )

        return trajectory_data_list


class Sender(mp.Process):
    """Background process that aggregates and forwards rollout trajectories."""

    def __init__(
        self,
        rollout_samples_pull_address: str,
        samples_push_address: str,
        stats_rollout_pub_address: str,
        data_timeout: int,
        samples_buffer: int,
    ) -> None:
        super().__init__(daemon=True)
        self.rollout_samples_pull_address = rollout_samples_pull_address
        self.samples_push_address = samples_push_address
        self.stats_rollout_pub_address = stats_rollout_pub_address
        self.data_timeout = data_timeout
        self.samples_buffer = samples_buffer

        self.ongoing_data: dict[int, dict[int, OpeningGroupData]] = defaultdict(
            lambda: defaultdict(OpeningGroupData)
        )  # global_engine_id x group_id -> group_data
        self.ready_data: list[TrajectoryData] = []

        self.timeout_check_interval = 5
        self.last_timeout_check = time.time()

    def _serialize(
        self, trajectory_data_list: list[TrajectoryData]
    ) -> tuple[pa.Buffer, np.ndarray]:
        """Serialize a list of trajectories into a PyArrow IPC buffer and a lengths array."""
        lengths = np.array([len(data.token_ids) for data in trajectory_data_list], dtype=np.int64)

        rb = pa.record_batch(
            [
                pa.array([data.start_index for data in trajectory_data_list], type=pa.int32()),
                pa.array(
                    [data.end_opening for data in trajectory_data_list],
                    type=pa.int32(),
                ),
                pa.array([data.white_wins for data in trajectory_data_list], type=pa.int32()),
                pa.array([data.black_wins for data in trajectory_data_list], type=pa.int32()),
                pa.array([data.draws for data in trajectory_data_list], type=pa.int32()),
                pa.array(
                    [data.token_ids for data in trajectory_data_list],
                    type=pa.list_(pa.int32()),
                ),
                pa.array(
                    [data.token_logprobs for data in trajectory_data_list],
                    type=pa.list_(pa.float32()),
                ),
                pa.array(
                    [data.allowed_tokens_flat for data in trajectory_data_list],
                    type=pa.list_(pa.int32()),
                ),
                pa.array(
                    [data.allowed_tokens_offsets for data in trajectory_data_list],
                    type=pa.list_(pa.int32()),
                ),
            ],
            names=[
                "start_index",
                "end_opening",
                "white_wins",
                "black_wins",
                "draws",
                "token_ids",
                "token_logprobs",
                "allowed_tokens_flat",
                "allowed_tokens_offsets",
            ],
        )

        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, rb.schema) as writer:
            writer.write_batch(rb)

        buf = sink.getvalue()

        return buf, lengths

    def _startup(self) -> tuple[zmq.Socket, zmq.Socket, StatsPub]:
        """Create ZMQ sockets and the StatsPub instance."""
        ctx = zmq.Context()
        self._ctx = ctx
        pull_socket = ctx.socket(zmq.PULL)
        pull_socket.bind(self.rollout_samples_pull_address)

        push_socket = ctx.socket(zmq.PUSH)
        push_socket.connect(self.samples_push_address)

        stats_pub = StatsPub(ctx, self.stats_rollout_pub_address)

        return pull_socket, push_socket, stats_pub

    def _cleanup(self) -> None:
        """Terminate the ZMQ context if it was created."""
        if hasattr(self, "_ctx"):
            self._ctx.term()

    def run(self) -> None:
        """Main loop: receive partial data, assemble trajectories, push to trainer."""
        atexit.register(self._cleanup)
        pull_socket, push_socket, stats_pub = self._startup()

        while True:
            # pulling data
            info_raw, *bufs = pull_socket.recv_multipart()

            info = msgpack.unpackb(info_raw)
            group_data = self.ongoing_data[info["global_engine_id"]][info["group_id"]]
            group_data.last_update = time.time()

            if info["type"] == "group":
                group_data.start_index = info["start_index"]
                group_data.end_opening = info["end_opening"]
                group_data.token_ids = [np.frombuffer(buf, dtype=np.int32) for buf in bufs]

            if info["type"] == "rust":
                traj_id = info["traj_id"]
                group_data.token_logprobs[traj_id] = np.frombuffer(bufs[0], dtype=np.float32)
                group_data.allowed_tokens_flat[traj_id] = np.frombuffer(bufs[1], dtype=np.int32)
                group_data.allowed_tokens_offsets[traj_id] = np.frombuffer(bufs[2], dtype=np.int32)

            if group_data.is_complete():
                stats_pub.update(group_data)
                self.ready_data.extend(group_data.to_trajectory_data_list())
                self.ongoing_data[info["global_engine_id"]].pop(info["group_id"])

            if len(self.ready_data) >= self.samples_buffer:
                push_socket.send_multipart(self._serialize(self.ready_data[: self.samples_buffer]))
                self.ready_data = self.ready_data[self.samples_buffer :]

            if time.time() - self.last_timeout_check > self.timeout_check_interval:
                for groups in self.ongoing_data.values():
                    for group_id, group_data in list(groups.items()):
                        if time.time() - group_data.last_update > self.data_timeout:
                            groups.pop(group_id)
                            stats_pub.update_timeout(group_data)

                self.last_timeout_check = time.time()
