"""Batch server for coordinating data sampling and distribution."""

import atexit
import json
import logging
import multiprocessing as mp
import os
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import zmq

from plynder.core import setup_logging
from plynder.core.config.infrastructure import BufferConfig, LmdbConfig
from plynder.core.config.networking import NetworkingConfig
from plynder.core.config.training import BatchingConfig
from plynder.train.data.receiver import Receiver

logger = logging.getLogger(__name__)


class BatchServer:
    """Server coordinating Receiver and BatchSampler processes."""

    def __init__(
        self,
        networking: NetworkingConfig,
        buffers: BufferConfig,
        lmdb: LmdbConfig,
        batching: BatchingConfig,
        num_replicas: int,
        num_workers: int,
        lengths_shm_name: str = "msgbuf_lengths",
        meta_shm_name: str = "msgbuf_meta",
    ) -> None:
        self.num_workers = num_workers
        self.num_replicas = num_replicas
        self.lengths_shm_name = lengths_shm_name
        self.meta_shm_name = meta_shm_name

        self.receiver = Receiver(
            samples_pull_address=networking.samples_pull,
            db_path=lmdb.db_path,
            lengths_shm_name=lengths_shm_name,
            meta_shm_name=meta_shm_name,
            map_size=lmdb.map_size,
            max_readers=2 * num_replicas * num_workers,
            ring_record_capacity=buffers.ring_record_capacity,
            samples_buffer_size=buffers.samples_buffer,
        )

        self.batch_sampler = BatchSampler(
            networking=networking,
            buffers=buffers,
            batching=batching,
            num_replicas=num_replicas,
            num_workers=num_workers,
            lengths_shm_name=lengths_shm_name,
            meta_shm_name=meta_shm_name,
            N=self.receiver.get_max_slots(),
        )

    def start(self) -> None:
        if os.path.exists("/dev/shm/" + self.lengths_shm_name):
            os.remove("/dev/shm/" + self.lengths_shm_name)

        lengths_shm = SharedMemory(
            name=self.lengths_shm_name,
            create=True,
            size=self.receiver.get_max_slots() * 8,
        )
        atexit.register(lengths_shm.unlink)
        atexit.register(lengths_shm.close)

        if os.path.exists("/dev/shm/" + self.meta_shm_name):
            os.remove("/dev/shm/" + self.meta_shm_name)

        meta_shm = SharedMemory(name=self.meta_shm_name, create=True, size=1 * 8)
        meta = np.ndarray((1,), dtype=np.int64, buffer=meta_shm.buf)
        meta[0] = 0
        atexit.register(meta_shm.unlink)
        atexit.register(meta_shm.close)

        self.receiver.start()
        self.batch_sampler.start()

    def shutdown(self) -> None:
        self.receiver.terminate()
        self.batch_sampler.terminate()
        self.receiver.join()
        self.batch_sampler.join()


class BatchSampler(mp.Process):
    """Distributes batches to workers based on token lengths."""

    def __init__(
        self,
        networking: NetworkingConfig,
        buffers: BufferConfig,
        batching: BatchingConfig,
        num_replicas: int,
        num_workers: int,
        lengths_shm_name: str,
        meta_shm_name: str,
        N: int,
    ) -> None:
        super().__init__(daemon=True)
        self.networking = networking
        self.buffers = buffers
        self.batching = batching
        self.num_replicas = num_replicas
        self.num_workers = num_workers
        self.lengths_shm_name = lengths_shm_name
        self.meta_shm_name = meta_shm_name
        self.N = N

        self.local_batch_size = batching.total_batch_size // num_replicas
        self.buffer_size = buffers.samples_buffering_sampler
        self.microbatch_max_tokens = batching.microbatch_max_tokens

    def _build_microbatch_assignments(
        self, ring_lengths: np.ndarray, seq: int
    ) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], int]:
        """
        Benchmark gives ~5 ms for 16k buffer_size
        """
        slots = np.arange(seq - self.buffer_size, seq, dtype=np.dtype("<u4")) % self.N
        lengths = ring_lengths[slots]

        order = np.argsort(lengths)

        # Shape W,N,B => World, Number of batches, Batch_size
        lengths = (
            lengths[order]
            .reshape((self.num_replicas, -1), order="F")
            .reshape((self.num_replicas, -1, self.local_batch_size))
        )
        slots = (
            slots[order]
            .reshape((self.num_replicas, -1), order="F")
            .reshape((self.num_replicas, -1, self.local_batch_size))
        )

        num_replicas, num_batches, batch_size = lengths.shape

        perm = np.random.permutation(num_batches)
        lengths = lengths[:, perm, :]
        slots = slots[:, perm, :]

        microbatch_assignments = [[[] for _ in range(num_batches)] for _ in range(num_replicas)]
        microbatch_lengths = [[[] for _ in range(num_batches)] for _ in range(num_replicas)]
        last_cut = np.zeros((num_replicas, num_batches), dtype=np.int64)

        for b in range(batch_size):
            col = lengths[:, :, b]
            overflow = (col * (b + 1 - last_cut)) > self.microbatch_max_tokens  # shape (W,N,) bool
            # We must split if any rank can not fit the tokens cause DDP requires same nb of backward (so microbatch) between two sync
            overflow = np.any(overflow, axis=0)  # shape N bool
            if overflow.any():
                for n in np.nonzero(overflow)[0]:
                    for w in range(num_replicas):
                        microbatch_assignments[w][n].append(
                            np.ascontiguousarray(slots[w, n, last_cut[w, n] : b])
                        )
                        microbatch_lengths[w][n].append(
                            np.ascontiguousarray(lengths[w, n, last_cut[w, n] : b])
                        )
                        last_cut[w, n] = b

        # Last micro_batch + rebalance logic with penultimate micro_batch
        for w in range(num_replicas):
            for n in range(num_batches):
                if len(microbatch_assignments[w][n]) > 0:
                    last_microbatch_len = batch_size - last_cut[w, n]
                    second_to_last_microbatch_len = len(microbatch_assignments[w][n][-1])
                    swapped_nb = min(
                        self.microbatch_max_tokens // lengths[w, n, -1] - last_microbatch_len,
                        (second_to_last_microbatch_len - last_microbatch_len) // 2,
                    )
                else:
                    swapped_nb = 0

                if swapped_nb > 0:
                    microbatch_assignments[w][n][-1] = np.ascontiguousarray(
                        microbatch_assignments[w][n][-1][:-swapped_nb]
                    )
                    microbatch_lengths[w][n][-1] = np.ascontiguousarray(
                        microbatch_lengths[w][n][-1][:-swapped_nb]
                    )

                microbatch_assignments[w][n].append(
                    np.ascontiguousarray(slots[w, n, last_cut[w, n] - swapped_nb :])
                )
                microbatch_lengths[w][n].append(
                    np.ascontiguousarray(lengths[w, n, last_cut[w, n] - swapped_nb :])
                )

        return microbatch_assignments, microbatch_lengths, num_batches

    def _startup_shm(
        self,
    ) -> tuple[SharedMemory, np.ndarray, SharedMemory, np.ndarray]:
        lengths_shm = SharedMemory(name=self.lengths_shm_name, create=False)
        lengths = np.ndarray((self.N,), dtype=np.int64, buffer=lengths_shm.buf)

        meta_shm = SharedMemory(name=self.meta_shm_name, create=False)
        meta = np.ndarray((1,), dtype=np.int64, buffer=meta_shm.buf)  # 0 -> seq

        return lengths_shm, lengths, meta_shm, meta

    def _startup_zmq(
        self,
    ) -> tuple[
        zmq.Socket,
        zmq.Socket,
        dict[tuple[int, int], bytes],
        dict[bytes, tuple[int, int]],
    ]:
        self.ctx = zmq.Context()
        socket = self.ctx.socket(zmq.ROUTER)
        socket.bind(self.networking.microbatch_assignment_router)

        stats_socket = self.ctx.socket(zmq.PUB)
        stats_socket.connect(self.networking.stats_trainer_pub)

        identities = {}
        rank_wid = {}

        while len(identities) < self.num_replicas * self.num_workers:
            identity, raw = socket.recv_multipart()
            msg = json.loads(raw)
            identities[(msg["rank"], msg["worker_id"])] = identity
            rank_wid[identity] = (msg["rank"], msg["worker_id"])
            logger.info(f"[BatchSampler] rank {msg['rank']} worker {msg['worker_id']} ready")

        return socket, stats_socket, identities, rank_wid

    def run(self) -> None:
        setup_logging()
        lengths_shm, lengths, meta_shm, meta = self._startup_shm()
        socket, stats_socket, identities, rank_wid = self._startup_zmq()

        ack_stacks = {
            (rank, wid): [] for wid in range(self.num_workers) for rank in range(self.num_replicas)
        }
        workers_turn = dict.fromkeys(range(self.num_replicas), 0)

        logger.info("[BatchSampler] Waiting for first data")
        while int(meta[0]) < self.buffer_size:
            time.sleep(0.001)
        logger.info("[BatchSampler] Sending batches")

        last_seq = 0
        t_sampler_idle = None

        try:
            while True:
                seq = meta[0]
                (
                    microbatch_assignments,
                    microbatch_lengths,
                    num_batches,
                ) = self._build_microbatch_assignments(lengths, seq)

                # Send async all slots to workers
                for batch_idx in range(num_batches):
                    for rank in range(self.num_replicas):
                        for microbatch_idx, microbatch_slot in enumerate(
                            microbatch_assignments[rank][batch_idx]
                        ):
                            sync_step = (
                                b"\x01"
                                if microbatch_idx
                                == len(microbatch_assignments[rank][batch_idx]) - 1
                                else b"\x00"
                            )

                            worker_id = workers_turn[rank]
                            identity = identities[(rank, worker_id)]

                            socket.send_multipart(
                                [identity, memoryview(microbatch_slot), sync_step]
                            )
                            ack_stacks[rank, worker_id].append(None)

                            workers_turn[rank] = (worker_id + 1) % self.num_workers

                if t_sampler_idle is not None:
                    total_tokens = sum(
                        len(microbatch) * microbatch[-1]  # cause microbatch is len-sorted
                        for rank_batches in microbatch_lengths
                        for batch in rank_batches
                        for microbatch in batch
                    )

                    real_tokens = sum(
                        np.sum(microbatch)
                        for rank_batches in microbatch_lengths
                        for batch in rank_batches
                        for microbatch in batch
                    )

                    padding_ratio = 1 - real_tokens / total_tokens
                    data_usage = float(self.buffer_size / (seq - last_seq + 1e-8))

                    stats_socket.send_json(
                        {
                            "throughput_trainer": total_tokens
                            / (time.perf_counter() - t_sampler_idle),
                            "padding_ratio": padding_ratio,
                            "data_usage": data_usage,
                        }
                    )

                t_sampler_idle = time.perf_counter()
                last_seq = seq

                # Wait until at least one worker wants a new micro_batch
                while all(len(stack) > 0 for stack in ack_stacks.values()):
                    identity, ack = socket.recv_multipart()
                    assert ack == b"ACK"
                    ack_stacks[rank_wid[identity]].pop()
        finally:
            socket.close()
            stats_socket.close()
            self.ctx.term()

            lengths_shm.close()
            meta_shm.close()
