"""LMDB-based dataset for training."""

import struct

import lmdb
import numpy as np
import pyarrow as pa
import zmq
from torch.utils.data import IterableDataset, get_worker_info

from plynder.train.data.receiver import RECORD_NS, SLOT_NS


class LMDBDataset(IterableDataset):
    """Iterable dataset reading from LMDB via ZeroMQ sampler."""

    def __init__(
        self,
        rank: int,
        db_path: str = "/dev/shm/msgbuf",
        db_map_size: int = 10_000_000_000,
        sampler_address: str = "tcp://localhost:5558",
    ) -> None:
        self.rank = rank
        self.db_path = db_path
        self.sampler_address = sampler_address
        self.db_map_size = db_map_size

    def _startup(self) -> tuple[lmdb.Environment, zmq.Socket]:
        env = lmdb.open(
            self.db_path,
            readonly=True,
            lock=False,  # readers never write the lock file
            readahead=False,  # data is hot in page cache; readahead wastes RAM
            meminit=False,
            map_size=self.db_map_size,
        )

        ctx = zmq.Context()

        socket = ctx.socket(zmq.DEALER)
        socket.connect(self.sampler_address)

        wid = get_worker_info().id if get_worker_info() else 0
        socket.send_json({"rank": self.rank, "worker_id": wid})

        return env, socket

    def _deserialize(self, record_rows: dict[int, tuple[pa.RecordBatch, list]]) -> list[dict]:
        data = []
        for rb, row_idx in record_rows.values():
            n = len(row_idx)
            if not n:
                continue

            chunk = rb.take(pa.array(row_idx, type=pa.int32()))

            # Scalars are cheap, so a plain pylist is fine
            end_openings = chunk.column("end_opening").to_pylist()
            start_indices = chunk.column("start_index").to_pylist()
            white_wins = chunk.column("white_wins").to_pylist()
            black_wins = chunk.column("black_wins").to_pylist()
            draws = chunk.column("draws").to_pylist()

            # token_ids : ListArray<int32>
            tok_col = chunk.column("token_ids")
            tok_vals = tok_col.values.to_numpy(zero_copy_only=False)  # flat int32
            tok_off = tok_col.offsets.to_numpy(zero_copy_only=True)  # int32 offsets

            # token_logprobs : ListArray<float32>
            lp_col = chunk.column("token_logprobs")
            lp_vals = lp_col.values.to_numpy(zero_copy_only=False)
            lp_off = lp_col.offsets.to_numpy(zero_copy_only=True)

            # allowed_tokens_flat : ListArray<int32>
            at_flat_col = chunk.column("allowed_tokens_flat")
            at_flat_vals = at_flat_col.values.to_numpy(zero_copy_only=False)
            at_flat_off = at_flat_col.offsets.to_numpy(zero_copy_only=True)

            # allowed_tokens_offsets : ListArray<int32>
            at_offsets_col = chunk.column("allowed_tokens_offsets")
            at_offsets_vals = at_offsets_col.values.to_numpy(zero_copy_only=False)
            at_offsets_off = at_offsets_col.offsets.to_numpy(zero_copy_only=True)

            for i in range(n):
                data.append(
                    {
                        "end_opening": end_openings[i],
                        "start_index": start_indices[i],
                        "white_wins": white_wins[i],
                        "black_wins": black_wins[i],
                        "draws": draws[i],
                        # Numpy views avoid allocation and copying
                        "token_ids": tok_vals[tok_off[i] : tok_off[i + 1]].copy(),
                        "token_logprobs": lp_vals[lp_off[i] : lp_off[i + 1]].copy(),
                        "at_flat": at_flat_vals[at_flat_off[i] : at_flat_off[i + 1]].copy(),
                        "at_offsets": at_offsets_vals[
                            at_offsets_off[i] : at_offsets_off[i + 1]
                        ].copy(),
                    }
                )

        return data

    def __iter__(self):
        env, socket = self._startup()

        while True:
            slots_raw, sync_raw = socket.recv_multipart()

            slots = np.frombuffer(slots_raw, dtype="<u4")
            sync_step = sync_raw == b"\x01"

            record_rows = {}
            with env.begin(write=False, buffers=True) as txn:
                for slot in slots:
                    record_idx, row_idx = struct.unpack("<II", txn.get(SLOT_NS + slot))

                    if record_idx not in record_rows:
                        raw = txn.get(RECORD_NS + struct.pack("<I", record_idx))
                        record_rows[record_idx] = (
                            pa.ipc.open_stream(pa.py_buffer(raw)).read_next_batch(),
                            [],
                        )

                    record_rows[record_idx][1].append(row_idx)

            res = self._deserialize(record_rows), sync_step

            yield res

            socket.send(b"ACK")
