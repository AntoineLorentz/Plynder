"""Receiver process for ingesting data into LMDB."""

import logging
import multiprocessing as mp
import struct
from multiprocessing.shared_memory import SharedMemory

import lmdb
import numpy as np
import zmq

from plynder.core import setup_logging

logger = logging.getLogger(__name__)

RECORD_NS = b"\x00"  # namespace prefix for raw record keys
SLOT_NS = b"\x01"  # namespace prefix for slot→(record_idx, row) keys


class Receiver(mp.Process):
    """Receive data via ZeroMQ and store in LMDB."""

    def __init__(
        self,
        samples_pull_address: str,
        db_path: str,
        map_size: int,
        max_readers: int,
        ring_record_capacity: int,
        samples_buffer_size: int,
        lengths_shm_name: str,
        meta_shm_name: str,
    ) -> None:
        super().__init__(daemon=True)
        self.samples_pull_address = samples_pull_address
        self.db_path = db_path
        self.map_size = map_size
        self.max_readers = max_readers
        self.ring_record_capacity = ring_record_capacity
        self.samples_buffer_size = samples_buffer_size
        self.lengths_shm_name = lengths_shm_name
        self.meta_shm_name = meta_shm_name

        self.max_slots = ring_record_capacity * samples_buffer_size

    def get_max_slots(self) -> int:
        return self.max_slots

    def run(self) -> None:
        setup_logging()

        ctx = zmq.Context()

        socket = ctx.socket(zmq.PULL)
        socket.bind(self.samples_pull_address)
        logger.info(f"[Receiver] Binded to {self.samples_pull_address}")

        env = lmdb.open(
            self.db_path,
            max_readers=self.max_readers,
            writemap=True,
            map_async=True,
            map_size=self.map_size,
        )
        logger.info(f"[Receiver] Opened lmdb at {self.db_path} with map_size {self.map_size}")

        lengths_shm = SharedMemory(name=self.lengths_shm_name, create=False)
        lengths = np.ndarray((self.max_slots,), dtype=np.int64, buffer=lengths_shm.buf)

        meta_shm = SharedMemory(name=self.meta_shm_name, create=False)
        meta = np.ndarray((1,), dtype=np.int64, buffer=meta_shm.buf)

        seq = 0

        try:
            while True:
                record_raw, lengths_raw = socket.recv_multipart(copy=False)

                record_idx = (seq // self.samples_buffer_size) % self.ring_record_capacity
                record_key = RECORD_NS + struct.pack("<I", record_idx)  # Little endian, uint32

                slots = np.arange(seq, seq + self.samples_buffer_size, dtype="<u4") % self.max_slots

                with env.begin(write=True) as txn:
                    txn.put(record_key, bytes(record_raw))

                    #### For every row, map  slot → (record_idx, row_index)
                    #    Reader step-1: get slot  → 8-byte value
                    #    Reader step-2: get record → raw IPC, then slice row
                    for i, slot in enumerate(slots):
                        slot_key = SLOT_NS + slot
                        value = struct.pack("<II", record_idx, i)
                        txn.put(slot_key, value)

                lengths_array = np.frombuffer(bytes(lengths_raw), dtype=np.int64)
                lengths[slots] = lengths_array

                seq += len(lengths_array)
                meta[0] += len(lengths_array)
        finally:
            socket.close()
            ctx.term()
            env.close()
            lengths_shm.close()
            meta_shm.close()
