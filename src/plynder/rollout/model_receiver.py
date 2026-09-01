"""Model receiver process: subscribes to trainer model updates and writes them to shared memory."""

import atexit
import multiprocessing as mp
import time
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

import zmq


class ModelReceiver(mp.Process):
    """Receives model weights via ZMQ SUB, writes them to a 3-slot shared-memory pool."""

    def __init__(
        self,
        model_sub_address: str,
        model_sub_topic: str,
        rollout_model_pub_address: str,
        pool_size: int = 3,
    ) -> None:
        super().__init__(daemon=True)
        self.model_sub_address = model_sub_address
        self.model_sub_topic = model_sub_topic
        self.rollout_model_pub_address = rollout_model_pub_address
        self.base_shm_name = "plynder_model_shm"
        self.pool_size = pool_size
        self.current_slot = 0

    def _startup(self) -> tuple[zmq.Socket, zmq.Socket]:
        """Create the ZMQ SUB (trainer) and PUB (rollout workers) sockets."""
        ctx = zmq.Context()
        self._ctx = ctx

        sub_socket = ctx.socket(zmq.SUB)
        sub_socket.connect(self.model_sub_address)
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, self.model_sub_topic)
        time.sleep(1)  # slow-joiner guard

        pub_socket = ctx.socket(zmq.PUB)
        pub_socket.bind(self.rollout_model_pub_address)
        time.sleep(1)

        return sub_socket, pub_socket

    def _cleanup(self) -> None:
        """Terminate ZMQ context and unlink any leftover shared-memory segments."""
        if hasattr(self, "_ctx"):
            self._ctx.term()

        for slot in range(self.pool_size):
            shm_name = f"{self.base_shm_name}_{slot}"
            try:
                existing_shm = SharedMemory(name=shm_name)
                resource_tracker.unregister(f"/{shm_name}", "shared_memory")
                existing_shm.close()
                existing_shm.unlink()
            except (FileNotFoundError, OSError):
                pass

    def run(self) -> None:
        """Main loop: receive model weights, write to shared memory, publish shm metadata."""
        atexit.register(self._cleanup)
        sub_socket, pub_socket = self._startup()

        while True:
            topic, step, msg = sub_socket.recv_multipart()

            # Compute slot name
            slot = self.current_slot
            shm_name = f"{self.base_shm_name}_{slot}"

            # Cleanup previous shm in this slot (if exists)
            try:
                existing_shm = SharedMemory(name=shm_name)
                existing_shm.close()
                existing_shm.unlink()
            except FileNotFoundError:
                pass

            shm = SharedMemory(name=shm_name, create=True, size=len(msg))
            # We manage this segment's lifetime manually in _cleanup,
            # so opt out of automatic tracking.
            resource_tracker.unregister(f"/{shm_name}", "shared_memory")
            shm.buf[: len(msg)] = msg
            shm.close()

            data = {"shm_name": shm_name, "size": len(msg)}
            pub_socket.send_json(data)

            self.current_slot = (self.current_slot + 1) % self.pool_size
