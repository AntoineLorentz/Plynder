"""Statistics collector for aggregating training metrics."""

import statistics
import threading
import time
from typing import Any

import zmq


class StatsCollector(threading.Thread):
    """Background thread that collects stats via ZeroMQ subscription."""

    def __init__(self, stats_trainer_sub_address: str, ctx: zmq.Context) -> None:
        super().__init__(daemon=True)
        self.stats_trainer_sub_address = stats_trainer_sub_address
        self.ctx = ctx
        self.lock = threading.Lock()
        self.rollout_data: list[dict[str, Any]] = []
        self.eval_data: dict[str, Any] | None = None

    def get_evaluation(self) -> dict[str, float | int] | None:
        with self.lock:
            if self.eval_data:
                res = self.eval_data
                self.eval_data = None
                return res

    def get_rollout(self) -> dict[str, float]:
        with self.lock:
            aggregated = {}

            if len(self.rollout_data) == 0:
                return aggregated

            all_keys = set().union(*(d.keys() for d in self.rollout_data))

            for key in all_keys:
                vals = [w[key] for w in self.rollout_data if key in w]

                numeric_vals = [v for v in vals if isinstance(v, (int, float))]

                aggregated[key] = statistics.mean(numeric_vals) if numeric_vals else float("nan")

            self.rollout_data = []

        return aggregated

    _SHUTDOWN = {"__shutdown__": True}

    def shutdown(self) -> None:
        pub = self.ctx.socket(zmq.PUB)
        pub.connect(self.stats_trainer_sub_address.replace("*", "localhost"))
        time.sleep(0.05)  # ZMQ slow-joiner: give SUB time to see the publisher
        pub.send_json(self._SHUTDOWN)
        pub.close()
        self.join()

    def run(self) -> None:
        socket = self.ctx.socket(zmq.SUB)
        socket.bind(self.stats_trainer_sub_address)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        while (msg := socket.recv_json()) != self._SHUTDOWN:
            with self.lock:
                if "step" in msg:
                    self.eval_data = msg
                else:
                    self.rollout_data.append(msg)

        socket.close()
