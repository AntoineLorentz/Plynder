"""Model sender for broadcasting checkpoints via ZeroMQ."""

from __future__ import annotations

import io
import logging
import os
import queue
import shutil
import struct
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import zmq

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from accelerate import Accelerator

logger = logging.getLogger(__name__)


class ModelSender(threading.Thread):
    """Background thread for sending and saving model checkpoints."""

    def __init__(
        self,
        model_pub_address: str,
        ctx: zmq.Context,
        checkpoint_dir: str | Path,
        archive_dir: str | Path,
        max_keep: int = 3,
        archive_steps: int | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.model_pub_address = model_pub_address
        self.ctx = ctx
        self.queue = queue.Queue()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.archive_dir = Path(archive_dir)
        self.max_keep = max_keep
        self.archive_steps = archive_steps

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Rolling window: tracks (step, path) of saved checkpoint *directories*
        self._recent: deque[tuple[int, Path, Path]] = deque()
        self._archived_milestones: set[int] = set()

        logger.info(f"[ModelSender] Publishing to    {self.model_pub_address}")
        logger.info(f"[ModelSender] Checkpoints  ->  {self.checkpoint_dir}  (keep last {max_keep})")
        if archive_steps:
            logger.info(
                f"[ModelSender] Archiving every {archive_steps} steps -> {self.archive_dir}"
            )

    def send(self, model: torch.nn.Module, topic: str, step: int) -> None:
        """Enqueue a broadcast of the model weights over ZMQ."""
        state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        self.queue.put(
            (
                "send",
                {
                    "topic": topic.encode(),
                    "step": struct.pack("<i", step),
                    "state_dict": state_dict,
                },
            )
        )

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        head: torch.nn.Module,
        optimizers: Sequence[torch.optim.Optimizer],
        accelerator: Accelerator,
    ) -> None:
        """Enqueue a checkpoint save (model + head + optimizers) to the background thread."""
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_head = accelerator.unwrap_model(head)
        model_sd = accelerator.get_state_dict(model)  # handles FSDP/DeepSpeed sharding
        opt_sds = [opt.state_dict() for opt in optimizers]  # plain .state_dict(), no wrapper needed
        save_fn = accelerator.save  # replaces torch.save for distributed safety

        def _to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_to_cpu(v) for v in obj)
            return obj

        opt_sds = [_to_cpu(opt_sd) for opt_sd in opt_sds]

        self.queue.put(
            (
                "save",
                {
                    "step": step,
                    "model": unwrapped_model,
                    "head": unwrapped_head,
                    "model_sd": model_sd,
                    "optimizer_sds": opt_sds,
                    "save_fn": save_fn,
                },
            )
        )

    def shutdown(self) -> None:
        self.queue.put(None)
        self.join()

    # ------------------------------------------------------------------ #
    #  Background thread                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        socket = self.ctx.socket(zmq.PUB)
        socket.bind(self.model_pub_address)

        for kind, data in iter(self.queue.get, None):  # stops when None is dequeued
            if kind == "send":
                self._do_send(socket, data)
            elif kind == "save":
                self._do_save(data)

        socket.close()

    def _do_send(self, socket: zmq.Socket, payload: dict[str, Any]) -> None:
        buffer = io.BytesIO()
        torch.save(payload["state_dict"], buffer)
        socket.send_multipart([payload["topic"], payload["step"], buffer.getvalue()])

    def _do_save(self, payload: dict[str, Any]) -> None:
        step = payload["step"]
        model = payload["model"]
        head = payload["head"]
        model_sd = payload["model_sd"]

        ckpt_dir = self.checkpoint_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        head_path = self.checkpoint_dir / f"head-checkpoint-{step}"

        model.save_pretrained(ckpt_dir, state_dict=model_sd)
        torch.save(head, head_path)

        self._recent.append((step, ckpt_dir, head_path))
        self._maybe_archive(step, ckpt_dir, head_path)
        self._evict_old_checkpoints()

    # ------------------------------------------------------------------ #
    #  Rolling eviction                                                    #
    # ------------------------------------------------------------------ #

    def _evict_old_checkpoints(self) -> None:
        """Remove checkpoint directories beyond the rolling window.

        Archived copies live under self.archive_dir, so we only remove
        the entry from self.checkpoint_dir; the archive is untouched.
        """
        while len(self._recent) > self.max_keep:
            _, old_dir, old_head = self._recent.popleft()
            if old_dir.exists():
                shutil.rmtree(old_dir)
            if old_head.exists():
                os.remove(old_head)

    # ------------------------------------------------------------------ #
    #  Archiving                                                           #
    # ------------------------------------------------------------------ #

    def _maybe_archive(self, step: int, ckpt_dir: Path, head_path: Path) -> None:
        """Archive ckpt_dir when it is the first checkpoint to exceed
        archive_steps * k for a milestone k not yet archived.

        The archive is a full copy (not a hard-link) because directories
        cannot be hard-linked portably, and we want the archive to survive
        the rolling eviction of self.checkpoint_dir.
        """
        if not self.archive_steps or step == 0:
            return

        k = step // self.archive_steps

        if k < 1 or k in self._archived_milestones:
            return

        self._archived_milestones.add(k)
        dest = self.archive_dir / ckpt_dir.name
        dest_head = self.archive_dir / head_path.name
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(ckpt_dir, dest)
        shutil.copy(head_path, dest_head)
