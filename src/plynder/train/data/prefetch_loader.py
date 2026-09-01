"""Prefetch loader for overlapping data transfer with computation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from accelerate.utils import send_to_device

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class PrefetchLoader:
    """Wraps a DataLoader and implements one-batch-ahead prefetching.

    Moves batch N to GPU non-blocking (async H→D via DMA) while
    CPU workers fetch batch N+1.
    """

    def __init__(self, dataloader: DataLoader, device: str | torch.device) -> None:
        self.dataloader = dataloader
        self.device = device

    def __iter__(self) -> Iterator[tuple[dict, bool]]:
        it = iter(self.dataloader)

        # Fetch batch 0 into pinned RAM.
        try:
            batch, sync_step = next(it)
        except StopIteration:
            return

        while True:
            # Schedule async H→D copy for current batch (returns immediately)
            batch = send_to_device(batch, self.device, non_blocking=True)
            try:
                # While GPU consumes current batch, workers are already
                # reading + collating the next one from LMDB
                next_batch, next_sync = next(it)
                yield batch, sync_step
                batch, sync_step = next_batch, next_sync
            except StopIteration:
                yield batch, sync_step
                break
