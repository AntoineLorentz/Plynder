"""Shared chess utilities for evaluation and Stockfish tournaments.

Provides the castling-notation mapping, token lookup tables, the ``safe_uci``
helper, and the async ``EnginePool`` used by both the evaluation worker pool
and the Stockfish tournament script.
"""

import asyncio
import contextlib
from contextlib import asynccontextmanager

import chess
import chess.engine

from plynder.core import TOKEN_ID_TO_UCI

#: Plynder uses king-captures-rook UCI notation for castling (e.g. "e1h1"
#: instead of the standard "e1g1").  This map converts standard UCI castling
#: moves to the Plynder convention.
CASTLING_MAP: dict[str, str] = {
    "e1g1": "e1h1",  # White kingside
    "e1c1": "e1a1",  # White queenside
    "e8g8": "e8h8",  # Black kingside
    "e8c8": "e8a8",  # Black queenside
}

#: Reverse lookup: UCI move string → token ID.
UCI_TO_TOKEN_ID: dict[str, int] = {v: k for k, v in TOKEN_ID_TO_UCI.items()}


def safe_uci(move: chess.Move, board: chess.Board) -> str:
    """Convert a chess.Move to Plynder's UCI convention (king-captures-rook for castling)."""
    uci = move.uci()
    if board.is_castling(move):
        return CASTLING_MAP.get(uci, uci)
    return uci


class EnginePool:
    """Fixed-size pool of persistent Stockfish engines, avoiding spawn and quit per game.

    Each engine is configured once at startup and reused across games via an
    asyncio.Queue.  Use the ``acquire`` async context manager to check out an
    engine and automatically return it after the game.
    """

    def __init__(self, stockfish_path: str, size: int, sf_config: dict[str, str]) -> None:
        self._path = stockfish_path
        self._size = size
        self._config = sf_config
        self._queue: asyncio.Queue = asyncio.Queue()

    async def start(self) -> None:
        """Spawn ``size`` Stockfish engines and configure each one."""
        for _ in range(self._size):
            _, engine = await chess.engine.popen_uci(self._path)
            await engine.configure(self._config)
            self._queue.put_nowait(engine)

    @asynccontextmanager
    async def acquire(self):
        """Check out an engine from the pool; return it on exit."""
        engine = await self._queue.get()
        try:
            yield engine
        finally:
            self._queue.put_nowait(engine)

    async def shutdown(self) -> None:
        """Quit all engines in the pool."""
        while not self._queue.empty():
            engine = self._queue.get_nowait()
            with contextlib.suppress(Exception):
                await engine.quit()
