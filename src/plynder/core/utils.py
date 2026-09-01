"""Chess token mapping and PGN utilities.

Provides the deterministic mapping between chess moves (UCI strings) and the
~1968-token action space, plus the ``to_pgn`` helper for converting token
sequences to PGN games.
"""

import logging
import sys

import chess
import chess.pgn
import torch


def setup_logging(log_file: str | None = None) -> None:
    """Configure logging to both console and a file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any handlers already registered to avoid duplicates
    logger.handlers.clear()

    # Common format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # INFO (and below) → stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(lambda r: r.levelno < logging.ERROR)
    logger.addHandler(stdout_handler)

    # ERROR and above → stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.ERROR)
    logger.addHandler(stderr_handler)

    # File handler (append mode, keeps the full log)
    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)


FILES = "abcdefgh"
RANKS = "12345678"

SLIDE_DIRS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),  # rook-like
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),  # bishop-like
]

KNIGHT_OFFS = [
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
]

# Sort-key order for promotion pieces: q < r < b < n, matching the Rust
# promotion_variant_index (Queen=1, Rook=2, Bishop=3, Knight=4) in
# rust_extension/src/utils.rs.
PROMO_PIECES = ["q", "r", "b", "n"]


def in_bounds(x: int, y: int) -> bool:
    """Check whether (x, y) is a valid board coordinate (0..7)."""
    return 0 <= x < 8 and 0 <= y < 8


def square_idx(file: int, rank: int) -> int:
    """Index 0..63 where 0 == a1, 1 == b1, ... 8 == a2, ..."""
    return rank * 8 + file


def idx_to_uci(idx: int) -> str:
    """Convert a square index (0..63) to a UCI square string (e.g. 'e4')."""
    f = idx % 8
    r = idx // 8
    return f"{FILES[f]}{RANKS[r]}"


def file_rank_to_uci(file: int, rank: int) -> str:
    """Convert (file, rank) indices to a UCI square string."""
    return f"{FILES[file]}{RANKS[rank]}"


def promo_index(promo: str | None) -> int:
    """Map None -> 0, 'n'->1, 'b'->2, 'r'->3, 'q'->4 for sorting key."""
    if promo is None:
        return 0
    return 1 + PROMO_PIECES.index(promo)


def generate_all_moves() -> list[tuple[int, int, str | None]]:
    """
    Returns list of moves as tuples (from_idx, to_idx, promotion or None)
    Mirrors the Rust logic: slide moves, knight moves, explicit promotions,
    then dedupe & sort deterministically.
    """
    tmp: list[tuple[int, int, str | None]] = []

    # iterate every source square (file: 0..7, rank: 0..7)
    for fx in range(8):
        for fy in range(8):
            from_idx = square_idx(fx, fy)

            # 1) sliding moves (queen-like covering rook, bishop, king single-step)
            for dx, dy in SLIDE_DIRS:
                for step in range(1, 8):
                    x = fx + dx * step
                    y = fy + dy * step
                    if not in_bounds(x, y):
                        break
                    to_idx = square_idx(x, y)
                    tmp.append((from_idx, to_idx, None))

            # 2) knight moves
            for dx, dy in KNIGHT_OFFS:
                x = fx + dx
                y = fy + dy
                if in_bounds(x, y):
                    to_idx = square_idx(x, y)
                    tmp.append((from_idx, to_idx, None))

    # 3) explicit promotion moves
    # Rank::Seventh (index 6) with dy = +1 (to rank 7)
    # Rank::Second  (index 1) with dy = -1 (to rank 0)
    for f in range(8):
        for r, dy in ((6, 1), (1, -1)):
            from_file = f
            from_rank = r
            for dx in (-1, 0, 1):
                x = from_file + dx
                y = from_rank + dy
                if not in_bounds(x, y):
                    continue
                from_idx = square_idx(from_file, from_rank)
                to_idx = square_idx(x, y)
                for promo in PROMO_PIECES:
                    tmp.append((from_idx, to_idx, promo))

    # Deduplicate using a set
    unique: set[tuple[int, int, str | None]] = set(tmp)

    # Convert to list and sort deterministically to match Rust sort key:
    # key = (from_i << 16) | (to_i << 8) | promo_index
    all_moves = list(unique)
    all_moves.sort(key=lambda m: (m[0] << 16) | (m[1] << 8) | promo_index(m[2]))
    return all_moves


def moves_to_uci_dict() -> dict[int, str]:
    """
    Returns dict mapping token_id (0-based index into the sorted move list)
    to UCI move string, e.g. "e2e4" or "e7e8q".
    """
    moves = generate_all_moves()
    uci_list: list[str] = []
    for from_idx, to_idx, promo in moves:
        from_sq = idx_to_uci(from_idx)
        to_sq = idx_to_uci(to_idx)
        uci = f"{from_sq}{to_sq}" if promo is None else f"{from_sq}{to_sq}{promo}"
        uci_list.append(uci)

    # map token_id -> uci
    return dict(enumerate(uci_list))


TOKEN_ID_TO_UCI = moves_to_uci_dict()


def to_pgn(token_ids: list[int] | torch.Tensor) -> chess.pgn.Game:
    """Convert a sequence of token IDs to a chess.pgn.Game.

    The token sequence starts with a BOS token (index 0), followed by move
    tokens, and ends with a result token (DRAW / WHITE_WIN / BLACK_WIN).
    Only the move tokens are added as variations to the PGN game.
    """
    from plynder.core.special_tokens import SpecialTokens

    board = chess.Board()
    game = chess.pgn.Game()

    start = 1
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.Tensor(token_ids).long()

    # First result token after the BOS: DRAW / WHITE_WIN / BLACK_WIN.
    result_tokens = torch.tensor(
        [SpecialTokens.DRAW, SpecialTokens.WHITE_WIN, SpecialTokens.BLACK_WIN]
    )
    is_result = torch.isin(token_ids[start:], result_tokens)
    end = int(torch.nonzero(is_result)[0].item()) + start if is_result.any() else len(token_ids)

    # Add moves to the game
    node = game
    for move_id in token_ids[start:end]:
        uci_move = TOKEN_ID_TO_UCI[int(move_id)]
        board.push_uci(uci_move)
        node = node.add_variation(chess.Move.from_uci(uci_move))

    return game


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print the move-token mapping.")
    parser.add_argument(
        "-n", "--num", type=int, default=30, help="Number of token→UCI pairs to print (default: 30)"
    )
    args = parser.parse_args()

    token_to_uci = moves_to_uci_dict()
    print(f"Total moves generated: {len(token_to_uci)}")
    for i in range(min(args.num, len(token_to_uci))):
        print(i, token_to_uci[i])
