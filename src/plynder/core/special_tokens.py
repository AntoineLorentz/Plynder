"""Special token IDs for the Plynder chess vocabulary.

The vocabulary is 1972 tokens: 1968 chess moves (indices 0-1967) followed by
4 special tokens that signal game states.
"""

from enum import IntEnum


class SpecialTokens(IntEnum):
    """Special token IDs appended after the 1968 move tokens."""

    BOS = 1968
    """Beginning-of-sequence token."""

    DRAW = 1969
    """Game ended in a draw."""

    WHITE_WIN = 1970
    """White won the game."""

    BLACK_WIN = 1971
    """Black won the game."""
