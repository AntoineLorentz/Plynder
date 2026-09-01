import json

import torch  # noqa: F401
from plynder_rs import SequenceStatesAsync, TerminalTokens
from tqdm import tqdm

from plynder.core import SpecialTokens

sequences_states = SequenceStatesAsync(
    vocab_size=1972,
    terminal_tokens=TerminalTokens(
        draw_id=SpecialTokens.DRAW,
        terminal_token_win_0=SpecialTokens.WHITE_WIN,
        terminal_token_win_1=SpecialTokens.BLACK_WIN,
    ),
    device=None,
    rollout_address=None,
    global_engine_id=None,
)

results = []
seq_id = "seq"
seq_ids = [seq_id]


all_moves = [[SpecialTokens.BOS]]
depth_max = 5

pbar = tqdm(desc="Discovering openings")

while len(all_moves) > 0:
    moves = all_moves.pop()

    sequences_states.add_sequence(seq_id, moves)
    sequences_states.spawn_valid_tokens_vec(seq_ids)
    legal_moves = sequences_states.join_get_valid_tokens_vec()[0]
    sequences_states.remove_sequence(seq_id)

    if len(moves) >= depth_max:
        if legal_moves[0] not in (
            SpecialTokens.DRAW,
            SpecialTokens.WHITE_WIN,
            SpecialTokens.BLACK_WIN,
        ):
            results.append(moves)

        pbar.update(1)
        continue

    for m in legal_moves:
        new_moves = moves + [m]
        all_moves.append(new_moves)


with open("openings.jsonl", "w") as f:
    for moves in results:
        f.write(json.dumps(moves) + "\n")
