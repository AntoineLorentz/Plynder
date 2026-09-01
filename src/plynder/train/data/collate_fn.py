"""Collation function for batching training examples."""

import logging
from typing import Any

import numpy as np
import torch

from plynder.core import SpecialTokens

logger = logging.getLogger(__name__)

LOG_EPS = float(np.log(1e-8))


def collate_fn(
    examples: list[dict[str, Any]],
    sync_step: bool,
    microbatch_max_tokens: int,
    pad_token_id: int,
    vocab_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Collate examples into padded batches.

    Args:
        examples: List of example dictionaries.
        sync_step: Whether to synchronize gradients after this batch.
        microbatch_max_tokens: Maximum number of tokens in the micro-batch.
        Controls VRAM allocation only; does NOT affect total batch size or gradient updates.
        pad_token_id: Token ID for padding.
        vocab_size: Vocabulary size.
        dtype: Data type for tensors (token_logprobs are always float32, see below).

    Returns:
        Tuple of (buffers dict, sync_step flag).
    """

    seq_length = max(len(ex["token_ids"]) for ex in examples) - 1
    batch_size = len(examples)

    # We use microbatch_max_tokens as the flat buffer size to reduce memory fragmentation
    buffers = {
        "batch_size": batch_size,
        "seq_length": seq_length,
        "token_ids": torch.full((microbatch_max_tokens,), pad_token_id, dtype=torch.long),
        # FP32 reference logprobs (per MiniMax-M1 paper §3.2): behavior-policy
        # logprobs must match the precision of the policy logprobs in the loss.
        "token_logprobs": torch.full((microbatch_max_tokens,), LOG_EPS, dtype=torch.float32),
        "advantage": torch.zeros((microbatch_max_tokens,), dtype=dtype),
        "logits_mask": torch.ones((microbatch_max_tokens, vocab_size), dtype=torch.bool),
        "attention_mask": torch.zeros((microbatch_max_tokens,), dtype=torch.bool),
        "rules_informative_tokens": torch.zeros((microbatch_max_tokens,), dtype=dtype),
    }

    # Happen only if the batch is somehow corrupted
    if batch_size * seq_length > microbatch_max_tokens:
        logger.warning(
            f"Batch overflow: {batch_size=} * {seq_length=} = {batch_size * seq_length} "
            f"> {microbatch_max_tokens=}. Skipping signal."
        )
        return buffers, sync_step

    real = reshape_padded_buffers(buffers)
    token_ids = real["token_ids"]  # (B, S)
    token_logprobs = real["token_logprobs"]  # (B, S-1)
    advantage = real["advantage"]  # (B, S-1)
    logits_mask = real["logits_mask"]  # (B, S-1, V)
    attention_mask = real["attention_mask"]  # (B, S-1)
    rules_informative_tokens = real["rules_informative_tokens"]  # (B, S-1)

    # pre-compute per-example scalars
    end_openings = [ex["end_opening"] - 1 for ex in examples]
    start_idxs = [ex["start_index"] - 1 for ex in examples]
    lengths = [
        min(len(ex["token_ids"]) - 1, seq_length) for ex in examples
    ]  # we remove EOS (result token) as it is not an action

    # batch advantage (no Python loop inside)
    adv_arrays = compute_group_advantages(examples, end_openings, lengths)

    for i, ex in enumerate(examples):
        seq_length_i = lengths[i]
        end_opening = end_openings[i]
        start_idx = start_idxs[i]
        num_steps = seq_length_i - 1 - end_opening

        # token_ids
        token_ids[i, :seq_length_i] = torch.as_tensor(
            ex["token_ids"][:seq_length_i].astype(np.int64)
        )

        eos = ex["token_ids"][-1]
        if (
            eos
            not in (
                SpecialTokens.DRAW,
                SpecialTokens.WHITE_WIN,
                SpecialTokens.BLACK_WIN,
            )
            or len(ex["token_logprobs"]) < num_steps
        ):
            logits_mask[i, :, :] = False
            attention_mask[i, :] = (
                True  # Advantage is zero so grad will also be zero for this example
            )
            continue

        # Probabilities & advantages are one shorter than L (because L does not account for EOS so just bos_token)
        token_logprobs[i, end_opening : seq_length_i - 1] = (
            torch.as_tensor(ex["token_logprobs"][:num_steps]) + 1e-8
        )

        # advantage
        adv = adv_arrays[i]
        if adv is not None:
            advantage[i, end_opening : seq_length_i - 1] = torch.as_tensor(adv, dtype=dtype)

        # Build the logits mask with a sentinel, vectorized scatter, and range expansion
        allowed_offsets = ex["at_offsets"]
        allowed_flat_mask = ex["at_flat"]

        end_offset = int(allowed_offsets[num_steps])
        step_lens = torch.diff(torch.as_tensor(allowed_offsets[: num_steps + 1], dtype=torch.long))
        t_idx = torch.arange(
            end_opening, end_opening + num_steps, dtype=torch.long
        ).repeat_interleave(step_lens)
        tok_idx = torch.as_tensor(allowed_flat_mask[:end_offset], dtype=torch.long)

        logits_mask[i, t_idx, tok_idx] = False

        logits_mask[i, :end_opening, :] = False
        logits_mask[i, seq_length_i - 1 :, :] = False

        # attention + rules
        attention_mask[i, start_idx : seq_length_i - 1] = True
        group_size_inv = 1.0 / (ex["white_wins"] + ex["black_wins"] + ex["draws"])
        rules_informative_tokens[i, end_opening:start_idx] = group_size_inv
        rules_informative_tokens[i, start_idx : seq_length_i - 1] = 1.0

    return buffers, sync_step


def compute_group_advantages(
    examples: list[dict[str, Any]],
    end_openings: list[int],
    lengths: list[int],
) -> list[np.ndarray | None]:
    """
    Returns one float32 advantage array per example,
    length = L-1-end_opening, aligned to the action window.
    """
    results = []
    for ex, end_opening, seq_length_i in zip(examples, end_openings, lengths, strict=True):
        eos = ex["token_ids"][-1]
        if eos not in (
            SpecialTokens.DRAW,
            SpecialTokens.WHITE_WIN,
            SpecialTokens.BLACK_WIN,
        ):
            results.append(None)
            continue

        white_wins, black_wins, draws = (
            ex["white_wins"],
            ex["black_wins"],
            ex["draws"],
        )
        group_size = white_wins + black_wins + draws  # group size (e.g. 8)

        # white reward stats
        r_white_group = np.empty(group_size, dtype=np.float32)
        r_white_group[:white_wins] = 1.0
        r_white_group[white_wins : white_wins + black_wins] = -1.0
        r_white_group[white_wins + black_wins :] = 0.0
        mean_w = r_white_group.mean()
        std_w = float(np.sqrt(r_white_group.var() + 1e-8))

        # black reward stats
        r_black_group = np.empty(group_size, dtype=np.float32)
        r_black_group[:black_wins] = 1.0
        r_black_group[black_wins : black_wins + white_wins] = -1.0
        r_black_group[black_wins + white_wins :] = 0.0
        mean_b = r_black_group.mean()
        std_b = float(np.sqrt(r_black_group.var() + 1e-8))

        r_w = {
            SpecialTokens.DRAW: 0.0,
            SpecialTokens.WHITE_WIN: 1.0,
            SpecialTokens.BLACK_WIN: -1.0,
        }[eos]
        r_b = {
            SpecialTokens.DRAW: 0.0,
            SpecialTokens.WHITE_WIN: -1.0,
            SpecialTokens.BLACK_WIN: 1.0,
        }[eos]

        adv_w = np.float32((r_w - mean_w) / std_w)
        adv_b = np.float32((r_b - mean_b) / std_b)

        num_steps = seq_length_i - 1 - end_opening

        # Even indices (0,2,4,...) -> white; odd -> black
        adv = np.empty(num_steps, dtype=np.float32)
        adv[0::2] = adv_w
        adv[1::2] = adv_b

        results.append(adv)

    return results


def reshape_padded_buffers(
    buffers: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """For memory fragmentation"""
    batch_size = buffers["batch_size"]
    seq_length = buffers["seq_length"]

    return {
        "token_ids": buffers["token_ids"][: batch_size * seq_length].view(batch_size, seq_length),
        "token_logprobs": buffers["token_logprobs"][: batch_size * (seq_length - 1)].view(
            batch_size, seq_length - 1
        ),
        "advantage": buffers["advantage"][: batch_size * (seq_length - 1)].view(
            batch_size, seq_length - 1
        ),
        "logits_mask": buffers["logits_mask"][: batch_size * (seq_length - 1), :].view(
            batch_size, seq_length - 1, -1
        ),
        "attention_mask": buffers["attention_mask"][: batch_size * (seq_length - 1)].view(
            batch_size, seq_length - 1
        ),
        "rules_informative_tokens": buffers["rules_informative_tokens"][
            : batch_size * (seq_length - 1)
        ].view(batch_size, seq_length - 1),
    }
