"""Warmup pass to trigger torch.compile graph tracing before the training loop.

Tensor defaults are chosen to:
  - produce finite (non-NaN) values throughout every kernel
  - exercise the non-degenerate code paths (both clip branches, non-zero gradients,
    rules loss actually computed) so dynamo traces the real computation graph
"""

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from plynder.core.config import Config
from plynder.train.data.collate_fn import reshape_padded_buffers
from plynder.train.losses import build_loss_kwargs, compute_cispo_loss, compute_rules_loss


def _make_warmup_batch(
    B: int,
    L: int,
    vocab_size: int,
    microbatch_max_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build a synthetic micro-batch of shape (B, L) backed by flat buffers of
    size microbatch_max_tokens.

    Value rationale
    ---------------
    token_ids
        Token id 1 (never the pad/EOS sentinel 0).

    token_logprobs
        log(1/vocab_size), the log-probability a uniform model assigns to any token.
        This makes prob_ratio = exp(model_logprob - log(1/V)) = model_prob x V,
        which floats around 1 and thus exercises both the lower *and* upper clip
        branches of CISPO depending on the random model weights.

    advantage
        Alternates +1 / -1 so that torch.min sees both orderings
        (prob_ratio x adv  vs  clipped x adv) across the batch.
        Using only +1 would always favour the un-clipped branch for small
        prob_ratios and never trace the clipped-positive path; only -1 would
        miss the clipped-negative path.

    logits_mask
        All False (no token masked).  Setting any *entire row* to True would
        push every logit to -inf and make log_softmax return NaN.

    attention_mask
        All True so that mask_lengths > 0 everywhere and the loss is non-zero,
        ensuring a meaningful backward pass during warmup.

    rules_informative_tokens
        1.0 everywhere so that informative_tokens is all-True and the full
        rules-loss branch is compiled (importance_coeff is non-empty, KL is
        actually computed).  Using 0.0 would skip the rules loss entirely.
    """
    M = microbatch_max_tokens

    token_ids = torch.ones(M, dtype=torch.long, device=device)

    # Span the range [1e-10, 1.0] so to trigger every branch: 1 - eps, 1, 1 + eps
    # FP32 to match the real collate buffers (behavior-policy logprobs are FP32).
    token_logprobs = torch.log(torch.linspace(1e-10, 1.0, M, dtype=torch.float32, device=device))

    # Alternating sign so both clip branches are exercised.
    advantage = torch.ones(M, dtype=dtype, device=device)
    advantage[1::2] = -1.0

    # All False: any fully-masked row → log_softmax(-inf) → NaN.
    logits_mask = torch.zeros(M, vocab_size, dtype=torch.bool, device=device)

    # All True: loss is non-zero, backward exercises the real gradient path.
    attention_mask = torch.ones(M, dtype=torch.bool, device=device)

    # Uniform weight = 1: rules loss is non-trivially computed.
    rules_informative_tokens = torch.ones(M, dtype=dtype, device=device)

    flat = {
        "batch_size": B,
        "seq_length": L,
        "token_ids": token_ids,
        "token_logprobs": token_logprobs,
        "advantage": advantage,
        "logits_mask": logits_mask,
        "attention_mask": attention_mask,
        "rules_informative_tokens": rules_informative_tokens,
    }
    batch = reshape_padded_buffers(flat)

    # Mark both dims dynamic so dynamo emits symbolic guards instead of
    # baking in concrete sizes.
    torch._dynamo.mark_dynamic(batch["token_ids"], 0)
    torch._dynamo.mark_dynamic(batch["token_ids"], 1)
    return batch


def run_warmup(
    model: nn.Module,
    head: nn.Module,
    optimizer_muon: torch.optim.Optimizer,
    optimizer_adam: torch.optim.Optimizer,
    accelerator: Any,
    cfg: Config,
    vocab_size: int,
    no_sync_context: Callable,
    hidden_layer_index: int = 6,
) -> None:
    """Run two (shape x sync) warmup passes to trigger torch.compile tracing
    for both the model/head forward and the two loss functions.

    Two shapes are used to straddle inductor's B*L threshold so that the
    compiler emits guards for the dynamic-shape branches we actually hit
    during training:
      - large  : B * L = microbatch_max_tokens         (at/above threshold)
      - small  : B * L ≤ 16_384                        (below threshold)

    Each shape is run twice: once without gradient sync (no_sync) and once
    with it, because the wrapped-forward guard is different in each case.
    """
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.assume_static_by_default = False

    dtype = next(model.parameters()).dtype
    device = accelerator.device
    T = cfg.training.batching.microbatch_max_tokens

    warmup_shapes = [
        (2, T // 2),
        (4, min(T // 4, 4096)),
    ]

    for B, L in warmup_shapes:
        batch = _make_warmup_batch(
            B=B,
            L=L,
            vocab_size=vocab_size,
            microbatch_max_tokens=T,
            dtype=dtype,
            device=device,
        )

        for sync in (False, True):
            ctx = nullcontext if sync else no_sync_context
            with ctx():
                input_ids = batch["token_ids"][:, :-1]  # (B, L-1)
                output_ids = batch["token_ids"][:, 1:]  # (B, L-1)

                out = model(input_ids=input_ids, output_hidden_states=True)

                # ── rules loss ────────────────────────────────────────────

                # head is already compiled; call it outside compute_rules_loss
                # to avoid double-compilation.
                logits_rules = head(out.hidden_states[hidden_layer_index])
                loss_rules = compute_rules_loss(
                    logits_rules,
                    batch["logits_mask"],
                    batch["rules_informative_tokens"],
                )

                # ── Policy loss (CISPO) ──────────────────────────────────────────
                base_kwargs = build_loss_kwargs(cfg, batch, output_ids, out.logits)
                loss_policy, _ = compute_cispo_loss(**base_kwargs)
                loss = loss_policy + loss_rules
                accelerator.backward(loss)

            optimizer_muon.zero_grad()
            optimizer_adam.zero_grad()
