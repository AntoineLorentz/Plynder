"""Compiled loss functions: CISPO policy loss + rules-distillation loss.

CISPO (Clipped Importance-Sampling weight Policy Optimization, MiniMax-M1
Eq.4-5) is a REINFORCE-style objective: the importance-sampling ratio is
clipped, detached (stop-gradient), and used as a constant multiplier while the
gradient flows through the policy logprob. The released model additionally
uses the Eq.7 IS mask to zero the gradient on over-confident good tokens
(A>0, r > 1+eps_high) and abandoned bad tokens (A<=0, r < 1-eps_low). The
rules loss is a KL-divergence distillation that teaches an auxiliary head to
predict legal-move masks.

Both functions are ``@torch.compile``-compatible and share the base signature:
    (logits, logits_mask, token_logprobs_ref, advantage, attention_mask,
     output_ids, ...hyperparameters)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from plynder.core.config import Config


@torch.compile(dynamic=True)
def compute_cispo_loss(
    logits: torch.Tensor,  # (B, L, V)
    logits_mask: torch.Tensor,  # (B, L, V), bool, True = forbidden token
    token_logprobs_ref: torch.Tensor,  # (B, L), behavior-policy logprobs (fp32)
    advantage: torch.Tensor,  # (B, L)
    attention_mask: torch.Tensor,  # (B, L)
    output_ids: torch.Tensor,  # (B, L), target token ids
    cispo_epsilon_low: float = 0.1,
    cispo_epsilon_high: float = 0.2,
    cispo_use_mask: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CISPO: Clipped Importance-Sampling weight Policy Optimization
    (Wang et al. 2026, "MiniMax-M1: Scaling Test-Time Compute Efficiently...").

    Per Eq.4-5 of the paper:
        L = -mean( sg(clip(r, 1-eps_low^IS, 1+eps_high^IS)) * A * logp_theta )

    where:
    - r = exp(logp_theta - logp_ref)  (importance sampling weight)
    - sg = stop-gradient: clipped weight is a constant multiplier
    - The gradient flows through logp_theta only (REINFORCE-style)

    With `cispo_use_mask=True` (the released model's recipe), the Eq.7 mask
    additionally zeroes the gradient on over-confident good tokens
    (A>0, r > 1+eps_high) and abandoned bad tokens (A<=0, r < 1-eps_low).
    """
    masked_logits = logits.masked_fill(logits_mask, float("-inf"))
    all_token_logprobs = F.log_softmax(masked_logits, dim=-1)
    token_logprobs = all_token_logprobs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)

    prob_ratio = torch.exp(token_logprobs - token_logprobs_ref)
    clipped_r = torch.clamp(prob_ratio, 1.0 - cispo_epsilon_low, 1.0 + cispo_epsilon_high)

    # For logging
    clip_fraction = (
        ((prob_ratio < 1.0 - cispo_epsilon_low) | (prob_ratio > 1.0 + cispo_epsilon_high))
        & attention_mask
    ).sum() / attention_mask.sum().clamp(min=1.0)

    # REINFORCE-style: detached clipped_r is a constant weight multiplier.
    # Gradient flows through token_logprobs (= logp_theta).
    grad_coeff = clipped_r.detach() * advantage * token_logprobs

    if cispo_use_mask:
        # Eq.7 mask: zero the gradient on over-confident good tokens
        # (A>0, r > 1+eps_high) and abandoned bad tokens (A<=0, r < 1-eps_low).
        pos_adv = advantage > 0
        mask = torch.ones_like(prob_ratio)
        mask[pos_adv & (prob_ratio > 1.0 + cispo_epsilon_high)] = 0.0
        mask[(~pos_adv) & (prob_ratio < 1.0 - cispo_epsilon_low)] = 0.0
        grad_coeff = grad_coeff * mask

    # Sequence-level aggregation
    masked_sum = (grad_coeff * attention_mask).sum(dim=1).float()
    mask_lengths = attention_mask.sum(dim=1).clamp(min=1).float()
    loss = -torch.mean(masked_sum / mask_lengths)

    return loss, clip_fraction


@torch.compile(dynamic=True)
def compute_rules_loss(
    logits_rules: torch.Tensor,  # (B, L, V), auxiliary head logits
    logits_mask: torch.Tensor,  # (B, L, V), bool, True = forbidden token
    importance_coeff: torch.Tensor,  # (B, L), per-token weight
) -> torch.Tensor:
    """KL-divergence between auxiliary head and rules-derived teacher."""
    student_logprobs = F.log_softmax(logits_rules, dim=-1)
    teacher_logprobs = F.log_softmax(logits_mask.float() * -100.0, dim=-1)

    kl = F.kl_div(
        input=student_logprobs,
        target=teacher_logprobs,
        log_target=True,
        reduction="none",
    )
    kl = importance_coeff.unsqueeze(-1) * kl
    return kl.sum().float() / importance_coeff.sum().clamp(min=1.0)


def build_loss_kwargs(
    cfg: Config,
    batch: dict[str, torch.Tensor],
    output_ids: torch.Tensor,
    logits: torch.Tensor,
) -> dict[str, Any]:
    """Build the kwargs dict for ``compute_cispo_loss`` from a collated batch."""
    return {
        "logits": logits,
        "logits_mask": batch["logits_mask"],
        "token_logprobs_ref": batch["token_logprobs"],
        "advantage": batch["advantage"],
        "attention_mask": batch["attention_mask"],
        "output_ids": output_ids,
        "cispo_epsilon_low": cfg.training.cispo_epsilon_low,
        "cispo_epsilon_high": cfg.training.cispo_epsilon_high,
        "cispo_use_mask": cfg.training.cispo_use_mask,
    }
