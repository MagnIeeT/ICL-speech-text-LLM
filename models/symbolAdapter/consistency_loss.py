"""Surface-form equivariance objective for definition-guided tuning.

Why
---
Surrogate tuning shows each example under exactly one legend, so the objective
is satisfiable by learning several surface-specific routines instead of one
surface-independent procedure. Measured consequence: the regime mixture reaches
instruction-tuning accuracy on *trained* tasks and fails to transfer to held-out
ones (``docs/e0_results.md``, section E1).

What we want is equivariance under label-surface relabelling. If a bijection
``f`` rewrites the legend's surfaces, the correct answer becomes ``f(answer)``
rather than staying fixed. Mapping both outputs back through their own ``f``
into *definition* space turns that into an invariance we can penalise:

    L = NLL(answer_a | x, legend_a)
      + NLL(answer_b | x, legend_b)
      + weight * D(p_a, p_b)

``p_a`` and ``p_b`` are distributions over the same K definitions, in the same
order, so column k means the same category under both legends. ``D`` is a
symmetric divergence.

The two NLL terms leave the passes independent; ``D`` is the only term that
charges the model for answering identical content differently. The NLL terms are
equally non-negotiable: consistency alone is minimised by emitting one confident
answer under every legend, right or wrong, and we have measured that failure
mode. The full-data mixture pushed canonical/symbolic agreement to 86.3 percent
while held-out accuracy fell.

How p is computed
-----------------
A forward pass returns logits ``[B, T, V]``. Row ``prompt_length - 1`` predicts
the first answer token and is conditioned on the prompt only, so it scores every
candidate surface fairly in one pass. We gather the K first-token ids, softmax
over just those K, and that is ``p``.

This is a distribution over first tokens, not over full label strings. It is a
good stand-in while the K surfaces begin with distinct tokens, because once the
first token is chosen the rest of the spelling is near-deterministic. Verified on
the Qwen2-Audio tokenizer: CREMA-D 6/6, MInDS-14 en 14/14, Skit-S2I 14/14,
RAVDESS-Song 6/6 distinct. HVB is the exception at 10 distinct out of 18
(``answer_*``, ``question_*`` and ``statement_*`` each collide), and HVB is also
the only multi-label set, so a K-way distribution is the wrong object for it
regardless. Callers should skip HVB rather than feed it colliding ids;
``check_collisions`` exists so that is a deliberate choice and not a silent one.

Swapping the objective
----------------------
``DIVERGENCES`` maps a name to a function of two log-probability tensors. Add an
entry and it becomes selectable by flag without touching the training loop.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "DIVERGENCES",
    "candidate_first_token_ids",
    "check_collisions",
    "definition_log_probs",
    "consistency_term",
    "combined_loss",
]


# ---------------------------------------------------------------------------
# Divergences: two [B, K] log-prob tensors -> [B]
# ---------------------------------------------------------------------------

def _sym_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """KL(p||q) + KL(q||p). Unbounded, strongest gradient on disagreement."""
    p, q = log_p.exp(), log_q.exp()
    return (p * (log_p - log_q)).sum(-1) + (q * (log_q - log_p)).sum(-1)


def _js(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon. Bounded by log 2, so safer if the KL version destabilises."""
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    p, q = log_p.exp(), log_q.exp()
    return 0.5 * (p * (log_p - log_m)).sum(-1) + 0.5 * (q * (log_q - log_m)).sum(-1)


def _l2(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Squared distance between probability vectors. Mildest of the three."""
    return ((log_p.exp() - log_q.exp()) ** 2).sum(-1)


DIVERGENCES = {
    "sym_kl": _sym_kl,
    "js": _js,
    "l2": _l2,
}


# ---------------------------------------------------------------------------
# Surfaces -> first-token ids
# ---------------------------------------------------------------------------

def candidate_first_token_ids(
    surfaces: Sequence[str],
    tokenizer,
    prefix: str = "",
) -> List[int]:
    """First token id of each surface, preserving the given order.

    ``prefix`` is prepended before tokenising. Most tokenizers encode ``"angry"``
    and ``" angry"`` differently, so pass whatever the template places
    immediately before the answer; otherwise the ids will not match the tokens
    the model is actually asked to emit.
    """
    ids: List[int] = []
    for s in surfaces:
        toks = tokenizer.encode(prefix + s, add_special_tokens=False)
        if not toks:
            raise ValueError(f"surface {s!r} tokenised to nothing")
        ids.append(int(toks[0]))
    return ids


def check_collisions(
    first_ids: Sequence[int],
    surfaces: Sequence[str],
) -> List[Tuple[str, ...]]:
    """Groups of surfaces sharing a first token, which the objective cannot separate.

    Empty list means the K surfaces are safe to use. A non-empty result means
    those definitions share a logit, so the divergence is blind to the
    difference between them.
    """
    buckets: Dict[int, List[str]] = {}
    for tid, s in zip(first_ids, surfaces):
        buckets.setdefault(tid, []).append(s)
    return [tuple(v) for v in buckets.values() if len(v) > 1]


# ---------------------------------------------------------------------------
# Logits -> distribution over definitions
# ---------------------------------------------------------------------------

def definition_log_probs(
    logits: torch.Tensor,
    prompt_lengths: Sequence[int],
    first_token_ids: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    input_len: Optional[int] = None,
) -> torch.Tensor:
    """Log-probabilities over the K candidate definitions. Returns ``[B, K]``.

    Args:
        logits: ``[B, T, V]`` from the forward pass.
        prompt_lengths: per-sample prompt length. Row ``prompt_length - 1``
            predicts the first answer token.
        first_token_ids: ``[B, K]`` first token of each candidate surface,
            ordered by definition so column k is the same category in both
            legends.
        valid_mask: ``[B, K]`` bool; False marks padded candidate slots.
        input_len: length of ``input_ids`` fed to the model. When given, the
            row is shifted by ``logits.size(1) - input_len`` to undo the
            in-forward audio expansion (batch size 1 only).

    Renormalising over the K candidates is deliberate: the term compares the
    model's relative preference among the legend's own labels. Mass placed
    outside the inventory is a different failure, and the NLL terms handle it.
    """
    if logits.dim() != 3:
        raise ValueError(f"expected [B, T, V] logits, got {tuple(logits.shape)}")
    if first_token_ids.dim() != 2:
        raise ValueError(f"expected [B, K] first_token_ids, got {tuple(first_token_ids.shape)}")

    # Qwen2-Audio expands <|AUDIO|> inside forward, so logits are longer than
    # input_ids. Audio sits before the answer, so every later row moves by the
    # same amount. One shift per batch is only valid for batch size 1.
    shift = 0
    if input_len is not None:
        if logits.size(0) != 1:
            raise ValueError("input_len shift is only valid for batch size 1")
        shift = logits.size(1) - int(input_len)
    rows = []
    for i, plen in enumerate(prompt_lengths):
        idx = max(0, min(int(plen) - 1 + shift, logits.size(1) - 1))
        rows.append(logits[i, idx])
    # float32: fp16 log-probs underflow once the distribution sharpens.
    row = torch.stack(rows, dim=0).float()                     # [B, V]

    gathered = row.gather(1, first_token_ids.to(row.device))   # [B, K]
    if valid_mask is not None:
        gathered = gathered.masked_fill(~valid_mask.to(gathered.device), float("-inf"))
    return F.log_softmax(gathered, dim=-1)


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------

def consistency_term(
    log_p_a: torch.Tensor,
    log_p_b: torch.Tensor,
    divergence: str = "sym_kl",
) -> torch.Tensor:
    """Scalar divergence between the two legends' definition distributions."""
    try:
        fn = DIVERGENCES[divergence]
    except KeyError:
        raise ValueError(
            f"unknown divergence {divergence!r}; available: {sorted(DIVERGENCES)}"
        ) from None
    if log_p_a.shape != log_p_b.shape:
        raise ValueError(
            "both legends must expose the same K definitions in the same order, got "
            f"{tuple(log_p_a.shape)} and {tuple(log_p_b.shape)}"
        )
    return fn(log_p_a, log_p_b).mean()


def combined_loss(
    nll_a: torch.Tensor,
    nll_b: torch.Tensor,
    log_p_a: Optional[torch.Tensor] = None,
    log_p_b: Optional[torch.Tensor] = None,
    weight: float = 0.0,
    divergence: str = "sym_kl",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``nll_a + nll_b + weight * D(p_a, p_b)``, with components for logging.

    Degrades to ``nll_a + nll_b`` when ``weight`` is zero or either distribution
    is missing, which is the intended path for HVB batches and for the
    weight-zero control run. A non-finite divergence is dropped for that step
    with a warning rather than poisoning the gradient.
    """
    total = nll_a + nll_b
    parts = {
        "nll_a": float(nll_a.detach()),
        "nll_b": float(nll_b.detach()),
        "consistency": 0.0,
        "applied": 0.0,
    }

    if weight and log_p_a is not None and log_p_b is not None:
        cons = consistency_term(log_p_a, log_p_b, divergence)
        if torch.isfinite(cons):
            total = total + weight * cons
            parts["consistency"] = float(cons.detach())
            parts["applied"] = 1.0
        else:
            logging.warning("consistency term non-finite; skipped for this step")

    parts["total"] = float(total.detach())
    return total, parts
