"""DG-Trap: definition-trap training objective.

Goal: a model that picks a class from user-given definitions, whatever the class
is called. Each training instance is shown in one of three namings:

    orig      all names original; a name that fits the audio is often right
    goldswap  one gold class swaps names with one non-gold class; the gold's
              real name (the "trap") now sits on a wrong definition
    pairswap  two non-gold classes swap names; mismatched lines do not point
              to the answer (blocks the "odd line = answer" leak)

Losses (reference-free, single stage):

    L_nll  = -log p(y | audio, legend)                        every instance
    L_trap = -log sigmoid(beta * (log p(gold tok) - log p(trap tok)) - margin)
             at the first completion token where the shown gold name and the
             trap name differ (goldswap only; same forward pass)
    L_cond = -log sigmoid(beta * (log p(y | legend) - log p(y | legend with the
             gold's deciding words removed)) - margin)        second pass
    L = L_nll + lambda_t * L_trap + lambda_c * L_cond

Harmless word dropout removes NON-deciding words from random legend lines
(gold included) in every instance, so a damaged-looking line is not a cue and
the conditional term cannot be satisfied by "avoid damaged lines".

Logit alignment (important)
---------------------------
Qwen2-Audio (transformers 4.45) expands each <|AUDIO|> token into the audio
frames INSIDE the forward pass, so `logits` is longer than `input_ids` and
`prompt_length` (counted on input_ids) does not index it. The completion is
always the last tokens of an unpadded sequence, so it is located from the END:
completion row j sits at logits.size(1) - n_completion + j. Batch size must be
1 (no padding). `check_alignment` recomputes the model's NLL from these rows so
the training log proves the indexing is right.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "LEGEND_LINE",
    "load_deciding_index",
    "map_prompt_text",
    "edit_legend",
    "drop_words",
    "remove_words",
    "completion_rows",
    "token_logprobs",
    "check_alignment",
    "first_divergence",
    "trap_token_diff",
    "margin_loss",
    "swap_completion_slot",
]

# "- name: definition" bullet. Guideline bullets have no colon and never match.
LEGEND_LINE = re.compile(r'(?m)^([ \t]*-[ \t]*)([^:\n]+?)([ \t]*:[ \t]*)(\S[^\n]*?)[ \t]*$')
_SUBWORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Deciding-word index (built offline by scripts/build_deciding_words.py)
# ---------------------------------------------------------------------------

def load_deciding_index(path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{dataset: {normalised definition: {label, neighbour, deciding(set)}}}.

    Keyed by definition text so every wording variant in the JSON is covered;
    the prompt is matched by its actual definition text, not by the variant name.
    """
    with open(path) as f:
        raw = json.load(f)
    index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for ds, variants in raw.items():
        ds_idx = index.setdefault(ds, {})
        for _variant, labels in variants.items():
            for label, r in labels.items():
                ds_idx[_norm(r["definition"])] = {
                    "label": label,
                    "neighbour": r["neighbour"],
                    "deciding": set(r["deciding_words"]),
                }
    return index


# ---------------------------------------------------------------------------
# Prompt text editing
# ---------------------------------------------------------------------------

def map_prompt_text(prompt_obj: Any, fn: Callable[[str], str]) -> Any:
    """Copy of a prompt (str or chat dict) with `fn` applied to every text part.

    New objects at every level, so the caller's prompt (shared with other views
    and mutated later by tokenize_batch_with_audio) is never touched.
    """
    if isinstance(prompt_obj, str):
        return fn(prompt_obj)
    if isinstance(prompt_obj, dict) and "conversation" in prompt_obj:
        new_obj = dict(prompt_obj)
        new_conv = []
        for msg in prompt_obj.get("conversation", []):
            new_msg = dict(msg)
            content = new_msg.get("content")
            if isinstance(content, str):
                new_msg["content"] = fn(content)
            elif isinstance(content, list):
                new_msg["content"] = [dict(p, text=fn(p["text"])) if p.get("type") == "text" else p
                                      for p in content]
            new_conv.append(new_msg)
        new_obj["conversation"] = new_conv
        return new_obj
    if isinstance(prompt_obj, dict):
        return dict(prompt_obj)
    return prompt_obj


def edit_legend(text: str, fn: Callable[[str, str], Optional[str]]) -> str:
    """Rewrite legend definitions: fn(name, definition) -> new definition or None (keep)."""
    def _sub(m: re.Match) -> str:
        new_def = fn(m.group(2).strip(), m.group(4))
        if new_def is None:
            return m.group(0)
        return f"{m.group(1)}{m.group(2)}{m.group(3)}{new_def}".rstrip()
    return LEGEND_LINE.sub(_sub, text)


def _is_deciding(token: str, deciding: Set[str]) -> bool:
    return any(w in deciding for w in _SUBWORD.findall(token.lower()))


def drop_words(definition: str, deciding: Set[str], rate: float,
               rng: random.Random) -> Tuple[str, int]:
    """Drop each NON-deciding token with probability `rate`; keep at least one token.

    Returns (new definition, number of tokens dropped).
    """
    tokens = definition.split()
    if rate <= 0 or len(tokens) <= 1:
        return definition, 0
    kept = [t for t in tokens if _is_deciding(t, deciding) or rng.random() >= rate]
    if not kept:
        kept = [tokens[0]]
    return " ".join(kept), len(tokens) - len(kept)


def remove_words(definition: str, remove: Set[str]) -> Tuple[str, int]:
    """Remove every token containing a word in `remove`. Returns (new definition, n removed)."""
    tokens = definition.split()
    kept = [t for t in tokens if not _is_deciding(t, remove)]
    return " ".join(kept), len(tokens) - len(kept)


def swap_completion_slot(completion: str, shown: str, replacement: str) -> Optional[str]:
    """Replace the label `shown` in a comma-separated completion by `replacement`."""
    parts = [p.strip() for p in completion.split(",")]
    if shown not in parts:
        return None
    return ", ".join(replacement if p == shown else p for p in parts)


# ---------------------------------------------------------------------------
# Logit indexing and losses
# ---------------------------------------------------------------------------

def completion_rows(logits: torch.Tensor, n_completion: int) -> torch.Tensor:
    """Rows that predict the n completion tokens: [n, V]. End-aligned, see module doc."""
    if logits.dim() != 3 or logits.size(0) != 1:
        raise ValueError(f"expected [1, T, V] logits, got {tuple(logits.shape)}")
    T = logits.size(1)
    if n_completion < 1 or n_completion >= T:
        raise ValueError(f"n_completion={n_completion} invalid for T={T}")
    return logits[0, T - n_completion - 1: T - 1]


def token_logprobs(logits: torch.Tensor, completion_ids: Sequence[int]) -> torch.Tensor:
    """log p of each completion token under the model: [n] (float32)."""
    rows = completion_rows(logits, len(completion_ids)).float()
    ids = torch.tensor(list(completion_ids), device=rows.device, dtype=torch.long)
    return F.log_softmax(rows, dim=-1).gather(1, ids[:, None]).squeeze(1)


def check_alignment(logits: torch.Tensor, completion_ids: Sequence[int],
                    model_loss: torch.Tensor) -> float:
    """|our NLL - model NLL|; ~0 when the end-aligned rows are the right ones."""
    with torch.no_grad():
        ours = -token_logprobs(logits, completion_ids).mean()
        return float((ours - model_loss.float()).abs())


def first_divergence(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    """First index where two token lists differ; None if one is a prefix of the other."""
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None


def trap_token_diff(logits: torch.Tensor, gold_ids: Sequence[int],
                    trap_ids: Sequence[int]) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """log p(gold token) - log p(trap token) at their first differing completion position.

    The shared prefix is on the gold path (teacher forcing), so one pass suffices
    and the prefix's probability cancels out of the comparison.
    """
    k = first_divergence(gold_ids, trap_ids)
    if k is None or k >= len(gold_ids):
        return None, {"k": k}
    rows = completion_rows(logits, len(gold_ids))
    logp = F.log_softmax(rows[k].float(), dim=-1)
    g, t = int(gold_ids[k]), int(trap_ids[k])
    diff = logp[g] - logp[t]
    return diff, {"k": k, "gold_tok": g, "trap_tok": t,
                  "logp_gold": float(logp[g].detach()), "logp_trap": float(logp[t].detach())}


def margin_loss(diff: torch.Tensor, beta: float, margin: float) -> torch.Tensor:
    """-log sigmoid(beta * diff - margin)."""
    return -F.logsigmoid(beta * diff - margin)
