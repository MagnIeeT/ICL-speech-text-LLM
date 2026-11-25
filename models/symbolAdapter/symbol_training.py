"""Simple Symbol Adapter trainer: LoRA-only, epoch-based, symbol strategy driven."""

import json
import logging
import random
import re
import os
import time
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config.data_config.master_config import DATASET_CONFIGS
from config.train_config.training_configs import TrainingConfig, SymbolUpdateStrategy
from .symbol_manager import SymbolManager, generate_fresh_symbols, get_dataset_info, tokenize_batch_with_audio
from .consistency_loss import (
    candidate_first_token_ids,
    check_collisions,
    combined_loss,
    consistency_term,
    definition_log_probs,
)
from .trap_loss import (
    check_alignment,
    drop_words,
    edit_legend,
    load_deciding_index,
    map_prompt_text,
    margin_loss,
    remove_words,
    swap_completion_slot,
    token_logprobs,
    trap_token_diff,
)
from .validation import ValidationManager


# A legend entry is a '- label: definition' bullet. Guideline bullets carry no
# colon, so they never match and are never moved.
_LEGEND_LINE = re.compile(r'^[ \t]*-[ \t]*[^:\n]+?[ \t]*:[ \t]*\S.*$')


def _shuffle_legend_text(text: str) -> str:
    """Randomly reorder the first contiguous block of legend bullets in `text`.

    Only line order changes; every line keeps its own surface and definition,
    so the label-to-definition binding is untouched. Text with no legend block
    of two or more lines is returned unchanged.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if not _LEGEND_LINE.match(lines[i]):
            i += 1
            continue
        j = i
        while j < len(lines) and _LEGEND_LINE.match(lines[j]):
            j += 1
        if j - i >= 2:
            block = lines[i:j]
            random.shuffle(block)
            lines[i:j] = block
            return "\n".join(lines)
        i = j
    return text


def _shuffle_legend_in_prompt(prompt_obj: Any) -> Any:
    """Copy of one prompt (str or chat dict) with its legend order shuffled.

    New dicts are built at every level, so the caller's prompt objects, which
    the other consistency view may share, are never mutated.
    """
    if isinstance(prompt_obj, str):
        return _shuffle_legend_text(prompt_obj)
    if isinstance(prompt_obj, dict) and "conversation" in prompt_obj:
        new_obj = dict(prompt_obj)
        new_conv = []
        for msg in prompt_obj.get("conversation", []):
            new_msg = dict(msg)
            content = new_msg.get("content")
            if isinstance(content, str):
                new_msg["content"] = _shuffle_legend_text(content)
            elif isinstance(content, list):
                new_msg["content"] = [
                    dict(part, text=_shuffle_legend_text(part["text"]))
                    if part.get("type") == "text" else part
                    for part in content]
            new_conv.append(new_msg)
        new_obj["conversation"] = new_conv
        return new_obj
    return prompt_obj


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def _shuffle_legend_in_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Batch copy with an independent legend order drawn for every instance."""
    out = dict(batch)
    out["prompt"] = [_shuffle_legend_in_prompt(p) for p in batch.get("prompt", [])]
    return out


class SymbolTrainingOrchestrator:
    """Single-path trainer with optional dynamic symbol replacement."""

    def __init__(
        self,
        config: TrainingConfig,
        model,
        processor,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tokenizer=None,
        symbol_manager: SymbolManager = None,
        train_dataset_names: set = None,
    ):
        self.config = config
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.symbol_manager = symbol_manager
        self.train_dataset_names = train_dataset_names or set()
        self.optimizer = None
        self.global_step = 0
        self._epoch_log_count = 0  # logs first 2 batches per epoch
        self._swap_cache: dict = {}       # per-dataset swap mapping; reset each epoch

        # ---- surface-form consistency objective (off unless weight > 0) ----
        _sc = config.symbol_config
        self._cons_weight = float(getattr(_sc, "consistency_weight", 0.0) or 0.0)
        self._cons_div = getattr(_sc, "consistency_divergence", "js")
        _views = (getattr(_sc, "consistency_views", "sym,flip") or "sym,flip").split(",")
        self._cons_views = tuple(v.strip() for v in _views if v.strip())
        # Independent legend order per view, so "same position" is not a valid
        # invariant and only the definition is shared across the two views.
        self._cons_shuffle = bool(getattr(_sc, "consistency_shuffle_legend", False))
        # Control: the two-view path at weight 0. Same data and same two passes
        # per step, no coupling, so any gain of the weighted run over this one is
        # the divergence term and not the doubled supervision.
        self._cons_control = bool(getattr(_sc, "consistency_control", False))
        self._cons_on = self._cons_weight > 0 or self._cons_control
        # Multi-label targets, and colliding canonical surfaces, cannot carry the
        # divergence (HVB is both). Those take one ordinary pass instead of two.
        self._cons_skipped_ds: set = set()          # structurally ineligible: never gets it
        self._cons_collided_ds: set = set()         # eligible, but some drawing collided
        self._cons_eligible: Dict[str, bool] = {}   # per-dataset, decided once
        self._cons_stats = {"steps": 0, "applied": 0, "measured": 0, "single": 0, "collided": 0}
        # Running means logged every _cons_log_every batches (the per-step
        # CONSISTENCY line only fires for the first two batches of an epoch).
        self._cons_log_every = 200
        self._cons_run = self._new_cons_run()
        if self._cons_on:
            if config.data_config.batch_size != 1:
                raise ValueError("consistency needs batch_size 1 (the answer row is shifted by "
                                 "the audio expansion, which differs per sample)")
            if len(self._cons_views) != 2:
                raise ValueError(
                    f"consistency_views needs exactly two views, got {self._cons_views}")
            bad = [v for v in self._cons_views if v not in ("sym", "flip", "orig", "same")]
            if bad or self._cons_views[0] == "same":
                raise ValueError(
                    f"consistency_views {self._cons_views}: views are sym|flip|orig, and "
                    "'same' is allowed only as the second view")
            if self._cons_views[1] == "same" and not self._cons_shuffle:
                raise ValueError(
                    "consistency view 'same' without consistency_shuffle_legend gives two "
                    "identical prompts; the pair would carry no signal")
            logging.info(
                "Consistency objective ON: views=%s weight=%.3f divergence=%s "
                "shuffle_legend=%s control=%s",
                "+".join(self._cons_views), self._cons_weight, self._cons_div,
                self._cons_shuffle, self._cons_control)
            # The divergence reads one logit row, so symbols sharing a first token
            # would merge two definitions. Rebuild the pool to prevent it.
            if self.symbol_manager is not None:
                self.symbol_manager.regenerate_pool_distinct_first_tokens()

        # ---- DG-Trap objective (off unless trap_enabled) ----
        self._trap_on = bool(getattr(_sc, "trap_enabled", False))
        if self._trap_on:
            self._init_trap(config)

        self.validator = ValidationManager(
            config=config,
            symbol_manager=self.symbol_manager,
            tokenizer=tokenizer,
            processor=self.processor,
            max_val_samples=config.data_config.val_max_samples,
        )

        self._setup_training_environment()

    # ------------------------------------------------------------------
    # DG-Trap objective
    # ------------------------------------------------------------------

    def _init_trap(self, config: TrainingConfig) -> None:
        sc = config.symbol_config
        if self._cons_on:
            raise ValueError("trap_enabled and the consistency objective are mutually exclusive")
        if sc.swap_labels or (getattr(sc, "surrogate_mix", 0.0) or 0.0) > 0:
            raise ValueError("trap_enabled uses its own naming; turn off --swap_labels and --surrogate_mix")
        if config.data_config.batch_size != 1:
            raise ValueError("trap_enabled needs batch_size 1 (completion rows are located from the "
                             "end of an unpadded sequence; see trap_loss.py)")
        # Renaming rewrites label words anywhere outside definitions. With a transcript or
        # text few-shot examples in the prompt, a goldswap of balance<->address would turn
        # "check my balance" into "check my address": a lexical leak that defeats the trap.
        if getattr(config.data_config, "input_mode", "speech_only") != "speech_only":
            raise ValueError("trap_enabled supports input_mode=speech_only only (renaming would "
                             "rewrite label words inside the transcript)")
        if int(getattr(config.data_config, "num_examples", 0) or 0) > 0:
            raise ValueError("trap_enabled supports zero-shot prompts only (NUM_EXAMPLES=0); renaming "
                             "would rewrite label words inside few-shot example text")
        probs = [sc.trap_p_orig, sc.trap_p_goldswap, sc.trap_p_pairswap,
                 float(getattr(sc, "trap_p_symbol", 0.0) or 0.0)]
        if min(probs) < 0 or sum(probs) <= 0:
            raise ValueError(f"trap naming probabilities must be >= 0 with a positive sum, got {probs}")
        total = sum(probs)
        self._trap_kinds = ["orig", "goldswap", "pairswap", "symbol"]
        self._trap_probs = [p / total for p in probs]
        # Strings training must never draw (the test symbol files), so test symbols stay unseen.
        self._trap_symbol_exclude: set = set()
        for f in (x.strip() for x in (getattr(sc, "trap_symbol_exclude", "") or "").split(",")):
            if f:
                with open(f) as fh:
                    for m in json.load(fh).values():
                        self._trap_symbol_exclude.update(s.lower() for s in m.values())
        if self._trap_symbol_exclude:
            logging.info("DG-Trap symbol naming: %d test strings excluded from training draws",
                         len(self._trap_symbol_exclude))
        self._trap_nn_frac = float(sc.trap_nn_frac)
        self._trap_dropout_max = float(sc.trap_dropout_max)
        self._trap_lambda = float(sc.trap_lambda)
        self._trap_cond_lambda = float(sc.trap_cond_lambda)
        self._trap_cond_frac = float(sc.trap_cond_frac)
        self._trap_cond_mode = sc.trap_cond_mode
        self._trap_cond_span = getattr(sc, "trap_cond_span", "all")
        self._trap_beta = float(sc.trap_beta)
        self._trap_margin = float(sc.trap_margin)
        self._trap_cond_margin = float(sc.trap_cond_margin)
        path = sc.trap_deciding_words
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), path)
        self._trap_index = load_deciding_index(path)
        # Dedicated RNG so the naming draws are reproducible under --seed and do not
        # shift the global random stream used elsewhere.
        self._trap_rng = random.Random(config.seed)
        self._trap_log_every = 200
        self._trap_detail_every = 1000         # one full TRAP-DETAIL block every N batches
        self._trap_run = self._new_trap_run()
        self._trap_totals = self._new_trap_run()
        self._trap_warned: set = set()
        logging.info(
            "DG-Trap ON: p(orig,goldswap,pairswap)=%s nn_frac=%.2f dropout_max=%.2f "
            "lambda_trap=%.3f lambda_cond=%.3f cond_frac=%.2f cond_mode=%s cond_span=%s beta=%.2f "
            "margin=%.2f cond_margin=%.2f deciding_words=%s (datasets=%s)",
            [round(p, 3) for p in self._trap_probs], self._trap_nn_frac, self._trap_dropout_max,
            self._trap_lambda, self._trap_cond_lambda, self._trap_cond_frac, self._trap_cond_mode,
            self._trap_cond_span,
            self._trap_beta, self._trap_margin, self._trap_cond_margin, path,
            sorted(self._trap_index))

    @staticmethod
    def _new_trap_run() -> Dict[str, float]:
        keys = ["steps", "orig", "goldswap", "pairswap", "symbol", "nn", "random", "nll",
                "trap_n", "trap_loss", "trap_diff", "trap_win", "cond_n", "cond_loss",
                "cond_diff", "cond_win", "dropped", "removed", "align_err_max",
                "skip_no_gold", "skip_no_partner", "skip_trap_k", "skip_cond_words",
                "skip_cond_empty", "skip_nonfinite", "unmatched_lines", "cond_span_fallback"]
        return {k: 0.0 for k in keys}

    def _trap_draw_symbols(self, n: int) -> List[str]:
        """n distinct fresh 2-token symbols (same generator as Sym training and the test
        files), none from the excluded test files. Seeded from the trap RNG; the global
        `random` state is restored, so other random draws are unaffected."""
        state = random.getstate()
        random.seed(self._trap_rng.getrandbits(64))
        try:
            out: List[str] = []
            for _ in range(20):
                for s in generate_fresh_symbols(self.tokenizer, n - len(out) + 4, token_size=2):
                    if s.lower() not in self._trap_symbol_exclude and s not in out:
                        out.append(s)
                if len(out) >= n:
                    return out[:n]
            raise RuntimeError(f"could not draw {n} symbols outside the exclusion list")
        finally:
            random.setstate(state)

    def _trap_label_span(self, completion: str, label: str, c_ids: Sequence[int]):
        """(start, end) of `label`'s tokens inside the completion ids, or None.

        The label must occur exactly once as a comma-separated part, and the ids of the
        text up to and including it must be a prefix of c_ids (BPE boundary unchanged);
        otherwise the caller falls back to scoring the whole completion.
        """
        spans = [(m.start(), m.end()) for m in re.finditer(r"[^,]+", completion)]
        hits = [(s, e) for s, e in spans if completion[s:e].strip() == label]
        if len(hits) != 1:
            return None
        s, e = hits[0]
        # Start right after the comma: Qwen merges the following space into the label's
        # first token (", answer" -> "," + " answer"), so splitting after the space breaks it.
        e = s + (len(completion[s:e]) - len(completion[s:e].lstrip())) + len(label)
        tok = lambda t: self.tokenizer(t, add_special_tokens=False).input_ids if t else []
        a, b = tok(completion[:s]), tok(completion[:e])
        if len(b) <= len(a) or b[:len(a)] != a or list(c_ids[:len(b)]) != b:
            return None
        return len(a), len(b)

    def _trap_warn_once(self, key: str, msg: str, *args) -> None:
        if key not in self._trap_warned:
            self._trap_warned.add(key)
            logging.warning("DG-Trap: " + msg, *args)

    def _trap_build(self, raw_batch: Dict[str, Any], prompt_text_fn, mapping: Dict[str, str]):
        """Tokenised copy of raw_batch with the prompt edited, then labels renamed by `mapping`."""
        staged = dict(raw_batch)
        staged["prompt"] = [map_prompt_text(p, prompt_text_fn) for p in raw_batch["prompt"]]
        staged["completion"] = list(raw_batch["completion"])
        if mapping:
            staged = self.symbol_manager.replace_symbols_in_batch(staged, mapping, mapping)
        return tokenize_batch_with_audio(staged, self.processor, staged.get("completion"))

    def _trap_step(self, raw_batch: Dict[str, Any], epoch: int, batch_idx: int):
        """One DG-Trap step. Returns (batch, loss, parts); loss None means skip the batch."""
        rng = self._trap_rng
        run = self._trap_run
        ds_name_str, relevant_labels = get_dataset_info(raw_batch, self.symbol_manager.original_labels)
        ds_index = self._trap_index.get(ds_name_str, {})
        if not ds_index:
            self._trap_warn_once(f"noidx_{ds_name_str}",
                                 "no deciding words for %s; dropout and the conditional term are off "
                                 "for it (naming and trap still apply)", ds_name_str)
        completion = raw_batch["completion"][0]
        gold = [g for g in (p.strip() for p in completion.split(",")) if g in relevant_labels]
        non_gold = [l for l in relevant_labels if l not in gold]
        # Per-label info from THIS prompt's definition text, so the neighbour and the
        # deciding words follow whichever wording variant the prompt uses.
        line_info: Dict[str, Dict[str, Any]] = {}

        def _collect(text: str) -> str:
            def _line(name: str, definition: str):
                info = ds_index.get(" ".join(definition.lower().split()))
                if info is not None:
                    line_info[name.lower()] = info
                return None
            edit_legend(text, _line)
            return text
        for p in raw_batch["prompt"]:
            map_prompt_text(p, _collect)

        # 1. naming
        kind = rng.choices(self._trap_kinds, weights=self._trap_probs, k=1)[0]
        mapping: Dict[str, str] = {}
        g_flip = partner = None
        partner_src = ""
        if not gold:
            run["skip_no_gold"] += 1
            self._trap_warn_once("nogold", "completion %r has no valid label of %s; plain step",
                                 completion, ds_name_str)
            kind = "orig"
        if kind == "goldswap":
            g_flip = rng.choice(gold)
            nb = (line_info.get(g_flip.lower()) or {}).get("neighbour")
            if nb in non_gold and rng.random() < self._trap_nn_frac:
                partner, partner_src = nb, "nn"
            elif non_gold:
                partner, partner_src = rng.choice(non_gold), "random"
            if partner is None:
                run["skip_no_partner"] += 1
                kind, g_flip = "orig", None
            else:
                mapping = {g_flip: partner, partner: g_flip}
        elif kind == "pairswap":
            if len(non_gold) >= 2:
                a, b = rng.sample(non_gold, 2)
                mapping = {a: b, b: a}
            else:
                run["skip_no_partner"] += 1
                kind = "orig"
        elif kind == "symbol":
            # Every label of the task gets a fresh meaningless string for this instance:
            # copy practice, and the name gives no hint, so only the definition helps.
            mapping = dict(zip(relevant_labels, self._trap_draw_symbols(len(relevant_labels))))

        # 2. harmless dropout (canonical names, before renaming; identical in both views)
        rate = rng.uniform(0.0, self._trap_dropout_max) if self._trap_dropout_max > 0 else 0.0
        drop_stats = {"dropped": 0, "unmatched": 0}

        def _dropout(text: str) -> str:
            def _line(name: str, definition: str):
                info = ds_index.get(" ".join(definition.lower().split()))
                if info is None:
                    drop_stats["unmatched"] += 1
                    return None
                new_def, n = drop_words(definition, info["deciding"], rate, rng)
                drop_stats["dropped"] += n
                return new_def
            return edit_legend(text, _line) if ds_index else text

        dropped_prompts = [map_prompt_text(p, _dropout) for p in raw_batch["prompt"]]
        if drop_stats["unmatched"]:
            run["unmatched_lines"] += drop_stats["unmatched"]
            self._trap_warn_once(f"unmatched_{ds_name_str}",
                                 "%d legend line(s) of %s not found in the deciding-word index "
                                 "(left unedited). Rebuild with scripts/build_deciding_words.py if "
                                 "the definitions changed.", drop_stats["unmatched"], ds_name_str)
        base = dict(raw_batch)
        base["prompt"] = dropped_prompts

        # 3. conditional view: remove the gold line's deciding words
        cond_on = (self._trap_cond_lambda > 0 and bool(gold) and bool(ds_index)
                   and rng.random() < self._trap_cond_frac)
        g_cond = None
        removed_words: List[str] = []
        if cond_on:
            g_cond = g_flip if g_flip else rng.choice(gold)
            info = line_info.get(g_cond.lower())
            if info is None or not info["deciding"]:
                run["skip_cond_words"] += 1
                cond_on = False
        rem_stats = {"removed": 0, "empty": False}

        def _damage(text: str) -> str:
            def _line(name: str, definition: str):
                if name.lower() != g_cond.lower():
                    return None
                if self._trap_cond_mode == "random":
                    toks = definition.split()
                    k = min(len([t for t in toks if any(w in info["deciding"] for w in
                                                       re.findall(r"[a-z]+(?:'[a-z]+)?", t.lower()))]),
                            len(toks))
                    drop_idx = set(rng.sample(range(len(toks)), k)) if k else set()
                    if len(drop_idx) == len(toks):
                        # An empty line exists only in the damaged view: a shortcut. Skip.
                        rem_stats["empty"] = True
                        return None
                    removed_words.extend(toks[i] for i in sorted(drop_idx))
                    rem_stats["removed"] += len(drop_idx)
                    return " ".join(t for i, t in enumerate(toks) if i not in drop_idx)
                new_def, n = remove_words(definition, info["deciding"])
                if not new_def.strip():
                    # An empty line exists only in the damaged view: a shortcut. Skip.
                    rem_stats["empty"] = True
                    return None
                removed_words.extend(t for t in definition.split() if t not in new_def.split())
                rem_stats["removed"] += n
                return new_def
            return edit_legend(text, _line)

        # 4. forward (full view)
        batch_full = self._trap_build(base, lambda t: t, mapping)
        shown_completion = batch_full["completion"][0]
        c_ids = self.tokenizer(f"{shown_completion}{self.tokenizer.eos_token}",
                               add_special_tokens=False).input_ids
        if batch_idx < 2 or (batch_idx + 1) % self._trap_detail_every == 0:
            self._log_train_prompt(batch_full, epoch, batch_idx, tag=f"trap {kind} {ds_name_str}")
        tail = batch_full["input_ids"][0][-len(c_ids):].tolist()
        if tail != list(c_ids):
            self._trap_warn_once("cids", "completion ids %s do not match the input tail %s; "
                                 "trap/cond terms disabled for such steps", c_ids, tail)
        batch_full = self._move_batch_to_device(batch_full)
        out = self.model(batch_full)
        nll, logits = out.get("loss"), out.get("logits")
        if nll is None or not torch.isfinite(nll) or logits is None:
            run["skip_nonfinite"] += 1
            self._trap_warn_once("nll", "model returned no usable loss/logits at batch %d "
                                 "(non-finite NLL or dummy-loss path); step skipped", batch_idx)
            return batch_full, None, None
        ids_ok = tail == list(c_ids)
        align_err = check_alignment(logits, c_ids, nll) if ids_ok else float("nan")
        if ids_ok and align_err > 1e-2:
            self._trap_warn_once("align", "end-aligned NLL differs from the model loss by %.4f "
                                 "(batch %d); completion rows may be mis-indexed", align_err, batch_idx)
        loss = nll
        parts: Dict[str, Any] = {"kind": kind, "nll": float(nll.detach()), "align_err": align_err,
                                 "rate": rate, "dropped": drop_stats["dropped"]}

        # 5. trap margin (same pass)
        if kind == "goldswap" and ids_ok:
            trap_completion = swap_completion_slot(shown_completion, partner, g_flip)
            t_ids = (self.tokenizer(f"{trap_completion}{self.tokenizer.eos_token}",
                                    add_special_tokens=False).input_ids if trap_completion else None)
            diff, tinfo = (trap_token_diff(logits, c_ids, t_ids) if t_ids else (None, {"k": None}))
            if diff is None:
                run["skip_trap_k"] += 1
            elif not torch.isfinite(diff):
                run["skip_nonfinite"] += 1
                self._trap_warn_once("trapnan", "non-finite trap diff at batch %d; term skipped", batch_idx)
            else:
                l_trap = margin_loss(diff, self._trap_beta, self._trap_margin)
                if self._trap_lambda > 0:
                    loss = loss + self._trap_lambda * l_trap
                parts.update(trap_loss=float(l_trap.detach()), trap_diff=float(diff.detach()),
                             trap_k=tinfo["k"],
                             trap_gold_tok=self.tokenizer.decode([tinfo["gold_tok"]]),
                             trap_trap_tok=self.tokenizer.decode([tinfo["trap_tok"]]),
                             trap_completion=trap_completion)

        # 6. conditional margin (second pass)
        if cond_on and ids_ok:
            batch_dam = self._trap_build(base, _damage, mapping)
            if rem_stats["empty"]:
                run["skip_cond_empty"] += 1
            elif rem_stats["removed"] == 0:
                run["skip_cond_words"] += 1
            else:
                if batch_idx < 2 or (batch_idx + 1) % self._trap_detail_every == 0:
                    self._log_train_prompt(batch_dam, epoch, batch_idx,
                                           tag=f"trap cond-damaged {ds_name_str} gold={g_cond}")
                batch_dam = self._move_batch_to_device(batch_dam)
                out_d = self.model(batch_dam)
                if out_d.get("logits") is not None:
                    tok_full = token_logprobs(logits, c_ids)
                    tok_dam = token_logprobs(out_d["logits"], c_ids)
                    span = None
                    if self._trap_cond_span == "gold":
                        # Score only the damaged label (multi-label hvb: not the other golds,
                        # commas or EOS). g_cond is shown under its mapped name, if renamed.
                        span = self._trap_label_span(shown_completion, mapping.get(g_cond, g_cond), c_ids)
                        if span is None:
                            run["cond_span_fallback"] += 1
                            self._trap_warn_once("condspan", "could not locate %r in completion %r; "
                                                 "cond term scores the whole completion for such steps",
                                                 mapping.get(g_cond, g_cond), shown_completion)
                    if span is not None:
                        lp_full = tok_full[span[0]:span[1]].sum()
                        lp_dam = tok_dam[span[0]:span[1]].sum()
                        parts["cond_span"] = span
                    else:
                        lp_full = tok_full.sum()
                        lp_dam = tok_dam.sum()
                    cdiff = lp_full - lp_dam
                    if not torch.isfinite(cdiff):
                        run["skip_nonfinite"] += 1
                        self._trap_warn_once("condnan", "non-finite cond diff at batch %d; term "
                                             "skipped", batch_idx)
                    else:
                        l_cond = margin_loss(cdiff, self._trap_beta, self._trap_cond_margin)
                        loss = loss + self._trap_cond_lambda * l_cond
                        parts.update(cond_loss=float(l_cond.detach()), cond_diff=float(cdiff.detach()),
                                     cond_gold=g_cond, cond_removed=removed_words)

        parts["total"] = float(loss.detach())
        if batch_idx < 2 or (batch_idx + 1) % self._trap_detail_every == 0:
            logging.info(
                "TRAP-DETAIL epoch=%d batch=%d ds=%s kind=%s mapping=%s partner_src=%s gold=%s "
                "shown_completion=%r dropout_rate=%.3f dropped=%d align_err=%.2e nll=%.4f "
                "trap=[k=%s gold_tok=%r trap_tok=%r diff=%s loss=%s] cond=[gold=%s removed=%s diff=%s "
                "loss=%s] total=%.4f",
                epoch + 1, batch_idx, ds_name_str, kind, mapping, partner_src or "-", gold,
                shown_completion, rate, drop_stats["dropped"], align_err, parts["nll"],
                parts.get("trap_k"), parts.get("trap_gold_tok"), parts.get("trap_trap_tok"),
                _fmt(parts.get("trap_diff")), _fmt(parts.get("trap_loss")), parts.get("cond_gold"),
                parts.get("cond_removed"), _fmt(parts.get("cond_diff")), _fmt(parts.get("cond_loss")),
                parts["total"])
        self._trap_track(parts, partner_src, rem_stats["removed"], epoch, batch_idx)
        return batch_full, loss, parts

    def _trap_track(self, parts: Dict[str, Any], partner_src: str, removed: int,
                    epoch: int, batch_idx: int) -> None:
        for r in (self._trap_run, self._trap_totals):
            r["steps"] += 1
            r[parts["kind"]] += 1
            if partner_src:
                r[partner_src] += 1
            r["nll"] += parts["nll"]
            r["dropped"] += parts["dropped"]
            if parts["align_err"] == parts["align_err"]:   # not NaN
                r["align_err_max"] = max(r["align_err_max"], parts["align_err"])
            if "trap_diff" in parts:
                r["trap_n"] += 1
                r["trap_loss"] += parts["trap_loss"]
                r["trap_diff"] += parts["trap_diff"]
                r["trap_win"] += float(parts["trap_diff"] > 0)
            if "cond_diff" in parts:
                r["cond_n"] += 1
                r["cond_loss"] += parts["cond_loss"]
                r["cond_diff"] += parts["cond_diff"]
                r["cond_win"] += float(parts["cond_diff"] > 0)
                r["removed"] += removed
        if (batch_idx + 1) % self._trap_log_every == 0:
            logging.info("TRAP-RUN epoch=%d batch=%d %s", epoch + 1, batch_idx + 1,
                         self._trap_summary(self._trap_run))
            self._trap_run = self._new_trap_run()

    def _trap_summary(self, r: Dict[str, float]) -> str:
        n = max(r["steps"], 1)
        tn, cn = max(r["trap_n"], 1), max(r["cond_n"], 1)
        return (
            f"n={int(r['steps'])} kinds(orig/gold/pair/sym)={int(r['orig'])}/{int(r['goldswap'])}/"
            f"{int(r['pairswap'])}/{int(r['symbol'])} partner(nn/rand)={int(r['nn'])}/{int(r['random'])} "
            f"nll={r['nll'] / n:.4f} | trap n={int(r['trap_n'])} loss={r['trap_loss'] / tn:.4f} "
            f"diff={r['trap_diff'] / tn:.3f} win={r['trap_win'] / tn:.3f}"
            f"{' (measured, not trained)' if self._trap_lambda == 0 else ''} | "
            f"cond n={int(r['cond_n'])} loss={r['cond_loss'] / cn:.4f} diff={r['cond_diff'] / cn:.3f} "
            f"win={r['cond_win'] / cn:.3f} removed/step={r['removed'] / cn:.2f} | "
            f"dropped/step={r['dropped'] / n:.2f} align_err_max={r['align_err_max']:.2e} | "
            f"skips(no_gold/no_partner/trap_k/cond_words/cond_empty)={int(r['skip_no_gold'])}/"
            f"{int(r['skip_no_partner'])}/{int(r['skip_trap_k'])}/{int(r['skip_cond_words'])}/"
            f"{int(r['skip_cond_empty'])} "
            f"cond_span_fallback={int(r['cond_span_fallback'])} "
            f"nonfinite={int(r['skip_nonfinite'])} "
            f"unmatched_lines={int(r['unmatched_lines'])}")

    def _setup_training_environment(self):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.metrics_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)
        checkpoint_dir = self.config.get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logging.info("Training environment setup complete")

    def _setup_lora_optimizer(self, remaining_epochs: int = None, lr_override: float = None):
        base_lr = lr_override if lr_override is not None else self.config.lora_config.learning_rate
        # Collect ALL unfrozen model params, not just LoRA-named ones. SALMONN unfreezes
        # the speech Q-Former + speech_llama_proj (freeze_*=False in custom_salmonn.py);
        # a name filter on "lora" would silently drop them from the optimizer so they
        # never train. For qwen/flamingo only LoRA is unfrozen, so this is equivalent.
        lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        if not lora_params:
            raise ValueError("No trainable parameters found")
        n_model = sum(p.numel() for n, p in self.model.named_parameters() if p.requires_grad)
        n_lora = sum(p.numel() for n, p in self.model.named_parameters() if p.requires_grad and "lora" in n.lower())
        logging.info("Optimizer: model_params=%d (of which lora=%d)  lr=%.2e", n_model, n_lora, base_lr)
        self.optimizer = torch.optim.AdamW([{"params": lora_params, "lr": base_lr}], lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.config.lora_config.weight_decay)
        t_max = remaining_epochs if remaining_epochs is not None else self.config.lora_config.epochs
        steps_per_epoch = max(len(self.train_dataloader) // max(self.config.lora_config.gradient_accumulation_steps, 1), 1)
        total_steps = max(t_max, 1) * steps_per_epoch
        warmup = self.config.lora_config.warmup_steps
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
        logging.info("CosineWithWarmup: warmup=%d  total_steps=%d  lr=%.2e", warmup, total_steps, base_lr)

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor): device_batch[key] = value.to(self.config.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                device_batch[key] = [v.to(self.config.device) for v in value]
            else: device_batch[key] = value
        return device_batch

    def _get_swap_mappings(self, ds_name_str: str, relevant_labels: list, epoch: int, batch_idx: int,
                           force_real_labels: bool = False) -> dict:
        """Return label->surface mapping for swap mode, cached per epoch or fresh per instance.

        force_real_labels=True permutes the REAL label words even when symbols are
        enabled. The surrogate mixture needs this: its flipped regime must supply
        real vocabulary, which is the whole reason it preserves canonical ability.
        Without it, no_symbols=False would permute the synthetic symbols instead and
        both regimes would be synthetic.
        """
        per_instance = self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE
        if per_instance or ds_name_str not in self._swap_cache:
            base_mapping = None
            if not self.config.symbol_config.no_symbols and not force_real_labels:
                full_base = self.symbol_manager._pure_symbol_mappings.get(ds_name_str, {})
                base_mapping = {l: full_base[l] for l in relevant_labels if l in full_base}
            mapping = self.symbol_manager.generate_swap_mapping_for_labels(
                relevant_labels,
                base_symbol_mapping=base_mapping,
                epoch=epoch if not per_instance else None,
            )
            if not per_instance:
                self._swap_cache[ds_name_str] = mapping
        else:
            mapping = self._swap_cache[ds_name_str]
        if batch_idx < 2:
            mode = "symbol-swap" if not self.config.symbol_config.no_symbols else "label-swap"
            logging.info("Swap mapping [%s] (epoch=%d batch=%d dataset=%s): %s",
                         mode, epoch + 1, batch_idx, ds_name_str, mapping)
        return mapping

    def _set_epoch_symbol_maps_on_dataset(self, epoch: int) -> None:
        """Push symbol maps onto the dataset before the DataLoader iterator forks workers.
        Workers inherit the maps at fork time — no IPC needed after this point.
        Only used for non-swap modes (fixed, per_epoch, per_instance, no_symbols).
        """
        dataset = getattr(self.train_dataloader, 'dataset', None)
        if dataset is None or not hasattr(dataset, 'set_epoch_symbol_maps'):
            return

        no_symbols = self.config.symbol_config.no_symbols
        per_instance = (self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE)

        if no_symbols:
            dataset.set_epoch_symbol_maps({}, no_symbols=True)
        elif per_instance:
            pool_list = self.symbol_manager._pregenerated_mappings or [self.symbol_manager.fixed_mappings]
            ds_pool: Dict[str, List] = {}
            for full_map in pool_list:
                for ds_name, label_map in full_map.items():
                    ds_pool.setdefault(ds_name, []).append(label_map)
            dataset.set_epoch_symbol_maps({}, no_symbols=False, per_instance=True, pool=ds_pool)
        else:
            maps = self.symbol_manager.get_symbols_for_epoch(epoch, force_new_symbols=False)
            dataset.set_epoch_symbol_maps(maps, no_symbols=False, per_instance=False)

        logging.info(
            "Symbol maps pushed to dataset workers (epoch=%d no_symbols=%s per_instance=%s)",
            epoch + 1, no_symbols, per_instance,
        )

    def _apply_symbol_replacement(self, batch: Dict[str, Any], epoch: int, batch_idx: int) -> Dict[str, Any]:
        ds_name_str, relevant_labels = get_dataset_info(batch, self.symbol_manager.original_labels)

        # Surrogate regime for this instance. Both regimes are surrogate tuning: the
        # flipped one assigns permuted REAL label words, never the canonical label, so
        # no instruction-tuning targets enter training.
        _mix = getattr(self.config.symbol_config, "surrogate_mix", 0.0)
        _use_flip = self.config.symbol_config.swap_labels or (_mix > 0 and random.random() < _mix)

        if _use_flip:
            # From the mixture, the flipped regime must use permuted REAL words.
            _from_mix = not self.config.symbol_config.swap_labels
            p_mappings = c_mappings = self._get_swap_mappings(
                ds_name_str, relevant_labels, epoch, batch_idx, force_real_labels=_from_mix)
        else:
            force_new = self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE
            all_ds_mappings = self.symbol_manager.get_symbols_for_epoch(epoch, force_new_symbols=force_new)
            p_mappings = c_mappings = all_ds_mappings.get(ds_name_str) or all_ds_mappings.get("") or {}
            if batch_idx < 2:
                logging.info("Symbol mapping (epoch=%d batch=%d dataset=%s): %s", epoch + 1, batch_idx, ds_name_str, p_mappings)

        replaced = self.symbol_manager.replace_symbols_in_batch(batch, prompt_mappings=p_mappings, completion_mappings=c_mappings)
        return tokenize_batch_with_audio(replaced, self.processor, replaced.get("completion"))

    # ------------------------------------------------------------------
    # Surface-form consistency objective
    # ------------------------------------------------------------------

    def _log_train_prompt(self, batch: Dict[str, Any], epoch: int, batch_idx: int,
                          tag: str = "") -> None:
        """Decode and log the first instance's full prompt and completion."""
        ids = batch["input_ids"][0]
        p_len = int(batch["prompt_length"][0])
        full_prompt = self.tokenizer.decode(ids[:p_len], skip_special_tokens=False)
        completion_ids = [t for t in ids[p_len:].tolist()
                          if t != -100 and t != self.tokenizer.pad_token_id]
        completion_txt = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        logging.info("=" * 60)
        logging.info("TRAIN%s epoch=%d batch=%d  prompt_len=%d  seq_len=%d",
                     f" [{tag}]" if tag else "", epoch + 1, batch_idx, p_len, len(ids))
        logging.info("--- Full prompt ---\n%s", full_prompt)
        logging.info("--- Completion text : %r", completion_txt)
        logging.info("--- Completion IDs  : %s", completion_ids)
        logging.info("=" * 60)

    @staticmethod
    def _new_cons_run() -> Dict[str, float]:
        return {"two": 0, "nll_a": 0.0, "nll_b": 0.0, "d": 0.0, "d_n": 0,
                "single": 0, "nll_single": 0.0}

    def _cons_track(self, parts: Dict[str, float], epoch: int, batch_idx: int) -> None:
        """Accumulate one step and log running means every _cons_log_every batches."""
        r = self._cons_run
        if parts.get("single"):
            r["single"] += 1
            r["nll_single"] += parts["nll_a"]
        else:
            r["two"] += 1
            r["nll_a"] += parts["nll_a"]
            r["nll_b"] += parts["nll_b"]
            if parts.get("applied") or parts.get("measured"):
                r["d"] += parts["consistency"]
                r["d_n"] += 1
        if (batch_idx + 1) % self._cons_log_every:
            return
        view_a, view_b = self._cons_views
        two = max(r["two"], 1)
        logging.info(
            "CONS-RUN epoch=%d batch=%d | two-view n=%d nll[%s]=%.4f nll[%s]=%.4f "
            "D=%.4f%s | single-pass n=%d nll=%.4f",
            epoch + 1, batch_idx + 1, r["two"], view_a, r["nll_a"] / two,
            view_b, r["nll_b"] / two, r["d"] / max(r["d_n"], 1),
            " (measured, not trained)" if self._cons_weight == 0 else "",
            r["single"], r["nll_single"] / max(r["single"], 1))
        self._cons_run = self._new_cons_run()

    def _build_view(self, batch: Dict[str, Any], view: str, ds_name_str: str,
                    relevant_labels: list, epoch: int, batch_idx: int,
                    reuse_mapping: Dict[str, str] = None):
        """Tokenised batch for one legend, plus its {label: surface} mapping.

        `same` reuses `reuse_mapping` (view A's surfaces), so with the legend
        shuffled the two views differ only in line order.

        An empty mapping means the canonical labels are used as-is. The prompt
        dicts are shallow-copied for that case because tokenize_batch_with_audio
        writes `_audio` and `_precomputed` into them in place, and the two views
        must not share those objects.
        """
        if view == "orig":
            mapping: Dict[str, str] = {}
            staged = dict(batch)
            staged["prompt"] = [dict(p) if isinstance(p, dict) else p
                                for p in batch.get("prompt", [])]
        elif view == "flip":
            mapping = self._get_swap_mappings(
                ds_name_str, relevant_labels, epoch, batch_idx, force_real_labels=True)
            staged = self.symbol_manager.replace_symbols_in_batch(batch, mapping, mapping)
        elif view == "sym":
            all_ds = self.symbol_manager.get_symbols_for_epoch(epoch, force_new_symbols=True)
            mapping = all_ds.get(ds_name_str) or all_ds.get("") or {}
            staged = self.symbol_manager.replace_symbols_in_batch(batch, mapping, mapping)
        elif view == "same":
            mapping = dict(reuse_mapping or {})
            if mapping:
                staged = self.symbol_manager.replace_symbols_in_batch(batch, mapping, mapping)
            else:
                staged = dict(batch)
                staged["prompt"] = [dict(p) if isinstance(p, dict) else p
                                    for p in batch.get("prompt", [])]
        else:
            raise ValueError(f"unknown consistency view {view!r}; expected sym|flip|orig|same")

        if self._cons_shuffle:
            staged = _shuffle_legend_in_batch(staged)
        tokenized = tokenize_batch_with_audio(staged, self.processor, staged.get("completion"))
        return tokenized, mapping

    def _cons_candidate_ids(self, ds_name_str: str, relevant_labels: list,
                            mapping: Dict[str, str]):
        """First-token ids of this view's surfaces, ordered by definition.

        `relevant_labels` is sorted, so column k is the same definition in both
        views. Returns None when two surfaces share a first token, which makes
        the divergence blind to the difference between those definitions.
        """
        surfaces = [mapping.get(l, l) for l in relevant_labels]
        if not surfaces:
            return None
        ids = candidate_first_token_ids(surfaces, self.tokenizer)
        collisions = check_collisions(ids, surfaces)
        if collisions:
            # This drawing only. The dataset stays eligible and later drawings
            # normally carry the term, so this is counted, not treated as a skip.
            self._cons_stats["collided"] += 1
            if ds_name_str not in self._cons_collided_ds:
                self._cons_collided_ds.add(ds_name_str)
                logging.warning(
                    "Consistency term dropped for one drawing on %s: surfaces share a "
                    "first token %s. This step keeps both NLL terms without the "
                    "divergence; later drawings are unaffected.",
                    ds_name_str, collisions)
            return None
        return ids

    def _cons_dataset_eligible(self, ds_name_str: str, relevant_labels: list) -> bool:
        """Can this dataset carry the divergence term? Decided once, then cached.

        Two disqualifiers:
          * multi-label (HVB) — the answer is a set, so a distribution over K
            definitions is the wrong object no matter how it is tokenised.
          * colliding canonical surfaces, when `orig` is one of the views — HVB's
            answer_*, question_* and statement_* groups each share a first token,
            so those definitions would share a logit.

        An ineligible dataset takes a single forward pass through the ordinary
        `_apply_symbol_replacement` path, so its regime follows `surrogate_mix`:
        0 gives pure Sym, 0.5 gives the per-instance mixture. One pass rather
        than two, since the second buys nothing without the divergence term.
        """
        if ds_name_str in self._cons_eligible:
            return self._cons_eligible[ds_name_str]

        reason = None
        for dt_enum, ds_cfg in DATASET_CONFIGS.items():
            if dt_enum.value == ds_name_str:
                if getattr(ds_cfg, "is_multi_label", False):
                    reason = "multi-label target"
                break

        if reason is None and "orig" in self._cons_views and relevant_labels:
            ids = candidate_first_token_ids(relevant_labels, self.tokenizer)
            collisions = check_collisions(ids, relevant_labels)
            if collisions:
                reason = f"canonical surfaces share a first token: {collisions}"

        eligible = reason is None
        self._cons_eligible[ds_name_str] = eligible
        if not eligible:
            self._cons_skipped_ds.add(ds_name_str)
            logging.warning(
                "Consistency term off for %s (%s). Single pass via the standard "
                "regime path (surrogate_mix=%.2f).",
                ds_name_str, reason,
                getattr(self.config.symbol_config, "surrogate_mix", 0.0) or 0.0)
        return eligible

    def _consistency_step(self, raw_batch: Dict[str, Any], epoch: int, batch_idx: int):
        """Two legends, two forward passes, one coupled loss.

        Returns (view_a_batch, loss, parts). The two passes are deliberately in
        the same step with no optimizer.step() between them: the divergence term
        is a function of both outputs at once and cannot be expressed otherwise.

        Datasets that cannot carry the term take a single pass instead.
        """
        ds_name_str, relevant_labels = get_dataset_info(
            raw_batch, self.symbol_manager.original_labels)
        view_a, view_b = self._cons_views

        if not self._cons_dataset_eligible(ds_name_str, relevant_labels):
            src = _shuffle_legend_in_batch(raw_batch) if self._cons_shuffle else raw_batch
            batch_a = self._apply_symbol_replacement(src, epoch, batch_idx)
            if batch_idx < 2:
                self._log_train_prompt(batch_a, epoch, batch_idx, tag=f"single-pass {ds_name_str}")
            batch_a = self._move_batch_to_device(batch_a)
            nll = self.model(batch_a).get("loss")
            if nll is None or torch.isnan(nll):
                return batch_a, None, None
            self._cons_stats["steps"] += 1
            self._cons_stats["single"] += 1
            return batch_a, nll, {"nll_a": float(nll.detach()), "nll_b": 0.0,
                                  "consistency": 0.0, "applied": 0.0, "single": 1.0,
                                  "total": float(nll.detach())}

        batch_a, map_a = self._build_view(raw_batch, view_a, ds_name_str,
                                          relevant_labels, epoch, batch_idx)
        batch_b, map_b = self._build_view(raw_batch, view_b, ds_name_str,
                                          relevant_labels, epoch, batch_idx,
                                          reuse_mapping=map_a)

        # Two identical drawings carry no signal (p ~1% with a 100-map pool).
        if view_a == view_b and map_a == map_b:
            for _ in range(5):
                batch_b, map_b = self._build_view(raw_batch, view_b, ds_name_str,
                                                  relevant_labels, epoch, batch_idx)
                if map_b != map_a:
                    break

        if batch_idx < 2:
            self._log_train_prompt(batch_a, epoch, batch_idx, tag=f"view A={view_a} {ds_name_str}")
            self._log_train_prompt(batch_b, epoch, batch_idx, tag=f"view B={view_b} {ds_name_str}")

        batch_a = self._move_batch_to_device(batch_a)
        batch_b = self._move_batch_to_device(batch_b)

        out_a = self.model(batch_a)
        out_b = self.model(batch_b)
        nll_a, nll_b = out_a.get("loss"), out_b.get("loss")
        if nll_a is None or nll_b is None or torch.isnan(nll_a) or torch.isnan(nll_b):
            return batch_a, None, None

        log_p_a = log_p_b = None
        ids_a = self._cons_candidate_ids(ds_name_str, relevant_labels, map_a)
        ids_b = self._cons_candidate_ids(ds_name_str, relevant_labels, map_b)
        if ids_a and ids_b and out_a.get("logits") is not None and out_b.get("logits") is not None:
            n = len(batch_a["prompt_length"])
            t_a = torch.tensor([ids_a] * n, dtype=torch.long)
            t_b = torch.tensor([ids_b] * n, dtype=torch.long)
            log_p_a = definition_log_probs(out_a["logits"], batch_a["prompt_length"], t_a,
                                           input_len=batch_a["input_ids"].size(1))
            log_p_b = definition_log_probs(out_b["logits"], batch_b["prompt_length"], t_b,
                                           input_len=batch_b["input_ids"].size(1))

        loss, parts = combined_loss(
            nll_a, nll_b, log_p_a, log_p_b,
            weight=self._cons_weight, divergence=self._cons_div)
        # Control: measure the divergence for the log, never add it to the loss.
        if self._cons_weight == 0 and log_p_a is not None and log_p_b is not None:
            with torch.no_grad():
                _d = consistency_term(log_p_a, log_p_b, self._cons_div)
            if torch.isfinite(_d):
                parts["consistency"] = float(_d)
                parts["measured"] = 1.0

        self._cons_stats["steps"] += 1
        self._cons_stats["applied"] += int(parts["applied"])
        self._cons_stats["measured"] += int(parts.get("measured", 0))
        if batch_idx < 2:
            logging.info(
                "CONSISTENCY epoch=%d batch=%d ds=%s views=%s/%s nll=%.4f/%.4f D=%.4f total=%.4f",
                epoch + 1, batch_idx, ds_name_str, view_a, view_b,
                parts["nll_a"], parts["nll_b"], parts["consistency"], parts["total"])
            logging.info("  surfaces[%s]: %s", view_a,
                         [map_a.get(l, l) for l in relevant_labels])
            logging.info("  surfaces[%s]: %s", view_b,
                         [map_b.get(l, l) for l in relevant_labels])
        return batch_a, loss, parts

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self._swap_cache = {}  # reset per-epoch swap assignments
        total_loss, num_batches = 0.0, 0
        accumulation_steps = self.config.lora_config.gradient_accumulation_steps

        # For non-swap modes: push maps onto dataset so workers do substitution.
        # Must happen before progress_bar is created — that's when workers fork.
        # Mixing decides the regime per instance, so the maps cannot be pushed to the
        # dataloader workers ahead of time; substitute in the main thread as swap mode does.
        # The consistency objective builds both legends in the main thread from the
        # raw prompt, so workers must not pre-substitute or view B would be a
        # relabelling of view A rather than of the canonical prompt.
        # DG-Trap renames real labels per instance in the main thread, from the canonical prompt.
        _use_worker_symbols = not (self.config.symbol_config.swap_labels
                                   or getattr(self.config.symbol_config, "surrogate_mix", 0.0) > 0
                                   or self._cons_on
                                   or self._trap_on)
        if _use_worker_symbols:
            self._set_epoch_symbol_maps_on_dataset(epoch)

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)

        _PROFILE = epoch == 0  # only profile first epoch
        _t_batch_start = time.time()

        _torch_prof = None
        if _PROFILE and getattr(self.config, "torch_profile", False):
            import torch.profiler as _tp
            _prof_out = os.path.join(self.config.output_dir, f"{self.config.run_name}_torchprof")
            _torch_prof = _tp.profile(
                activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
                on_trace_ready=_tp.tensorboard_trace_handler(_prof_out),
                schedule=_tp.schedule(wait=2, warmup=2, active=20, repeat=1),
            )
            _torch_prof.start()
            logging.info("PyTorch profiler started → %s", _prof_out)

        for batch_idx, raw_batch in enumerate(progress_bar):
            try:
                _t0 = time.time()
                t_dataloader = _t0 - _t_batch_start  # time waiting for DataLoader

                if "completion" not in raw_batch or not raw_batch["completion"]:
                    for key in ["label", "true_label", "target"]:
                        if key in raw_batch and raw_batch[key]: raw_batch["completion"] = raw_batch[key]; break

                _t1 = time.time()
                if self._trap_on:
                    # One or two passes in one step; _trap_step owns the forwards.
                    updated_batch, loss, _tparts = self._trap_step(raw_batch, epoch, batch_idx)
                    if loss is None:
                        _t_batch_start = time.time()
                        continue
                    (loss / accumulation_steps).backward()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                    total_loss += loss.item()
                    num_batches += 1
                    progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.6f}",
                                              "kind": _tparts["kind"]})
                    _t_batch_start = time.time()
                    if _torch_prof is not None:
                        _torch_prof.step()
                    continue
                if self._cons_on:
                    # Two legends in one step; _consistency_step owns the forwards.
                    updated_batch, loss, _cparts = self._consistency_step(
                        raw_batch, epoch, batch_idx)
                    _t2 = _t3 = _t4 = time.time()
                    t_tokenize = t_to_device = 0.0
                    t_forward = _t4 - _t1
                    if loss is None:
                        _t_batch_start = time.time()
                        continue
                    (loss / accumulation_steps).backward()
                    torch.cuda.synchronize()
                    t_backward = time.time() - _t4
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                    total_loss += loss.item()
                    num_batches += 1
                    progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.6f}",
                                              "D": f"{_cparts['consistency']:.3f}"})
                    self._cons_track(_cparts, epoch, batch_idx)
                    _t_batch_start = time.time()
                    if _torch_prof is not None:
                        _torch_prof.step()
                    continue

                if _use_worker_symbols:
                    if "input_ids" in raw_batch:
                        # Flamingo (File 3): worker tokenized — main thread is no-op.
                        updated_batch = raw_batch
                    else:
                        # Qwen (worker tokenization not yet applied): tokenize in main thread.
                        updated_batch = tokenize_batch_with_audio(raw_batch, self.processor, raw_batch.get("completion"))
                else:
                    # swap_labels: substitution + tokenization in main thread.
                    updated_batch = self._apply_symbol_replacement(raw_batch, epoch, batch_idx)
                _t2 = time.time()
                t_tokenize = _t2 - _t1

                if self._epoch_log_count < 2:
                    self._log_train_prompt(updated_batch, epoch, batch_idx)
                    self._epoch_log_count += 1

                updated_batch = self._move_batch_to_device(updated_batch)
                torch.cuda.synchronize()
                _t3 = time.time()
                t_to_device = _t3 - _t2

                outputs = self.model(updated_batch)
                torch.cuda.synchronize()
                _t4 = time.time()
                t_forward = _t4 - _t3

                loss = outputs.get("loss")
                if loss is None or torch.isnan(loss): continue
                (loss / accumulation_steps).backward()
                torch.cuda.synchronize()
                _t5 = time.time()
                t_backward = _t5 - _t4

                if _PROFILE and batch_idx < 50:
                    seq_len = updated_batch["input_ids"].shape[-1]
                    logging.info(
                        "TIMING batch=%d seq_len=%d | dataloader=%.3fs tokenize=%.3fs "
                        "to_device=%.3fs forward=%.3fs backward=%.3fs total=%.3fs",
                        batch_idx, seq_len,
                        t_dataloader, t_tokenize, t_to_device, t_forward, t_backward,
                        time.time() - _t_batch_start,
                    )

                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.6f}"})
                _t_batch_start = time.time()
                if _torch_prof is not None:
                    _torch_prof.step()
            except Exception as exc:
                logging.error(f"Batch {batch_idx} failed: {exc}")
                if self._trap_on:
                    # A bug in the trap step would otherwise silently skip every batch.
                    import traceback
                    _trap_fail = getattr(self, "_trap_fail", 0) + 1
                    self._trap_fail = _trap_fail
                    if _trap_fail <= 5:
                        logging.error("DG-Trap step traceback:\n%s", traceback.format_exc())
                    if _trap_fail >= 20 and _trap_fail > 0.2 * (batch_idx + 1):
                        raise RuntimeError(
                            f"DG-Trap: {_trap_fail} of {batch_idx + 1} batches failed; aborting") from exc

        # Accumulation Cleanup
        if num_batches > 0 and (num_batches % accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        if _torch_prof is not None:
            _torch_prof.stop()
            logging.info("PyTorch profiler saved.")
        progress_bar.close()
        if self._cons_on and self._cons_stats["steps"]:
            _s = self._cons_stats["steps"]
            _a = self._cons_stats["applied"]
            _1 = self._cons_stats["single"]
            _c = self._cons_stats["collided"]
            _two = max(_s - _1, 1)
            if self._cons_weight == 0:
                _m = self._cons_stats["measured"]
                _what = "CONTROL (weight 0): D measured on %d/%d two-view steps (%.1f%%), not trained"
            else:
                _m = _a
                _what = "Consistency: %d/%d two-view steps got the divergence (%.1f%%)"
            logging.info(
                _what + "; %d lost it to a first-token collision; %d single-pass steps. "
                "Never eligible: %s. Had a colliding drawing: %s.",
                _m, _two, 100.0 * _m / _two, _c, _1,
                sorted(self._cons_skipped_ds) or "none",
                sorted(self._cons_collided_ds) or "none")
            self._cons_stats = {"steps": 0, "applied": 0, "measured": 0, "single": 0, "collided": 0}
            self._cons_run = self._new_cons_run()
        if self._trap_on:
            logging.info("TRAP-EPOCH epoch=%d failed_batches=%d %s", epoch + 1,
                         getattr(self, "_trap_fail", 0), self._trap_summary(self._trap_totals))
            self._trap_totals = self._new_trap_run()
            self._trap_run = self._new_trap_run()
            self._trap_fail = 0
        return total_loss / max(num_batches, 1)

    def _build_current_symbol_map(self, epoch: int) -> Dict[str, Dict[str, str]]:
        """Build {ds_name: {label: symbol}} for 'fixed' validation — always reflects current training state."""
        return self.symbol_manager.get_symbols_for_epoch(epoch)

    def _run_validation(self, epoch: int) -> Dict[str, Any]:
        # symbol_manager covers every label of every configured dataset, so the map
        # needs no extension for cross-task validation datasets.
        symbol_map = self._build_current_symbol_map(epoch)
        return self.validator.run_comprehensive_validation(
            model=self.model,
            val_dataloader=self.val_dataloader,
            epoch=epoch,
            symbol_map=symbol_map,
        )

    def _save_checkpoint(self, epoch: int, checkpoint_type: str):
        checkpoint_dir = self.config.get_training_output_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"lora_epoch{epoch + 1}_{checkpoint_type}.pt")
        trainable_state = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        checkpoint_data = {
            "model_state": trainable_state,
            # optimizer_state intentionally NOT saved — it's never loaded (no resume path)
            # and unused at inference; storing it ~tripled checkpoint size.
            "optimizer_state": None,
            "config": self.config,
            "symbol_mappings": {
                "current_epoch_mappings": self.symbol_manager.get_symbols_for_epoch(epoch) if self.symbol_manager else {},
                "original_labels": self.symbol_manager.original_labels if self.symbol_manager else [],
                "symbol_token_size": self.symbol_manager.token_size if self.symbol_manager else 2,
                "dynamic_per_epoch": self.symbol_manager.dynamic_per_epoch if self.symbol_manager else False,
            },
        }
        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Saved checkpoint: {os.path.basename(checkpoint_path)}")

    def run_complete_training(self) -> Dict[str, Any]:
        total_epochs = self.config.lora_config.epochs
        history = []

        if self.config.validate_before_training:
            logging.info("Running baseline validation before training (epoch 0)")
            baseline_scores = self._run_validation(epoch=-1)
            logging.info(f"Baseline validation (pre-training): {baseline_scores}")
            history.append({"epoch": 0, "train_loss": None, "validation": baseline_scores})

        self._setup_lora_optimizer()
        self.optimizer.zero_grad(set_to_none=True)
        logging.info("Starting LoRA training for %d epochs", total_epochs)

        for epoch in range(total_epochs):
            self._epoch_log_count = 0
            logging.info(f"Epoch {epoch + 1}/{total_epochs}")
            epoch_loss = self._train_one_epoch(epoch)
            validation_scores = self._run_validation(epoch)
            logging.info(f"Epoch {epoch + 1} loss: {epoch_loss:.6f}")
            logging.info(f"Epoch {epoch + 1} validation: {validation_scores}")

            history.append({"epoch": epoch + 1, "train_loss": epoch_loss, "validation": validation_scores})
            if self.config.checkpoint_frequency > 0 and (epoch + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch, "periodic")

        # 1. Consolidated Validation Summary Table
        logging.info("=" * 30 + " Consolidated Validation Summary " + "=" * 30)
        all_ds_names, all_mode_names = [], []
        for entry in history:
            modes = entry.get("validation", {}).get("all_modes", {})
            for mode, datasets in modes.items():
                if mode not in all_mode_names: all_mode_names.append(mode)
                for ds in datasets.keys():
                    if ds not in all_ds_names: all_ds_names.append(ds)

        if all_ds_names:
            train_ds = [n for n in all_ds_names if n in self.train_dataset_names]
            val_ds = [n for n in all_ds_names if n not in self.train_dataset_names]
            header = f"{'Epoch':<8} | {'Mode':<12}"
            for ds in all_ds_names:
                col = f"{ds}(T)" if ds in train_ds else ds
                header += f" | {col:<15}"
            if len(train_ds) > 1: header += " | Avg (train)"
            if len(val_ds) > 1: header += " | Avg (val)"
            logging.info(header)
            logging.info("-" * len(header))
            for i, entry in enumerate(history):
                modes = entry.get("validation", {}).get("all_modes", {})
                for mode in all_mode_names:
                    datasets = modes.get(mode, {})
                    row = f"{entry['epoch']:<8} | {mode:<12}"
                    t_scores, v_scores = [], []
                    for ds in all_ds_names:
                        s = datasets.get(ds, {}).get("score")
                        row += f" | {s:<15.6f}" if s is not None else " | -"
                        if s is not None: (t_scores if ds in train_ds else v_scores).append(s)
                    if len(train_ds) > 1: row += f" | {sum(t_scores)/len(t_scores):<12.6f}" if t_scores else " | -"
                    if len(val_ds) > 1: row += f" | {sum(v_scores)/len(v_scores):<12.6f}" if v_scores else " | -"
                    logging.info(row)
                if i < len(history) - 1: logging.info("-" * len(header))
        logging.info("=" * 80)

        self._save_checkpoint(total_epochs - 1, "final")
        return {"history": history}
