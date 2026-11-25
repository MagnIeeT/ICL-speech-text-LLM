"""CPU test suite for DG-Trap (models/symbolAdapter/trap_loss.py + _trap_step).

Uses the real Qwen2-Audio tokenizer (local HF cache), real canonical training
prompts (tests/fixtures/trap_prompts.json, taken from an IT training log), the
real label-renaming code, and a small mock model that reproduces Qwen2-Audio's
in-forward expansion of <|AUDIO|> into many rows (transformers 4.45), so the
logit-alignment logic is tested against the real failure mode.

Run:  HF_HUB_OFFLINE=1 python tests/test_dg_trap.py
"""
import json
import logging
import math
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from config.train_config.training_configs import TrainingConfig, SymbolConfig, DataConfig, ModelType  # noqa: E402
from dataload.qwen_processor import QwenProcessor  # noqa: E402
from models.symbolAdapter.consistency_loss import definition_log_probs  # noqa: E402
from models.symbolAdapter.symbol_manager import SymbolManager  # noqa: E402
from models.symbolAdapter.symbol_training import SymbolTrainingOrchestrator  # noqa: E402
from models.symbolAdapter.trap_loss import (  # noqa: E402
    LEGEND_LINE, check_alignment, drop_words, edit_legend, first_divergence,
    load_deciding_index, map_prompt_text, remove_words, swap_completion_slot,
    token_logprobs, trap_token_diff)
from models.symbolAdapter.validation import ValidationManager  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
torch.manual_seed(0)
AUDIO_EXPANSION = 57          # rows one <|AUDIO|> token becomes inside the mock forward
PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"PASS  {name}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
FIX = json.load(open("tests/fixtures/trap_prompts.json"))
# Multi-label HVB instance (same prompt, two gold labels, as in real HVB data).
FIX["hvb_multi"] = {"prompt": FIX["hvb"]["prompt"], "completion": "backchannel, answer_dis", "ds": "hvb"}
for k in ("cremad", "hvb", "minds14_en"):
    FIX[k]["ds"] = k
INDEX = load_deciding_index("config/data_config/deciding_words.json")
TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
AUDIO_ID = TOK.convert_tokens_to_ids("<|AUDIO|>")


class _Proc:
    """Real QwenProcessor.tokenize_batch, without loading the audio feature extractor."""
    def __init__(self, tok):
        self.tokenizer, self.max_length = tok, 4096

    def tokenize_batch(self, prompts, completions=None, padding_side="right"):
        return QwenProcessor.tokenize_batch(self, prompts, completions, padding_side)


class MockQwenAudio(nn.Module):
    """Causal toy LM with Qwen2-Audio's audio-token expansion and its loss convention."""
    def __init__(self, vocab, dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.audio_rows = nn.Parameter(torch.randn(AUDIO_EXPANSION, dim) * 0.1)
        self.head = nn.Linear(dim, vocab)

    def forward(self, batch):
        ids = batch["input_ids"].long()
        labels = ids.clone()
        labels[0, : int(batch["prompt_length"][0])] = -100
        h = self.emb(ids)[0]
        pos = (ids[0] == AUDIO_ID).nonzero().flatten().tolist()
        if pos:                                   # expand like HF _merge_input_ids_with_audio_features
            p = pos[0]
            h = torch.cat([h[:p], self.audio_rows, h[p + 1:]], 0)
            labels = torch.cat([labels[0, :p], torch.full((AUDIO_EXPANSION,), -100), labels[0, p + 1:]])[None]
        ctx = torch.cumsum(h, 0) / torch.arange(1, h.size(0) + 1)[:, None]   # causal context
        logits = self.head(h + ctx)[None]
        loss = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100)
        return {"loss": loss, "logits": logits}


def make_symbol_manager():
    sm = SymbolManager.__new__(SymbolManager)
    sm.original_labels = []
    sm.no_symbols, sm.swap_labels = True, False
    return sm


def make_orchestrator(**trap):
    sc = SymbolConfig(trap_enabled=True, **trap)
    cfg = TrainingConfig(model_type=ModelType.QWEN, symbol_config=sc, data_config=DataConfig(batch_size=1, num_examples=0),
                         device="cpu", seed=0, output_dir="/tmp", checkpoint_dir="/tmp",
                         metrics_dir="/tmp", logs_dir="/tmp")
    o = SymbolTrainingOrchestrator.__new__(SymbolTrainingOrchestrator)
    o.config, o.tokenizer, o.processor = cfg, TOK, _Proc(TOK)
    o.symbol_manager = make_symbol_manager()
    o.model = MockQwenAudio(len(TOK))
    o._cons_on = False
    o._init_trap(cfg)
    return o


def raw_batch(key):
    f = FIX[key]
    return {"prompt": [f["prompt"]], "completion": [f["completion"]], "dataset_type": [f["ds"]],
            "text": ["x"], "audio": None}


def legend(text):
    return {m.group(2).strip(): m.group(4) for m in LEGEND_LINE.finditer(text)}


# ---------------------------------------------------------------------------
# 1. Text operations
# ---------------------------------------------------------------------------
def test_text_ops():
    rng = random.Random(0)
    d = "the customer asks where a bank branch is located"
    dec = {"located", "branch"}
    for _ in range(200):
        new, n = drop_words(d, dec, 0.9, rng)
        assert "located" in new and "branch" in new and new.split(), new
    assert drop_words(d, dec, 0.0, rng) == (d, 0)
    new, n = remove_words(d, dec)
    assert "branch" not in new and "located" not in new and n == 2, new
    new, n = remove_words('Brief verbal/textual feedback (like "uh-huh", "mm-hmm")', {"uh", "huh"})
    assert "uh-huh" not in new and n == 1, new
    prompt = FIX["hvb"]["prompt"]
    out = edit_legend(prompt, lambda name, dd: dd.upper())
    assert "- Consider the banking context when analyzing" in out          # guideline untouched
    assert len(legend(out)) == len(legend(prompt)) == 18
    p_obj = {"conversation": [{"role": "system", "content": [{"type": "text", "text": "a"}]}]}
    new_obj = map_prompt_text(p_obj, lambda t: t + "b")
    assert p_obj["conversation"][0]["content"][0]["text"] == "a" and \
        new_obj["conversation"][0]["content"][0]["text"] == "ab"
    assert swap_completion_slot("backchannel, answer_dis", "answer_dis", "thanks") == "backchannel, thanks"
    assert swap_completion_slot("happy", "sad", "x") is None
    ok("text operations (dropout keeps deciding words, removal, legend-only edits, no mutation)")


def test_index_covers_prompts():
    for key in ("cremad", "hvb", "minds14_en"):
        defs = legend(FIX[key]["prompt"])
        miss = [n for n, dd in defs.items() if " ".join(dd.lower().split()) not in INDEX[key]]
        assert not miss, (key, miss)
        assert all(INDEX[key][" ".join(dd.lower().split())]["label"] == n for n, dd in defs.items())
    ok("deciding-word index matches every legend line of the real prompts (3 datasets)")


# ---------------------------------------------------------------------------
# 2. Renaming with the real replacement code
# ---------------------------------------------------------------------------
def test_renaming():
    sm = make_symbol_manager()
    b = raw_batch("hvb_multi")
    before = legend(b["prompt"][0])
    m = {"answer_dis": "thanks", "thanks": "answer_dis"}
    out = sm.replace_symbols_in_batch(b, m, m)
    after = legend(out["prompt"][0])
    assert after["thanks"] == before["answer_dis"] and after["answer_dis"] == before["thanks"]
    assert after["question_check"] == before["question_check"]              # prefix label untouched
    assert out["completion"][0] == "backchannel, thanks"
    assert b["completion"][0] == "backchannel, answer_dis"                   # original not mutated
    ok("goldswap via replace_symbols_in_batch: names swapped, definitions fixed, completion renamed")


# ---------------------------------------------------------------------------
# 3. Logit alignment under audio expansion (and the E2 indexing bug)
# ---------------------------------------------------------------------------
def test_alignment():
    model = MockQwenAudio(len(TOK))
    proc = _Proc(TOK)
    for key in ("cremad", "hvb_multi", "minds14_en"):
        b = raw_batch(key)
        tb = proc.tokenize_batch(b["prompt"], b["completion"])
        tb["prompt_length"] = tb["prompt_length"]
        out = model(tb)
        c_ids = TOK(f"{b['completion'][0]}{TOK.eos_token}", add_special_tokens=False).input_ids
        assert tb["input_ids"][0][-len(c_ids):].tolist() == c_ids
        assert out["logits"].size(1) == tb["input_ids"].size(1) - 1 + AUDIO_EXPANSION
        err = check_alignment(out["logits"], c_ids, out["loss"])
        assert err < 1e-5, (key, err)
        # E2's indexing: row prompt_length-1 on the expanded logits is NOT the answer row.
        plen = int(tb["prompt_length"][0])
        e2_row = out["logits"][0, plen - 1]
        true_row = out["logits"][0, out["logits"].size(1) - len(c_ids) - 1]
        assert not torch.allclose(e2_row, true_row), "expected E2 row to differ under expansion"
        lp = definition_log_probs(out["logits"], [plen], torch.tensor([[c_ids[0]]]))
        assert lp.shape == (1, 1)
    ok(f"end-aligned completion rows reproduce the model NLL (err<1e-5) with {AUDIO_EXPANSION}x audio "
       "expansion; prompt_length-1 indexing (E2) reads a different row")


def test_trap_diff_shared_prefix():
    model = MockQwenAudio(len(TOK))
    proc = _Proc(TOK)
    b = {"prompt": [FIX["hvb"]["prompt"]], "completion": ["question_general"]}
    tb = proc.tokenize_batch(b["prompt"], b["completion"])
    out = model(tb)
    g = TOK("question_general" + TOK.eos_token, add_special_tokens=False).input_ids
    t = TOK("question_check" + TOK.eos_token, add_special_tokens=False).input_ids
    k = first_divergence(g, t)
    assert k is not None and k > 0, (g, t)                                   # shared "question" prefix
    diff, info = trap_token_diff(out["logits"], g, t)
    rows = out["logits"][0, -len(g) - 1:-1]
    lp = F.log_softmax(rows[k].float(), -1)
    assert abs(float(diff) - float(lp[g[k]] - lp[t[k]])) < 1e-5
    assert TOK.decode([info["gold_tok"]]) != TOK.decode([info["trap_tok"]])
    ok(f"trap diff taken at first differing token k={k} "
       f"({TOK.decode([g[k]])!r} vs {TOK.decode([t[k]])!r}) for shared-prefix labels")


# ---------------------------------------------------------------------------
# 4. Full _trap_step
# ---------------------------------------------------------------------------
def test_trap_step_end_to_end():
    o = make_orchestrator(trap_cond_frac=1.0, trap_lambda=0.5, trap_cond_lambda=0.5)
    seen = Counter()
    for step in range(60):
        for key in ("cremad", "hvb", "hvb_multi", "minds14_en"):
            o.model.zero_grad()
            _, loss, parts = o._trap_step(raw_batch(key), 0, step + 2)
            assert loss is not None and math.isfinite(float(loss)), (key, parts)
            loss.backward()
            gn = sum(p.grad.abs().sum() for p in o.model.parameters() if p.grad is not None)
            assert float(gn) > 0
            assert parts["align_err"] < 1e-4, parts
            seen[(key, parts["kind"])] += 1
            if parts["kind"] == "goldswap":
                assert "trap_diff" in parts, parts
            if "cond_diff" in parts:
                assert parts["cond_removed"], parts
                seen["cond"] += 1
    assert all(seen[(k, kind)] > 0 for k in ("cremad", "hvb_multi", "minds14_en")
               for kind in ("orig", "goldswap", "pairswap")), seen
    assert seen["cond"] > 100, seen
    tot = o._trap_totals
    assert tot["unmatched_lines"] == 0 and tot["skip_no_gold"] == 0, tot
    ok(f"_trap_step: 240 steps, all kinds on all datasets incl. multi-label HVB, finite loss, "
       f"gradients flow, cond term on {seen['cond']} steps, no unmatched legend lines")


def test_views_and_dropout():
    """Inspect the actual prompts the step builds: renaming, dropout, damaged gold line."""
    o = make_orchestrator(trap_p_orig=0, trap_p_goldswap=1, trap_p_pairswap=0, trap_cond_frac=1.0,
                          trap_dropout_max=0.5)
    built = []
    orig_build = o._trap_build

    def spy(rb, fn, mapping):
        tb = orig_build(rb, fn, mapping)
        built.append((rb["prompt"][0], fn(rb["prompt"][0]), mapping, tb["completion"][0]))
        return tb
    o._trap_build = spy
    canon = legend(FIX["minds14_en"]["prompt"])
    for step in range(30):
        built.clear()
        _, _, parts = o._trap_step(raw_batch("minds14_en"), 0, step + 2)
        (_, full_txt, mapping, comp), (_, dam_txt, mapping2, comp2) = built
        assert mapping == mapping2 and comp == comp2
        g, partner = "address", comp
        assert mapping == {g: partner, partner: g}
        full_after = legend(full_txt)
        dam_after = legend(dam_txt)
        info = INDEX["minds14_en"][" ".join(canon[g].lower().split())]
        for name, dd in full_after.items():                     # dropout kept deciding words
            src = INDEX["minds14_en"][" ".join(canon[name].lower().split())]
            for w in src["deciding"]:
                assert w in dd.lower(), (name, w, dd)
        diff_lines = [n for n in full_after if full_after[n] != dam_after.get(n)]
        assert diff_lines == [g], diff_lines                    # only the gold line damaged
        assert not any(w in dam_after[g].lower().split() for w in info["deciding"])
    ok("built views: goldswap mapping consistent across views, dropout never removes deciding words, "
       "conditional view damages only the gold line")


def test_lambda_zero_is_data_only():
    o = make_orchestrator(trap_lambda=0.0, trap_cond_lambda=0.0, trap_p_orig=0, trap_p_goldswap=1,
                          trap_p_pairswap=0)
    calls = {"n": 0}
    fwd = o.model.forward

    def counting(b):
        calls["n"] += 1
        return fwd(b)
    o.model.forward = counting
    for step in range(10):
        calls["n"] = 0
        _, loss, parts = o._trap_step(raw_batch("cremad"), 0, step + 2)
        assert calls["n"] == 1 and abs(float(loss) - parts["nll"]) < 1e-6 and "trap_diff" in parts
    ok("A0 (lambdas 0): loss == NLL exactly, one forward pass, trap diff still measured for the log")


def test_proportions_and_partners():
    o = make_orchestrator(trap_cond_lambda=0.0)
    kinds, src = Counter(), Counter()
    for step in range(1500):
        _, _, parts = o._trap_step(raw_batch("cremad"), 0, step + 2)
        kinds[parts["kind"]] += 1
    tot = o._trap_totals
    for k in ("orig", "goldswap", "pairswap"):
        assert abs(kinds[k] / 1500 - 1 / 3) < 0.05, kinds
    nn_share = tot["nn"] / max(tot["nn"] + tot["random"], 1)
    assert 0.40 < nn_share < 0.60, nn_share          # happy's neighbour (sad) is non-gold: P(nn) = nn_frac = 0.5
    ok(f"naming proportions {dict(kinds)} ~1/3 each; nearest-neighbour partner share {nn_share:.2f}")


def test_cond_random_mode():
    o = make_orchestrator(trap_cond_frac=1.0, trap_cond_mode="random")
    n = 0
    for step in range(20):
        _, loss, parts = o._trap_step(raw_batch("hvb"), 0, step + 2)
        n += "cond_diff" in parts
    assert n > 0
    ok("ablation A5 (cond_mode=random) runs")


def test_guards():
    for bad in ({"trap_lambda": 0.5, "swap": True}, {"bs": 2}):
        try:
            sc = SymbolConfig(trap_enabled=True, swap_labels=bad.get("swap", False))
            cfg = TrainingConfig(model_type=ModelType.QWEN, symbol_config=sc,
                                 data_config=DataConfig(batch_size=bad.get("bs", 1), num_examples=0), device="cpu",
                                 output_dir="/tmp", checkpoint_dir="/tmp", metrics_dir="/tmp", logs_dir="/tmp")
            o = SymbolTrainingOrchestrator.__new__(SymbolTrainingOrchestrator)
            o._cons_on = False
            o._init_trap(cfg)
            raise AssertionError(f"guard did not fire for {bad}")
        except ValueError:
            pass
    ok("guards: swap_labels and batch_size>1 rejected")


def test_wording_variant_lookup():
    """Neighbour and deciding words follow the prompt's own wording (review finding 2)."""
    from config.data_config.master_config import DATASET_CONFIGS
    cfg = {dt.value: c for dt, c in DATASET_CONFIGS.items()}["hvb"]
    terse = cfg.prompt_variants["terse"]
    o = make_orchestrator(trap_p_orig=0, trap_p_goldswap=1, trap_p_pairswap=0, trap_cond_frac=1.0,
                          trap_dropout_max=0.0)
    seen = 0
    for step in range(20):
        b = {"prompt": [terse], "completion": ["thanks"], "dataset_type": ["hvb"], "text": ["x"], "audio": None}
        _, loss, parts = o._trap_step(b, 0, step + 2)
        if "cond_removed" in parts:
            seen += 1
            info = INDEX["hvb"][" ".join(legend(terse)["thanks"].lower().split())]
            assert info["deciding"] == {"thanks"} and parts["cond_removed"] == ["thanks"], parts
    assert seen > 0
    ok("terse wording: conditional term removes the terse definition's own deciding words")


def test_guards_text_and_fewshot():
    for dc in (DataConfig(batch_size=1, input_mode="speech_text"), DataConfig(batch_size=1, num_examples=2)):
        cfg = TrainingConfig(model_type=ModelType.QWEN, symbol_config=SymbolConfig(trap_enabled=True),
                             data_config=dc, device="cpu", output_dir="/tmp", checkpoint_dir="/tmp",
                             metrics_dir="/tmp", logs_dir="/tmp")
        o = SymbolTrainingOrchestrator.__new__(SymbolTrainingOrchestrator)
        o._cons_on = False
        try:
            o._init_trap(cfg)
            raise AssertionError("guard did not fire")
        except ValueError:
            pass
    ok("guards: text input modes and few-shot prompts rejected (renaming would leak into text)")


def test_nonfinite_skipped():
    o = make_orchestrator(trap_p_orig=0, trap_p_goldswap=1, trap_p_pairswap=0, trap_cond_frac=1.0)
    fwd = o.model.forward
    state = {"n": 0}

    def nan_second(b):
        state["n"] += 1
        out = fwd(b)
        if state["n"] % 2 == 0:                   # poison the conditional (second) pass
            out["logits"] = out["logits"] * float("nan")
        return out
    o.model.forward = nan_second
    for step in range(5):
        _, loss, parts = o._trap_step(raw_batch("minds14_en"), 0, step + 2)
        assert math.isfinite(float(loss)) and "cond_loss" not in parts, parts
    assert o._trap_totals["steps"] == 5
    ok("non-finite conditional term is skipped, total loss stays finite")


# ---------------------------------------------------------------------------
# 5. Validation 'index' mode
# ---------------------------------------------------------------------------
def test_index_mode():
    sm = make_symbol_manager()
    labels = sorted({n for n in legend(FIX["hvb"]["prompt"])})
    sm.labels_per_dataset = {"hvb": labels}
    vm = ValidationManager.__new__(ValidationManager)
    vm.symbol_manager = sm
    vm.config = TrainingConfig(model_type=ModelType.QWEN, device="cpu", output_dir="/tmp",
                               checkpoint_dir="/tmp", metrics_dir="/tmp", logs_dir="/tmp")
    vm.config.symbol_config.validation_modes = "original,index"
    modes = vm._resolve_validation_modes()
    assert modes == [("original", True, False, False, False), ("index", False, False, False, True)]
    m = vm._build_index_map()["hvb"]
    assert m[labels[0]] == "label1" and m[labels[9]] == "label10" and len(set(m.values())) == 18
    out = sm.replace_symbols_in_batch({"prompt": [FIX["hvb"]["prompt"]], "completion": [labels[0]]}, m, m)
    assert "- label1:" in out["prompt"][0] and out["completion"][0] == "label1"
    for l in labels:                                      # round trip incl. label1 vs label10..18
        assert sm.convert_symbols_back(m[l], mappings=m) == l, l
    assert sm.convert_symbols_back("label1, label12", mappings=m) == f"{labels[0]}, {labels[11]}"
    ok("validation 'index' mode: label1..label18 applied to legend and target, round-trips back exactly")


def test_cond_empty_skipped():
    """A gold definition made only of deciding words (hvb backchannel) must not become an empty
    legend line in the damaged view: the conditional term is skipped and counted instead."""
    info = INDEX["hvb"][" ".join(legend(FIX["hvb"]["prompt"])["backchannel"].lower().split())]
    assert not remove_words(legend(FIX["hvb"]["prompt"])["backchannel"], info["deciding"])[0].strip()
    for mode in ("deciding", "random"):
        o = make_orchestrator(trap_p_orig=1, trap_p_goldswap=0, trap_p_pairswap=0, trap_cond_frac=1.0,
                              trap_dropout_max=0.0, trap_cond_mode=mode)
        calls = {"n": 0}
        fwd = o.model.forward

        def counting(b):
            calls["n"] += 1
            return fwd(b)
        o.model.forward = counting
        b = raw_batch("hvb")
        b["completion"] = ["backchannel"]
        for step in range(5):
            calls["n"] = 0
            _, loss, parts = o._trap_step(b, 0, step + 2)
            assert calls["n"] == 1 and "cond_diff" not in parts, (mode, parts)
        assert o._trap_run["skip_cond_empty"] == 5 and o._trap_run["skip_cond_words"] == 0, mode
        assert "cond_empty" in o._trap_summary(o._trap_run)
    ok("all-deciding gold definition: conditional term skipped (no empty line, one forward pass), "
       "counted as skip_cond_empty, in both deciding and random modes")


def test_e2_fixed_row():
    """E2 fix: definition_log_probs(..., input_len=) reads the true answer row."""
    model = MockQwenAudio(len(TOK))
    proc = _Proc(TOK)
    for key in ("cremad", "hvb", "minds14_en"):
        b = raw_batch(key)
        tb = proc.tokenize_batch(b["prompt"], b["completion"])
        out = model(tb)
        logits = out["logits"]
        c_ids = TOK(f"{b['completion'][0]}{TOK.eos_token}", add_special_tokens=False).input_ids
        plen = int(tb["prompt_length"][0])
        true_row = logits.size(1) - len(c_ids) - 1
        # The shifted row is exactly the row that predicts the first answer token.
        shift = logits.size(1) - tb["input_ids"].size(1)
        assert plen - 1 + shift == true_row, (key, plen - 1 + shift, true_row)
        # Its log-prob of the gold first token equals the model's own NLL term for that token.
        model_lp0 = token_logprobs(logits, c_ids)[0]
        k = torch.tensor([[c_ids[0], 0]])                  # gold first token + a dummy candidate
        row = logits[0, true_row].float()
        expected = F.log_softmax(row[k[0]], -1)
        got = definition_log_probs(logits, [plen], k, input_len=tb["input_ids"].size(1))[0]
        assert torch.allclose(got, expected, atol=1e-6), (key, got, expected)
        assert torch.allclose(F.log_softmax(row, -1)[c_ids[0]], model_lp0, atol=1e-5)
        # Old call (no input_len) is unchanged and still reads the wrong row.
        old = definition_log_probs(logits, [plen], k)[0]
        assert not torch.allclose(old, expected), key
    # Batch size > 1 is refused: one shift cannot be right for different audio lengths.
    try:
        definition_log_probs(torch.zeros(2, 10, 5), [3, 3], torch.zeros(2, 2, dtype=torch.long), input_len=8)
        raise AssertionError("expected ValueError for batch size 2")
    except ValueError:
        pass
    # The orchestrator wiring passes input_len at both call sites, and init guards batch size.
    src = open("models/symbolAdapter/symbol_training.py").read()
    assert src.count("input_len=batch_a[\"input_ids\"].size(1)") == 1
    assert src.count("input_len=batch_b[\"input_ids\"].size(1)") == 1
    assert "consistency needs batch_size 1" in src
    ok("E2 fix: input_len shift reads the true answer row (matches model token log-prob); "
       "batch>1 refused; both call sites wired")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nALL {len(PASSED)} TESTS PASSED")
